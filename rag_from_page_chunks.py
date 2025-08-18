import json
import os

import hashlib
from typing import List, Dict, Any
from tqdm import tqdm
import sys
import concurrent.futures
import random
import torch

from get_text_embedding import get_text_embedding

from dotenv import load_dotenv
from openai import OpenAI
import time
from openai import APIConnectionError, RateLimitError, APITimeoutError, APIError


# 统一加载项目根目录的.env
load_dotenv()

import os, requests

class SiliconFlowReranker:
    def __init__(self, api_key: str = None,
                 model: str = "BAAI/bge-reranker-v2-m3",
                 endpoint: str = "https://api.siliconflow.cn/v1/rerank",
                 timeout: int = 30):
        self.api_key = os.getenv('LOCAL_API_KEY')
        if not self.api_key:
            raise ValueError("Missing SILICONFLOW_API_KEY")
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout

    def rerank(self, question: str, candidates: list):
        # candidates: list of dicts with keys: 'content', 'metadata': {'file_name','page'}
        docs = [c["content"] for c in candidates]
        payload = {"model": self.model, "query": question, "documents": docs}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        r = requests.post(self.endpoint, json=payload, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()

        # SiliconFlow returns a list of results with {index, relevance_score}
        # (Long docs are auto-chunked server-side; highest sub-chunk score is used per doc.) :contentReference[oaicite:1]{index=1}
        results = data.get("results") or data.get("data") or []
        # attach scores back to your candidate objects
        scored = []
        for res in results:
            idx = res.get("index")
            if idx is None or idx >= len(candidates): 
                continue
            c = dict(candidates[idx])  # shallow copy
            c["rerank_score"] = float(res.get("relevance_score", 0.0))
            scored.append(c)

        # If API didn’t return for all docs, keep the rest with score 0
        seen = {res.get("index") for res in results if res.get("index") is not None}
        for i, c in enumerate(candidates):
            if i not in seen:
                cc = dict(c)
                cc["rerank_score"] = cc.get("rerank_score", 0.0)
                scored.append(cc)

        return sorted(scored, key=lambda x: x["rerank_score"], reverse=True)


class PageChunkLoader:
    def __init__(self, json_path: str):
        self.json_path = json_path
    def load_chunks(self) -> List[Dict[str, Any]]:
        with open(self.json_path, 'r', encoding='utf-8') as f:
            return json.load(f)


class EmbeddingModel:
    def __init__(self, batch_size: int = 64):
        self.api_key = os.getenv('LOCAL_API_KEY')
        self.base_url = os.getenv('LOCAL_BASE_URL')
        self.embedding_model = os.getenv('LOCAL_EMBEDDING_MODEL')
        self.batch_size = batch_size
        if not self.api_key or not self.base_url:
            raise ValueError('请在.env中配置LOCAL_API_KEY和LOCAL_BASE_URL')

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return get_text_embedding(
            texts,
            api_key=self.api_key,
            base_url=self.base_url,
            embedding_model=self.embedding_model,
            batch_size=self.batch_size
        )

    def embed_text(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]
    
class SimpleVectorStore:
    def __init__(self):
        self.embeddings = []
        self.chunks = []
    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        self.chunks.extend(chunks)
        self.embeddings.extend(embeddings)
    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        from numpy import dot
        from numpy.linalg import norm
        import numpy as np
        if not self.embeddings:
            return []
        emb_matrix = np.array(self.embeddings)
        query_emb = np.array(query_embedding)
        sims = emb_matrix @ query_emb / (norm(emb_matrix, axis=1) * norm(query_emb) + 1e-8)
        idxs = sims.argsort()[::-1][:top_k]
        return [self.chunks[i] for i in idxs]


class SimpleRAG:
    def __init__(
        self,
        chunk_json_path: str,
        model_path: str = None,
        batch_size: int = 32,
        use_rerank: bool = False,
        candidate_k: int = 120,
        final_k: int = 5,
        reranker=None,  # expect your HybridReranker instance
    ):
        self.loader = PageChunkLoader(chunk_json_path)
        self.embedding_model = EmbeddingModel(batch_size=batch_size)
        self.vector_store = SimpleVectorStore()

        # Rerank controls
        self.use_rerank = use_rerank
        self.candidate_k = candidate_k
        self.final_k = final_k
        self.reranker = reranker

    def setup(self):
        print("加载所有页chunk...")
        chunks = self.loader.load_chunks()
        print(f"共加载 {len(chunks)} 个chunk")
        print("生成嵌入...")
        embeddings = self.embedding_model.embed_texts([c["content"] for c in chunks])
        print("存储向量...")
        self.vector_store.add_chunks(chunks, embeddings)
        print("RAG向量库构建完成！")

    def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        q_emb = self.embedding_model.embed_text(question)
        results = self.vector_store.search(q_emb, top_k)
        return {"question": question, "chunks": results}

    def _build_context(self, items: List[Dict[str, Any]]) -> str:
        return "\n".join(
            [
                f"[文件名]{c['metadata']['file_name']} [页码]{c['metadata']['page']}\n{c['content']}"
                for c in items
            ]
        )

    def generate_answer(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """
        检索+大模型生成式回答，返回结构化结果
        """
        qwen_api_key = os.getenv("LOCAL_API_KEY")
        qwen_base_url = os.getenv("LOCAL_BASE_URL")
        qwen_model = os.getenv("LOCAL_TEXT_MODEL")
        if not qwen_api_key or not qwen_base_url or not qwen_model:
            raise ValueError("请在.env中配置LOCAL_API_KEY、LOCAL_BASE_URL、LOCAL_TEXT_MODEL")

        # ------ Retrieval (+ optional rerank) ------
        q_emb = self.embedding_model.embed_text(question)

        if self.use_rerank and self.reranker is not None:
            # Stage A: larger pool
            candidates = self.vector_store.search(q_emb, self.candidate_k)
            if candidates:
                # Stage B: rerank
                ranked = self.reranker.rerank(question, candidates)
                chunks = ranked[: self.final_k]
            else:
                chunks = []
        else:
            # Baseline
            chunks = self.vector_store.search(q_emb, top_k)

        # Build LLM context
        context = self._build_context(chunks)

        # ------ LLM call (unchanged) ------
        prompt = (
            "你是一名专业的金融分析助手，请根据以下检索到的内容回答用户问题。\n"
            "请严格按照如下JSON格式输出：\n"
            '{"answer": "你的简洁回答", "filename": "来源文件名", "page": "来源页码"}\n'
            f"检索内容：\n{context}\n\n问题：{question}\n"
            "请确保输出内容为合法JSON字符串，不要输出多余内容。"
        )

        client = OpenAI(api_key=qwen_api_key, base_url=qwen_base_url)
        completion = client.chat.completions.create(
            model=qwen_model,
            messages=[
                {"role": "system", "content": "你是一名专业的金融分析助手。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=1024,
        )

        import json as pyjson
        from extract_json_array import extract_json_array

        raw = completion.choices[0].message.content.strip()
        json_str = extract_json_array(raw, mode="objects")

        if json_str:
            try:
                arr = pyjson.loads(json_str)
                if isinstance(arr, list) and arr:
                    j = arr[0]
                    answer = j.get("answer", "")
                    filename = j.get("filename", "")
                    page = j.get("page", "")
                else:
                    answer = raw
                    filename = chunks[0]["metadata"]["file_name"] if chunks else ""
                    page = chunks[0]["metadata"]["page"] if chunks else ""
            except Exception:
                answer = raw
                filename = chunks[0]["metadata"]["file_name"] if chunks else ""
                page = chunks[0]["metadata"]["page"] if chunks else ""
        else:
            answer = raw
            filename = chunks[0]["metadata"]["file_name"] if chunks else ""
            page = chunks[0]["metadata"]["page"] if chunks else ""

        # ------ Source fallback/override from best evidence (place it HERE) ------
        if chunks:
            best_file = chunks[0]["metadata"]["file_name"]
            best_page = chunks[0]["metadata"]["page"]
            # Fallback: only fill if model didn't provide values
            if not filename:
                filename = best_file
            if not page:
                page = best_page
            # If you prefer to ALWAYS trust retrieval/rerank for grading:
            # filename, page = best_file, best_page

        # Final return
        return {
            "question": question,
            "answer": answer,
            "filename": filename,
            "page": page,
            "retrieval_chunks": chunks,
        }


# if __name__ == '__main__':
#     # 路径可根据实际情况调整
#     chunk_json_path = os.path.join(os.path.dirname(__file__), 'all_pdf_page_chunks.json')
#     rag = SimpleRAG(chunk_json_path)
#     rag.setup()

#     # 控制测试时读取的题目数量，默认只随机抽取10个，实际跑全部时设为None
#     TEST_SAMPLE_NUM = None  # 设置为None则全部跑
#     FILL_UNANSWERED = True  # 未回答的也输出默认内容

#     # 批量评测脚本：读取测试集，检索+大模型生成，输出结构化结果
#     test_path = os.path.join(os.path.dirname(__file__), 'datas/多模态RAG图文问答挑战赛测试集.json')
#     if os.path.exists(test_path):
#         with open(test_path, 'r', encoding='utf-8') as f:
#             test_data = json.load(f)
#         import concurrent.futures
#         import random

#         # 记录所有原始索引
#         all_indices = list(range(len(test_data)))
#         # 随机抽取部分题目用于测试
#         selected_indices = all_indices
#         if TEST_SAMPLE_NUM is not None and TEST_SAMPLE_NUM > 0:
#             if len(test_data) > TEST_SAMPLE_NUM:
#                 selected_indices = sorted(random.sample(all_indices, TEST_SAMPLE_NUM))

#         def process_one(idx):
#             item = test_data[idx]
#             question = item['question']
#             tqdm.write(f"[{selected_indices.index(idx)+1}/{len(selected_indices)}] 正在处理: {question[:30]}...")
#             result = rag.generate_answer(question, top_k=5)
#             return idx, result

#         results = []
#         if selected_indices:
#             with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
#                 results = list(tqdm(executor.map(process_one, selected_indices), total=len(selected_indices), desc='并发批量生成'))

#         # 先输出一份未过滤的原始结果（含 idx）
#         import json
#         raw_out_path = os.path.join(os.path.dirname(__file__), 'rag_top1_pred_raw.json')
#         with open(raw_out_path, 'w', encoding='utf-8') as f:
#             json.dump(results, f, ensure_ascii=False, indent=2)
#         print(f'已输出原始未过滤结果到: {raw_out_path}')

#         # 只保留结果部分，并去除 retrieval_chunks 字段
#         idx2result = {idx: {k: v for k, v in r.items() if k != 'retrieval_chunks'} for idx, r in results}
#         filtered_results = []
#         for idx, item in enumerate(test_data):
#             if idx in idx2result:
#                 filtered_results.append(idx2result[idx])
#             elif FILL_UNANSWERED:
#                 # 未被回答的，补默认内容
#                 filtered_results.append({
#                     "question": item.get("question", ""),
#                     "answer": "",
#                     "filename": "",
#                     "page": "",
#                 })
#         # 输出结构化结果到json
#         out_path = os.path.join(os.path.dirname(__file__), 'rag_top1_pred.json')
#         with open(out_path, 'w', encoding='utf-8') as f:
#             json.dump(filtered_results, f, ensure_ascii=False, indent=2)
#         print(f'已输出结构化检索+大模型生成结果到: {out_path}')
    
        
        
if __name__ == '__main__':
    from pathlib import Path

    # 路径可根据实际情况调整
    chunk_json_path = "./all_pdf_page_chunks.json"
    reranker = SiliconFlowReranker()
    
    #rag = SimpleRAG(chunk_json_path)
    rag = SimpleRAG(
    chunk_json_path=chunk_json_path,  # or your page/block chunk file
    use_rerank=True,
    candidate_k=60,
    final_k=5,
    reranker=reranker,
)
    rag.setup()

    # 控制测试时读取的题目数量，默认只随机抽取10个，实际跑全部时设为None
    TEST_SAMPLE_NUM = None  # 设置为None则全部跑
    FILL_UNANSWERED = True  # 未回答的也输出默认内容

    # 批量评测脚本：读取测试集，检索+大模型生成，输出结构化结果
    test_path = "./datas/test.json"
    if not os.path.exists(test_path):
        print("datas/test.json 不存在")
        sys.exit(1)

    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # ============= 新增：通过“索引区间”来只处理一段 =============
    # 支持三种设置方式：
    # 1) 环境变量：BATCH_START, BATCH_SIZE
    # 2) 命令行：python rag_from_page_chunks.py 0 200
    # 3) 默认：全量（不建议长跑；建议分段跑）
    BATCH_START = int(os.getenv("BATCH_START", "-1"))
    BATCH_SIZE  = int(os.getenv("BATCH_SIZE",  "-1"))
    if len(sys.argv) >= 3:
        BATCH_START = int(sys.argv[1])
        BATCH_SIZE  = int(sys.argv[2])

    all_indices = list(range(len(test_data)))

    # 随机抽样（可选）
    if TEST_SAMPLE_NUM is not None and TEST_SAMPLE_NUM > 0 and len(test_data) > TEST_SAMPLE_NUM:
        all_indices = sorted(random.sample(all_indices, TEST_SAMPLE_NUM))

    # 计算本次要处理的 selected_indices
    if BATCH_START >= 0 and BATCH_SIZE > 0:
        end = min(BATCH_START + BATCH_SIZE, len(test_data))
        selected_indices = list(range(BATCH_START, end))
        print(f"仅处理索引区间 [{BATCH_START}, {end-1}] ，共 {len(selected_indices)} 条")
    else:
        selected_indices = all_indices
        print(f"未指定分段，默认处理全部：共 {len(selected_indices)} 条（不建议长跑）")

    # parts 目录：每段单独落盘
    PART_DIR = Path("./parts")
    PART_DIR.mkdir(exist_ok=True)

    def process_one(idx):
        item = test_data[idx]
        question = item['question']
        tqdm.write(f"[{selected_indices.index(idx)+1}/{len(selected_indices)}] 正在处理: {question[:30]}...")
        time.sleep(1)  # 维持你原来的最小限流
        # 如需更稳，可在这里加 try/except 返回空答以避免整批失败
        result = rag.generate_answer(question, top_k=5)
        return idx, result

    # 并发执行（保持你现有写法与并发度）
    results = []
    if selected_indices:
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = list(tqdm(
                executor.map(process_one, selected_indices),
                total=len(selected_indices),
                desc='并发批量生成'
            ))

    # —— 本段原始结果（含 idx）单独落盘 —— #
    part_tag = f"{selected_indices[0]:06d}_{selected_indices[-1]:06d}" if selected_indices else "empty"
    raw_out_path = PART_DIR / f"raw_{part_tag}.json"
    with open(raw_out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f'已输出本段原始结果: {raw_out_path}')

    # —— 本段结构化（去掉 retrieval_chunks）也单独落盘，便于最终合并 —— #
    idx2result = {idx: {k: v for k, v in r.items() if k != 'retrieval_chunks'} for idx, r in results}
    filtered_results = []
    for idx in selected_indices:
        if idx in idx2result:
            filtered_results.append(idx2result[idx])
        elif FILL_UNANSWERED:
            filtered_results.append({
                "question": test_data[idx].get("question", ""),
                "answer": "",
                "filename": "",
                "page": "",
            })

    part_out_path = PART_DIR / f"part_{part_tag}.json"
    tmp_path = str(part_out_path) + ".tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_results, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, part_out_path)
    print(f'已输出本段结构化结果: {part_out_path}')

    # ——（可选）如果你恰好跑的是全量且只想直接生成最终文件，也可以在此合并 —— #
    # 否则建议使用我提供的 merge_parts.py 独立一步合并所有 part_*.json
