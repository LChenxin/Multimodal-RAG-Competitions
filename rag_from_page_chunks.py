import json
import os

import hashlib
from typing import List, Dict, Any
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(__file__))
from get_text_embedding import get_text_embedding

from dotenv import load_dotenv
from openai import OpenAI
# 统一加载项目根目录的.env
load_dotenv()

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
    def __init__(self, chunk_json_path: str, model_path: str = None, batch_size: int = 32):
        self.loader = PageChunkLoader(chunk_json_path)
        self.embedding_model = EmbeddingModel(batch_size=batch_size)
        self.vector_store = SimpleVectorStore()
    def setup(self):
        print("加载所有页chunk...")
        chunks = self.loader.load_chunks()
        print(f"共加载 {len(chunks)} 个chunk")
        print("生成嵌入...")
        embeddings = self.embedding_model.embed_texts([c['content'] for c in chunks])
        print("存储向量...")
        self.vector_store.add_chunks(chunks, embeddings)
        print("RAG向量库构建完成！")
    def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        q_emb = self.embedding_model.embed_text(question)
        results = self.vector_store.search(q_emb, top_k)
        return {
            "question": question,
            "chunks": results
        }

    def generate_answer(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """
        检索+大模型生成式回答，返回结构化结果
        """
        qwen_api_key = os.getenv('LOCAL_API_KEY')
        qwen_base_url = os.getenv('LOCAL_BASE_URL')
        qwen_model = os.getenv('LOCAL_TEXT_MODEL')
        if not qwen_api_key or not qwen_base_url or not qwen_model:
            raise ValueError('请在.env中配置LOCAL_API_KEY、LOCAL_BASE_URL、LOCAL_TEXT_MODEL')
        q_emb = self.embedding_model.embed_text(question)
        chunks = self.vector_store.search(q_emb, top_k)
        # 拼接检索内容，带上元数据
        context = "\n".join([
            f"[文件名]{c['metadata']['file_name']} [页码]{c['metadata']['page']}\n{c['content']}" for c in chunks
        ])
        # 明确要求输出JSON格式 answer/page/filename
        prompt = (
            f"你是一名专业的金融分析助手，请根据以下检索到的内容回答用户问题。\n"
            f"请严格按照如下JSON格式输出：\n"
            f'{{"answer": "你的简洁回答", "filename": "来源文件名", "page": "来源页码"}}'"\n"
            f"检索内容：\n{context}\n\n问题：{question}\n"
            f"请确保输出内容为合法JSON字符串，不要输出多余内容。"
        )
        client = OpenAI(api_key=qwen_api_key, base_url=qwen_base_url)
        completion = client.chat.completions.create(
            model=qwen_model,
            messages=[
                {"role": "system", "content": "你是一名专业的金融分析助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1024
        )
        import json as pyjson
        sys.path.append(os.path.dirname(__file__))
        from extract_json_array import extract_json_array
        raw = completion.choices[0].message.content.strip()
        # 用 extract_json_array 提取 JSON 对象
        json_str = extract_json_array(raw, mode='objects')
        if json_str:
            try:
                arr = pyjson.loads(json_str)
                # 只取第一个对象
                if isinstance(arr, list) and arr:
                    j = arr[0]
                    answer = j.get('answer', '')
                    filename = j.get('filename', '')
                    page = j.get('page', '')
                else:
                    answer = raw
                    filename = chunks[0]['metadata']['file_name'] if chunks else ''
                    page = chunks[0]['metadata']['page'] if chunks else ''
            except Exception:
                answer = raw
                filename = chunks[0]['metadata']['file_name'] if chunks else ''
                page = chunks[0]['metadata']['page'] if chunks else ''
        else:
            answer = raw
            filename = chunks[0]['metadata']['file_name'] if chunks else ''
            page = chunks[0]['metadata']['page'] if chunks else ''
        # 结构化输出
        return {
            "question": question,
            "answer": answer,
            "filename": filename,
            "page": page,
            "retrieval_chunks": chunks
        }


if __name__ == '__main__':
    # 路径可根据实际情况调整
    chunk_json_path = os.path.join(os.path.dirname(__file__), 'all_pdf_page_chunks.json')
    rag = SimpleRAG(chunk_json_path)
    rag.setup()

    # 控制测试时读取的题目数量，默认只随机抽取10个，实际跑全部时设为None
    TEST_SAMPLE_NUM = 10  # 设置为None则全部跑
    FILL_UNANSWERED = True  # 未回答的也输出默认内容

    # 批量评测脚本：读取测试集，检索+大模型生成，输出结构化结果
    test_path = os.path.join(os.path.dirname(__file__), 'datas/多模态RAG图文问答挑战赛测试集.json')
    if os.path.exists(test_path):
        with open(test_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        import concurrent.futures
        import random

        # 记录所有原始索引
        all_indices = list(range(len(test_data)))
        # 随机抽取部分题目用于测试
        selected_indices = all_indices
        if TEST_SAMPLE_NUM is not None and TEST_SAMPLE_NUM > 0:
            if len(test_data) > TEST_SAMPLE_NUM:
                selected_indices = sorted(random.sample(all_indices, TEST_SAMPLE_NUM))

        def process_one(idx):
            item = test_data[idx]
            question = item['question']
            tqdm.write(f"[{selected_indices.index(idx)+1}/{len(selected_indices)}] 正在处理: {question[:30]}...")
            result = rag.generate_answer(question, top_k=5)
            return idx, result

        results = []
        if selected_indices:
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                results = list(tqdm(executor.map(process_one, selected_indices), total=len(selected_indices), desc='并发批量生成'))

        # 先输出一份未过滤的原始结果（含 idx）
        import json
        raw_out_path = os.path.join(os.path.dirname(__file__), 'rag_top1_pred_raw.json')
        with open(raw_out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f'已输出原始未过滤结果到: {raw_out_path}')

        # 只保留结果部分，并去除 retrieval_chunks 字段
        idx2result = {idx: {k: v for k, v in r.items() if k != 'retrieval_chunks'} for idx, r in results}
        filtered_results = []
        for idx, item in enumerate(test_data):
            if idx in idx2result:
                filtered_results.append(idx2result[idx])
            elif FILL_UNANSWERED:
                # 未被回答的，补默认内容
                filtered_results.append({
                    "question": item.get("question", ""),
                    "answer": "",
                    "filename": "",
                    "page": "",
                })
        # 输出结构化结果到json
        out_path = os.path.join(os.path.dirname(__file__), 'rag_top1_pred.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_results, f, ensure_ascii=False, indent=2)
        print(f'已输出结构化检索+大模型生成结果到: {out_path}')
    
        