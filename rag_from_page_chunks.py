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

from rank_bm25 import BM25Okapi
import jieba, re

def _preprocess(doc: str) -> str:
    # 切分 6-8 位数字（日期/代码），弱化连字符/下划线
    doc = re.sub(r'(\d{6,8})', r' \1 ', doc)
    doc = doc.replace('-', ' ').replace('_', ' ')
    return doc

class FileBM25Index:
    """
    文件级 BM25：用 文件名 + 前3页文本 + 标题/目录 行 来增强“版本可分性”
    """
    def __init__(self, pages_per_file: int = 3, per_page_chars: int = 1500,
                 repeat_filename_tokens: int = 2):
        self.pages_per_file = pages_per_file
        self.per_page_chars = per_page_chars
        self.repeat_filename_tokens = repeat_filename_tokens
        self.files = []
        self.tokens = []
        self.fn_to_idx = {}
        self.bm25 = None

    def build(self, chunks):
        # 收集每文件的若干最早页 & 抓取标题行（MinerU markdown 常有 #/##/表/图）
        pages_by_file = {}
        titles_by_file = {}
        for c in chunks:
            fn = c["metadata"]["file_name"]
            pg = int(c["metadata"]["page"])
            txt = c["content"]
            # 标题/目录/图表行
            titles = "\n".join(
                ln for ln in txt.splitlines()
                if ln.lstrip().startswith(('#','##','表','图'))
            )
            if titles:
                titles_by_file.setdefault(fn, []).append(titles[:500])

            pages_by_file.setdefault(fn, {})
            # 只收每页一次
            if pg not in pages_by_file[fn]:
                pages_by_file[fn][pg] = txt

        self.files, self.tokens, self.fn_to_idx = [], [], {}
        for fn, pgmap in pages_by_file.items():
            first_pages = [pgmap[p] for p in sorted(pgmap)[: self.pages_per_file]]
            titles = "\n".join(titles_by_file.get(fn, []))[:1500]
            # 文档：文件名 + 标题/目录摘要 + 前3页各1500字
            doc = f"{fn}\n{titles}\n" + "\n".join(fp[: self.per_page_chars] for fp in first_pages)
            doc = _preprocess(doc)
            toks = list(jieba.cut(doc))
            # 额外重复文件名分词，放大“版本词/日期码”权重
            toks += list(jieba.cut(_preprocess(fn))) * self.repeat_filename_tokens

            self.fn_to_idx[fn] = len(self.files)
            self.files.append(fn)
            self.tokens.append(toks)

        self.bm25 = BM25Okapi(self.tokens)

    def top_files(self, query, n=25):
        if not self.bm25:
            return []
        q = list(jieba.cut(_preprocess(query)))
        scores = self.bm25.get_scores(q)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(self.files[i], float(scores[i])) for i in order[:n]]

import re
def rrf(rank, k=90):
    """
    Reciprocal Rank Fusion 基本项。
    约定 rank 为 0-based，这里内部转为 1-based 以避免极端放大。
    k 越大越“温和”，推荐 90。
    """
    r = int(rank) + 1
    return 1.0 / (k + r)


THEME_TOKENS = ["二次创业","三条增长曲线","深度复盘","稳定成长","公司深度报告","再谈","走在修复的道路上"]
def extract_years(q): return set(re.findall(r"20\d{2}", q))

def soft_bias(ranked, question, year_boost=0.15, theme_boost=0.05):
    years = extract_years(question)
    themed = any(t in question for t in THEME_TOKENS)
    for c in ranked:
        fn = c["metadata"]["file_name"]
        s  = c.get("rerank_score", 0.0)
        if years and any(y in fn for y in years): s += year_boost
        if themed: s += theme_boost if any(t in fn for t in THEME_TOKENS) else 0.0
        c["rerank_score"] = s
    ranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return ranked

def soft_bias_filewise(ranked, question, year_boost=0.03, theme_boost=0.01):
    years = set(re.findall(r"20\d{2}", question))
    THEME_TOKENS = ["二次创业","三条增长曲线","深度复盘","稳定成长","公司深度报告","再谈","走在修复的道路上"]
    file_bonus = {}
    for c in ranked:
        fn = c["metadata"]["file_name"]
        if fn in file_bonus: 
            continue
        b = 0.0
        if years and any(y in fn for y in years): 
            b += year_boost
        if any(t in fn for t in THEME_TOKENS):   
            b += theme_boost
        file_bonus[fn] = b
    for c in ranked:
        fn = c["metadata"]["file_name"]
        c["rerank_score"] = float(c.get("rerank_score", 0.0)) + file_bonus.get(fn, 0.0)
    ranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return ranked



from collections import defaultdict
import numpy as np

def file_vote(ranked_chunks, top_n_files=1, agg="sum"):
    buckets = defaultdict(list)
    for c in ranked_chunks:
        fn = c["metadata"]["file_name"]
        s  = c.get("rerank_score", c.get("ret_score", 0.0))
        buckets[fn].append(float(s))

    def agg_fn(v):
        v = np.asarray(v, float)
        return v.sum() if agg=="sum" else (v.mean() if agg=="mean" else v.max())

    scored = sorted([(fn, agg_fn(v)) for fn, v in buckets.items()], key=lambda x: x[1], reverse=True)
    keep = set(fn for fn,_ in scored[:top_n_files])
    kept = [c for c in ranked_chunks if c["metadata"]["file_name"] in keep]
    return kept, (scored[0][0] if scored else None, dict(scored))


from collections import defaultdict
import numpy as np

def page_vote(ranked_chunks, top_m_pages=1, agg="max"):
    """
    ranked_chunks: list already reranked by your reranker (desc)
    Aggregate scores per (file_name, page) and return chunks from the winning page(s).
    """
    buckets = defaultdict(list)
    for c in ranked_chunks:
        key = (c["metadata"]["file_name"], int(c["metadata"]["page"]))
        s = c.get("rerank_score", c.get("ret_score", 0.0))
        buckets[key].append(float(s))

    def agg_fn(v):
        v = np.asarray(v, float)
        if agg == "sum": return v.sum()
        if agg == "mean": return v.mean()
        return v.max()

    scored_pages = sorted([(k, agg_fn(v)) for k, v in buckets.items()], key=lambda x: x[1], reverse=True)
    keep_keys = set(k for k,_ in scored_pages[:top_m_pages])
    return [c for c in ranked_chunks if (c["metadata"]["file_name"], int(c["metadata"]["page"])) in keep_keys]

def expand_neighbors_on_page(page_chunks, radius=1, max_total=8):
    # Assumes all chunks are from the same (file,page); use their widx to add neighbors
    page_chunks = sorted(page_chunks, key=lambda x: x["metadata"].get("widx", 0))
    out = []
    picked = set()
    for c in page_chunks[:max_total]:
        w = c["metadata"].get("widx", 0)
        for j in range(max(0, w - radius), min(len(page_chunks), w + radius + 1)):
            cj = page_chunks[j]
            key = cj["metadata"].get("widx", j)
            if key in picked: 
                continue
            picked.add(key); out.append(cj)
            if len(out) >= max_total: 
                return out
    return out or page_chunks[:max_total]


import os, re, time, requests

class SiliconFlowReranker:
    def __init__(self, api_key: str = None,
                 model: str = "BAAI/bge-reranker-v2-m3",
                 endpoint: str = "https://api.siliconflow.cn/v1/rerank",
                 timeout: int = 30,
                 max_retries: int = 4,
                 add_header: bool = True,
                 header_snippet_chars: int = 300,
                 body_max_chars: int = 800):
        self.api_key = api_key or os.getenv('LOCAL_API_KEY')
        if not self.api_key:
            raise ValueError("Missing SILICONFLOW_API_KEY")
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout
        self.max_retries = max_retries

        # NEW
        self.add_header = add_header
        self.header_snippet_chars = header_snippet_chars
        self.body_max_chars = body_max_chars

    def _extract_date_from_filename(self, fn: str) -> str:
        """
        Try to parse YYYY[-_.]?MM[-_.]?DD in filename and normalize to YYYY-MM-DD.
        Returns '' if not found.
        """
        m = re.search(r'(20\d{2})[-_.]?(0[1-9]|1[0-2])[-_.]?(0[1-9]|[12]\d|3[01])', fn)
        if not m: return ''
        y, mo, d = m.groups()
        return f"{y}-{mo}-{d}"

    def _make_header(self, c: dict) -> str:
        meta = c.get("metadata", {})
        fn = str(meta.get("file_name", ""))
        pg = str(meta.get("page", ""))
        dt = self._extract_date_from_filename(fn)
        snippet = c.get("content", "")[: self.header_snippet_chars]
        parts = [f"【文件】{fn}", f"【页】{pg}"]
        if dt: parts.append(f"【日期】{dt}")
        parts.append(f"【摘要】{snippet}")
        return "\n".join(parts) + "\n"

    def rerank(self, question: str, candidates: list, return_meta: bool = False):
        """
        Returns:
          ranked_chunks, meta where meta = {"status": http_status or None, "attempts": N}
        Retries on 429/5xx with exponential backoff.
        """
        # Build documents (with optional header)
        if self.add_header:
            documents = []
            for c in candidates:
                header = self._make_header(c)               # uses header_snippet_chars already
                body   = c["content"][: self.body_max_chars]
                documents.append(header + body)
        else:
            documents = [c["content"][: self.body_max_chars] for c in candidates]

        payload = {"model": self.model, "query": question, "documents": documents}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        status = None
        attempts = 0
        data = None

        for attempt in range(self.max_retries):
            attempts = attempt + 1
            try:
                r = requests.post(self.endpoint, json=payload, headers=headers, timeout=self.timeout)
                status = r.status_code
                if status in (429, 500, 502, 503, 504):
                    time.sleep(2 ** attempt)  # backoff
                    continue
                r.raise_for_status()
                data = r.json()
                break
            except requests.RequestException:
                time.sleep(2 ** attempt)

        ranked = []
        if data:
            results = data.get("results") or data.get("data") or []
            seen = set()
            for res in results:
                idx = res.get("index")
                if idx is None or idx >= len(candidates):
                    continue
                seen.add(idx)
                c = dict(candidates[idx])
                c["rerank_score"] = float(res.get("relevance_score", 0.0))
                ranked.append(c)
            # fill any missing
            for i, c in enumerate(candidates):
                if i not in seen:
                    cc = dict(c); cc["rerank_score"] = 0.0
                    ranked.append(cc)
            ranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        meta = {"status": status, "attempts": attempts}
        return (ranked, meta) if return_meta else ranked



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


from collections import defaultdict

def pick_best_page_and_expand(file_scoped, final_k=5, radius=1):
    # group by page
    by_page = defaultdict(list)
    for c in file_scoped:
        by_page[int(c["metadata"]["page"])].append(c)
    # score each page by max rerank_score
    page_scores = [(pg, max(float(x.get("rerank_score", 0.0)) for x in cs))
                   for pg, cs in by_page.items()]
    best_pg, _ = max(page_scores, key=lambda t: t[1])

    page_chunks = sorted(by_page[best_pg], key=lambda x: x["metadata"].get("widx", 0))
    # expand around the highest-scored chunk on that page
    top_idx = max(range(len(page_chunks)),
                  key=lambda i: float(page_chunks[i].get("rerank_score", 0.0)))
    w0 = page_chunks[top_idx]["metadata"].get("widx", top_idx)

    # collect neighbors by widx
    wanted = {w0}
    for r in range(1, radius+1):
        wanted.add(w0 - r); wanted.add(w0 + r)

    chosen = [c for c in page_chunks if c["metadata"].get("widx", -10**9) in wanted]
    # backfill with other chunks on same page if needed
    if len(chosen) < final_k:
        for c in page_chunks:
            if c not in chosen:
                chosen.append(c)
            if len(chosen) >= final_k: break
    return chosen[:final_k]


from typing import Dict, Any, List, Optional

class SimpleRAG:
    def __init__(
        self,
        chunk_json_path: str,
        model_path: str = None,
        batch_size: int = 32,
        use_rerank: bool = False,
        candidate_k: int = 120,
        final_k: int = 5,
        reranker=None,
        
    ):
        self.loader = PageChunkLoader(chunk_json_path)
        self.embedding_model = EmbeddingModel(batch_size=batch_size)
        self.vector_store = SimpleVectorStore()
        

        # Rerank controls
        self.use_rerank = use_rerank
        self.candidate_k = candidate_k
        self.final_k = final_k
        self.reranker = reranker
        

        # Behavior flags
        self.lock_source     = bool(int(os.getenv("LOCK_SOURCE",  "0")))
        self.strict_rerank   = bool(int(os.getenv("STRICT_RERANK","0")))
        self.debug_telemetry = bool(int(os.getenv("DEBUG_RAG",    "1")))

        self.limit_per_file = int(os.getenv("LIMIT_PER_FILE", "12"))


        # # Retrieval helpers
        # self.file_bm25 = FileBM25Index(pages_per_file=3, per_page_chars=1500, repeat_filename_tokens=2)
        # self.file_bm25.build(self.chunks)


        # Hold corpus if needed elsewhere
        self.chunks = None                 # <-- set in setup()

    def setup(self):
        print("加载所有页chunk...")
        self.chunks = self.loader.load_chunks()
        print(f"共加载 {len(self.chunks)} 个chunk")

        print("生成嵌入...")
        embeddings = self.embedding_model.embed_texts([c["content"] for c in self.chunks])

        print("存储向量...")
        self.vector_store.add_chunks(self.chunks, embeddings)

        # Build file-level BM25 now that chunks exist
        print("构建文件级BM25索引...")
        self.file_bm25 = FileBM25Index(pages_per_file=3, per_page_chars=1500, repeat_filename_tokens=2)
        self.file_bm25.build(self.chunks)

        print("RAG向量库构建完成！")

    def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        q_emb = self.embedding_model.embed_text(question)
        results = self.vector_store.search(q_emb, top_k)
        return {"question": question, "chunks": results}

    def _build_context(self, items: List[Dict[str, Any]]) -> str:
        return "\n".join(
            f"[文件名]{c['metadata']['file_name']} [页码]{c['metadata']['page']}\n{c['content']}"
            for c in items
        )

    def generate_answer(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        qwen_api_key = os.getenv("LOCAL_API_KEY")
        qwen_base_url = os.getenv("LOCAL_BASE_URL")
        qwen_model    = os.getenv("LOCAL_TEXT_MODEL")
        if not qwen_api_key or not qwen_base_url or not qwen_model:
            raise ValueError("请在.env中配置LOCAL_API_KEY、LOCAL_BASE_URL、LOCAL_TEXT_MODEL")

        tele = {"path":"", "rerank_ok":False, "rerank_attempts":0, "rerank_http":None,
                "cand_n":0, "ranked_n":0, "file_vote_best":None, "file_vote_n":0, "final_n":0}

        # Stage-A dense
        q_emb = self.embedding_model.embed_text(question)
        candidates = self.vector_store.search(q_emb, self.candidate_k)
        tele["cand_n"] = len(candidates)

        # Optional per-file diversity cap
        if candidates and self.limit_per_file > 0:
            by_file, diverse = {}, []
            for c in candidates:
                fn = c["metadata"]["file_name"]
                by_file.setdefault(fn, 0)
                if by_file[fn] < self.limit_per_file:
                    diverse.append(c); by_file[fn] += 1
            candidates = diverse

        # RRF fuse with file-level BM25 (only if built)
        if self.file_bm25 and candidates:
            # 记录 BEFORE (先存，再融合)
            pre_rrf_files = list(dict.fromkeys([c["metadata"]["file_name"] for c in candidates]))
            tele["first3_files_before_rrf"] = pre_rrf_files[:3]

            # 参数
            BM25_TOPN = int(os.getenv("BM25_TOPN", "25"))
            RRF_K     = int(os.getenv("RRF_K", "90"))

            # BM25 文件排名
            top_files = self.file_bm25.top_files(question, n=BM25_TOPN)
            tele["bm25_top_files"] = [fn for fn, _ in top_files[:5]]

            # 融合（统一用 RRF_K；rank 约定 0-based -> rrf 内部会 +1）
            file_to_rank = {fn: r for r, (fn, _) in enumerate(top_files)}
            BIG = 10_000

            fused = []
            for i, c in enumerate(candidates):
                fn = c["metadata"]["file_name"]
                dense_rank = i
                bm25_rank  = file_to_rank.get(fn, BIG)
                score = rrf(dense_rank, k=RRF_K) + rrf(bm25_rank, k=RRF_K)
                cc = dict(c); cc["fused_score"] = score
                fused.append(cc)

            candidates = sorted(fused, key=lambda x: x["fused_score"], reverse=True)

        # 记录 AFTER（放在 if 外也行，candidates 一定存在）
        tele["first3_files_after_rrf"] = list(dict.fromkeys([c["metadata"]["file_name"] for c in candidates]))[:3]

        # ---- Edition router: 文件级 rerank（先选报告/版本） --------------------
        # 从 RRF 后的 candidates 里，每个文件取一个代表 chunk（首次出现即可）
        by_file = {}
        for c in candidates:
            fn = c["metadata"]["file_name"]
            if fn not in by_file:
                by_file[fn] = c
        file_profiles = list(by_file.values())

        # 调用同一个 reranker 做“文件级”打分（它会看到你的 header+日期等关键信息）
        file_ranked, _ = self.reranker.rerank(question, file_profiles, return_meta=True)
        if file_ranked:
            chosen_file = file_ranked[0]["metadata"]["file_name"]
            tele["file_router_top1"] = chosen_file
            # 之后的页面级流程只在该文件内部进行
            candidates = [c for c in candidates if c["metadata"]["file_name"] == chosen_file]
            tele["unique_files_after_router"] = list({c["metadata"]["file_name"] for c in candidates})
        else:
            tele["file_router_top1"] = None
        # -----------------------------------------------------------------------



        # Rerank → soft bias → file_vote → neighbor expansion
        if self.use_rerank and self.reranker is not None and candidates:
            tele["path"] = "rerank"
            ranked, rr_meta = self.reranker.rerank(question, candidates, return_meta=True)
            tele["rerank_http"] = rr_meta.get("status"); tele["rerank_attempts"] = rr_meta.get("attempts")
            tele["ranked_n"] = len(ranked); tele["rerank_ok"] = bool(ranked)

            if not ranked:
                if self.strict_rerank:
                    raise RuntimeError(f"Rerank failed (status={tele['rerank_http']}) and STRICT_RERANK=1")
                chunks = candidates[: self.final_k]
            else:
                # 可选：微弱文件级 bias，而不是逐 chunk 大幅 bias
                ranked = soft_bias_filewise(ranked, question)
                # 在“被路由出的文件”的页面里选最佳页并邻域扩展
                chunks = pick_best_page_and_expand(ranked, final_k=self.final_k, radius=1)


                # no page_vote; just take neighbors within the chosen file
                # try:
                #     chunks = expand_neighbors_on_page(file_scoped, radius=1, max_total=self.final_k)
                # except Exception:
                #     chunks = file_scoped[: self.final_k]
        else:
            tele["path"] = "baseline"
            chunks = candidates[: top_k]

        tele["final_n"] = len(chunks)

        # Early exit
        if not chunks:
            out = {"question": question, "answer": "", "filename": "", "page": "", "retrieval_chunks": []}
            if self.debug_telemetry: out["debug"] = tele
            return out

        # Evidence for LOCK_SOURCE
        top_file = chunks[0]['metadata']['file_name']
        top_page = chunks[0]['metadata']['page']

        # Build context
        context = self._build_context(chunks)

        # LLM call
        client = OpenAI(api_key=qwen_api_key, base_url=qwen_base_url)
        prompt = (
            "你是一名专业的金融分析助手，请根据以下检索到的内容回答用户问题。\n"
            "请严格按照如下JSON格式输出：\n"
            '{"answer": "你的简洁回答", "filename": "来源文件名", "page": "来源页码"}\n'
            f"检索内容：\n{context}\n\n问题：{question}\n"
            "请确保输出内容为合法JSON字符串，不要输出多余内容。"
        )
        completion = client.chat.completions.create(
            model=qwen_model,
            messages=[{"role": "system", "content": "你是一名专业的金融分析助手。"},
                      {"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=1024,
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
                    answer = j.get("answer", ""); filename = j.get("filename", ""); page = j.get("page", "")
                else:
                    answer, filename, page = raw, top_file, top_page
            except Exception:
                answer, filename, page = raw, top_file, top_page
        else:
            answer, filename, page = raw, top_file, top_page

        # normalize page & apply offset
        PAGE_OFFSET = int(os.getenv("PAGE_OFFSET", "0"))
        try:
            if page not in ("", None): page = int(page) + PAGE_OFFSET
        except Exception:
            try: page = int(top_page) + PAGE_OFFSET
            except Exception: page = top_page

        # LOCK_SOURCE
        model_file, model_page = filename, page
        if self.lock_source and chunks:
            filename, page = top_file, (int(top_page) + PAGE_OFFSET if str(top_page).isdigit() else top_page)

        out = {"question": question, "answer": answer, "filename": filename, "page": page, "retrieval_chunks": chunks}
        if self.debug_telemetry:
            tele.update({"top_file": top_file, "top_page": top_page, "model_file": model_file, "model_page": model_page})
            out["debug"] = tele
        return out





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
    chunk_json_path = "./all_pdf_windows_mineru.json"
    reranker = SiliconFlowReranker(
    add_header=True,
    header_snippet_chars=int(os.getenv("HEADER_SNIPPET","120")),
    body_max_chars=int(os.getenv("RERANK_BODY_MAX","800")),
)
    
    #rag = SimpleRAG(chunk_json_path)
    rag = SimpleRAG(
    chunk_json_path=chunk_json_path,  # or your page/block chunk file
    use_rerank=True,
    candidate_k=100,
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
