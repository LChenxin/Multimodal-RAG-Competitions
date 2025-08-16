# ✅ Block-level chunking with your exact schema
import re, fitz, hashlib
from pathlib import Path
import json

CAPTION_RE = re.compile(r'^(图|表|Figure|Table)\s*[\d一二三四五六七八九十]+', re.IGNORECASE)

def _norm(s: str) -> str:
    return re.sub(r'\s+', ' ', (s or '').strip())

def extract_blocks(page, strip_header_footer=True, top_ratio=0.12, bottom_ratio=0.10):
    """Return normalized block texts (in reading order). Optionally drop header/footer."""
    blocks = page.get_text("blocks")  # [(x0,y0,x1,y1,text, ...)]
    blocks = sorted(blocks, key=lambda b: (round(b[1],1), round(b[0],1)))
    out = []
    if strip_header_footer:
        h = float(page.rect.height)
        top_y = h * top_ratio
        bot_y = h * (1 - bottom_ratio)
        for (x0, y0, x1, y1, txt, *_rest) in blocks:
            t = _norm(txt)
            if not t:
                continue
            # drop header/footer unless it looks like a caption
            if (y1 <= top_y or y0 >= bot_y) and not CAPTION_RE.match(t):
                continue
            out.append(t)
    else:
        out = [_norm(b[4]) for b in blocks if _norm(b[4])]
    return out

def attach_captions(block_texts):
    """Attach caption-looking blocks to the next block to keep figure/table context."""
    attached, i = [], 0
    while i < len(block_texts):
        t = block_texts[i]
        if CAPTION_RE.match(t) and i + 1 < len(block_texts):
            attached.append((t + " " + block_texts[i+1]).strip())
            i += 2
        else:
            attached.append(t)
            i += 1
    return attached

def merge_to_windows(block_texts, target_chars=900, overlap_chars=150):
    """Greedy merge of block texts into sliding windows with small overlap."""
    chunks, buf = [], ""
    for t in block_texts:
        if len(buf) + len(t) + 1 <= target_chars:
            buf = (buf + " " + t).strip()
        else:
            if buf:
                chunks.append(buf)
            tail = buf[-overlap_chars:] if overlap_chars and buf else ""
            buf = (tail + " " + t).strip()
    if buf:
        chunks.append(buf)
    return chunks

def process_pdfs_to_chunks(datas_dir: Path, output_json_path: Path):
    all_chunks = []
    seen = set()  

    # Set to 1 if your labels/pages are 1-based in ground truth
    PAGE_BASE = 0
    
    # 递归查找 datas_dir 目录下的所有 .pdf 文件
    pdf_files = list(datas_dir.rglob('*.pdf'))
    if not pdf_files:
        print(f"警告：在目录 '{datas_dir}' 中未找到任何 PDF 文件。")
        return

    for pdf_path in pdf_files:
        file_name_stem = pdf_path.stem
        full_file_name = pdf_path.name
        print(f"📄 正在处理: {full_file_name}")

        try:
            with fitz.open(pdf_path) as doc:
                for page_idx, page in enumerate(doc):
                    blocks = extract_blocks(page, strip_header_footer=True)
                    if not blocks:
                        continue
                    blocks = attach_captions(blocks)
                    windows = merge_to_windows(blocks, target_chars=900, overlap_chars=150)

                    for widx, w in enumerate(windows):
                        # dedup by hash of normalized text
                        h = hashlib.md5(_norm(w).encode("utf-8")).hexdigest()
                        if h in seen:
                            continue
                        seen.add(h)

                        chunk = {
                            "id": f"{file_name_stem}_page_{page_idx + PAGE_BASE}_w{widx}",
                            "content": w,
                            "metadata": {
                                "page": page_idx + PAGE_BASE,
                                "file_name": full_file_name
                            }
                        }
                        all_chunks.append(chunk)
        except Exception as e:
            print(f"⚠️ 处理文件 '{pdf_path}' 时发生错误: {e}")
    # 确保输出目录存在
    
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    # 将所有 chunks 写入一个 JSON 文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\n处理完成！所有内容已保存至: {output_json_path}")



def main():
    base_dir = Path(__file__).parent
    datas_dir = base_dir / 'datas'
    chunk_json_path = base_dir / 'all_pdf_page_chunks.json'
    
    process_pdfs_to_chunks(datas_dir, chunk_json_path)

if __name__ == '__main__':
    main()