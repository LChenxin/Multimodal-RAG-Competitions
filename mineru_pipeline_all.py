
import os
from pathlib import Path
import json
from collections import defaultdict
import asyncio
from image_utils.async_image_analysis import AsyncImageAnalysis

def parse_all_pdfs(datas_dir, output_base_dir, backend="pipeline"):
    """
    步骤1：解析所有PDF，输出内容到 data_base_json_content/
    """
    from mineru_parse_pdf import do_parse
    datas_dir = Path(datas_dir)
    output_base_dir = Path(output_base_dir)
    pdf_files = list(datas_dir.rglob('*.pdf'))
    if not pdf_files:
        print(f"未找到PDF文件于: {datas_dir}")
        return
    for pdf_path in pdf_files:
        file_name = pdf_path.stem
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        output_dir = output_base_dir / file_name
        output_dir.mkdir(parents=True, exist_ok=True)
        do_parse(
            output_dir=str(output_dir),
            pdf_file_names=[file_name],
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=["ch"],
            backend=backend,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_dump_md=False,
            f_dump_middle_json=False,
            f_dump_model_output=False,
            f_dump_orig_pdf=False,
            f_dump_content_list=True
        )
        subdir = "auto" if backend == "pipeline" else "vlm"
        print(f"已输出: {output_dir / subdir / (file_name + '_content_list.json')}")

def group_by_page(content_list):
    pages = defaultdict(list)
    for item in content_list:
        page_idx = item.get('page_idx', 0)
        pages[page_idx].append(item)
    return dict(pages)

def item_to_markdown(item, enable_image_caption=True):
    """
    enable_image_caption: 是否启用多模态视觉分析（图片caption补全），默认True。
    """
    # 默认API参数：硅基流动Qwen/Qwen2.5-VL-32B-Instruct
    vision_provider = "guiji"
    vision_model = "Pro/Qwen/Qwen2.5-VL-7B-Instruct"
    vision_api_key = os.getenv("LOCAL_API_KEY")
    vision_base_url = os.getenv("LOCAL_BASE_URL")
    
    if item['type'] == 'text':
        level = item.get('text_level', 0)
        text = item.get('text', '')
        if level == 1:
            return f"# {text}\n\n"
        elif level == 2:
            return f"## {text}\n\n"
        else:
            return f"{text}\n\n"
    elif item['type'] == 'image':
        orig_caption_list = item.get('image_caption', [])
        orig_caption = orig_caption_list[0] if orig_caption_list else ''
        img_path = item.get('img_path', '')
        
        abs_img_path = Path(img_path)
        if not abs_img_path.is_absolute():
            if img_base is not None:
                abs_img_path = (img_base / img_path).resolve()
            else:
                abs_img_path = abs_img_path.resolve()  # 尝试用当前工作目录解析        

        sem_caption = ''
        chart_facts = {}    # 结构化结果
        if enable_image_caption and img_path and os.path.exists(img_path):
            print(f"[VLM] start  -> {abs_img_path}")
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                async def get_caption():
                    async with AsyncImageAnalysis(
                        provider="guiji",
                        api_key=os.getenv("LOCAL_API_KEY"),
                        base_url=os.getenv("LOCAL_BASE_URL"),
                        vision_model="Pro/Qwen/Qwen2.5-VL-7B-Instruct"
                    ) as analyzer:
                        # 优先用“图表专用”提示词；若失败再用通用
                        from image_utils.prompts import get_chart_analysis_prompt, get_image_analysis_prompt
                        prompt = get_chart_analysis_prompt()
                        r = await analyzer.analyze_image(local_image_path=img_path, prompt=prompt)
                        if not r or ("title" not in r and "description" not in r):
                            r = await analyzer.analyze_image(
                                local_image_path=img_path,
                                prompt=get_image_analysis_prompt(20, 80)
                            )
                        return r or {}
                r = loop.run_until_complete(get_caption())
                loop.close()
                print(f"[VLM] done   -> {abs_img_path} | " f"title={bool(r.get('title'))} desc={bool(r.get('description'))}" )
                # 语义 caption（title 优先，其次 description）
                sem_caption = (r.get('title') or r.get('description') or '').strip()

                # 结构化信息（若有）
                chart_facts = {
                    "type": r.get("type") or "",
                    "series": r.get("series") or [],
                    "x_range": r.get("x_range") or "",
                    "y_unit": r.get("y_unit") or "",
                    "bullets": r.get("bullets") or []
                }

                # 回写到 item，供后续 embed_text 使用
                item["orig_caption"] = orig_caption
                item["sem_caption"] = sem_caption
                item["chart_facts"] = chart_facts

            except Exception as e:
                print(f"图片解释失败: {img_path}, {e}")

        # 渲染给人看的 MD：仍旧使用“原始 caption”为主，避免把模型结果当权威
        show_caption = orig_caption or sem_caption
        md = f"![{show_caption}]({img_path})\n\n"
        return md
    elif item['type'] == 'table':
        cap = item.get('table_caption', [])
        if isinstance(cap, list):
            caption = cap[0] if cap else ''
        elif isinstance(cap, str):
            caption = cap.strip()
        else:
            caption = ''
        table_html = item.get('table_body', '')
        img_path = item.get('img_path', '')
        md = ''
        if caption:
            md += f"**{caption}**\n"
        if img_path:
            md += f"![{caption}]({img_path})\n"
        md += f"{table_html}\n\n"
        return md

    else:
        return '\n'

def assemble_pages_to_markdown(pages, img_base: Path | None = None):
    page_md = {}
    for page_idx in sorted(pages.keys()):
        md = ''
        for item in pages[page_idx]:
            md += item_to_markdown(item, enable_image_caption=True, img_base=img_base)
        page_md[page_idx] = md
    return page_md

def _first_text(v):
    """从 list[str]/str/None 中安全取第一个非空字符串"""
    if isinstance(v, list):
        for x in v:
            if isinstance(x, str) and x.strip():
                return x.strip()
        return ""
    if isinstance(v, str):
        return v.strip()
    return ""

def build_embed_text_for_page(items: list) -> str:
    parts = []
    for it in items:
        t = it.get('type')
        if t == 'text':
            txt = it.get('text', '')
            if txt:
                parts.append(txt)
        elif t == 'image':
            oc = _first_text(it.get('orig_caption') or it.get('image_caption') or "")
            sc = _first_text(it.get('sem_caption') or "")
            cf = it.get('chart_facts') or {}
            bullets_list = cf.get('bullets') or []
            if not isinstance(bullets_list, list):
                bullets_list = [str(bullets_list)]
            bullets = "\n".join(f"- {str(b)}" for b in bullets_list if str(b).strip())
            series_list = cf.get('series') or []
            if not isinstance(series_list, list):
                series_list = [str(series_list)]
            series = ", ".join(s for s in (str(x).strip() for x in series_list) if s)

            block = (
                "【图像】\n"
                f"原始caption：{oc}\n"
                f"语义caption：{sc}\n"
                f"系列：{series}\n"
                f"横轴：{_first_text(cf.get('x_range'))}\n"
                f"单位：{_first_text(cf.get('y_unit'))}\n"
                f"要点：\n{bullets}\n"
            ).strip()
            parts.append(block)
        elif t == 'table':
            # 关键：table_caption 可能是 [], None, 或 str
            cap = _first_text(it.get('table_caption'))
            if cap:
                parts.append(f"【表】{cap}")
            else:
                # 兜底：如果没有 caption，尽量从其他字段给一点可检索的文本
                title = _first_text(it.get('title') or it.get('summary'))
                if title:
                    parts.append(f"【表】{title}")
                else:
                    parts.append("【表】")
        else:
            # 其他类型忽略或做兜底
            pass
    return "\n".join(p for p in parts if p).strip()


def process_all_pdfs_to_page_json(input_base_dir, output_base_dir):
    """
    步骤2：将 content_list.json 转为 page_content.json
    """
    input_base_dir = Path(input_base_dir)
    output_base_dir = Path(output_base_dir)
    pdf_dirs = [d for d in input_base_dir.iterdir() if d.is_dir()]
    for pdf_dir in pdf_dirs:
        file_name = pdf_dir.name
        candidates = [
            pdf_dir / 'auto' / f'{file_name}_content_list.json',
            pdf_dir / 'vlm'  / f'{file_name}_content_list.json',
            pdf_dir / file_name / 'auto' / f'{file_name}_content_list.json',
            pdf_dir / file_name / 'vlm'  / f'{file_name}_content_list.json',
        ]
        json_path = next((p for p in candidates if p.exists()), None)
        if json_path is None:
            # 兜底：全盘搜
            hits = list(pdf_dir.rglob(f"*{file_name}_content_list.json"))
            json_path = hits[0] if hits else None
        if json_path is None:
            print(f"未找到 content_list.json 于 {pdf_dir}（尝试了 auto/ 与 vlm/）")
            continue
        with open(json_path, 'r', encoding='utf-8') as f:
            content_list = json.load(f)
        pages = group_by_page(content_list)
        page_md = assemble_pages_to_markdown(pages, img_base=json_path.parent)
        page_embed = {str(pg): build_embed_text_for_page(pages[pg]) for pg in pages}# 之后的 get(str(pg)) 保持不变

        
        output_dir = output_base_dir / file_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_json_path = output_dir / f'{file_name}_page_content.json'

        payload = {"md": page_md, "embed": page_embed}  # 合包
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"已输出: {output_json_path}")

def process_page_content_to_chunks(input_base_dir, output_json_path):
    """
    步骤3：将 page_content.json 合并为 all_pdf_page_chunks.json
    """
    input_base_dir = Path(input_base_dir)
    input_base_dir.mkdir(parents=True, exist_ok=True)
    if not any(input_base_dir.iterdir()):
        print(f"{input_base_dir} 为空，跳过合并。请先确认第2步已产出 page_content.json")
        return
    
    all_chunks = []
    for pdf_dir in input_base_dir.iterdir():
        if not pdf_dir.is_dir():
            continue
        file_name = pdf_dir.name
        page_content_path = pdf_dir / f"{file_name}_page_content.json"
        if not page_content_path.exists():
            sub_dir = pdf_dir / file_name
            page_content_path2 = sub_dir / f"{file_name}_page_content.json"
            if page_content_path2.exists():
                page_content_path = page_content_path2
            else:
                print(f"未找到: {page_content_path} 也未找到: {page_content_path2}")
                continue
        # process_page_content_to_chunks()
        with open(page_content_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        page_md = payload.get("md", {})
        page_embed = payload.get("embed", {})

        for page_idx, content in page_md.items():
            pg = int(page_idx) if isinstance(page_idx, str) else page_idx
            chunk = {
                "id": f"{file_name}.pdf_page_{pg}",
                "content": content,                       # 渲染给用户
                "embed_text": page_embed.get(str(pg), ""),# 只用于向量
                "metadata": {"file_name": f"{file_name}.pdf", "page": pg, "page_base": 0, "widx": 0}
            }
            all_chunks.append(chunk)
            
            
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"已输出: {output_json_path}")
    


def main():
    base_dir = Path(__file__).parent
    datas_dir = base_dir / 'pdftest'
    content_dir = base_dir / 'data_base_json_content'
    page_dir = base_dir / 'data_base_json_page_content'
    chunk_json_path = base_dir / 'all_pdf_page_chunks.json'
    # 步骤1：PDF → content_list.json
    parse_all_pdfs(datas_dir, content_dir, backend="vlm-transformers")
    # 步骤2：content_list.json → page_content.json
    process_all_pdfs_to_page_json(content_dir, page_dir)
    # 步骤3：page_content.json → all_pdf_page_chunks.json
    process_page_content_to_chunks(page_dir, chunk_json_path)
    print("全部处理完成！")

if __name__ == '__main__':
    main()
