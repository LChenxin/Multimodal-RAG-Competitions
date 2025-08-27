
import os
from pathlib import Path
import json
from collections import defaultdict
import asyncio
from image_utils.async_image_analysis import AsyncImageAnalysis

def parse_all_pdfs(datas_dir, output_base_dir, backend="pipeline", force=False):
    from mineru_parse_pdf import do_parse
    datas_dir = Path(datas_dir)
    output_base_dir = Path(output_base_dir)
    pdf_files = [p for p in datas_dir.rglob('*.pdf') if "checkpoint" not in p.as_posix().lower()]
    if not pdf_files:
        print(f"未找到PDF文件于: {datas_dir}")
        return

    subdir = "auto" if backend == "pipeline" else "vlm"

    for pdf_path in pdf_files:
        file_name = pdf_path.stem
        output_dir = output_base_dir / file_name

        # —— 关键：同时考虑两种目录结构（MinerU 有时会多套一层 file_name）——
        candidate1 = output_dir / subdir / f"{file_name}_content_list.json"
        candidate2 = output_dir / file_name / subdir / f"{file_name}_content_list.json"

        # ✅ 已存在则跳过（除非 force=True）
        if (candidate1.exists() or candidate2.exists()) and not force:
            print(f"[跳过] 已有解析结果: {candidate1 if candidate1.exists() else candidate2}")
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        do_parse(
            output_dir=str(output_dir),             # 保持不变；MinerU 可能会套一层
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

        # 跑完后再检测实际落地在哪个路径并打印
        actual = candidate1 if candidate1.exists() else (candidate2 if candidate2.exists() else None)
        print(f"已输出: {actual or (output_dir / subdir / f'{file_name}_content_list.json')}")



def group_by_page(content_list):
    pages = defaultdict(list)
    for item in content_list:
        page_idx = item.get('page_idx', 0)
        pages[page_idx].append(item)
    return dict(pages)

def item_to_markdown(item, enable_image_caption=True, img_base: Path | str | None = None):
    """
    enable_image_caption: 是否启用多模态视觉分析（图片caption补全），默认True。
    """
    if isinstance(img_base, str):
        img_base = Path(img_base)
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

        abs_img_path = None
        if img_path:  # 只在非空时解析
            p = Path(img_path)
            if p.is_absolute():
                abs_img_path = p
            else:
                abs_img_path = (Path(img_base) / p).resolve() if img_base else p.resolve()


        sem_caption = ''
        chart_facts = {}    # 结构化结果
        if enable_image_caption and abs_img_path and abs_img_path.exists()and abs_img_path.is_file():
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
                        r = await analyzer.analyze_image(local_image_path=str(abs_img_path), prompt=prompt)
                        if not r or ("title" not in r and "description" not in r):
                            r = await analyzer.analyze_image(
                                local_image_path=str(abs_img_path),
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

def assemble_pages_to_markdown(pages, img_base: Path | str | None = None):
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

def _norm_val(v: str) -> str:
    s = v.strip().replace(",", "")
    # 缺失占位统一为 NA（不要 0）
    if s in ("—", "-", "N/A", "NA", "null", "None", ""):
        return "NA"
    # 百分比：保留原值 + 提供一个 0-1 的小数，利于检索
    if s.endswith("%"):
        try:
            p = float(s[:-1])
            return f"{s} {p/100:g}"
        except:
            return s
    return s

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
            import re, html as _html

            def _strip_tags(s: str) -> str:
                s = re.sub(r"<[^>]+>", "", s)
                return _html.unescape(s).strip()

            def _parse_table(html: str):
                if not html:
                    return [], []
                trs = re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.IGNORECASE|re.DOTALL)
                rows = []
                for tr in trs:
                    cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", tr, flags=re.IGNORECASE|re.DOTALL)
                    cells = [_strip_tags(c) for c in cells]
                    if any(c.strip() for c in cells):
                        rows.append(cells)

                if not rows:
                    return [], []

                def _looks_like_years(cols):
                    hits = 0
                    for c in cols:
                        if re.search(r"20\d{2}\s*[AE]?$", c) or re.search(r"20\d{2}\s*[AE]\b", c):
                            hits += 1
                    return hits >= max(2, len(cols)//2)

                header = []
                if rows and (("会计年度" in rows[0][0]) or _looks_like_years(rows[0][1:])):
                    header = rows[0]
                    data_rows = rows[1:]
                else:
                    data_rows = rows
                return header, data_rows

            cap = _first_text(it.get('table_caption')) or _first_text(it.get('title') or it.get('summary')) or ""
            html_body = it.get('table_body') or ""
            header, data_rows = _parse_table(html_body)

            lines = []
            if cap:
                lines.append(f"【表】{cap}")

            if header and len(header) > 1:
                years = " ".join(h.strip() for h in header[1:] if h.strip())
                if years:
                    lines.append(f"列: {years}")

            for row in data_rows:
                if not row:
                    continue
                name = row[0].strip()
                if not name or name in ("会计年度",):
                    continue
                vals = [_norm_val(c) for c in row[1:] if str(c).strip() != ""]
                if vals:
                    lines.append(f"{name}: " + " ".join(vals))
                else:
                    lines.append(name)

            if not lines:
                lines.append(f"【表】{cap or '表'}")

            parts.append("\n".join(lines))
        else:
            # 其他类型忽略或做兜底
            pass
        
    out = "\n".join(p for p in parts if p).strip()

    # 最后再做一次统一清洗（去掉可能混入的 Markdown/HTML）
# 最后再做一次统一清洗（去掉可能混入的 Markdown/HTML）
    import re, html as _html, unicodedata

    out = re.sub(r"<[^>]+>", " ", out)              # 去 HTML 标签
    out = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", out) # 去图片语法 ![...](...)
    # 去掉行首的标题/列表/引用标记（但不影响数值里的'-'）
    out = re.sub(r"^\s{0,3}#{1,6}\s+", "", out, flags=re.MULTILINE)  # # 标题
    out = re.sub(r"^\s{0,3}>\s+", "", out, flags=re.MULTILINE)       # 引用 >
    out = re.sub(r"^\s{0,3}[-*+]\s+", "", out, flags=re.MULTILINE)   # 列表 -,*,+
    out = out.replace("`", "")                                        # 反引号
    out = _html.unescape(out)
    out = unicodedata.normalize("NFKC", out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out




def process_all_pdfs_to_page_json(input_base_dir, output_base_dir):
    """
    步骤2：将 content_list.json 转为 page_content.json
    """
    input_base_dir = Path(input_base_dir)
    output_base_dir = Path(output_base_dir)
    pdf_dirs = [d for d in input_base_dir.iterdir()
                if d.is_dir() and "checkpoint" not in d.name.lower()]
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
        page_embed = {str(pg): build_embed_text_for_page(pages[pg]) for pg in sorted(pages)}
        
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
        if not pdf_dir.is_dir() or "checkpoint" in pdf_dir.name.lower():
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
    


def main(force=False):
    base_dir = Path(__file__).parent
    datas_dir = base_dir / 'datas'
    content_dir = base_dir / 'data_base_json_content'
    page_dir = base_dir / 'data_base_json_page_content'
    chunk_json_path = base_dir / 'all_pdf_page_chunks.json'
    # 步骤1：PDF → content_list.json
    parse_all_pdfs(datas_dir, content_dir, backend="vlm-transformers", force=False)
    # 步骤2：content_list.json → page_content.json
    process_all_pdfs_to_page_json(content_dir, page_dir)
    # 步骤3：page_content.json → all_pdf_page_chunks.json
    process_page_content_to_chunks(page_dir, chunk_json_path)
    print("全部处理完成！")

if __name__ == '__main__':
    main()
