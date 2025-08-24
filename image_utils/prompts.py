\
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图片分析相关的提示词模板
"""

def get_image_analysis_prompt(title_max_length: int, description_max_length: int) -> str:
    """
    生成图片分析的提示词。

    Args:
        title_max_length: 标题最大长度。
        description_max_length: 描述最大长度。

    Returns:
        格式化的提示词字符串。
    """
    return f"""请分析这张图片并生成一个{title_max_length}字以上的标题、{description_max_length}字以上的图片描述，使用JSON格式输出。

分析以下方面:
1. 图像类型（图表、示意图、照片等）
2. 主要内容/主题
3. 包含的关键信息点
4. 图像的可能用途

输出格式必须严格为:
{{
  "title": "标题({title_max_length}字以内)",
  "description": "详细描述({description_max_length}字以内)"
}}

只返回JSON，不要有其他说明文字。
"""
# image_utils/prompts.py
def get_chart_analysis_prompt(max_bullets: int = 6) -> str:
    return f"""你将看到一张图表（可能包含多条折线/柱状图）。请仅根据图中可见内容，提取结构化信息并返回 JSON：

必须字段：
- "title": 简短标题（不超过 40 字）
- "type": 图表类型（如 折线图/柱状图/饼图/散点图/其他）
- "series": 数组，每个元素是图例/系列名称（若无法看清，用""）
- "x_range": 横轴范围或时间跨度（如"2020–2025Q1"，未知则""）
- "y_unit": 纵轴单位（如"%" 或 "亿元"，未知则""）
- "bullets": 不超过 {max_bullets} 条要点，聚焦趋势/极值/交叉（如"2024毛利率约26%"）；仅写你能从图上读出的信息

严格输出 JSON，不要多余文本。若看不清请用空字符串或空数组，不要编造数值。
"""
