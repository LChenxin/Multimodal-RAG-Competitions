import json, os
from pathlib import Path

PART_DIR = Path("./parts")
TEST_PATH = "./datas/test.json"
FINAL_PATH = "./rag_top1_pred.json"

def main():
    # 读测试集，确定总长度与顺序
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    N = len(test_data)

    # 读所有 part_*.json 并合并为 idx->result
    idx2result = {}
    for p in sorted(PART_DIR.glob("part_*.json")):
        with open(p, "r", encoding="utf-8") as f:
            part_list = json.load(f)
        # part_* 文件里是顺序段，但不含 idx；我们需要从文件名推断区间，再按顺序映射
        # 文件名形如 part_000000_000199.json
        name = p.stem  # part_000000_000199
        try:
            _, start_end = name.split("part_")
            start_str, end_str = start_end.split("_")
            start, end = int(start_str), int(end_str)
        except Exception:
            print(f"跳过无法解析区间的文件: {p.name}")
            continue
        # 将该段的条目顺序映射回全局 idx
        for offset, item in enumerate(part_list):
            idx = start + offset
            if idx < N:
                idx2result[idx] = item

    # 组装最终结果，保持原顺序 & 补空
    final = []
    for idx in range(N):
        if idx in idx2result:
            final.append(idx2result[idx])
        else:
            final.append({
                "question": test_data[idx].get("question",""),
                "answer": "",
                "filename": "",
                "page": "",
            })

    tmp = FINAL_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    os.replace(tmp, FINAL_PATH)
    print(f"✅ 合并完成: {FINAL_PATH} （共 {len(final)} 条）")

if __name__ == "__main__":
    main()
