## package
import json
import pathlib
from typing import Any, Dict, List, Tuple
import datasets         # pip install datasets

## data processing
# ────────────────────── 讀檔工具
def load_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(l) if l.strip() else {}
                for l in fh]

# ────────────────────── 型別修正工具
def ensure_str_list(obj: Any) -> List[str]:
    if obj is None:
        return []
    if isinstance(obj, list):
        flat = []
        for x in obj:
            flat.extend(ensure_str_list(x))      # 遞迴處理巢狀 list
        return [str(s) for s in flat]
    return [str(obj)]

def ensure_int_list(obj: Any) -> List[int]:
    flat: List[int] = []
    def _walk(o):
        if o is None:
            return
        if isinstance(o, list):
            for item in o:
                _walk(item)
        else:
            try:
                flat.append(int(o))
            except (TypeError, ValueError):
                pass
    _walk(obj)
    return flat

def context_to_string(ctx: Any) -> str:
    if isinstance(ctx, str):
        return ctx
    # 將最終元素攤平成 str，再用換行符連接句子
    lines = ensure_str_list(ctx)
    return "\n".join(lines)

# ────────────────────── 建立全文 / highlight 對照表
def build_lookup_maps() -> Tuple[
    Dict[Tuple[str, int], str],
    Dict[Tuple[str, int], List[int]]
]:
    ctx_map = {
        (d["id"], d["av_num"]): context_to_string(d["context"])
        for d in load_jsonl(ROOT / "overall_context.txt")
    }
    hl_map = {
        (d["id"], d["av_num"]): ensure_int_list(d["highlights"])
        for d in load_jsonl(ROOT / "overall_highlights.txt")
    }
    return ctx_map, hl_map

## get data

def get_vcsum(save_type):
    #!/usr/bin/env python3
    """
    build_hf_dataset.py
    把 vcsum_data/ 轉成 HuggingFace DatasetDict（train / dev / test）。
    """
    # ──────────────────────
    ROOT   = pathlib.Path("vcsum_data")
    SPLITS = ["train", "dev", "test"]

    FEATURES = datasets.Features({
        "id"        : datasets.Value("string"),
        "av_num"    : datasets.Value("int32"),
        "context"   : datasets.Value("string"),
        "summary"   : datasets.Value("string"),
        "agenda"    : datasets.Sequence(datasets.Value("string")),
        "discussion": datasets.Sequence(datasets.Value("string")),
        "eos_index" : datasets.Sequence(datasets.Value("int32")),
        "highlights": datasets.Sequence(datasets.Value("int8")),
        "split"     : datasets.Value("string"),
    })

    ctx_map, hl_map = build_lookup_maps()
    ds_dict: Dict[str, datasets.Dataset] = {}

    for sp in SPLITS:
        long_items  = {(d["id"], d["av_num"]): d
                       for d in load_jsonl(ROOT / f"long_{sp}.txt")}
        short_items = {(d["id"], d["av_num"]): d
                       for d in load_jsonl(ROOT / f"short_{sp}.txt")}

        rows = []
        for key, s in short_items.items():
            l = long_items.get(key, {})
            rows.append({
                "id"        : key[0],
                "av_num"    : int(key[1]),
                "context"   : ctx_map.get(key,
                               context_to_string(s.get("context", l.get("context", "")))),
                "summary"   : l.get("summary", ""),
                "agenda"    : ensure_str_list(s.get("agenda")),
                "discussion": ensure_str_list(s.get("discussion")),
                "eos_index" : ensure_int_list(s.get("eos_index")),
                "highlights": hl_map.get(key, []),
                "split"     : sp,
            })

        ds_dict[sp] = datasets.Dataset.from_list(rows).cast(FEATURES)

    dataset = datasets.DatasetDict(ds_dict)
    print(dataset)           # 完整結構
    print(dataset["train"][0])  # 範例檢查

    out_dir = "content"
    dataset.save_to_disk(out_dir)
    print(f"✓ HuggingFace Dataset saved under directory: {out_dir}")



def get_LR_Sum():
    ds = load_dataset("bltlab/lr-sum", "Chinese")

    return ds