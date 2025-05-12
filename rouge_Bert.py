# ============================================================
# 綜合指標函式：composite_score(text_a, text_b,
#                               alpha=0.6, top_k=3, chunk=512)
# ============================================================

import re, jieba, numpy as np
from typing import List
from rouge_score import rouge_scorer, tokenizers
from transformers import AutoTokenizer
from evaluate import load as load_metric

# ---------- 共用資源 (僅初始化一次) ----------
STOPWORDS = {"的", "了", "是", "在", "和", "也", "有", "就", "不", "與"}

def _normalize(text: str) -> str:
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[　]", "", text)
    text = re.sub(r"[“”「」『』]", "\"", text)
    text = re.sub(r"[‘’]", "'", text)
    return text.strip()

def _remove_sw(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in STOPWORDS]

class _JiebaTok(tokenizers.Tokenizer):
    def tokenize(self, txt: str):
        toks = list(jieba.cut(_normalize(txt)))
        return _remove_sw(toks)

_ROUGE = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=False,
    tokenizer=_JiebaTok()
)

_BERT_TOK = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B",
                                          trust_remote_code=True)
_BERT_METRIC = load_metric("bertscore")

# ---------- 主函式 ----------
def composite_score(text_a: str,
                    text_b: str,
                    alpha: float = 0.6,
                    top_k: int = 3,
                    chunk: int = 512) -> float:
    """
    綜合分數 = alpha * BERTScore_F1 + (1-alpha) * Weighted_ROUGE_F1

    text_a: 原文（長篇）
    text_b: 摘要或候選文本
    alpha : BERTScore 權重 (0~1)
    top_k : BERTScore 取前 k 高段平均
    chunk : BERTScore 每段最長 token 數
    """
    # -------- ROUGE --------
    segs = ["".join(p) for p in np.array_split(list(_normalize(text_a)), 2)]
    r_scores = []
    for s in segs:
        r = _ROUGE.score(s, _normalize(text_b))
        r_scores.append(0.4*r["rouge1"].fmeasure +
                        0.3*r["rouge2"].fmeasure +
                        0.3*r["rougeL"].fmeasure)
    rouge_f1 = float(np.mean(r_scores))

    # -------- BERTScore --------
    ids = _BERT_TOK.encode(text_a, add_special_tokens=False)
    segments = [_BERT_TOK.decode(ids[i:i+chunk])
                for i in range(0, len(ids), chunk)]
    f1_vals = [_BERT_METRIC.compute(predictions=[seg],
                                    references=[text_b],
                                    lang="zh")["f1"][0]
               for seg in segments]
    bert_f1 = float(np.mean(sorted(f1_vals, reverse=True)[:min(top_k, len(f1_vals))]))

    # -------- 組合 --------
    return alpha * bert_f1 + (1 - alpha) * rouge_f1


if __name__ == "__main__":
    score = composite_score(long_context, summary,
                        alpha=0.65, top_k=3, chunk=512)
    print("Composite Score:", score)
