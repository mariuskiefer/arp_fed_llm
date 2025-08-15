# eval_sentiment_both.py
# Evaluate sentiment scoring (nlp + llm) on the holdout set using gold entities.
# No aliasing/canonicalization; duplicates handled by per-sentence position index.

import ast
import json
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error

from get_sentiment_nlp import extract_nlp_sentiment
from get_sentiment_llm import extract_llm_sentiment

HOLDOUT_CSV = "/Users/mariuskiefer/Desktop/arp_fed_llm/holdout_eval_set.csv"

# ---------------------------
# parse the "Entities" cell -> ordered list[(entity, score)]
def parse_entities_cell(s):
    try:
        raw = ast.literal_eval(s)
    except Exception:
        return []
    out = []
    for tup in raw or []:
        if not isinstance(tup, (list, tuple)) or len(tup) < 2:
            continue
        ent, scr = tup[0], tup[1]
        if ent in (None, "") or scr in ("", None):
            continue
        try:
            out.append((str(ent), float(scr)))
        except Exception:
            pass
    return out

# ---------------------------
# flatten results -> DataFrame(sentence, pos, entity, pred_score)
def flatten_pred_struct(results):
    rows = []
    for item in results:
        s = item["sentence"]
        for i, ent in enumerate(item["entities"]):
            rows.append({"sentence": s, "pos": i, "entity": ent["name"], "pred_score": ent["sentiment"]})
    return pd.DataFrame(rows)

# ---------------------------
# compute and print metrics (overall + per-entity) including MAE and baseline MAE
def compute_and_print_metrics(name, merged, score_to_label):
    def to_label(x):
        try:
            return score_to_label[float(x)]
        except Exception:
            return None

    df = merged.copy()
    df = df[(df["pred_score"] != "") & (~df["pred_score"].isna())].copy()
    if df.empty:
        print(f"\n== {name} ==\nNo comparable predictions after alignment.")
        return

    df["y_true"] = df["true_score"].apply(to_label)
    df["y_pred"] = df["pred_score"].apply(to_label)
    df = df.dropna(subset=["y_true", "y_pred"]).astype({"y_true": int, "y_pred": int})
    if df.empty:
        print(f"\n== {name} ==\nNo valid label pairs after mapping.")
        return

    labels_sorted = sorted(set(score_to_label.values()))
    macro = f1_score(df["y_true"], df["y_pred"], average="macro", labels=labels_sorted)
    micro = f1_score(df["y_true"], df["y_pred"], average="micro", labels=labels_sorted)
    weighted = f1_score(df["y_true"], df["y_pred"], average="weighted", labels=labels_sorted)
    acc = accuracy_score(df["y_true"], df["y_pred"])

    # MAE for model predictions
    df_scores = df[["true_score", "pred_score"]].dropna().copy()
    mae = float(np.mean(np.abs(df_scores["pred_score"].astype(float) - df_scores["true_score"].astype(float))))

    # Baseline MAE: always predict 0.33
    baseline_preds = [0.33] * len(df_scores)
    baseline_mae = mean_absolute_error(df_scores["true_score"], baseline_preds)

    print(f"\n== {name} ==  (n={len(df)})")
    print(f"Accuracy        : {acc:.4f}")
    print(f"Macro F1        : {macro:.4f}")
    print(f"Micro F1        : {micro:.4f}")
    print(f"Weighted F1     : {weighted:.4f}")
    print(f"MAE (model)     : {mae:.4f}")
    print(f"MAE (baseline)  : {baseline_mae:.4f}")
    print(f"Improvement vs baseline: {baseline_mae - mae:.4f}")

    # per-entity breakdown
    print("\n-- Per-entity (macro F1 / acc / support) --")
    per_rows = []
    for ent, sub in df.groupby("entity"):
        f1_ent = f1_score(sub["y_true"], sub["y_pred"], average="macro", labels=labels_sorted)
        acc_ent = accuracy_score(sub["y_true"], sub["y_pred"])
        per_rows.append((ent, len(sub), f1_ent, acc_ent))
    per_rows.sort(key=lambda x: (-x[1], -x[2]))
    for ent, n, f1e, acce in per_rows:
        print(f"{ent:<28} macroF1={f1e:.3f}  acc={acce:.3f}  n={n}")

# ---------------------------
# main
if __name__ == "__main__":
    with open("score_to_label.json", "r") as fp:
        s2l = json.load(fp)
    score_to_label = {float(k): int(v) for k, v in s2l.items()}

    df = pd.read_csv(HOLDOUT_CSV)
    by_sent = defaultdict(list)
    for _, r in df.iterrows():
        s = r.get("Sentence", "")
        ents = parse_entities_cell(r.get("Entities", "[]"))
        if ents:
            by_sent[s].extend(ents)

    gold_rows = []
    for s, ents in by_sent.items():
        for i, (e, sc) in enumerate(ents):
            gold_rows.append({"sentence": s, "pos": i, "entity": e, "true_score": sc})
    gold = pd.DataFrame(gold_rows)
    if gold.empty:
        raise ValueError("Holdout has no valid (sentence, entity, score) rows.")

    items_llm = [{"sentence": s, "entities": [{"name": e} for e, _ in ents]} for s, ents in by_sent.items()]
    items_nlp = [(s, [e for e, _ in ents]) for s, ents in by_sent.items()]

    preds_llm = extract_llm_sentiment(items_llm)
    preds_nlp = extract_nlp_sentiment(items_nlp)

    df_llm = flatten_pred_struct(preds_llm)
    df_nlp = flatten_pred_struct(preds_nlp)

    merged_llm = gold.merge(df_llm, on=["sentence", "pos", "entity"], how="left")
    merged_nlp = gold.merge(df_nlp, on=["sentence", "pos", "entity"], how="left")

    compute_and_print_metrics("LLM", merged_llm, score_to_label)
    compute_and_print_metrics("NLP", merged_nlp, score_to_label)
