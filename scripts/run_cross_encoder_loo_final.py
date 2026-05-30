# -- IMPORTS --
from __future__ import annotations
import gc
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder
from tqdm.auto import tqdm
from transformers import AutoTokenizer


root = Path(__file__).resolve().parents[1]

pairs_file = root / "thesis_runs/cross_encoder/explanations/explanation_pairs_1_per_query_on_duot5_queries_seed42.csv"
out_dir = root / "thesis_runs/cross_encoder/explanations"
model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
max_length = 512
seed = 42


def load_pairs() -> pd.DataFrame:
    return pd.read_csv(pairs_file, dtype={"qid": str, "pid_i": str, "pid_j": str})


def setup_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ce = CrossEncoder(model_name, max_length=max_length)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = ce.model.to(device)
    model.eval()

    if tokenizer.mask_token_id is None:
        raise ValueError("Cross-encoder LOO requires a tokenizer with a mask token.")
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define a pad token.")

    return tokenizer, model, device


def merge_wordpiece_spans(tokens, positions):
    spans = []
    current_tokens, current_positions = [], []
    for token, pos in zip(tokens, positions):
        if token.startswith("##") and current_tokens:
            current_tokens.append(token)
            current_positions.append(int(pos))
        else:
            if current_tokens:
                spans.append((current_tokens, current_positions))
            current_tokens = [token]
            current_positions = [int(pos)]
    if current_tokens:
        spans.append((current_tokens, current_positions))

    words = []
    for toks, poss in spans:
        word = toks[0]
        for tok in toks[1:]:
            word += tok.replace("##", "")
        words.append({"word": word, "positions": poss, "tokens": toks})
    return words


def tokenize_ce(tokenizer, device, query, passage):
    encoded = tokenizer(
        query,
        passage,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors="pt")
    
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    token_type_ids = encoded.get("token_type_ids")
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    sep_positions = (input_ids[0] == tokenizer.sep_token_id).nonzero(as_tuple=True)[0].tolist()
    sep_idx = sep_positions[0] if sep_positions else len(tokens) - 1
    last_sep_idx = sep_positions[-1] if sep_positions else len(tokens) - 1

    doc_positions = list(range(sep_idx + 1, last_sep_idx))
    doc_tokens = [tokens[p] for p in doc_positions]
    doc_words = merge_wordpiece_spans(doc_tokens, doc_positions)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "tokens": tokens,
        "doc_words": doc_words}


def score_ce(model, tok):
    kwargs = {"input_ids": tok["input_ids"], "attention_mask": tok["attention_mask"]}
    if tok["token_type_ids"] is not None:
        kwargs["token_type_ids"] = tok["token_type_ids"]
    with torch.no_grad():
        outputs = model(**kwargs)
    return float(outputs.logits.squeeze(-1).detach().cpu().item())


def mask_positions(tokenizer, tok, positions):
    perturbed = {
        "input_ids": tok["input_ids"].clone(),
        "attention_mask": tok["attention_mask"].clone(),
        "token_type_ids": None if tok["token_type_ids"] is None else tok["token_type_ids"].clone()}
    perturbed["input_ids"][0, positions] = tokenizer.mask_token_id
    return perturbed


def compute_loo_pointwise(tokenizer, model, device, query, passage):
    tok = tokenize_ce(tokenizer, device, query, passage)
    original_score = score_ce(model, tok)
    rows = []

    for word_idx, span in enumerate(tok["doc_words"]):
        masked = mask_positions(tokenizer, tok, span["positions"])
        masked_score = score_ce(model, masked)
        rows.append(
            {
                "word": span["word"],
                "positions": span["positions"],
                "position": word_idx,
                "support_score": float(original_score - masked_score)})

    return {"method": "loo_pointwise", "score": float(original_score), "doc": rows}


def compute_loo_pairwise(tokenizer, model, device, query, passage_i, passage_j):
    tok_i = tokenize_ce(tokenizer, device, query, passage_i)
    tok_j = tokenize_ce(tokenizer, device, query, passage_j)
    score_i = score_ce(model, tok_i)
    score_j = score_ce(model, tok_j)
    original_g = score_i - score_j

    doc_i_rows = []
    for word_idx, span in enumerate(tok_i["doc_words"]):
        masked_i = mask_positions(tokenizer, tok_i, span["positions"])
        g_masked = score_ce(model, masked_i) - score_j
        doc_i_rows.append(
            {
                "word": span["word"],
                "positions": span["positions"],
                "position": word_idx,
                "support_score": float(original_g - g_masked)})

    doc_j_rows = []
    for word_idx, span in enumerate(tok_j["doc_words"]):
        masked_j = mask_positions(tokenizer, tok_j, span["positions"])
        g_masked = score_i - score_ce(model, masked_j)
        doc_j_rows.append(
            {
                "word": span["word"],
                "positions": span["positions"],
                "position": word_idx,
                "support_score": float(original_g - g_masked)})

    return {"method": "loo_pairwise", "score": float(original_g), "doc_i": doc_i_rows, "doc_j": doc_j_rows}


def flatten_pairwise_rows(record):
    rows = []
    for side, entries in [("doc_i", record["loo_pairwise"]["doc_i"]), ("doc_j", record["loo_pairwise"]["doc_j"])]:
        for entry in entries:
            rows.append(
                {
                    "qid": record["qid"],
                    "pid_i": record["pid_i"],
                    "pid_j": record["pid_j"],
                    "side": side,
                    "segment": "doc",
                    "position": entry["position"],
                    "token": entry["word"],
                    "support_score": entry["support_score"]})
            
    return rows


def flatten_pointwise_rows(record):
    rows = []
    for side, pid_key, attr_key in [
        ("doc_i", "pid_i", "loo_pointwise_i"),
        ("doc_j", "pid_j", "loo_pointwise_j")]:
        for entry in record[attr_key]["doc"]:
            rows.append(
                {
                    "qid": record["qid"],
                    "pid_i": record["pid_i"],
                    "pid_j": record["pid_j"],
                    "pid": record[pid_key],
                    "side": side,
                    "segment": "doc",
                    "position": entry["position"],
                    "token": entry["word"],
                    "support_score": entry["support_score"]})
            
    return rows


def main():
    np.random.seed(seed)
    torch.manual_seed(seed)

    tokenizer, model, device = setup_model()
    pairs_df = load_pairs()

    records = []
    failed = []

    for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc="Computing cross-encoder LOO"):
        try:
            loo_pairwise = compute_loo_pairwise(tokenizer, model, device, row["query"], row["passage_i"], row["passage_j"])
            loo_pointwise_i = compute_loo_pointwise(tokenizer, model, device, row["query"], row["passage_i"])
            loo_pointwise_j = compute_loo_pointwise(tokenizer, model, device, row["query"], row["passage_j"])

            records.append(
                {
                    "qid": row["qid"],
                    "query": row["query"],
                    "pid_i": row["pid_i"],
                    "pid_j": row["pid_j"],
                    "g_score": row["g_score"],
                    "correct_pref": row["correct_pref"],
                    "loo_pairwise": loo_pairwise,
                    "loo_pointwise_i": loo_pointwise_i,
                    "loo_pointwise_j": loo_pointwise_j,
                    "pointwise_score_gap": float(loo_pointwise_i["score"] - loo_pointwise_j["score"])})

        except Exception as exc:
            failed.append({"qid": row["qid"], "pid_i": row["pid_i"], "pid_j": row["pid_j"], "error": str(exc)})
            print(f"Error on qid={row['qid']} pid_i={row['pid_i']} pid_j={row['pid_j']}: {exc}")
        finally:
            gc.collect()
            if torch.backends.mps.is_available():
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "attributions_loo.pkl"
    with out_path.open("wb") as f:
        pickle.dump(records, f)

    pairwise_rows = [row for record in records for row in flatten_pairwise_rows(record)]
    pointwise_rows = [row for record in records for row in flatten_pointwise_rows(record)]

    pairwise_df = pd.DataFrame(pairwise_rows)
    pointwise_df = pd.DataFrame(pointwise_rows)
    pairwise_df.to_csv(out_dir / "attributions_loo_pairwise_rows.csv", index=False)
    pointwise_df.to_csv(out_dir / "attributions_loo_pointwise_rows.csv", index=False)

    summary_rows = []
    for record in records:
        pw_i = record["loo_pairwise"]["doc_i"]
        pw_j = record["loo_pairwise"]["doc_j"]
        pt_i = record["loo_pointwise_i"]["doc"]
        pt_j = record["loo_pointwise_j"]["doc"]

        best_pw_i = max(pw_i, key=lambda x: x["support_score"]) if pw_i else {"word": "", "support_score": np.nan}
        best_pw_j = max(pw_j, key=lambda x: x["support_score"]) if pw_j else {"word": "", "support_score": np.nan}
        best_pt_i = max(pt_i, key=lambda x: x["support_score"]) if pt_i else {"word": "", "support_score": np.nan}
        best_pt_j = max(pt_j, key=lambda x: x["support_score"]) if pt_j else {"word": "", "support_score": np.nan}

        summary_rows.append(
            {
                "qid": record["qid"],
                "pid_i": record["pid_i"],
                "pid_j": record["pid_j"],
                "g_score": record["g_score"],
                "correct_pref": record["correct_pref"],
                "loo_pairwise_score": record["loo_pairwise"]["score"],
                "loo_pointwise_i_score": record["loo_pointwise_i"]["score"],
                "loo_pointwise_j_score": record["loo_pointwise_j"]["score"],
                "pointwise_score_gap": record["pointwise_score_gap"],
                "pw_doc_i_top_word": best_pw_i["word"],
                "pw_doc_i_top_score": best_pw_i["support_score"],
                "pw_doc_j_top_word": best_pw_j["word"],
                "pw_doc_j_top_score": best_pw_j["support_score"],
                "pt_doc_i_top_word": best_pt_i["word"],
                "pt_doc_i_top_score": best_pt_i["support_score"],
                "pt_doc_j_top_word": best_pt_j["word"],
                "pt_doc_j_top_score": best_pt_j["support_score"]})

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "attributions_loo_summary.csv", index=False)

    if failed:
        pd.DataFrame(failed).to_csv(out_dir / "attributions_loo_failed.csv", index=False)

    print(f"Saved {len(records)} cross-encoder LOO records to {out_path}")
    print(f"Saved pairwise rows to {out_dir / 'attributions_loo_pairwise_rows.csv'}")
    print(f"Saved pointwise rows to {out_dir / 'attributions_loo_pointwise_rows.csv'}")
    print(f"Saved summary to {out_dir / 'attributions_loo_summary.csv'}")
    if failed:
        print(f"Failed pairs: {len(failed)}")


if __name__ == "__main__":
    main()