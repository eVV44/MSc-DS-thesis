# -- IMPORTS --
from __future__ import annotations
import gc
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


root = Path(__file__).resolve().parents[1]

pairs_file = root / "thesis_runs/duot5/explanations/explanation_pairs_1_per_query_seed42.csv"
ranked_results_file = root / "thesis_runs/duot5/reranking/duot5_final/ranked_results.csv"
out_dir = root / "thesis_runs/duot5/explanations"

model_name = "castorini/duot5-base-msmarco"
max_length = 512
seed = 42
force_cpu = True


def load_pairs() -> pd.DataFrame:
    return pd.read_csv(pairs_file, dtype={"qid": str, "pid_i": str, "pid_j": str})


def load_ranked_results() -> pd.DataFrame:
    df = pd.read_csv(ranked_results_file, dtype={"qid": str, "pid": str})
    df["qid"] = df["qid"].astype(str)
    df["pid"] = df["pid"].astype(str)
    return df


def setup_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    except Exception as exc:
        raise ImportError(
            "Failed to load the DuoT5 tokenizer. Install tokenizer dependencies "
            "(for example `pip install sentencepiece protobuf`) and try again.") from exc

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = torch.device("cpu") if force_cpu else (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
    model = model.to(device)
    model.eval()

    decoder_start_token_id = model.config.decoder_start_token_id
    if decoder_start_token_id is None:
        decoder_start_token_id = tokenizer.pad_token_id

    true_ids = tokenizer.encode("true", add_special_tokens=False)
    false_ids = tokenizer.encode("false", add_special_tokens=False)
    if len(true_ids) != 1 or len(false_ids) != 1:
        raise ValueError("Expected 'true' and 'false' to map to single tokens.")

    return tokenizer, model, device, int(decoder_start_token_id), int(true_ids[0]), int(false_ids[0])


def build_reference_pool(ranked_rows: pd.DataFrame):
    pool = {}
    for qid, group in ranked_rows.groupby("qid", sort=False):
        ordered = group.sort_values("rank", ascending=False)
        pool[qid] = [
            {
                "pid": str(row.pid),
                "passage": row.passage,
                "rank": int(row.rank),
                "score": float(row.score)}

            for row in ordered.itertuples(index=False)]
    return pool


def choose_reference(reference_pool, qid, excluded_pids=()):
    excluded = {str(pid) for pid in excluded_pids if pid is not None}
    candidates = reference_pool.get(str(qid), [])
    for candidate in candidates:
        if candidate["pid"] not in excluded:
            return candidate
    if not candidates:
        raise ValueError(f"No reference passage available for qid={qid}")
    return candidates[0]


def merge_sentencepiece_spans(tokenizer, tokens, positions):
    special_tokens = set(tokenizer.all_special_tokens)
    spans = []
    current_word, current_positions, current_tokens = "", [], []
    for token, pos in zip(tokens, positions):
        if token in special_tokens:
            if current_tokens:
                spans.append({"word": current_word or "".join(current_tokens), "positions": current_positions, "tokens": current_tokens})
                current_word, current_positions, current_tokens = "", [], []
            continue
        if token.startswith("▁"):
            if current_tokens:
                spans.append({"word": current_word or "".join(current_tokens), "positions": current_positions, "tokens": current_tokens})
            current_word = token.lstrip("▁") or token
            current_positions = [int(pos)]
            current_tokens = [token]
        else:
            current_word += token
            current_positions.append(int(pos))
            current_tokens.append(token)
    if current_tokens:
        spans.append({"word": current_word or "".join(current_tokens), "positions": current_positions, "tokens": current_tokens})
    return spans


def duo_input(query, doc0, doc1):
    return f"Query: {query} Document0: {doc0} Document1: {doc1} Relevant:"


def _tok_len(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def tokenize_duo_pairwise(tokenizer, device, query, doc0, doc1):
    text = duo_input(query, doc0, doc1)
    encoded = tokenizer(text, max_length=max_length, truncation=True, padding=False, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    c1 = "Query: "
    c2 = c1 + query
    c3 = c2 + " Document0: "
    c4 = c3 + doc0
    c5 = c4 + " Document1: "
    c6 = c5 + doc1

    doc0_positions = [pos for pos in range(_tok_len(tokenizer, c3), min(_tok_len(tokenizer, c4), input_ids.shape[1]))]
    doc1_positions = [pos for pos in range(_tok_len(tokenizer, c5), min(_tok_len(tokenizer, c6), input_ids.shape[1]))]
    doc0_words = merge_sentencepiece_spans(tokenizer, [tokens[p] for p in doc0_positions], doc0_positions)
    doc1_words = merge_sentencepiece_spans(tokenizer, [tokens[p] for p in doc1_positions], doc1_positions)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "tokens": tokens,
        "doc0_words": doc0_words,
        "doc1_words": doc1_words}


def tokenize_duo_pointwise(tokenizer, device, query, doc0, doc1):
    # same prompt format, but we conceptually attribute only Document0 against a fixed reference in Document1
    return tokenize_duo_pairwise(tokenizer, device, query, doc0, doc1)


def score_duot5_pairwise(model, device, decoder_start_token_id, true_token_id, false_token_id, tok):
    decoder_input_ids = torch.full((1, 1), decoder_start_token_id, dtype=torch.long, device=device)
    with torch.no_grad():
        outputs = model(
            input_ids=tok["input_ids"],
            attention_mask=tok["attention_mask"],
            decoder_input_ids=decoder_input_ids)
        
    logits = outputs.logits[:, 0, :]
    return float((logits[:, true_token_id] - logits[:, false_token_id]).detach().cpu().item())


def score_duot5_pointwise(model, device, decoder_start_token_id, true_token_id, tok):
    decoder_input_ids = torch.full((1, 1), decoder_start_token_id, dtype=torch.long, device=device)
    with torch.no_grad():
        outputs = model(
            input_ids=tok["input_ids"],
            attention_mask=tok["attention_mask"],
            decoder_input_ids=decoder_input_ids)
        
    logits = outputs.logits[:, 0, :]
    return float(logits[:, true_token_id].detach().cpu().item())


def mask_positions(tokenizer, tok, positions):
    perturbed = {
        "input_ids": tok["input_ids"].clone(),
        "attention_mask": tok["attention_mask"].clone()}
    
    perturbed["input_ids"][0, positions] = tokenizer.pad_token_id
    perturbed["attention_mask"][0, positions] = 0

    return perturbed


def compute_loo_duot5_pairwise(tokenizer, model, device, decoder_start_token_id, true_token_id, false_token_id, query, passage_i, passage_j):
    tok = tokenize_duo_pairwise(tokenizer, device, query, passage_i, passage_j)
    original_g = score_duot5_pairwise(model, device, decoder_start_token_id, true_token_id, false_token_id, tok)

    doc_i_rows = []
    for word_idx, span in enumerate(tok["doc0_words"]):
        masked = mask_positions(tokenizer, tok, span["positions"])
        g_masked = score_duot5_pairwise(model, device, decoder_start_token_id, true_token_id, false_token_id, masked)
        doc_i_rows.append(
            {
                "word": span["word"],
                "positions": span["positions"],
                "position": word_idx,
                "support_score": float(original_g - g_masked)})

    doc_j_rows = []
    for word_idx, span in enumerate(tok["doc1_words"]):
        masked = mask_positions(tokenizer, tok, span["positions"])
        g_masked = score_duot5_pairwise(model, device, decoder_start_token_id, true_token_id, false_token_id, masked)
        doc_j_rows.append(
            {
                "word": span["word"],
                "positions": span["positions"],
                "position": word_idx,
                "support_score": float(original_g - g_masked)})

    return {"method": "loo_pairwise", "score": float(original_g), "doc_i": doc_i_rows, "doc_j": doc_j_rows}


def compute_loo_duot5_pointwise(tokenizer, model, device, decoder_start_token_id, true_token_id, query, passage, ref_passage):
    tok = tokenize_duo_pointwise(tokenizer, device, query, passage, ref_passage)
    original_score = score_duot5_pointwise(model, device, decoder_start_token_id, true_token_id, tok)

    doc_rows = []
    for word_idx, span in enumerate(tok["doc0_words"]):
        masked = mask_positions(tokenizer, tok, span["positions"])
        masked_score = score_duot5_pointwise(model, device, decoder_start_token_id, true_token_id, masked)
        doc_rows.append(
            {
                "word": span["word"],
                "positions": span["positions"],
                "position": word_idx,
                "support_score": float(original_score - masked_score)})

    return {"method": "loo_pointwise", "score": float(original_score), "doc": doc_rows}


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
    for side, pid_key, ref_key, attr_key in [
        ("doc_i", "pid_i", "ref_pid_i", "loo_pointwise_i"),
        ("doc_j", "pid_j", "ref_pid_j", "loo_pointwise_j")]:
        for entry in record[attr_key]["doc"]:
            rows.append(
                {
                    "qid": record["qid"],
                    "pid_i": record["pid_i"],
                    "pid_j": record["pid_j"],
                    "pid": record[pid_key],
                    "ref_pid": record[ref_key],
                    "side": side,
                    "segment": "doc",
                    "position": entry["position"],
                    "token": entry["word"],
                    "support_score": entry["support_score"]})
            
    return rows


def main():
    np.random.seed(seed)
    torch.manual_seed(seed)

    tokenizer, model, device, decoder_start_token_id, true_token_id, false_token_id = setup_model()
    pairs_df = load_pairs()
    ranked_df = load_ranked_results()
    reference_pool = build_reference_pool(ranked_df)

    records = []
    failed = []

    for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc="Computing DuoT5 LOO"):
        try:
            loo_pairwise = compute_loo_duot5_pairwise(
                tokenizer,
                model,
                device,
                decoder_start_token_id,
                true_token_id,
                false_token_id,
                row["query"],
                row["passage_i"],
                row["passage_j"])

            ref_i = choose_reference(reference_pool, row["qid"], excluded_pids=[row["pid_i"]])
            ref_j = choose_reference(reference_pool, row["qid"], excluded_pids=[row["pid_j"]])

            loo_pointwise_i = compute_loo_duot5_pointwise(
                tokenizer,
                model,
                device,
                decoder_start_token_id,
                true_token_id,
                row["query"],
                row["passage_i"],
                ref_i["passage"])
            
            loo_pointwise_j = compute_loo_duot5_pointwise(
                tokenizer,
                model,
                device,
                decoder_start_token_id,
                true_token_id,
                row["query"],
                row["passage_j"],
                ref_j["passage"])

            records.append(
                {
                    "qid": row["qid"],
                    "query": row["query"],
                    "pid_i": row["pid_i"],
                    "pid_j": row["pid_j"],
                    "g_score": row["g_score"],
                    "correct_pref": row["correct_pref"],
                    "ref_pid_i": ref_i["pid"],
                    "ref_pid_j": ref_j["pid"],
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

    pairwise_rows = [r for record in records for r in flatten_pairwise_rows(record)]
    pointwise_rows = [r for record in records for r in flatten_pointwise_rows(record)]

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
                "ref_pid_i": record["ref_pid_i"],
                "ref_pid_j": record["ref_pid_j"],
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

    print(f"Device used: {'cpu' if force_cpu else ('mps' if torch.backends.mps.is_available() else 'cpu')}")
    print(f"Saved {len(records)} DuoT5 LOO records to {out_path}")
    print(f"Saved pairwise rows to {out_dir / 'attributions_loo_pairwise_rows.csv'}")
    print(f"Saved pointwise rows to {out_dir / 'attributions_loo_pointwise_rows.csv'}")
    print(f"Saved summary to {out_dir / 'attributions_loo_summary.csv'}")
    if failed:
        print(f"Failed pairs: {len(failed)}")


if __name__ == "__main__":
    main()