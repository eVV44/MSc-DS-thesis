# -- IMPORTS --
from __future__ import annotations
import gc
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


root = Path(__file__).resolve().parents[1]

pairs_file = root / "thesis_runs/duot5/explanations/explanation_pairs_1_per_query_seed42.csv"
out_dir = root / "thesis_runs/duot5/explanations"

model_name = "castorini/duot5-base-msmarco"
max_length = 512
n_steps = 50
seed = 42
force_cpu = True


def load_pairs() -> pd.DataFrame:
    return pd.read_csv(pairs_file, dtype={"qid": str, "pid_i": str, "pid_j": str})


def setup_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    except Exception as exc:
        raise ImportError(
            "Failed to load the DuoT5 tokenizer. Install tokenizer dependencies "
            "(for example `pip install sentencepiece protobuf`) and try again.") from exc

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()

    embedding_layer = model.get_input_embeddings()

    decoder_start_token_id = model.config.decoder_start_token_id
    if decoder_start_token_id is None:
        decoder_start_token_id = tokenizer.pad_token_id

    true_ids = tokenizer.encode("true", add_special_tokens=False)
    false_ids = tokenizer.encode("false", add_special_tokens=False)
    if len(true_ids) != 1 or len(false_ids) != 1:
        raise ValueError("Expected 'true' and 'false' to map to single tokens for DuoT5 scoring.")

    true_token_id = int(true_ids[0])
    false_token_id = int(false_ids[0])
    return tokenizer, model, embedding_layer, device, decoder_start_token_id, true_token_id, false_token_id


def duo_input(query: str, doc0: str, doc1: str) -> str:
    return f"Query: {query} Document0: {doc0} Document1: {doc1} Relevant:"


def ids_to_embeds(embedding_layer, input_ids):
    return embedding_layer(input_ids)


def _tok_len(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def tokenize_duo(tokenizer, device, query: str, doc0: str, doc1: str):
    text = duo_input(query, doc0, doc1)
    encoded = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors="pt")

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    query_prefix = "Query: "
    doc0_prefix = " Document0: "
    doc1_prefix = " Document1: "
    suffix = " Relevant:"

    c1 = query_prefix
    c2 = c1 + query
    c3 = c2 + doc0_prefix
    c4 = c3 + doc0
    c5 = c4 + doc1_prefix
    c6 = c5 + doc1
    c7 = c6 + suffix

    l1 = _tok_len(tokenizer, c1)
    l2 = _tok_len(tokenizer, c2)
    l3 = _tok_len(tokenizer, c3)
    l4 = _tok_len(tokenizer, c4)
    l5 = _tok_len(tokenizer, c5)
    l6 = _tok_len(tokenizer, c6)
    l7 = _tok_len(tokenizer, c7)

    expected_full_len = l7 + 1
    if input_ids.shape[1] != expected_full_len:
        print(
            f"Warning: segment length mismatch. Expected {expected_full_len}, got {input_ids.shape[1]}. "
            "This can happen due to truncation; segment positions will be clipped.")

    query_positions = [pos for pos in range(l1, min(l2, input_ids.shape[1]))]
    doc0_positions = [pos for pos in range(l3, min(l4, input_ids.shape[1]))]
    doc1_positions = [pos for pos in range(l5, min(l6, input_ids.shape[1]))]

    return {
        "text": text,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "tokens": tokens,
        "query_positions": query_positions,
        "doc0_positions": doc0_positions,
        "doc1_positions": doc1_positions,
    }


def forward_duo_from_embeds(model, decoder_start_token_id, true_token_id, false_token_id, input_embeds, attention_mask):
    batch_size = input_embeds.shape[0]
    decoder_input_ids = torch.full(
        (batch_size, 1),
        decoder_start_token_id,
        dtype=torch.long,
        device=input_embeds.device)

    outputs = model(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids)

    logits = outputs.logits[:, 0, :]
    margin = logits[:, true_token_id] - logits[:, false_token_id]
    return margin


def predict_duo_pair(model, tokenizer, device, decoder_start_token_id, true_token_id, false_token_id, query, doc0, doc1):
    tok = tokenize_duo(tokenizer, device, query, doc0, doc1)
    decoder_input_ids = torch.full(
        (1, 1),
        decoder_start_token_id,
        dtype=torch.long,
        device=device)

    with torch.no_grad():
        outputs = model(
            input_ids=tok["input_ids"],
            attention_mask=tok["attention_mask"],
            decoder_input_ids=decoder_input_ids)

    logits = outputs.logits[:, 0, :]
    tf_logits = logits[:, [false_token_id, true_token_id]]
    true_prob = torch.softmax(tf_logits, dim=-1)[:, 1].item()
    margin = (logits[:, true_token_id] - logits[:, false_token_id]).item()
    return true_prob, margin


def make_baseline_input_ids(tokenizer, input_ids):
    baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        for pos, token_id in enumerate(input_ids[0].tolist()):
            if token_id == eos_id:
                baseline_ids[0, pos] = eos_id
    return baseline_ids


def make_baseline_embeds(tokenizer, embedding_layer, input_ids):
    baseline_ids = make_baseline_input_ids(tokenizer, input_ids)
    return ids_to_embeds(embedding_layer, baseline_ids).detach()


def merge_sentencepiece(tokenizer, tokens, scores):
    special_tokens = set(tokenizer.all_special_tokens)
    word_tokens, word_scores = [], []
    current_word, current_score = "", 0.0

    for token, score in zip(tokens, scores):
        if token in special_tokens:
            if current_word:
                word_tokens.append(current_word)
                word_scores.append(current_score)
                current_word, current_score = "", 0.0
            continue

        if token.startswith("▁"):
            if current_word:
                word_tokens.append(current_word)
                word_scores.append(current_score)
            current_word = token.lstrip("▁") or token
            current_score = score
        else:
            current_word += token
            current_score += score

    if current_word:
        word_tokens.append(current_word)
        word_scores.append(current_score)

    return word_tokens, np.array(word_scores)


def aggregate_span(tokenizer, tokens, token_scores, positions):
    span_tokens = [tokens[pos] for pos in positions if pos < len(tokens)]
    span_scores = np.array([token_scores[pos] for pos in positions if pos < len(token_scores)])
    word_tokens, word_scores = merge_sentencepiece(tokenizer, span_tokens, span_scores)
    return {
        "tokens": span_tokens,
        "token_scores": span_scores,
        "word_tokens": word_tokens,
        "word_scores": word_scores,
        "positions": positions}


def aggregate_attributions(tokenizer, attributions, tok):
    token_scores = attributions[0].sum(dim=-1).detach().cpu().numpy()
    full_word_tokens, full_word_scores = merge_sentencepiece(tokenizer, tok["tokens"], token_scores)

    return {
        "tokens": tok["tokens"],
        "token_scores": token_scores,
        "word_tokens": full_word_tokens,
        "word_scores": full_word_scores,
        "query": aggregate_span(tokenizer, tok["tokens"], token_scores, tok["query_positions"]),
        "doc0": aggregate_span(tokenizer, tok["tokens"], token_scores, tok["doc0_positions"]),
        "doc1": aggregate_span(tokenizer, tok["tokens"], token_scores, tok["doc1_positions"])}


def compute_pairwise_ig_t5(
    tokenizer,
    model,
    embedding_layer,
    device,
    decoder_start_token_id,
    true_token_id,
    false_token_id,
    query,
    passage_i,
    passage_j,
):
    tok = tokenize_duo(tokenizer, device, query, passage_i, passage_j)
    input_embeds = ids_to_embeds(embedding_layer, tok["input_ids"]).detach()
    baseline_embeds = make_baseline_embeds(tokenizer, embedding_layer, tok["input_ids"])

    def forward_fn(input_embeds, attention_mask):
        return forward_duo_from_embeds(
            model,
            decoder_start_token_id,
            true_token_id,
            false_token_id,
            input_embeds,
            attention_mask)

    ig = IntegratedGradients(forward_fn)
    attributions, delta = ig.attribute(
        inputs=input_embeds,
        baselines=baseline_embeds,
        additional_forward_args=(tok["attention_mask"],),
        n_steps=n_steps,
        return_convergence_delta=True)

    true_prob, margin = predict_duo_pair(
        model,
        tokenizer,
        device,
        decoder_start_token_id,
        true_token_id,
        false_token_id,
        query,
        passage_i,
        passage_j)

    return {
        "method": "pairwise_ig",
        "input_text": tok["text"],
        "true_prob": float(true_prob),
        "margin": float(margin),
        **aggregate_attributions(tokenizer, attributions, tok),
        "convergence_delta": float(delta.detach().cpu().item()) if torch.is_tensor(delta) else float(delta)}


def main():
    np.random.seed(seed)
    torch.manual_seed(seed)

    tokenizer, model, embedding_layer, device, decoder_start_token_id, true_token_id, false_token_id = setup_model()
    pairs_df = load_pairs()

    attribution_records = []
    failed_pairs = []

    for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc="Computing DuoT5 pairwise IG"):
        try:
            pairwise_ig = compute_pairwise_ig_t5(
                tokenizer,
                model,
                embedding_layer,
                device,
                decoder_start_token_id,
                true_token_id,
                false_token_id,
                row["query"],
                row["passage_i"],
                row["passage_j"])

            attribution_records.append(
                {
                    "qid": row["qid"],
                    "query": row["query"],
                    "pid_i": row["pid_i"],
                    "pid_j": row["pid_j"],
                    "g_score": row["g_score"],
                    "correct_pref": row["correct_pref"],
                    "pairwise_ig": pairwise_ig})
            
        except Exception as exc:
            failed_pairs.append(
                {
                    "qid": row["qid"],
                    "pid_i": row["pid_i"],
                    "pid_j": row["pid_j"],
                    "error": str(exc)})
            
            print(f"Error on qid={row['qid']}, pid_i={row['pid_i']}, pid_j={row['pid_j']}: {exc}")
        finally:
            gc.collect()
            if torch.backends.mps.is_available():
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "attributions_pairwise_ig.pkl"
    with out_path.open("wb") as f:
        pickle.dump(attribution_records, f)

    summary = pd.DataFrame(
        [
            {
                "qid": record["qid"],
                "pid_i": record["pid_i"],
                "pid_j": record["pid_j"],
                "g_score": record["g_score"],
                "correct_pref": record["correct_pref"],
                "direct_true_prob": record["pairwise_ig"]["true_prob"],
                "direct_margin": record["pairwise_ig"]["margin"],
                "pw_ig_delta": record["pairwise_ig"]["convergence_delta"],
                "doc0_top_word": (
                    record["pairwise_ig"]["doc0"]["word_tokens"][
                        int(np.argmax(record["pairwise_ig"]["doc0"]["word_scores"]))]
                    if len(record["pairwise_ig"]["doc0"]["word_tokens"]) > 0
                    else ""),

                "doc1_top_word": (
                    record["pairwise_ig"]["doc1"]["word_tokens"][
                        int(np.argmax(record["pairwise_ig"]["doc1"]["word_scores"]))]
                    if len(record["pairwise_ig"]["doc1"]["word_tokens"]) > 0
                    else ""),}
            for record in attribution_records])
    
    summary_path = out_dir / "attributions_pairwise_ig_summary.csv"
    summary.to_csv(summary_path, index=False)

    if failed_pairs:
        pd.DataFrame(failed_pairs).to_csv(out_dir / "attributions_pairwise_ig_failed.csv", index=False)

    print(f"Device used: {'cpu' if force_cpu else ('mps' if torch.backends.mps.is_available() else 'cpu')}")
    print(f"IG integration steps: {n_steps}")
    print(f"Saved {len(attribution_records)} DuoT5 pairwise attribution records to {out_path}")
    print(f"Saved summary to {summary_path}")
    if failed_pairs:
        print(f"Failed pairs: {len(failed_pairs)}")


if __name__ == "__main__":
    main()