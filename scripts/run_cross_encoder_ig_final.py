# -- IMPORTS --
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from sentence_transformers import CrossEncoder
from tqdm.auto import tqdm
from transformers import AutoTokenizer


root = Path(__file__).resolve().parents[1]

pairs_file = root / "thesis_runs/cross_encoder/explanations/explanation_pairs_1_per_query_on_duot5_queries_seed42.csv"
out_dir = root / "thesis_runs/cross_encoder/explanations"

model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
max_length = 512
n_steps = 50
seed = 42


def setup_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ce = CrossEncoder(model_name, max_length=max_length)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    ce.model = ce.model.to(device)
    ce.model.eval()
    return tokenizer, ce.model, device


def load_pairs() -> pd.DataFrame:
    loaded_pairs = pd.read_csv(pairs_file, dtype={"qid": str, "pid_i": str, "pid_j": str})
    return loaded_pairs


def tokenize_pair(tokenizer, device, query: str, passage: str):
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

    tokenized_pair = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "tokens": tokens,
        "sep_idx": sep_idx}
    
    return tokenized_pair


def ids_to_embeds(embedding_layer, input_ids):
    return embedding_layer(input_ids)


def forward_from_embeds(bert_model, input_embeds, attention_mask, token_type_ids=None):
    kwargs = {
        "inputs_embeds": input_embeds,
        "attention_mask": attention_mask}
    
    if token_type_ids is not None:
        kwargs["token_type_ids"] = token_type_ids

    outputs = bert_model(**kwargs)
    return outputs.logits.squeeze(-1)


def make_baseline_input_ids(tokenizer, input_ids):
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define a pad token for the IG baseline.")

    baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
    special_token_ids = {
        token_id
        for token_id in [tokenizer.cls_token_id, tokenizer.sep_token_id]
        if token_id is not None}

    for pos, token_id in enumerate(input_ids[0].tolist()):
        if token_id in special_token_ids:
            baseline_ids[0, pos] = token_id

    return baseline_ids


def make_baseline_embeds(tokenizer, embedding_layer, input_ids):
    baseline_ids = make_baseline_input_ids(tokenizer, input_ids)
    return ids_to_embeds(embedding_layer, baseline_ids).detach()


def merge_wordpieces(tokenizer, tokens, scores):
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
        if token.startswith("##") and current_word:
            current_word += token[2:]
            current_score += score
        else:
            if current_word:
                word_tokens.append(current_word)
                word_scores.append(current_score)
            current_word = token.replace("##", "")
            current_score = score

    if current_word:
        word_tokens.append(current_word)
        word_scores.append(current_score)

    return word_tokens, np.array(word_scores)


def aggregate_attributions(tokenizer, attributions, tokens, sep_idx):
    token_scores = attributions[0].sum(dim=-1).detach().cpu().numpy()

    query_tokens = tokens[1:sep_idx]
    query_token_scores = token_scores[1:sep_idx]
    doc_tokens = tokens[sep_idx + 1:-1]
    doc_token_scores = token_scores[sep_idx + 1:-1]

    word_tokens, word_scores = merge_wordpieces(tokenizer, tokens, token_scores)
    query_word_tokens, query_word_scores = merge_wordpieces(tokenizer, query_tokens, query_token_scores)
    doc_word_tokens, doc_word_scores = merge_wordpieces(tokenizer, doc_tokens, doc_token_scores)

    aggregated = {
        "tokens": tokens,
        "token_scores": token_scores,
        "word_tokens": word_tokens,
        "word_scores": word_scores,
        "query_tokens": query_tokens,
        "query_token_scores": query_token_scores,
        "query_word_tokens": query_word_tokens,
        "query_word_scores": query_word_scores,
        "doc_tokens": doc_tokens,
        "doc_token_scores": doc_token_scores,
        "doc_word_tokens": doc_word_tokens,
        "doc_word_scores": doc_word_scores,
        "sep_idx": sep_idx}
    
    return aggregated


def compute_pointwise_ig(tokenizer, bert_model, device, embedding_layer, query, passage):
    tok = tokenize_pair(tokenizer, device, query, passage)
    input_embeds = ids_to_embeds(embedding_layer, tok["input_ids"]).detach()
    baseline_embeds = make_baseline_embeds(tokenizer, embedding_layer, tok["input_ids"])

    def pointwise_forward(input_embeds, attention_mask, token_type_ids):
        return forward_from_embeds(bert_model, input_embeds, attention_mask, token_type_ids)

    ig = IntegratedGradients(pointwise_forward)
    attributions, delta = ig.attribute(
        inputs=input_embeds,
        baselines=baseline_embeds,
        additional_forward_args=(tok["attention_mask"], tok["token_type_ids"]),
        n_steps=n_steps,
        return_convergence_delta=True)

    pairwise_ig = {
        "method": "pointwise_ig",
        **aggregate_attributions(tokenizer, attributions, tok["tokens"], tok["sep_idx"]),
        "convergence_delta": float(delta.detach().cpu().item()) if torch.is_tensor(delta) else float(delta)}

    return pairwise_ig


def compute_pairwise_ig(tokenizer, bert_model, device, embedding_layer, query, passage_i, passage_j):
    tok_i = tokenize_pair(tokenizer, device, query, passage_i)
    tok_j = tokenize_pair(tokenizer, device, query, passage_j)

    input_embeds_i = ids_to_embeds(embedding_layer, tok_i["input_ids"]).detach()
    input_embeds_j = ids_to_embeds(embedding_layer, tok_j["input_ids"]).detach()
    baseline_embeds_i = make_baseline_embeds(tokenizer, embedding_layer, tok_i["input_ids"])
    baseline_embeds_j = make_baseline_embeds(tokenizer, embedding_layer, tok_j["input_ids"])

    def forward_pairwise(input_embeds_i, input_embeds_j, attention_mask_i, token_type_ids_i, attention_mask_j, token_type_ids_j):
        score_i = forward_from_embeds(bert_model, input_embeds_i, attention_mask_i, token_type_ids_i)
        score_j = forward_from_embeds(bert_model, input_embeds_j, attention_mask_j, token_type_ids_j)
        return score_i - score_j

    ig = IntegratedGradients(forward_pairwise)
    attributions, delta = ig.attribute(
        inputs=(input_embeds_i, input_embeds_j),
        baselines=(baseline_embeds_i, baseline_embeds_j),
        additional_forward_args=(
            tok_i["attention_mask"],
            tok_i["token_type_ids"],
            tok_j["attention_mask"],
            tok_j["token_type_ids"]),
        n_steps=n_steps,
        return_convergence_delta=True)

    attr_i, attr_j = attributions
    pairwise_ig_result = {
        "method": "pairwise_ig",
        "doc_i": aggregate_attributions(tokenizer, attr_i, tok_i["tokens"], tok_i["sep_idx"]),
        "doc_j": aggregate_attributions(tokenizer, attr_j, tok_j["tokens"], tok_j["sep_idx"]),
        "convergence_delta": float(delta.detach().cpu().item()) if torch.is_tensor(delta) else float(delta)}

    return pairwise_ig_result

def main():
    np.random.seed(seed)
    torch.manual_seed(seed)

    tokenizer, bert_model, device = setup_model()
    embedding_layer = bert_model.get_input_embeddings()
    pairs_df = load_pairs()

    attribution_records = []
    failed_pairs = []

    for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc="Computing cross-encoder IG"):
        try:
            pt_ig_i = compute_pointwise_ig(tokenizer, bert_model, device, embedding_layer, row["query"], row["passage_i"])
            pt_ig_j = compute_pointwise_ig(tokenizer, bert_model, device, embedding_layer, row["query"], row["passage_j"])
            pw_ig = compute_pairwise_ig(
                tokenizer,
                bert_model,
                device,
                embedding_layer,
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
                    "pointwise_ig_i": pt_ig_i,
                    "pointwise_ig_j": pt_ig_j,
                    "pairwise_ig": pw_ig})
            
        except Exception as exc:
            failed_pairs.append({"qid": row["qid"], "pid_i": row["pid_i"], "pid_j": row["pid_j"], "error": str(exc)})
            print(f"Error on qid={row['qid']}, pid_i={row['pid_i']}, pid_j={row['pid_j']}: {exc}")

    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "attributions_ig.pkl"
    with out_path.open("wb") as f:
        pickle.dump(attribution_records, f)

    summary = pd.DataFrame(
        [
            {
                "qid": r["qid"],
                "pid_i": r["pid_i"],
                "pid_j": r["pid_j"],
                "g_score": r["g_score"],
                "correct_pref": r["correct_pref"],
                "pw_ig_delta": r["pairwise_ig"]["convergence_delta"],
                "pt_ig_i_delta": r["pointwise_ig_i"]["convergence_delta"],
                "pt_ig_j_delta": r["pointwise_ig_j"]["convergence_delta"],
                "pw_top_doc_i_word": (
                    r["pairwise_ig"]["doc_i"]["doc_word_tokens"][
                        int(np.argmax(r["pairwise_ig"]["doc_i"]["doc_word_scores"]))]
                    if len(r["pairwise_ig"]["doc_i"]["doc_word_tokens"]) > 0
                    else ""),
                "pw_top_doc_j_word": (
                    r["pairwise_ig"]["doc_j"]["doc_word_tokens"][
                        int(np.argmax(r["pairwise_ig"]["doc_j"]["doc_word_scores"]))]
                    if len(r["pairwise_ig"]["doc_j"]["doc_word_tokens"]) > 0
                    else "")}
            for r in attribution_records])
    
    summary.to_csv(out_dir / "attributions_ig_summary.csv", index=False)

    if failed_pairs:
        pd.DataFrame(failed_pairs).to_csv(out_dir / "attributions_ig_failed.csv", index=False)

    print(f"Saved {len(attribution_records)} attribution records to {out_path}")
    print(f"Saved summary to {out_dir / 'attributions_ig_summary.csv'}")
    if failed_pairs:
        print(f"Failed pairs: {len(failed_pairs)}")


if __name__ == "__main__":
    main()