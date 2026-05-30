# -- IMPORTS --
from __future__ import annotations
import argparse
import gc
import os
import pickle
import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from sentence_transformers import CrossEncoder
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


root = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(root / "thesis_runs/shared/mplconfig"))

from run_faithfulness_final import (
    build_support_table,
    load_cross_records,
    load_duot5_records,
    pair_key,
    record_target_score,
    support_rows_from_attr_cross,
    support_rows_from_attr_t5,
    support_rows_from_loo)

seed = 42
max_length = 512
ig_steps = 50
default_perturb_sizes = [1, 3]
force_duot5_cpu = True

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "could",
    "did", "do", "does", "doing", "down", "during",
    "each",
    "few", "for", "from", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "itself",
    "just",
    "me", "more", "most", "my", "myself",
    "no", "nor", "not", "now",
    "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own",
    "s", "same", "she", "should", "so", "some", "such",
    "t", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too",
    "under", "until", "up",
    "very",
    "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with",
    "you", "your", "yours", "yourself", "yourselves"}

EXPERIMENTS = {
    "cross_encoder": {
        "label": "Cross-encoder",
        "family": "cross_encoder",
        "pair_file": root / "thesis_runs/cross_encoder/stability/stability_pairs_50_queries_seed42.csv",
        "explanation_dir": root / "thesis_runs/cross_encoder/explanations",
        "stability_dir": root / "thesis_runs/cross_encoder/stability",
        "methods": ["pairwise_ig", "pointwise_ig", "loo_pairwise", "loo_pointwise"],
        "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2"},

    "duot5": {
        "label": "DuoT5",
        "family": "duot5",
        "pair_file": root / "thesis_runs/duot5/stability/stability_pairs_50_queries_seed42.csv",
        "explanation_dir": root / "thesis_runs/duot5/explanations",
        "stability_dir": root / "thesis_runs/duot5/stability",
        "ranked_results_file": root / "thesis_runs/duot5/reranking/duot5_final/ranked_results.csv",
        "methods": ["pairwise_ig", "pointwise_ig", "loo_pairwise", "loo_pointwise"],
        "model_name": "castorini/duot5-base-msmarco"}}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--perturb-sizes", type=int, nargs="+", default=default_perturb_sizes)
    return parser.parse_args()


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def normalize_text_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def query_lexicon(query: str) -> set[str]:
    return {normalize_text_token(tok) for tok in re.findall(r"\w+", query.lower()) if normalize_text_token(tok)}


def is_low_information_word(word: str, query_terms: set[str]) -> bool:
    norm = normalize_text_token(word)
    if not norm:
        return True
    if norm in query_terms:
        return False
    if norm in STOPWORDS:
        return True
    if len(norm) <= 2:
        return True
    return False


def bootstrapless_mean(values):
    values = pd.Series(values, dtype=float).dropna()
    return float(values.mean()) if not values.empty else np.nan


def spearman_similarity(x, y):
    x = pd.Series(x, dtype=float)
    y = pd.Series(y, dtype=float)
    valid = ~(x.isna() | y.isna())
    x = x[valid]
    y = y[valid]
    if len(x) < 2:
        return np.nan
    xr = x.rank(method="average")
    yr = y.rank(method="average")
    if xr.nunique() < 2 or yr.nunique() < 2:
        return np.nan
    return float(xr.corr(yr, method="pearson"))


def filter_subset_records(records, pair_file: Path):
    subset_df = pd.read_csv(pair_file, dtype={"qid": str, "pid_i": str, "pid_j": str})
    allowed = set(pair_key(r["qid"], r["pid_i"], r["pid_j"]) for r in subset_df.to_dict("records"))
    return [r for r in records if pair_key(r["qid"], r["pid_i"], r["pid_j"]) in allowed]


def setup_cross_runtime(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], local_files_only=True)
    ce = CrossEncoder(cfg["model_name"], max_length=max_length, local_files_only=True)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = ce.model.to(device)
    model.eval()
    embedding_layer = model.get_input_embeddings()
    return tokenizer, model, embedding_layer, device


def tokenize_cross(tokenizer, device, query, passage):
    encoded = tokenizer(query, passage, max_length=max_length, truncation=True, padding=False, return_tensors="pt")
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
        "sep_idx": sep_idx,
        "doc_positions": doc_positions,
        "doc_words": doc_words}


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
        words.append({"word": word, "positions": poss})
    return words


def score_cross(model, tok):
    kwargs = {"input_ids": tok["input_ids"], "attention_mask": tok["attention_mask"]}
    if tok["token_type_ids"] is not None:
        kwargs["token_type_ids"] = tok["token_type_ids"]
    with torch.no_grad():
        outputs = model(**kwargs)
    return float(outputs.logits.squeeze(-1).detach().cpu().item())


def ids_to_embeds(embedding_layer, input_ids):
    return embedding_layer(input_ids)


def forward_cross_from_embeds(model, input_embeds, attention_mask, token_type_ids=None):
    kwargs = {"inputs_embeds": input_embeds, "attention_mask": attention_mask}
    if token_type_ids is not None:
        kwargs["token_type_ids"] = token_type_ids
    outputs = model(**kwargs)
    return outputs.logits.squeeze(-1)


def make_cross_baseline_input_ids(tokenizer, input_ids):
    baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
    special_token_ids = {t for t in [tokenizer.cls_token_id, tokenizer.sep_token_id] if t is not None}
    for pos, token_id in enumerate(input_ids[0].tolist()):
        if token_id in special_token_ids:
            baseline_ids[0, pos] = token_id
    return baseline_ids


def make_cross_baseline_embeds(tokenizer, embedding_layer, input_ids):
    baseline_ids = make_cross_baseline_input_ids(tokenizer, input_ids)
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


def aggregate_cross_attributions(tokenizer, attributions, tokens, sep_idx):
    token_scores = attributions[0].sum(dim=-1).detach().cpu().numpy()
    query_tokens = tokens[1:sep_idx]
    query_token_scores = token_scores[1:sep_idx]
    doc_tokens = tokens[sep_idx + 1:-1]
    doc_token_scores = token_scores[sep_idx + 1:-1]
    word_tokens, word_scores = merge_wordpieces(tokenizer, tokens, token_scores)
    query_word_tokens, query_word_scores = merge_wordpieces(tokenizer, query_tokens, query_token_scores)
    doc_word_tokens, doc_word_scores = merge_wordpieces(tokenizer, doc_tokens, doc_token_scores)
    return {
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


def apply_cross_mask(tokenizer, tok, positions):
    perturbed = {
        "input_ids": tok["input_ids"].clone(),
        "attention_mask": tok["attention_mask"].clone(),
        "token_type_ids": None if tok["token_type_ids"] is None else tok["token_type_ids"].clone(),
        "tokens": list(tok["tokens"]),
        "sep_idx": tok["sep_idx"],
        "doc_positions": list(tok["doc_positions"]),
        "doc_words": list(tok["doc_words"])}
    
    positions = sorted({int(p) for p in positions if 0 <= int(p) < tok["input_ids"].shape[1]})
    if positions:
        perturbed["input_ids"][0, positions] = tokenizer.mask_token_id
        perturbed["tokens"] = tokenizer.convert_ids_to_tokens(perturbed["input_ids"][0].tolist())
    return perturbed


def compute_cross_pointwise_ig_from_tok(tokenizer, model, embedding_layer, tok):
    input_embeds = ids_to_embeds(embedding_layer, tok["input_ids"]).detach()
    baseline_embeds = make_cross_baseline_embeds(tokenizer, embedding_layer, tok["input_ids"])
    ig = IntegratedGradients(lambda embeds, mask, tt: forward_cross_from_embeds(model, embeds, mask, tt))
    attributions, delta = ig.attribute(
        inputs=input_embeds,
        baselines=baseline_embeds,
        additional_forward_args=(tok["attention_mask"], tok["token_type_ids"]),
        n_steps=ig_steps,
        return_convergence_delta=True)
    
    return {
        "method": "pointwise_ig",
        **aggregate_cross_attributions(tokenizer, attributions, tok["tokens"], tok["sep_idx"]),
        "convergence_delta": float(delta.detach().cpu().item()) if torch.is_tensor(delta) else float(delta)}


def compute_cross_pairwise_ig_from_tok(tokenizer, model, embedding_layer, tok_i, tok_j):
    input_embeds_i = ids_to_embeds(embedding_layer, tok_i["input_ids"]).detach()
    input_embeds_j = ids_to_embeds(embedding_layer, tok_j["input_ids"]).detach()
    baseline_embeds_i = make_cross_baseline_embeds(tokenizer, embedding_layer, tok_i["input_ids"])
    baseline_embeds_j = make_cross_baseline_embeds(tokenizer, embedding_layer, tok_j["input_ids"])

    def forward_pairwise(input_embeds_i, input_embeds_j, attention_mask_i, token_type_ids_i, attention_mask_j, token_type_ids_j):
        score_i = forward_cross_from_embeds(model, input_embeds_i, attention_mask_i, token_type_ids_i)
        score_j = forward_cross_from_embeds(model, input_embeds_j, attention_mask_j, token_type_ids_j)
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

        n_steps=ig_steps,
        return_convergence_delta=True)
    
    attr_i, attr_j = attributions

    return {
        "method": "pairwise_ig",
        "doc_i": aggregate_cross_attributions(tokenizer, attr_i, tok_i["tokens"], tok_i["sep_idx"]),
        "doc_j": aggregate_cross_attributions(tokenizer, attr_j, tok_j["tokens"], tok_j["sep_idx"]),
        "convergence_delta": float(delta.detach().cpu().item()) if torch.is_tensor(delta) else float(delta)}


def compute_cross_loo_pointwise_from_tok(tokenizer, model, tok):
    original_score = score_cross(model, tok)
    rows = []
    for word_idx, span in enumerate(tok["doc_words"]):
        masked = apply_cross_mask(tokenizer, tok, span["positions"])
        masked_score = score_cross(model, masked)
        rows.append({"word": span["word"], "positions": span["positions"], "position": word_idx, "support_score": float(original_score - masked_score)})
    return {"method": "loo_pointwise", "score": float(original_score), "doc": rows}


def compute_cross_loo_pairwise_from_tok(tokenizer, model, tok_i, tok_j):
    score_i = score_cross(model, tok_i)
    score_j = score_cross(model, tok_j)
    original_g = score_i - score_j
    doc_i_rows = []
    for word_idx, span in enumerate(tok_i["doc_words"]):
        masked_i = apply_cross_mask(tokenizer, tok_i, span["positions"])
        g_masked = score_cross(model, masked_i) - score_j
        doc_i_rows.append({"word": span["word"], "positions": span["positions"], "position": word_idx, "support_score": float(original_g - g_masked)})
    doc_j_rows = []
    for word_idx, span in enumerate(tok_j["doc_words"]):
        masked_j = apply_cross_mask(tokenizer, tok_j, span["positions"])
        g_masked = score_i - score_cross(model, masked_j)
        doc_j_rows.append({"word": span["word"], "positions": span["positions"], "position": word_idx, "support_score": float(original_g - g_masked)})
    return {"method": "loo_pairwise", "score": float(original_g), "doc_i": doc_i_rows, "doc_j": doc_j_rows}


def build_cross_candidates(tokenizer, device, query, passage_i, passage_j):
    q_terms = query_lexicon(query)
    tok_i = tokenize_cross(tokenizer, device, query, passage_i)
    tok_j = tokenize_cross(tokenizer, device, query, passage_j)
    candidates = []
    for side, tok in [("doc_i", tok_i), ("doc_j", tok_j)]:
        for idx, span in enumerate(tok["doc_words"]):
            if is_low_information_word(span["word"], q_terms):
                candidates.append({"side": side, "word": span["word"], "positions": list(span["positions"]), "position": idx})
    return tok_i, tok_j, candidates


def setup_duot5_runtime(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=False, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model_name"], local_files_only=True)
    device = torch.device("cpu") if force_duot5_cpu else (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
    model = model.to(device)
    model.eval()
    embedding_layer = model.get_input_embeddings()
    decoder_start_token_id = model.config.decoder_start_token_id
    if decoder_start_token_id is None:
        decoder_start_token_id = tokenizer.pad_token_id
    true_ids = tokenizer.encode("true", add_special_tokens=False)
    false_ids = tokenizer.encode("false", add_special_tokens=False)
    if len(true_ids) != 1 or len(false_ids) != 1:
        raise ValueError("Expected 'true' and 'false' to map to single tokens.")
    return tokenizer, model, embedding_layer, device, int(decoder_start_token_id), int(true_ids[0]), int(false_ids[0])


def duo_input(query, doc0, doc1):
    return f"Query: {query} Document0: {doc0} Document1: {doc1} Relevant:"


def _tok_len(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def merge_sentencepiece_spans(tokens, positions, special_tokens):
    spans = []
    current_tokens, current_positions = [], []
    for token, pos in zip(tokens, positions):
        if token in special_tokens:
            if current_tokens:
                spans.append((current_tokens, current_positions))
                current_tokens, current_positions = [], []
            continue
        if token.startswith("▁"):
            if current_tokens:
                spans.append((current_tokens, current_positions))
            current_tokens = [token]
            current_positions = [int(pos)]
        else:
            if not current_tokens:
                current_tokens = [token]
                current_positions = [int(pos)]
            else:
                current_tokens.append(token)
                current_positions.append(int(pos))
    if current_tokens:
        spans.append((current_tokens, current_positions))
    words = []
    for toks, poss in spans:
        word = toks[0].lstrip("▁") or toks[0]
        for tok in toks[1:]:
            word += tok
        words.append({"word": word, "positions": poss})
    return words


def tokenize_duot5(tokenizer, device, query, doc0, doc1):
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
    c7 = c6 + " Relevant:"

    l1 = _tok_len(tokenizer, c1)
    l2 = _tok_len(tokenizer, c2)
    l3 = _tok_len(tokenizer, c3)
    l4 = _tok_len(tokenizer, c4)
    l5 = _tok_len(tokenizer, c5)
    l6 = _tok_len(tokenizer, c6)
    l7 = _tok_len(tokenizer, c7)

    if input_ids.shape[1] != l7 + 1:
        pass

    query_positions = [pos for pos in range(l1, min(l2, input_ids.shape[1]))]
    doc0_positions = [pos for pos in range(l3, min(l4, input_ids.shape[1]))]
    doc1_positions = [pos for pos in range(l5, min(l6, input_ids.shape[1]))]
    special_tokens = set(tokenizer.all_special_tokens)
    doc0_words = merge_sentencepiece_spans([tokens[p] for p in doc0_positions], doc0_positions, special_tokens)
    doc1_words = merge_sentencepiece_spans([tokens[p] for p in doc1_positions], doc1_positions, special_tokens)

    return {
        "text": text,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "tokens": tokens,
        "query_positions": query_positions,
        "doc0_positions": doc0_positions,
        "doc1_positions": doc1_positions,
        "doc0_words": doc0_words,
        "doc1_words": doc1_words}


def make_duot5_baseline_input_ids(tokenizer, input_ids):
    baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        for pos, token_id in enumerate(input_ids[0].tolist()):
            if token_id == eos_id:
                baseline_ids[0, pos] = eos_id
    return baseline_ids


def make_duot5_baseline_embeds(tokenizer, embedding_layer, input_ids):
    baseline_ids = make_duot5_baseline_input_ids(tokenizer, input_ids)
    return embedding_layer(baseline_ids).detach()


def apply_duot5_mask(tokenizer, tok, positions):
    perturbed = {
        "input_ids": tok["input_ids"].clone(),
        "attention_mask": tok["attention_mask"].clone(),
        "tokens": list(tok["tokens"]),
        "query_positions": list(tok["query_positions"]),
        "doc0_positions": list(tok["doc0_positions"]),
        "doc1_positions": list(tok["doc1_positions"]),
        "doc0_words": list(tok["doc0_words"]),
        "doc1_words": list(tok["doc1_words"])}
    
    positions = sorted({int(p) for p in positions if 0 <= int(p) < tok["input_ids"].shape[1]})
    if positions:
        perturbed["input_ids"][0, positions] = tokenizer.pad_token_id
        perturbed["attention_mask"][0, positions] = 0
        perturbed["tokens"] = tokenizer.convert_ids_to_tokens(perturbed["input_ids"][0].tolist())
    return perturbed


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


def aggregate_duot5_span(tokenizer, tokens, token_scores, positions):
    span_tokens = [tokens[pos] for pos in positions if pos < len(tokens)]
    span_scores = np.array([token_scores[pos] for pos in positions if pos < len(token_scores)])
    word_tokens, word_scores = merge_sentencepiece(tokenizer, span_tokens, span_scores)
    return {"tokens": span_tokens, "token_scores": span_scores, "word_tokens": word_tokens, "word_scores": word_scores, "positions": positions}


def aggregate_duot5_attributions(tokenizer, attributions, tok):
    token_scores = attributions[0].sum(dim=-1).detach().cpu().numpy()
    full_word_tokens, full_word_scores = merge_sentencepiece(tokenizer, tok["tokens"], token_scores)
    return {
        "tokens": tok["tokens"],
        "token_scores": token_scores,
        "word_tokens": full_word_tokens,
        "word_scores": full_word_scores,
        "query": aggregate_duot5_span(tokenizer, tok["tokens"], token_scores, tok["query_positions"]),
        "doc0": aggregate_duot5_span(tokenizer, tok["tokens"], token_scores, tok["doc0_positions"]),
        "doc1": aggregate_duot5_span(tokenizer, tok["tokens"], token_scores, tok["doc1_positions"])}


def score_duot5_pairwise(runtime, tok):
    tokenizer, model, embedding_layer, device, decoder_start_token_id, true_token_id, false_token_id = runtime
    decoder_input_ids = torch.full((1, 1), decoder_start_token_id, dtype=torch.long, device=device)
    with torch.no_grad():
        outputs = model(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"], decoder_input_ids=decoder_input_ids)
    logits = outputs.logits[:, 0, :]
    return float((logits[:, true_token_id] - logits[:, false_token_id]).detach().cpu().item())


def score_duot5_pointwise(runtime, tok):
    tokenizer, model, embedding_layer, device, decoder_start_token_id, true_token_id, false_token_id = runtime
    decoder_input_ids = torch.full((1, 1), decoder_start_token_id, dtype=torch.long, device=device)
    with torch.no_grad():
        outputs = model(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"], decoder_input_ids=decoder_input_ids)
    logits = outputs.logits[:, 0, :]
    return float(logits[:, true_token_id].detach().cpu().item())


def forward_duot5_pairwise_from_embeds(model, decoder_start_token_id, true_token_id, false_token_id, input_embeds, attention_mask):
    decoder_input_ids = torch.full((input_embeds.shape[0], 1), decoder_start_token_id, dtype=torch.long, device=input_embeds.device)
    outputs = model(inputs_embeds=input_embeds, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
    logits = outputs.logits[:, 0, :]
    return logits[:, true_token_id] - logits[:, false_token_id]


def forward_duot5_true_logit_from_embeds(model, decoder_start_token_id, true_token_id, input_embeds, attention_mask):
    decoder_input_ids = torch.full((input_embeds.shape[0], 1), decoder_start_token_id, dtype=torch.long, device=input_embeds.device)
    outputs = model(inputs_embeds=input_embeds, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
    logits = outputs.logits[:, 0, :]
    return logits[:, true_token_id]


def compute_duot5_pairwise_ig_from_tok(runtime, tok):
    tokenizer, model, embedding_layer, device, decoder_start_token_id, true_token_id, false_token_id = runtime
    input_embeds = embedding_layer(tok["input_ids"]).detach()
    baseline_embeds = make_duot5_baseline_embeds(tokenizer, embedding_layer, tok["input_ids"])
    ig = IntegratedGradients(lambda embeds, mask: forward_duot5_pairwise_from_embeds(model, decoder_start_token_id, true_token_id, false_token_id, embeds, mask))
    attributions, delta = ig.attribute(
        inputs=input_embeds,
        baselines=baseline_embeds,
        additional_forward_args=(tok["attention_mask"],),
        n_steps=ig_steps,
        return_convergence_delta=True)
    
    return {
        "method": "pairwise_ig",
        **aggregate_duot5_attributions(tokenizer, attributions, tok),
        "convergence_delta": float(delta.detach().cpu().item()) if torch.is_tensor(delta) else float(delta)}


def compute_duot5_pointwise_ig_from_tok(runtime, tok):
    tokenizer, model, embedding_layer, device, decoder_start_token_id, true_token_id, false_token_id = runtime
    input_embeds = embedding_layer(tok["input_ids"]).detach()
    baseline_embeds = make_duot5_baseline_embeds(tokenizer, embedding_layer, tok["input_ids"])
    ig = IntegratedGradients(lambda embeds, mask: forward_duot5_true_logit_from_embeds(model, decoder_start_token_id, true_token_id, embeds, mask))
    attributions, delta = ig.attribute(
        inputs=input_embeds,
        baselines=baseline_embeds,
        additional_forward_args=(tok["attention_mask"],),
        n_steps=ig_steps,
        return_convergence_delta=True)
    
    return {
        "method": "pointwise_ig",
        **aggregate_duot5_attributions(tokenizer, attributions, tok),
        "convergence_delta": float(delta.detach().cpu().item()) if torch.is_tensor(delta) else float(delta)}


def compute_duot5_loo_pairwise_from_tok(runtime, tok):
    original_score = score_duot5_pairwise(runtime, tok)
    doc_i_rows = []
    for idx, span in enumerate(tok["doc0_words"]):
        masked = apply_duot5_mask(runtime[0], tok, span["positions"])
        masked_score = score_duot5_pairwise(runtime, masked)
        doc_i_rows.append({"word": span["word"], "positions": span["positions"], "position": idx, "support_score": float(original_score - masked_score)})
    doc_j_rows = []
    for idx, span in enumerate(tok["doc1_words"]):
        masked = apply_duot5_mask(runtime[0], tok, span["positions"])
        masked_score = score_duot5_pairwise(runtime, masked)
        doc_j_rows.append({"word": span["word"], "positions": span["positions"], "position": idx, "support_score": float(original_score - masked_score)})
    return {"method": "loo_pairwise", "score": float(original_score), "doc_i": doc_i_rows, "doc_j": doc_j_rows}


def compute_duot5_loo_pointwise_from_tok(runtime, tok):
    original_score = score_duot5_pointwise(runtime, tok)
    rows = []
    for idx, span in enumerate(tok["doc0_words"]):
        masked = apply_duot5_mask(runtime[0], tok, span["positions"])
        masked_score = score_duot5_pointwise(runtime, masked)
        rows.append({"word": span["word"], "positions": span["positions"], "position": idx, "support_score": float(original_score - masked_score)})
    return {"method": "loo_pointwise", "score": float(original_score), "doc": rows}


def build_duot5_candidates(tokenizer, device, query, passage_i, passage_j):
    q_terms = query_lexicon(query)
    tok = tokenize_duot5(tokenizer, device, query, passage_i, passage_j)
    candidates = []
    for side, key in [("doc_i", "doc0_words"), ("doc_j", "doc1_words")]:
        for idx, span in enumerate(tok[key]):
            if is_low_information_word(span["word"], q_terms):
                candidates.append({"side": side, "word": span["word"], "positions": list(span["positions"]), "position": idx})
    return tok, candidates


def select_perturbation(candidates, k, seed_value):
    if not candidates:
        return []
    rng = np.random.default_rng(seed_value)
    n = min(k, len(candidates))
    indices = rng.choice(len(candidates), size=n, replace=False)
    return [candidates[int(i)] for i in indices]


def positions_by_side(selected):
    out = {"doc_i": [], "doc_j": []}
    for row in selected:
        out[row["side"]].extend(int(p) for p in row["positions"])
    return {k: sorted(set(v)) for k, v in out.items()}


def exclude_perturbed_rows(table, perturbed_positions_by_side):
    if table.empty:
        return table
    keep = []
    for _, row in table.iterrows():
        blocked = set(int(p) for p in perturbed_positions_by_side.get(row["side"], []))
        row_positions = row.get("positions")
        if isinstance(row_positions, list):
            intersects = any(int(p) in blocked for p in row_positions)
        else:
            intersects = int(row["position"]) in blocked
        keep.append(not intersects)
    return table.loc[keep].reset_index(drop=True)


def align_side_vectors(original_table, perturbed_table, side):
    orig = original_table[original_table["side"] == side].copy()
    pert = perturbed_table[perturbed_table["side"] == side].copy()
    if orig.empty or pert.empty:
        return np.nan, 0
    orig = orig[["position", "support_score"]].rename(columns={"support_score": "orig"})
    pert = pert[["position", "support_score"]].rename(columns={"support_score": "pert"})
    merged = orig.merge(pert, on="position", how="inner").sort_values("position")
    if merged.empty:
        return np.nan, 0
    return spearman_similarity(merged["orig"].to_numpy(), merged["pert"].to_numpy()), int(len(merged))


def build_perturbed_support_table_cross(record, method, runtime, perturbed_positions_by_side):
    tokenizer, model, embedding_layer, device = runtime
    tok_i = tokenize_cross(tokenizer, device, record["query"], record["passage_i"])
    tok_j = tokenize_cross(tokenizer, device, record["query"], record["passage_j"])
    tok_i = apply_cross_mask(tokenizer, tok_i, perturbed_positions_by_side["doc_i"])
    tok_j = apply_cross_mask(tokenizer, tok_j, perturbed_positions_by_side["doc_j"])

    if method == "pairwise_ig":
        attr = compute_cross_pairwise_ig_from_tok(tokenizer, model, embedding_layer, tok_i, tok_j)
        rows = []
        rows.extend(support_rows_from_attr_cross(attr["doc_i"], "doc_i", sign=1.0))
        rows.extend(support_rows_from_attr_cross(attr["doc_j"], "doc_j", sign=1.0))
        score = score_cross(model, tok_i) - score_cross(model, tok_j)
    elif method == "pointwise_ig":
        attr_i = compute_cross_pointwise_ig_from_tok(tokenizer, model, embedding_layer, tok_i)
        attr_j = compute_cross_pointwise_ig_from_tok(tokenizer, model, embedding_layer, tok_j)
        rows = []
        rows.extend(support_rows_from_attr_cross(attr_i, "doc_i", sign=1.0))
        rows.extend(support_rows_from_attr_cross(attr_j, "doc_j", sign=-1.0))
        score = score_cross(model, tok_i) - score_cross(model, tok_j)
    elif method == "loo_pairwise":
        attr = compute_cross_loo_pairwise_from_tok(tokenizer, model, tok_i, tok_j)
        rows = []
        rows.extend(support_rows_from_loo(attr["doc_i"], "doc_i", sign=1.0))
        rows.extend(support_rows_from_loo(attr["doc_j"], "doc_j", sign=1.0))
        score = float(attr["score"])
    elif method == "loo_pointwise":
        attr_i = compute_cross_loo_pointwise_from_tok(tokenizer, model, tok_i)
        attr_j = compute_cross_loo_pointwise_from_tok(tokenizer, model, tok_j)
        rows = []
        rows.extend(support_rows_from_loo(attr_i["doc"], "doc_i", sign=1.0))
        rows.extend(support_rows_from_loo(attr_j["doc"], "doc_j", sign=-1.0))
        score = float(attr_i["score"] - attr_j["score"])
    else:
        raise ValueError(method)
    return pd.DataFrame(rows), float(score)


def build_perturbed_support_table_duot5(record, method, runtime, perturbed_positions_by_side):
    tokenizer, model, embedding_layer, device, decoder_start_token_id, true_token_id, false_token_id = runtime
    if method in {"pairwise_ig", "loo_pairwise"}:
        tok = tokenize_duot5(tokenizer, device, record["query"], record["passage_i"], record["passage_j"])
        mask_positions = set(perturbed_positions_by_side["doc_i"]) | set(perturbed_positions_by_side["doc_j"])
        tok = apply_duot5_mask(tokenizer, tok, mask_positions)
        if method == "pairwise_ig":
            attr = compute_duot5_pairwise_ig_from_tok(runtime, tok)
            rows = []
            rows.extend(support_rows_from_attr_t5(attr["doc0"], "doc_i"))
            rows.extend(support_rows_from_attr_t5(attr["doc1"], "doc_j"))
            score = score_duot5_pairwise(runtime, tok)
        else:
            attr = compute_duot5_loo_pairwise_from_tok(runtime, tok)
            rows = []
            rows.extend(support_rows_from_loo(attr["doc_i"], "doc_i", sign=1.0))
            rows.extend(support_rows_from_loo(attr["doc_j"], "doc_j", sign=1.0))
            score = float(attr["score"])
    else:
        tok_i = tokenize_duot5(tokenizer, device, record["query"], record["passage_i"], record["ref_passage_i"])
        tok_j = tokenize_duot5(tokenizer, device, record["query"], record["passage_j"], record["ref_passage_j"])
        tok_i = apply_duot5_mask(tokenizer, tok_i, perturbed_positions_by_side["doc_i"])
        tok_j = apply_duot5_mask(tokenizer, tok_j, perturbed_positions_by_side["doc_j"])
        if method == "pointwise_ig":
            attr_i = compute_duot5_pointwise_ig_from_tok(runtime, tok_i)
            attr_j = compute_duot5_pointwise_ig_from_tok(runtime, tok_j)
            rows = []
            rows.extend(support_rows_from_attr_t5(attr_i["doc0"], "doc_i"))
            rows.extend([{**row, "support_score": -row["support_score"]} for row in support_rows_from_attr_t5(attr_j["doc0"], "doc_j")])
            score = score_duot5_pointwise(runtime, tok_i) - score_duot5_pointwise(runtime, tok_j)
        else:
            attr_i = compute_duot5_loo_pointwise_from_tok(runtime, tok_i)
            attr_j = compute_duot5_loo_pointwise_from_tok(runtime, tok_j)
            rows = []
            rows.extend(support_rows_from_loo(attr_i["doc"], "doc_i", sign=1.0))
            rows.extend(support_rows_from_loo(attr_j["doc"], "doc_j", sign=-1.0))
            score = float(attr_i["score"] - attr_j["score"])
    return pd.DataFrame(rows), float(score)


def run_stability(records, cfg, perturb_sizes):
    family = cfg["family"]
    runtime = setup_cross_runtime(cfg) if family == "cross_encoder" else setup_duot5_runtime(cfg)
    rows = []

    for idx, record in enumerate(records):
        original_tables = {method: build_support_table(record, method, family) for method in cfg["methods"]}
        original_scores = {method: record_target_score(record, method, family) for method in cfg["methods"]}

        if family == "cross_encoder":
            tokenizer, model, embedding_layer, device = runtime
            _, _, candidates = build_cross_candidates(tokenizer, device, record["query"], record["passage_i"], record["passage_j"])
        else:
            tokenizer, model, embedding_layer, device, *_ = runtime
            _, candidates = build_duot5_candidates(tokenizer, device, record["query"], record["passage_i"], record["passage_j"])

        for k in perturb_sizes:
            selected = select_perturbation(candidates, k, seed + idx * 100 + k)
            if not selected:
                continue
            perturbed_positions_by_side = positions_by_side(selected)

            for method in cfg["methods"]:
                original_table = exclude_perturbed_rows(original_tables[method], perturbed_positions_by_side)
                if family == "cross_encoder":
                    perturbed_table, perturbed_score = build_perturbed_support_table_cross(record, method, runtime, perturbed_positions_by_side)
                else:
                    perturbed_table, perturbed_score = build_perturbed_support_table_duot5(record, method, runtime, perturbed_positions_by_side)
                perturbed_table = exclude_perturbed_rows(perturbed_table, perturbed_positions_by_side)

                doc_i_s, doc_i_n = align_side_vectors(original_table, perturbed_table, "doc_i")
                doc_j_s, doc_j_n = align_side_vectors(original_table, perturbed_table, "doc_j")
                mean_s = bootstrapless_mean([doc_i_s, doc_j_s])

                orig_score = original_scores[method]
                decision_preserved = int(np.sign(orig_score) == np.sign(perturbed_score))

                rows.append(
                    {
                        "experiment": family,
                        "model_label": cfg["label"],
                        "method": method,
                        "qid": record["qid"],
                        "pid_i": record["pid_i"],
                        "pid_j": record["pid_j"],
                        "perturbation_type": "low_info_mask",
                        "k_masked_words": k,
                        "candidate_pool_size": len(candidates),
                        "actual_masked_words": len(selected),
                        "g_original": float(orig_score),
                        "g_perturbed": float(perturbed_score),
                        "decision_preserved": decision_preserved,
                        "spearman_doc_i": doc_i_s,
                        "spearman_doc_j": doc_j_s,
                        "spearman_mean": mean_s,
                        "aligned_positions_doc_i": doc_i_n,
                        "aligned_positions_doc_j": doc_j_n})

        gc.collect()
        if torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    return pd.DataFrame(rows)


def summarize(stability_df):
    rows = []
    for (experiment, model_label, method, perturbation_type, k_masked_words), group in stability_df.groupby(
        ["experiment", "model_label", "method", "perturbation_type", "k_masked_words"], dropna=False):
        rows.append(
            {
                "experiment": experiment,
                "model_label": model_label,
                "method": method,
                "perturbation_type": perturbation_type,
                "k_masked_words": k_masked_words,
                "n_cases": int(len(group)),
                "decision_preserved_rate": float(group["decision_preserved"].mean()),
                "mean_spearman_all": bootstrapless_mean(group["spearman_mean"]),
                "mean_spearman_doc_i": bootstrapless_mean(group["spearman_doc_i"]),
                "mean_spearman_doc_j": bootstrapless_mean(group["spearman_doc_j"]),
                "mean_spearman_preserved_only": bootstrapless_mean(group.loc[group["decision_preserved"] == 1, "spearman_mean"]),
                "mean_aligned_positions_doc_i": bootstrapless_mean(group["aligned_positions_doc_i"]),
                "mean_aligned_positions_doc_j": bootstrapless_mean(group["aligned_positions_doc_j"])})
    return pd.DataFrame(rows).sort_values(["method", "k_masked_words"]).reset_index(drop=True)


def main():
    args = parse_args()
    cfg = EXPERIMENTS[args.experiment]
    cfg["stability_dir"].mkdir(parents=True, exist_ok=True)
    perturb_sizes = sorted(set(args.perturb_sizes))

    if cfg["family"] == "cross_encoder":
        records = filter_subset_records(load_cross_records(EXPERIMENTS["cross_encoder"]), cfg["pair_file"])
    else:
        records = filter_subset_records(load_duot5_records(EXPERIMENTS["duot5"]), cfg["pair_file"])
    if args.limit is not None:
        records = records[: args.limit]

    print(f"Experiment: {cfg['label']} ({cfg['family']})")
    print(f"Loaded records: {len(records)}")
    print(f"Perturbation: low-information document-word masking")
    print(f"Masked word budgets: {perturb_sizes}")
    print(f"IG integration steps: {ig_steps}")

    stability_df = run_stability(records, cfg, perturb_sizes)
    summary_df = summarize(stability_df)

    suffix = f"_limit{args.limit}" if args.limit is not None else ""
    detail_out = cfg["stability_dir"] / f"stability_results{suffix}.csv"
    summary_out = cfg["stability_dir"] / f"stability_summary{suffix}.csv"
    stability_df.to_csv(detail_out, index=False)
    summary_df.to_csv(summary_out, index=False)

    print(f"Saved detailed stability results to {detail_out}")
    print(f"Saved summary stability results to {summary_out}")


if __name__ == "__main__":
    main()