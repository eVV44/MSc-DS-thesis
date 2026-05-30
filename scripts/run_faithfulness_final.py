# -- IMPORTS --
from __future__ import annotations
import argparse
import gc
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]

SEED = 42
BUDGETS = [1, 3, 5, 10, 20]
N_RANDOM_TRIALS = 10
N_BOOTSTRAP = 1000
MAX_LENGTH = 512
TOKEN_SCOPE = "doc"


EXPERIMENTS = {
    "cross_encoder": {
        "label": "Cross-encoder",
        "family": "cross_encoder",
        "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "pair_file": ROOT / "thesis_runs/cross_encoder/explanations/explanation_pairs_1_per_query_on_duot5_queries_seed42.csv",
        "explanation_dir": ROOT / "thesis_runs/cross_encoder/explanations",
        "faithfulness_dir": ROOT / "thesis_runs/cross_encoder/faithfulness",
        "methods": {
            "pairwise_ig": "Pairwise IG",
            "pointwise_ig": "Pointwise IG",
            "loo_pairwise": "LOO pairwise",
            "loo_pointwise": "LOO pointwise",
        },
        "mask_strategy": "mask",
    },
    "duot5": {
        "label": "DuoT5",
        "family": "duot5",
        "model_name": "castorini/duot5-base-msmarco",
        "pair_file": ROOT / "thesis_runs/duot5/explanations/explanation_pairs_1_per_query_seed42.csv",
        "explanation_dir": ROOT / "thesis_runs/duot5/explanations",
        "faithfulness_dir": ROOT / "thesis_runs/duot5/faithfulness",
        "ranked_results_file": ROOT / "thesis_runs/duot5/reranking/duot5_final/ranked_results.csv",
        "methods": {
            "pairwise_ig": "Pairwise IG",
            "pointwise_ig": "Pointwise IG",
            "loo_pairwise": "LOO pairwise",
            "loo_pointwise": "LOO pointwise",
        },
        "mask_strategy": "pad",
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    return parser.parse_args()


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def pair_key(qid, pid_i, pid_j):
    return str(qid), str(pid_i), str(pid_j)


def bootstrap_mean_ci(values, rng, n_bootstrap=N_BOOTSTRAP, alpha=0.05):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    if len(values) == 1:
        value = float(values[0])
        return value, value, value

    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_means.append(float(np.mean(sample)))

    return (
        float(np.mean(values)),
        float(np.quantile(boot_means, alpha / 2)),
        float(np.quantile(boot_means, 1 - alpha / 2)),
    )


def load_cross_records(cfg):
    pair_df = pd.read_csv(cfg["pair_file"], dtype={"qid": str, "pid_i": str, "pid_j": str})
    ig_records = load_pickle(cfg["explanation_dir"] / "attributions_ig.pkl")
    loo_records = load_pickle(cfg["explanation_dir"] / "attributions_loo.pkl")

    pair_lookup = {
        pair_key(r["qid"], r["pid_i"], r["pid_j"]): {
            "query": r["query"],
            "passage_i": r["passage_i"],
            "passage_j": r["passage_j"],
            "g_score": float(r["g_score"]),
            "correct_pref": int(r["correct_pref"]),
        }
        for r in pair_df.to_dict("records")
    }
    ig_lookup = {pair_key(r["qid"], r["pid_i"], r["pid_j"]): r for r in ig_records}
    loo_lookup = {pair_key(r["qid"], r["pid_i"], r["pid_j"]): r for r in loo_records}

    records = []
    for key, text in pair_lookup.items():
        if key not in ig_lookup or key not in loo_lookup:
            continue
        ig = ig_lookup[key]
        loo = loo_lookup[key]
        records.append(
            {
                "qid": key[0],
                "pid_i": key[1],
                "pid_j": key[2],
                "query": text["query"],
                "passage_i": text["passage_i"],
                "passage_j": text["passage_j"],
                "g_score": text["g_score"],
                "correct_pref": text["correct_pref"],
                "pairwise_ig": ig["pairwise_ig"],
                "pointwise_ig_i": ig["pointwise_ig_i"],
                "pointwise_ig_j": ig["pointwise_ig_j"],
                "loo_pairwise": loo["loo_pairwise"],
                "loo_pointwise_i": loo["loo_pointwise_i"],
                "loo_pointwise_j": loo["loo_pointwise_j"],
            }
        )
    return records


def load_duot5_records(cfg):
    pair_df = pd.read_csv(cfg["pair_file"], dtype={"qid": str, "pid_i": str, "pid_j": str})
    ranked_df = pd.read_csv(cfg["ranked_results_file"], dtype={"qid": str, "pid": str})
    pairwise_ig_records = load_pickle(cfg["explanation_dir"] / "attributions_pairwise_ig.pkl")
    pointwise_ig_records = load_pickle(cfg["explanation_dir"] / "attributions_pointwise_ig.pkl")
    loo_records = load_pickle(cfg["explanation_dir"] / "attributions_loo.pkl")

    pair_lookup = {
        pair_key(r["qid"], r["pid_i"], r["pid_j"]): {
            "query": r["query"],
            "passage_i": r["passage_i"],
            "passage_j": r["passage_j"],
            "g_score": float(r["g_score"]),
            "correct_pref": int(r["correct_pref"]),
        }
        for r in pair_df.to_dict("records")
    }
    pairwise_ig_lookup = {pair_key(r["qid"], r["pid_i"], r["pid_j"]): r for r in pairwise_ig_records}
    pointwise_ig_lookup = {pair_key(r["qid"], r["pid_i"], r["pid_j"]): r for r in pointwise_ig_records}
    loo_lookup = {pair_key(r["qid"], r["pid_i"], r["pid_j"]): r for r in loo_records}
    ranked_passage_lookup = {
        (str(r["qid"]), str(r["pid"])): r["passage"]
        for r in ranked_df.to_dict("records")
    }

    records = []
    for key, text in pair_lookup.items():
        if key not in pairwise_ig_lookup or key not in pointwise_ig_lookup or key not in loo_lookup:
            continue
        pw = pairwise_ig_lookup[key]
        pt = pointwise_ig_lookup[key]
        loo = loo_lookup[key]
        records.append(
            {
                "qid": key[0],
                "pid_i": key[1],
                "pid_j": key[2],
                "query": text["query"],
                "passage_i": text["passage_i"],
                "passage_j": text["passage_j"],
                "g_score": text["g_score"],
                "correct_pref": text["correct_pref"],
                "pairwise_ig": pw["pairwise_ig"],
                "pointwise_ig_i": pt["pointwise_ig_i"],
                "pointwise_ig_j": pt["pointwise_ig_j"],
                "ref_pid_i": pt["ref_pid_i"],
                "ref_pid_j": pt["ref_pid_j"],
                "ref_passage_i": ranked_passage_lookup[(key[0], str(pt["ref_pid_i"]))],
                "ref_passage_j": ranked_passage_lookup[(key[0], str(pt["ref_pid_j"]))],
                "loo_pairwise": loo["loo_pairwise"],
                "loo_pointwise_i": loo["loo_pointwise_i"],
                "loo_pointwise_j": loo["loo_pointwise_j"],
            }
        )
    return records


def record_target_score(record, method, family):
    if family == "cross_encoder":
        if method in {"pairwise_ig", "pointwise_ig", "loo_pairwise", "loo_pointwise"}:
            return float(record["g_score"])
    elif family == "duot5":
        if method in {"pairwise_ig", "loo_pairwise"}:
            return float(record["pairwise_ig"]["margin"])
        if method in {"pointwise_ig", "loo_pointwise"}:
            return float(record["pointwise_ig_i"]["true_logit"] - record["pointwise_ig_j"]["true_logit"])
    raise ValueError(f"Unsupported method/family combination: {method}/{family}")


def support_rows_from_attr_cross(attr, side, sign=1.0):
    sep_idx = int(attr["sep_idx"])
    doc_positions = list(range(sep_idx + 1, sep_idx + 1 + len(attr["doc_tokens"])))
    rows = []
    for local_idx, position in enumerate(doc_positions):
        rows.append(
            {
                "side": side,
                "segment": "doc",
                "position": int(position),
                "token": attr["doc_tokens"][local_idx],
                "support_score": float(sign * attr["doc_token_scores"][local_idx]),
            }
        )
    return rows


def support_rows_from_attr_t5(span_attr, side):
    rows = []
    positions = span_attr.get("positions", [])
    tokens = span_attr.get("tokens", [])
    scores = span_attr.get("token_scores", [])
    for position, token, score in zip(positions, tokens, scores):
        rows.append(
            {
                "side": side,
                "segment": "doc",
                "position": int(position),
                "token": token,
                "support_score": float(score),
            }
        )
    return rows


def support_rows_from_loo(entries, side, sign=1.0):
    rows = []
    for row in entries:
        if not row.get("positions"):
            continue
        rows.append(
            {
                "side": side,
                "segment": "doc",
                "position": int(row["positions"][0]),
                "token": row["word"],
                "support_score": float(sign * row["support_score"]),
                "positions": [int(p) for p in row["positions"]],
            }
        )
    return rows


def build_support_table(record, method, family):
    rows = []
    if family == "cross_encoder":
        if method == "pairwise_ig":
            rows.extend(support_rows_from_attr_cross(record["pairwise_ig"]["doc_i"], "doc_i", sign=1.0))
            rows.extend(support_rows_from_attr_cross(record["pairwise_ig"]["doc_j"], "doc_j", sign=1.0))
        elif method == "pointwise_ig":
            rows.extend(support_rows_from_attr_cross(record["pointwise_ig_i"], "doc_i", sign=1.0))
            rows.extend(support_rows_from_attr_cross(record["pointwise_ig_j"], "doc_j", sign=-1.0))
        elif method == "loo_pairwise":
            rows.extend(support_rows_from_loo(record["loo_pairwise"]["doc_i"], "doc_i", sign=1.0))
            rows.extend(support_rows_from_loo(record["loo_pairwise"]["doc_j"], "doc_j", sign=1.0))
        elif method == "loo_pointwise":
            rows.extend(support_rows_from_loo(record["loo_pointwise_i"]["doc"], "doc_i", sign=1.0))
            rows.extend(support_rows_from_loo(record["loo_pointwise_j"]["doc"], "doc_j", sign=-1.0))
    elif family == "duot5":
        if method == "pairwise_ig":
            rows.extend(support_rows_from_attr_t5(record["pairwise_ig"]["doc0"], "doc_i"))
            rows.extend(support_rows_from_attr_t5(record["pairwise_ig"]["doc1"], "doc_j"))
        elif method == "pointwise_ig":
            rows.extend(support_rows_from_attr_t5(record["pointwise_ig_i"]["doc0"], "doc_i"))
            rows.extend([{**row, "support_score": -row["support_score"]} for row in support_rows_from_attr_t5(record["pointwise_ig_j"]["doc0"], "doc_j")])
        elif method == "loo_pairwise":
            rows.extend(support_rows_from_loo(record["loo_pairwise"]["doc_i"], "doc_i", sign=1.0))
            rows.extend(support_rows_from_loo(record["loo_pairwise"]["doc_j"], "doc_j", sign=1.0))
        elif method == "loo_pointwise":
            rows.extend(support_rows_from_loo(record["loo_pointwise_i"]["doc"], "doc_i", sign=1.0))
            rows.extend(support_rows_from_loo(record["loo_pointwise_j"]["doc"], "doc_j", sign=-1.0))
    else:
        raise ValueError(f"Unsupported family: {family}")

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values("support_score", ascending=False).reset_index(drop=True)


def eligible_support_pool(record, method, family, positive_only=True):
    table = build_support_table(record, method, family)
    if table.empty:
        return table
    if positive_only:
        table = table[table["support_score"] > 0].copy()
    return table.reset_index(drop=True)


def select_explanation_tokens(record, method, family, k):
    table = eligible_support_pool(record, method, family, positive_only=True)
    return table.head(k).reset_index(drop=True)


def select_random_tokens(record, method, family, k, rng):
    table = eligible_support_pool(record, method, family, positive_only=True)
    if table.empty:
        return table
    sample_size = min(k, len(table))
    return table.sample(n=sample_size, replace=False, random_state=int(rng.integers(0, 1_000_000_000))).reset_index(drop=True)


def selection_to_payload(selection):
    def collect_positions(side_name):
        subset = selection.loc[selection["side"] == side_name]
        positions = []
        for _, row in subset.iterrows():
            if "positions" in row and isinstance(row["positions"], list):
                positions.extend(int(p) for p in row["positions"])
            else:
                positions.append(int(row["position"]))
        return sorted(set(positions))

    return {"doc_i": collect_positions("doc_i"), "doc_j": collect_positions("doc_j")}


def setup_cross_runtime(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    ce = CrossEncoder(cfg["model_name"], max_length=MAX_LENGTH)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = ce.model.to(device)
    model.eval()
    return tokenizer, model, device


def tokenize_cross(tokenizer, device, query, passage):
    encoded = tokenizer(query, passage, max_length=MAX_LENGTH, truncation=True, padding=False, return_tensors="pt")
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
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "tokens": tokens,
        "doc_positions": doc_positions,
    }


def score_cross(model, tok):
    kwargs = {"input_ids": tok["input_ids"], "attention_mask": tok["attention_mask"]}
    if tok["token_type_ids"] is not None:
        kwargs["token_type_ids"] = tok["token_type_ids"]
    with torch.no_grad():
        outputs = model(**kwargs)
    return float(outputs.logits.squeeze(-1).detach().cpu().item())


def apply_cross_mask(tokenizer, tok, positions):
    perturbed = {
        "input_ids": tok["input_ids"].clone(),
        "attention_mask": tok["attention_mask"].clone(),
        "token_type_ids": None if tok["token_type_ids"] is None else tok["token_type_ids"].clone(),
    }
    positions = sorted({int(p) for p in positions if 0 <= int(p) < tok["input_ids"].shape[1]})
    if positions:
        perturbed["input_ids"][0, positions] = tokenizer.mask_token_id
    return perturbed


def perturb_cross(runtime, query, passage_i, passage_j, payload, check):
    tokenizer, model, device = runtime
    tok_i = tokenize_cross(tokenizer, device, query, passage_i)
    tok_j = tokenize_cross(tokenizer, device, query, passage_j)

    selected_i = set(int(p) for p in payload["doc_i"])
    selected_j = set(int(p) for p in payload["doc_j"])
    if check == "deletion":
        mask_i = selected_i
        mask_j = selected_j
    else:
        mask_i = set(tok_i["doc_positions"]) - selected_i
        mask_j = set(tok_j["doc_positions"]) - selected_j

    perturbed_i = apply_cross_mask(tokenizer, tok_i, mask_i)
    perturbed_j = apply_cross_mask(tokenizer, tok_j, mask_j)
    return score_cross(model, perturbed_i) - score_cross(model, perturbed_j)


def setup_duot5_runtime(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model_name"])
    device = torch.device("cpu")
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


def duo_input(query, doc0, doc1):
    return f"Query: {query} Document0: {doc0} Document1: {doc1} Relevant:"


def _tok_len(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def tokenize_duo_pair(tokenizer, device, query, doc0, doc1):
    text = duo_input(query, doc0, doc1)
    encoded = tokenizer(text, max_length=MAX_LENGTH, truncation=True, padding=False, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    c1 = "Query: "
    c2 = c1 + query
    c3 = c2 + " Document0: "
    c4 = c3 + doc0
    c5 = c4 + " Document1: "
    c6 = c5 + doc1
    doc0_positions = [pos for pos in range(_tok_len(tokenizer, c3), min(_tok_len(tokenizer, c4), input_ids.shape[1]))]
    doc1_positions = [pos for pos in range(_tok_len(tokenizer, c5), min(_tok_len(tokenizer, c6), input_ids.shape[1]))]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "doc0_positions": doc0_positions, "doc1_positions": doc1_positions}


def score_duot5_pairwise(runtime, tok):
    tokenizer, model, device, decoder_start_token_id, true_token_id, false_token_id = runtime
    decoder_input_ids = torch.full((1, 1), decoder_start_token_id, dtype=torch.long, device=device)
    with torch.no_grad():
        outputs = model(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"], decoder_input_ids=decoder_input_ids)
    logits = outputs.logits[:, 0, :]
    return float((logits[:, true_token_id] - logits[:, false_token_id]).detach().cpu().item())


def score_duot5_pointwise(runtime, tok):
    tokenizer, model, device, decoder_start_token_id, true_token_id, false_token_id = runtime
    decoder_input_ids = torch.full((1, 1), decoder_start_token_id, dtype=torch.long, device=device)
    with torch.no_grad():
        outputs = model(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"], decoder_input_ids=decoder_input_ids)
    logits = outputs.logits[:, 0, :]
    return float(logits[:, true_token_id].detach().cpu().item())


def apply_duot5_mask(tokenizer, tok, positions):
    perturbed = {"input_ids": tok["input_ids"].clone(), "attention_mask": tok["attention_mask"].clone()}
    positions = sorted({int(p) for p in positions if 0 <= int(p) < tok["input_ids"].shape[1]})
    if positions:
        perturbed["input_ids"][0, positions] = tokenizer.pad_token_id
        perturbed["attention_mask"][0, positions] = 0
    perturbed["doc0_positions"] = tok.get("doc0_positions", [])
    perturbed["doc1_positions"] = tok.get("doc1_positions", [])
    return perturbed


def perturb_duot5_pairwise(runtime, query, passage_i, passage_j, payload, check):
    tokenizer, model, device, *_ = runtime
    tok = tokenize_duo_pair(tokenizer, device, query, passage_i, passage_j)
    selected_i = set(int(p) for p in payload["doc_i"])
    selected_j = set(int(p) for p in payload["doc_j"])
    if check == "deletion":
        mask_i = selected_i
        mask_j = selected_j
    else:
        mask_i = set(tok["doc0_positions"]) - selected_i
        mask_j = set(tok["doc1_positions"]) - selected_j
    masked = apply_duot5_mask(tokenizer, tok, mask_i | mask_j)
    return score_duot5_pairwise(runtime, masked)


def perturb_duot5_pointwise(runtime, record, payload, check):
    tokenizer, model, device, *_ = runtime
    tok_i = tokenize_duo_pair(tokenizer, device, record["query"], record["passage_i"], record["ref_passage_i"])
    tok_j = tokenize_duo_pair(tokenizer, device, record["query"], record["passage_j"], record["ref_passage_j"])

    selected_i = set(int(p) for p in payload["doc_i"])
    selected_j = set(int(p) for p in payload["doc_j"])
    if check == "deletion":
        mask_i = selected_i
        mask_j = selected_j
    else:
        mask_i = set(tok_i["doc0_positions"]) - selected_i
        mask_j = set(tok_j["doc0_positions"]) - selected_j

    masked_i = apply_duot5_mask(tokenizer, tok_i, mask_i)
    masked_j = apply_duot5_mask(tokenizer, tok_j, mask_j)
    return score_duot5_pointwise(runtime, masked_i) - score_duot5_pointwise(runtime, masked_j)


def run_faithfulness(records, cfg):
    rng = np.random.default_rng(SEED)
    methods = cfg["methods"]
    family = cfg["family"]
    runtime = setup_cross_runtime(cfg) if family == "cross_encoder" else setup_duot5_runtime(cfg)

    rows = []
    for record in records:
        query = record["query"]
        passage_i = record["passage_i"]
        passage_j = record["passage_j"]

        for method in methods:
            original_score = record_target_score(record, method, family)
            for k in BUDGETS:
                selected = select_explanation_tokens(record, method, family, k)
                if selected.empty:
                    continue
                actual_k = len(selected)
                payload = selection_to_payload(selected)

                for check in ["deletion", "preservation"]:
                    if family == "cross_encoder":
                        perturbed_score = perturb_cross(runtime, query, passage_i, passage_j, payload, check)
                    elif method in {"pairwise_ig", "loo_pairwise"}:
                        perturbed_score = perturb_duot5_pairwise(runtime, query, passage_i, passage_j, payload, check)
                    else:
                        perturbed_score = perturb_duot5_pointwise(runtime, record, payload, check)

                    rows.append(
                        {
                            "experiment": family,
                            "model_label": cfg["label"],
                            "method": method,
                            "condition": "explanation",
                            "check": check,
                            "qid": record["qid"],
                            "pid_i": record["pid_i"],
                            "pid_j": record["pid_j"],
                            "k": k,
                            "actual_k": actual_k,
                            "trial": 0,
                            "g": original_score,
                            "g_perturbed": perturbed_score,
                            "delta_g": original_score - perturbed_score,
                            "abs_g_gap": abs(original_score - perturbed_score),
                            "preference_flipped": int(np.sign(original_score) != np.sign(perturbed_score)),
                            "sign_preserved": int(np.sign(original_score) == np.sign(perturbed_score)),
                        }
                    )

                    for trial in range(N_RANDOM_TRIALS):
                        random_selection = select_random_tokens(record, method, family, actual_k, rng)
                        if random_selection.empty:
                            continue
                        random_payload = selection_to_payload(random_selection)
                        if family == "cross_encoder":
                            random_score = perturb_cross(runtime, query, passage_i, passage_j, random_payload, check)
                        elif method in {"pairwise_ig", "loo_pairwise"}:
                            random_score = perturb_duot5_pairwise(runtime, query, passage_i, passage_j, random_payload, check)
                        else:
                            random_score = perturb_duot5_pointwise(runtime, record, random_payload, check)

                        rows.append(
                            {
                                "experiment": family,
                                "model_label": cfg["label"],
                                "method": method,
                                "condition": "random",
                                "check": check,
                                "qid": record["qid"],
                                "pid_i": record["pid_i"],
                                "pid_j": record["pid_j"],
                                "k": k,
                                "actual_k": len(random_selection),
                                "trial": trial,
                                "g": original_score,
                                "g_perturbed": random_score,
                                "delta_g": original_score - random_score,
                                "abs_g_gap": abs(original_score - random_score),
                                "preference_flipped": int(np.sign(original_score) != np.sign(random_score)),
                                "sign_preserved": int(np.sign(original_score) == np.sign(random_score)),
                            }
                        )

        gc.collect()
        if torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    return pd.DataFrame(rows)


def summarize(faithfulness_df):
    query_level = (
        faithfulness_df.groupby(["experiment", "model_label", "method", "condition", "check", "k", "qid"], as_index=False)
        .agg(
            mean_delta_g=("delta_g", "mean"),
            mean_abs_g_gap=("abs_g_gap", "mean"),
            flip_rate=("preference_flipped", "mean"),
            sign_preservation_rate=("sign_preserved", "mean"),
            mean_actual_k=("actual_k", "mean"),
        )
    )

    summary_rows = []
    summary_rng = np.random.default_rng(SEED)
    for (experiment, model_label, method, condition, check, k), group in query_level.groupby(
        ["experiment", "model_label", "method", "condition", "check", "k"]
    ):
        delta_mean, delta_low, delta_high = bootstrap_mean_ci(group["mean_delta_g"].to_numpy(), summary_rng)
        gap_mean, gap_low, gap_high = bootstrap_mean_ci(group["mean_abs_g_gap"].to_numpy(), summary_rng)
        flip_mean, flip_low, flip_high = bootstrap_mean_ci(group["flip_rate"].to_numpy(), summary_rng)
        preserve_mean, preserve_low, preserve_high = bootstrap_mean_ci(group["sign_preservation_rate"].to_numpy(), summary_rng)

        summary_rows.append(
            {
                "experiment": experiment,
                "model_label": model_label,
                "method": method,
                "condition": condition,
                "check": check,
                "k": k,
                "n_queries": group["qid"].nunique(),
                "mean_actual_k": float(group["mean_actual_k"].mean()),
                "mean_delta_g": delta_mean,
                "delta_g_ci_low": delta_low,
                "delta_g_ci_high": delta_high,
                "mean_abs_g_gap": gap_mean,
                "abs_g_gap_ci_low": gap_low,
                "abs_g_gap_ci_high": gap_high,
                "flip_rate": flip_mean,
                "flip_rate_ci_low": flip_low,
                "flip_rate_ci_high": flip_high,
                "sign_preservation_rate": preserve_mean,
                "sign_preservation_ci_low": preserve_low,
                "sign_preservation_ci_high": preserve_high,
            }
        )

    return pd.DataFrame(summary_rows).sort_values(["check", "method", "condition", "k"]).reset_index(drop=True)


def main():
    args = parse_args()
    cfg = EXPERIMENTS[args.experiment]
    cfg["faithfulness_dir"].mkdir(parents=True, exist_ok=True)

    if cfg["family"] == "cross_encoder":
        records = load_cross_records(cfg)
    else:
        records = load_duot5_records(cfg)

    print(f"Experiment: {cfg['label']} ({cfg['family']})")
    print(f"Loaded records: {len(records)}")

    faithfulness_df = run_faithfulness(records, cfg)
    summary_df = summarize(faithfulness_df)

    detail_out = cfg["faithfulness_dir"] / "faithfulness_results.csv"
    summary_out = cfg["faithfulness_dir"] / "faithfulness_summary.csv"
    faithfulness_df.to_csv(detail_out, index=False)
    summary_df.to_csv(summary_out, index=False)

    print(f"Saved detailed faithfulness results to {detail_out}")
    print(f"Saved summary faithfulness results to {summary_out}")


if __name__ == "__main__":
    main()