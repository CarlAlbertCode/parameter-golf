from __future__ import annotations

import argparse
import json
from pathlib import Path

import sentencepiece as spm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick local compatibility check for SentencePiece tokenizers.")
    parser.add_argument("--docs", required=True, help="Path to docs_selected.jsonl.")
    parser.add_argument("--baseline-tokenizer", required=True, help="Path to the current baseline SentencePiece .model.")
    parser.add_argument("--candidate-tokenizer", required=True, help="Path to the candidate SentencePiece .model.")
    parser.add_argument("--sample-docs", type=int, default=1024, help="Number of docs to sample from docs_selected.jsonl.")
    return parser.parse_args()


def iter_sample_docs(path: Path, limit: int):
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            payload = json.loads(line)
            text = payload.get("text")
            if isinstance(text, str):
                yield text


def summarize(name: str, tok: spm.SentencePieceProcessor, texts: list[str]) -> tuple[float, int, int]:
    vocab_size = int(tok.vocab_size())
    total_tokens = 0
    max_len = 0
    checked = 0
    for text in texts:
        ids = tok.encode(text, out_type=int)
        if ids:
            lo = min(ids)
            hi = max(ids)
            if lo < 0 or hi >= vocab_size:
                raise ValueError(f"{name} emitted token ids outside [0, {vocab_size}): min={lo} max={hi}")
        total_tokens += len(ids)
        max_len = max(max_len, len(ids))
        checked += 1
    avg_tokens = total_tokens / max(checked, 1)
    return avg_tokens, vocab_size, max_len


def main() -> None:
    args = parse_args()
    docs_path = Path(args.docs).expanduser().resolve()
    baseline_path = Path(args.baseline_tokenizer).expanduser().resolve()
    candidate_path = Path(args.candidate_tokenizer).expanduser().resolve()
    texts = list(iter_sample_docs(docs_path, args.sample_docs))
    if not texts:
        raise ValueError(f"no docs found in sample from {docs_path}")

    baseline = spm.SentencePieceProcessor(model_file=str(baseline_path))
    candidate = spm.SentencePieceProcessor(model_file=str(candidate_path))

    base_avg, base_vocab, base_max = summarize("baseline", baseline, texts)
    cand_avg, cand_vocab, cand_max = summarize("candidate", candidate, texts)

    print(f"sample_docs={len(texts)}")
    print(f"baseline_model={baseline_path}")
    print(f"baseline_vocab_size={base_vocab}")
    print(f"baseline_avg_tokens={base_avg:.4f}")
    print(f"baseline_max_tokens={base_max}")
    print(f"candidate_model={candidate_path}")
    print(f"candidate_vocab_size={cand_vocab}")
    print(f"candidate_avg_tokens={cand_avg:.4f}")
    print(f"candidate_max_tokens={cand_max}")
    print(f"avg_token_ratio_candidate_over_baseline={cand_avg / max(base_avg, 1e-9):.4f}")


if __name__ == "__main__":
    main()
