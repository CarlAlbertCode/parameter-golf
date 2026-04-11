from __future__ import annotations

import argparse
import glob
import importlib
import json
from pathlib import Path
from typing import Any, Iterator


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>"]


def load_tokenizers_api() -> tuple[Any, Any, Any, Any, Any]:
    try:
        tokenizers_mod = importlib.import_module("tokenizers")
        decoders_mod = importlib.import_module("tokenizers.decoders")
        models_mod = importlib.import_module("tokenizers.models")
        pre_tokenizers_mod = importlib.import_module("tokenizers.pre_tokenizers")
        trainers_mod = importlib.import_module("tokenizers.trainers")
    except ImportError as exc:
        raise ImportError("tokenizers is required to run train_tokenizer.py") from exc
    return (
        tokenizers_mod.Tokenizer,
        decoders_mod.ByteLevel,
        models_mod.BPE,
        pre_tokenizers_mod.ByteLevel,
        trainers_mod.BpeTrainer,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a deterministic byte-level BPE tokenizer.")
    parser.add_argument("--input", nargs="+", required=True, help="Input files, directories, or glob patterns.")
    parser.add_argument("--output-dir", required=True, help="Directory for tokenizer.json and vocab/merges.")
    parser.add_argument("--vocab-size", type=int, default=8192, help="Target vocabulary size.")
    parser.add_argument("--min-frequency", type=int, default=2, help="Minimum merge frequency.")
    parser.add_argument("--text-key", default="text", help="JSONL field used when reading .jsonl files.")
    return parser.parse_args()


def expand_inputs(inputs: list[str]) -> list[Path]:
    paths: set[Path] = set()
    for value in inputs:
        path = Path(value).expanduser()
        if path.exists():
            if path.is_dir():
                for pattern in ("*.jsonl", "*.txt"):
                    paths.update(candidate.resolve() for candidate in sorted(path.rglob(pattern)))
            else:
                paths.add(path.resolve())
            continue
        matches = sorted(Path(match).expanduser().resolve() for match in glob.glob(value, recursive=True))
        if not matches:
            raise FileNotFoundError(f"No files matched input {value!r}")
        paths.update(matches)
    ordered = sorted(paths, key=lambda p: p.as_posix())
    if not ordered:
        raise FileNotFoundError("No tokenizer training inputs found")
    return ordered


def iter_texts(paths: list[Path], text_key: str) -> Iterator[str]:
    for path in paths:
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    text = record[text_key]
                    if isinstance(text, str) and text:
                        yield text
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.rstrip("\n")
                if text:
                    yield text


def main() -> None:
    args = parse_args()
    input_paths = expand_inputs(args.input)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    Tokenizer, ByteLevelDecoder, BPE, ByteLevel, BpeTrainer = load_tokenizers_api()
    tokenizer = Tokenizer(BPE(unk_token=None, byte_fallback=True))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        show_progress=True,
        initial_alphabet=ByteLevel.alphabet(),
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.train_from_iterator(iter_texts(input_paths, args.text_key), trainer=trainer)

    tokenizer_json = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_json))
    tokenizer.model.save(str(output_dir), "tokenizer")

    print(f"tokenizer_json={tokenizer_json}")
    print(f"vocab_size={tokenizer.get_vocab_size(with_added_tokens=True)}")
    for token in SPECIAL_TOKENS:
        print(f"{token}_id={tokenizer.token_to_id(token)}")


if __name__ == "__main__":
    main()
