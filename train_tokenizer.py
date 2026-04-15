from __future__ import annotations

import argparse
import glob
import importlib
import json
import shutil
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
    parser = argparse.ArgumentParser(description="Train a deterministic tokenizer for local parameter-golf exports.")
    parser.add_argument("--input", nargs="+", required=True, help="Input files, directories, or glob patterns.")
    parser.add_argument("--output-dir", required=True, help="Directory for tokenizer.json and vocab/merges.")
    parser.add_argument(
        "--kind",
        choices=("bytelevel_bpe", "sentencepiece_bpe", "sentencepiece_unigram"),
        default="bytelevel_bpe",
        help="Tokenizer family to train. Defaults to bytelevel_bpe for backward compatibility.",
    )
    parser.add_argument("--vocab-size", type=int, default=8192, help="Target vocabulary size.")
    parser.add_argument("--min-frequency", type=int, default=2, help="Minimum merge frequency.")
    parser.add_argument("--text-key", default="text", help="JSONL field used when reading .jsonl files.")
    parser.add_argument(
        "--model-prefix",
        default=None,
        help="SentencePiece output prefix name. Defaults to the output dir name for SentencePiece trainers.",
    )
    parser.add_argument(
        "--normalization-rule-name",
        default="nfkc",
        help="SentencePiece normalization_rule_name. Defaults to nfkc.",
    )
    parser.add_argument(
        "--character-coverage",
        type=float,
        default=0.9995,
        help="SentencePiece character coverage. Defaults to 0.9995.",
    )
    parser.add_argument(
        "--byte-fallback",
        type=int,
        choices=(0, 1),
        default=1,
        help="Enable SentencePiece byte fallback when using SentencePiece trainers. Defaults to 1.",
    )
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


def load_sentencepiece_api() -> Any:
    try:
        return importlib.import_module("sentencepiece")
    except ImportError as exc:
        raise ImportError("sentencepiece is required to train SentencePiece tokenizers") from exc


def train_bytelevel_bpe(args: argparse.Namespace, input_paths: list[Path], output_dir: Path) -> None:
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


def train_sentencepiece(args: argparse.Namespace, input_paths: list[Path], output_dir: Path) -> None:
    spm = load_sentencepiece_api()
    model_type = "unigram" if args.kind == "sentencepiece_unigram" else "bpe"
    model_prefix = args.model_prefix or output_dir.name
    prefix_path = output_dir / model_prefix
    model_path = prefix_path.with_suffix(".model")
    vocab_path = prefix_path.with_suffix(".vocab")
    for artifact in (model_path, vocab_path):
        if artifact.exists():
            artifact.unlink()

    spm.SentencePieceTrainer.train(
        sentence_iterator=iter_texts(input_paths, args.text_key),
        model_prefix=str(prefix_path),
        model_type=model_type,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        byte_fallback=bool(args.byte_fallback),
        normalization_rule_name=args.normalization_rule_name,
        add_dummy_prefix=False,
        split_digits=True,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        hard_vocab_limit=False,
    )
    tokenizer = spm.SentencePieceProcessor(model_file=str(model_path))

    # Keep the requested canonical artifact name when the caller wants a single stable file path.
    canonical_model_path = output_dir / f"{model_prefix}.model"
    canonical_vocab_path = output_dir / f"{model_prefix}.vocab"
    if model_path != canonical_model_path:
        shutil.copy2(model_path, canonical_model_path)
        if vocab_path.exists():
            shutil.copy2(vocab_path, canonical_vocab_path)
        model_path = canonical_model_path
        vocab_path = canonical_vocab_path

    print(f"sentencepiece_model={model_path}")
    print(f"sentencepiece_vocab={vocab_path}")
    print(f"model_type={model_type}")
    print(f"vocab_size={int(tokenizer.vocab_size())}")
    print(f"<pad>_id={int(tokenizer.pad_id())}")
    print(f"<bos>_id={int(tokenizer.bos_id())}")
    print(f"<eos>_id={int(tokenizer.eos_id())}")
    print(f"<unk>_id={int(tokenizer.unk_id())}")


def main() -> None:
    args = parse_args()
    input_paths = expand_inputs(args.input)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.kind == "bytelevel_bpe":
        train_bytelevel_bpe(args, input_paths, output_dir)
        return
    train_sentencepiece(args, input_paths, output_dir)


if __name__ == "__main__":
    main()
