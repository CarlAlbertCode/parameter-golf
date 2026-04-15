from __future__ import annotations

import argparse
import json
from pathlib import Path

from download_hf_docs_and_tokenize import (
    APPEND_EOS,
    NUM_VAL_DOCS,
    SHARD_SIZE,
    VERSION,
    build_tokenizers,
    count_docs,
    docs_sidecar_path,
    export_shards,
    load_specs,
    maybe_load_docs_sidecar_meta,
    parse_reuse_sp_models,
    relativize_manifest_paths,
    write_tokenizer_config_export,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuild tokenizer shards locally from docs_selected.jsonl.")
    parser.add_argument("--docs-jsonl", required=True, help="Local docs_selected.jsonl path.")
    parser.add_argument("--output-root", required=True, help="Output root containing tokenizers/ and datasets/.")
    parser.add_argument(
        "--tokenizer-config",
        required=True,
        help="Tokenizer config JSON, for example data/tokenizer_specs_unigram_8192.json.",
    )
    parser.add_argument(
        "--num-val-docs",
        type=int,
        default=None,
        help="Validation document count. Defaults to docs sidecar when present, otherwise 50000.",
    )
    parser.add_argument("--chunk-tokens", type=int, default=SHARD_SIZE, help="Shard size in tokens.")
    parser.add_argument(
        "--tokenizer-train-docs",
        type=int,
        default=None,
        help="Optional cap on docs used when training the tokenizer.",
    )
    parser.add_argument(
        "--reuse-sp-model",
        action="append",
        default=[],
        metavar="VOCAB=MODEL",
        help="Reuse an existing SentencePiece model for the given vocab size instead of retraining it.",
    )
    parser.add_argument("--skip-byte", action="store_true", help="Skip pure-byte tokenizer exports.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.chunk_tokens <= 0:
        raise ValueError(f"--chunk-tokens must be positive, got {args.chunk_tokens}")

    docs_jsonl = Path(args.docs_jsonl).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    tokenizers_dir = output_root / "tokenizers"
    datasets_dir = output_root / "datasets"
    tokenizers_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    docs_sidecar = maybe_load_docs_sidecar_meta(docs_jsonl)
    docs_total = int(docs_sidecar["num_docs"]) if docs_sidecar is not None and docs_sidecar.get("num_docs") is not None else count_docs(docs_jsonl)
    if args.num_val_docs is not None:
        num_val_docs = int(args.num_val_docs)
    elif docs_sidecar is not None and docs_sidecar.get("docs_val") is not None:
        num_val_docs = int(docs_sidecar["docs_val"])
    else:
        num_val_docs = NUM_VAL_DOCS
    if not (0 <= num_val_docs <= docs_total):
        raise ValueError(f"num_val_docs must be in [0, {docs_total}], got {num_val_docs}")

    specs = load_specs(Path(args.tokenizer_config).expanduser().resolve())
    reuse_sp_models = parse_reuse_sp_models(args.reuse_sp_model)
    tokenizers, selected_specs = build_tokenizers(
        specs=specs,
        docs_jsonl=docs_jsonl,
        tokenizers_dir=tokenizers_dir,
        tokenizer_train_docs=args.tokenizer_train_docs,
        skip_byte=args.skip_byte,
        reuse_sp_models=reuse_sp_models,
    )
    write_tokenizer_config_export(output_root, selected_specs)

    manifest = {
        "version": VERSION,
        "num_docs": docs_total,
        "num_val_docs": num_val_docs,
        "shuffle_seed": None if docs_sidecar is None else docs_sidecar.get("shuffle_seed"),
        "shard_size": int(args.chunk_tokens),
        "append_eos": APPEND_EOS,
        "docs_jsonl": str(docs_jsonl),
        "docs_meta": {
            "num_docs": docs_total,
            "docs_sha256": None if docs_sidecar is None else docs_sidecar.get("docs_sha256"),
            "source_manifest": str(docs_sidecar_path(docs_jsonl)) if docs_sidecar is not None else None,
            "source_sidecar": docs_sidecar,
        },
        "tokenizer_specs": selected_specs,
        "tokenizers": [],
        "datasets": [],
    }

    for tok in tokenizers:
        output_dir = datasets_dir / tok["dataset_name"]
        print(f"Exporting dataset: {tok['dataset_name']}", flush=True)
        stats = export_shards(
            docs_jsonl,
            tok,
            output_dir,
            num_val_docs=num_val_docs,
            shard_size=int(args.chunk_tokens),
            docs_total=docs_total,
        )
        manifest["tokenizers"].append(tok["manifest"])
        manifest["datasets"].append(
            {
                "name": tok["dataset_name"],
                "tokenizer_name": tok["name"],
                "tokenizer_kind": tok["kind"],
                "path": str(output_dir),
                "train_glob": str(output_dir / "fineweb_train_*.bin"),
                "val_glob": str(output_dir / "fineweb_val_*.bin"),
                "vocab_size": tok["vocab_size"],
                "bos_id": tok["bos_id"],
                "eos_id": tok["eos_id"],
                "recommended_bigram_vocab_size": tok["recommended_bigram_vocab_size"],
                "stats": stats,
            }
        )

    manifest = relativize_manifest_paths(manifest, output_root)
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Done. Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
