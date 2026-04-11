"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: To keep readable for newcomers, let's make sure `train_gpt.py` and `train_gpt_mlx.py` never are longer than 1500 lines.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager, nullcontext
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path
from typing import Any, Iterator, cast

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------
# HYPERPARAMETERS
# -----------------------------
# Default Simple Baseline run:
# - 9 transformer blocks at width 512
# - 8 attention heads with 4 KV heads (GQA) and 2x MLP expansion
# - vocab size 1024, sequence length 1024, tied embeddings
# - 524,288 train tokens per step for 20,000 iterations with a ~10 minute cap

class Hyperparameters:
    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence and batch size. Validation always uses the full fineweb_val split.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 0))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 128))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.998))
    use_ema_for_eval = bool(int(os.environ.get("USE_EMA_FOR_EVAL", "1")))
    leaky_relu_slope = float(os.environ.get("LEAKY_RELU_SLOPE", 0.5))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 0))
    val_max_tokens = int(os.environ.get("VAL_MAX_TOKENS", 0))
    recurrent_layer_start = int(os.environ.get("RECURRENT_LAYER_START", -1))
    recurrent_layer_end = int(os.environ.get("RECURRENT_LAYER_END", -1))
    recurrent_loops = int(os.environ.get("RECURRENT_LOOPS", 0))
    parallel_residual_start = int(os.environ.get("PARALLEL_RESIDUAL_START", -1))
    attn_scale_init = float(os.environ.get("ATTN_SCALE_INIT", 0.5))
    mlp_scale_init = float(os.environ.get("MLP_SCALE_INIT", 0.5))
    skip_gate_init = float(os.environ.get("SKIP_GATE_INIT", 4.0))
    diag_enable = bool(int(os.environ.get("DIAG_ENABLE", "1")))
    diag_max_layers = int(os.environ.get("DIAG_MAX_LAYERS", 6))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

# -----------------------------
# MUON OPTIMIZER 
# -----------------------------
# 
# As borrowed from modded-nanogpt
# Background on Muon: https://kellerjordan.github.io/posts/muon/

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    # Scale correction from Muon reference implementations.
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION SETUP 
# -----------------------------
#
# It's common for small models have a large fraction of their parameters be embeddings, since the 2 * d_model * d_vocab vectors can be gigantic.
# Instead of locking the tokenizer, we let you bring your own and calculate our validation metrics on the average compression of the validation set.
# We calculate BPB (bits-per-byte) instead of validation loss, so we need methods to count the number of bits per token in the tokenizer.
# Note: Submissions that edit the tokenizer will be examined more carefully, since screwing this up might unjustly improve your score.

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_any = cast(Any, sp)
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp_any.is_control(token_id) or sp_any.is_unknown(token_id) or sp_any.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp_any.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = cast(str, sp_any.id_to_piece(token_id))
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int, max_tokens: int = 0) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if max_tokens > 0:
        usable = min(usable, (max_tokens // seq_len) * seq_len)
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1].contiguous()


def eval_val(
    args: Hyperparameters,
    model: Any,
    logits_model: Any,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    if args.eval_stride > 0:
        return eval_val_sliding(
            args,
            logits_model,
            rank,
            world_size,
            device,
            grad_accum_steps,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
        )
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            ensure_finite("val_batch_loss", batch_loss)
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


def eval_val_sliding(
    args: Hyperparameters,
    logits_model: Any,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    if args.eval_stride <= 0:
        raise ValueError(f"EVAL_STRIDE must be positive for sliding eval, got {args.eval_stride}")
    if args.eval_stride > args.train_seq_len:
        raise ValueError(
            f"EVAL_STRIDE must be <= TRAIN_SEQ_LEN for sliding eval, got "
            f"EVAL_STRIDE={args.eval_stride} TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    if not hasattr(logits_model, "forward_logits"):
        raise TypeError("Sliding eval requires a model with forward_logits")

    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    windows_per_batch = max(local_batch_tokens // args.train_seq_len, 1)
    score_tokens_per_batch = windows_per_batch * args.eval_stride
    total_targets = val_tokens.numel() - 1
    target_start = (total_targets * rank) // world_size
    target_end = (total_targets * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    logits_model.eval()
    with torch.inference_mode():
        batch_target_start = target_start
        while batch_target_start < target_end:
            batch_target_end = min(batch_target_start + score_tokens_per_batch, target_end)
            score_start = batch_target_start
            while score_start < batch_target_end:
                score_end = min(score_start + args.eval_stride, batch_target_end)
                score_len = score_end - score_start
                context_end = score_end
                context_start = max(0, context_end - args.train_seq_len)
                local = val_tokens[context_start : context_end + 1].to(
                    device=device,
                    dtype=torch.int64,
                    non_blocking=True,
                )
                x = local[:-1].unsqueeze(0)
                y = local[1:].unsqueeze(0)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    logits = logits_model.forward_logits(x)
                ensure_finite("val_logits", logits)
                scored_logits = logits[:, -score_len:, :].reshape(-1, logits.size(-1))
                scored_targets = y[:, -score_len:].reshape(-1)
                val_loss_sum += F.cross_entropy(scored_logits.float(), scored_targets, reduction="sum").to(torch.float64)
                val_token_count += float(score_len)
                prev_ids = x[:, -score_len:].reshape(-1)
                token_bytes = base_bytes_lut[scored_targets].to(dtype=torch.int16)
                token_bytes += (
                    has_leading_space_lut[scored_targets] & ~is_boundary_token_lut[prev_ids]
                ).to(dtype=torch.int16)
                val_byte_count += token_bytes.to(torch.float64).sum()
                score_start = score_end
            batch_target_start = batch_target_end

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    logits_model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)

# -----------------------------
# POST-TRAINING QUANTIZATION
# -----------------------------
#
# It's silly to export our model, which is trained in bf16 and fp32, at that same precision.
# Instead, we get approximately the same model (with a small hit) by quantizing the model to int8 & zlib compressing.
# We can then decompress the model and run in higher precision for evaluation, after closing in under the size limit.

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,skip_gate,skip_gate_logits",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())

def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t

def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        # Matrices get one scale per row, which usually tracks output-channel
        # ranges much better than a single tensor-wide scale.
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    # Vectors / scalars use a simpler per-tensor scale.
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale

def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    # Single supported clean-script export format:
    # - per-row int8 for 2D float tensors
    # - per-tensor int8 for other float tensors
    # - exact passthrough for non-floats
    # - passthrough for small float tensors, stored as fp16 to save bytes
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue

        # Small float tensors are cheap enough to keep directly. We still downcast
        # fp32/bf16 passthrough tensors to fp16 so metadata does not dominate size.
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats

def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = cast(dict[str, dict[str, object]], obj.get("qmeta", {}))
    passthrough_orig_dtypes = cast(dict[str, str], obj.get("passthrough_orig_dtypes", {}))
    quantized = cast(dict[str, Tensor], obj["quantized"])
    dtypes = cast(dict[str, str], obj["dtypes"])
    scales = cast(dict[str, Tensor], obj["scales"])
    passthrough = cast(dict[str, Tensor], obj["passthrough"])
    for name, q in quantized.items():
        dtype = getattr(torch, dtypes[name])
        s = scales[name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            # Broadcast the saved row scale back across trailing dimensions.
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in passthrough.items():
        # Restore small tensors, undoing the temporary fp16 storage cast if needed.
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# -----------------------------
# DATA LOADING 
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    # SHARD HEADER INTS & SHARD_MAGIC
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    # Reads shards sequentially and wraps around forever. The training loop therefore
    # has deterministic, simple streaming behavior with no sampling or workers.
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    # Each call consumes a contiguous chunk from the shared token stream, then slices out
    # one disjoint span per rank. The extra "+1" token lets us build (x, y) by shifting.
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.zero_init = False

    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


class BigramHashEmbedding(nn.Module):
    # Hash consecutive token pairs into a small learned side embedding.
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
        super().__init__()
        if bigram_vocab_size < 2:
            raise ValueError(f"BIGRAM_VOCAB_SIZE must be >= 2 when enabled, got {bigram_vocab_size}")
        if bigram_dim <= 0:
            raise ValueError(f"BIGRAM_DIM must be positive, got {bigram_dim}")
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)

    def _hash(self, token_ids: Tensor) -> Tensor:
        t = token_ids.to(torch.int32)
        sentinel = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = sentinel
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % sentinel
        return out.long()

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self._hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return self.scale.to(dtype=h.dtype) * h


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    # Keep small/control parameters in fp32 even when the model body runs in bf16.
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


def build_param_pairs(model: nn.Module, other_model: nn.Module) -> list[tuple[nn.Parameter, nn.Parameter]]:
    model_named_params = list(model.named_parameters())
    other_named_params = list(other_model.named_parameters())
    model_names = [name for name, _ in model_named_params]
    other_names = [name for name, _ in other_named_params]
    if model_names != other_names:
        raise RuntimeError("Model parameter structure mismatch")
    return [
        (param, other_param)
        for (_, param), (_, other_param) in zip(model_named_params, other_named_params, strict=True)
    ]


@torch.no_grad()
def update_ema_(param_pairs: list[tuple[nn.Parameter, nn.Parameter]], decay: float) -> None:
    one_minus_decay = 1.0 - decay
    for model_param, ema_param in param_pairs:
        ema_param.mul_(decay).add_(model_param.detach(), alpha=one_minus_decay)


@torch.no_grad()
def swap_param_pairs_(param_pairs: list[tuple[nn.Parameter, nn.Parameter]]) -> None:
    for model_param, other_param in param_pairs:
        tmp = model_param.detach().clone()
        model_param.copy_(other_param)
        other_param.copy_(tmp)


@contextmanager
def swapped_model_params(param_pairs: list[tuple[nn.Parameter, nn.Parameter]]) -> Iterator[None]:
    swap_param_pairs_(param_pairs)
    try:
        yield
    finally:
        swap_param_pairs_(param_pairs)


def ensure_finite(name: str, tensor: Tensor) -> None:
    if not torch.isfinite(tensor).all().item():
        raise FloatingPointError(f"{name} contains NaN or Inf")


def tensor_stats_str(tensor: Tensor) -> str:
    t = tensor.detach().float()
    if t.numel() == 0:
        return "mean:0.0000 std:0.0000 rms:0.0000"
    mean = t.mean().item()
    std = t.std(unbiased=False).item()
    rms = t.square().mean().sqrt().item()
    return f"mean:{mean:.4f} std:{std:.4f} rms:{rms:.4f}"


class Rotary(nn.Module):
    # Caches cos/sin tables per sequence length on the current device.
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        inv_freq = cast(Tensor, self.inv_freq)
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj.zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    # LeakyReLU^2 keeps the same parameterization/shape as the original relu^2 path.
    def __init__(self, dim: int, mlp_mult: int, leaky_relu_slope: float):
        super().__init__()
        hidden = mlp_mult * dim
        self.leaky_relu_slope = leaky_relu_slope
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj.zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = F.leaky_relu(self.fc(x), negative_slope=self.leaky_relu_slope)
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        leaky_relu_slope: float,
        attn_scale_init: float,
        mlp_scale_init: float,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult, leaky_relu_slope)
        self.attn_scale = nn.Parameter(torch.full((dim,), attn_scale_init, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.full((dim,), mlp_scale_init, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor, parallel_residual: bool = False) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        if parallel_residual:
            h_attn = self.attn_norm(x)
            h_mlp = self.mlp_norm(x)
            attn_out = self.attn(h_attn)
            mlp_out = self.mlp(h_mlp)
            x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
            x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * mlp_out
            return x
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        leaky_relu_slope: float,
        bigram_vocab_size: int,
        bigram_dim: int,
        recurrent_layer_start: int,
        recurrent_layer_end: int,
        recurrent_loops: int,
        parallel_residual_start: int,
        attn_scale_init: float,
        mlp_scale_init: float,
        skip_gate_init: float,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim) if bigram_vocab_size > 0 else None
        self.recurrent_layer_start = recurrent_layer_start
        self.recurrent_layer_end = recurrent_layer_end
        self.recurrent_loops = recurrent_loops
        self.parallel_residual_start = parallel_residual_start
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.skip_gate_logits = nn.Parameter(
            torch.full((self.num_skip_weights, model_dim), skip_gate_init, dtype=torch.float32)
        )
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    leaky_relu_slope,
                    attn_scale_init,
                    mlp_scale_init,
                )
                for i in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head.zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "zero_init", False):
                nn.init.zeros_(module.weight)

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []

        # First half stores skips; second half reuses them in reverse order.
        for i in range(self.num_encoder_layers):
            parallel_residual = self.parallel_residual_start >= 0 and i >= self.parallel_residual_start
            x = self.blocks[i](x, x0, parallel_residual=parallel_residual)
            skips.append(x)

        if self.recurrent_loops > 0:
            recurrent_encoder_updates: dict[int, Tensor] = {}
            for _ in range(self.recurrent_loops):
                for j in range(self.recurrent_layer_start, self.recurrent_layer_end):
                    parallel_residual = self.parallel_residual_start >= 0 and j >= self.parallel_residual_start
                    x = self.blocks[j](x, x0, parallel_residual=parallel_residual)
                    if j < self.num_encoder_layers:
                        recurrent_encoder_updates[j] = x
            for j, updated_skip in recurrent_encoder_updates.items():
                skips[j] = updated_skip

        for i in range(self.num_decoder_layers):
            if skips:
                gate = torch.sigmoid(self.skip_gate_logits[i]).to(dtype=x.dtype)[None, None, :]
                x = x + gate * self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            block_idx = self.num_encoder_layers + i
            parallel_residual = self.parallel_residual_start >= 0 and block_idx >= self.parallel_residual_start
            x = self.blocks[block_idx](x, x0, parallel_residual=parallel_residual)

        x = self.final_norm(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        targets = target_ids.reshape(-1)
        logits = self.forward_logits(input_ids)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets, reduction="mean")


def collect_control_stats(model: GPT, log0) -> None:
    attn_scales = torch.cat([block.attn_scale.detach().float().flatten().cpu() for block in model.blocks], dim=0)
    mlp_scales = torch.cat([block.mlp_scale.detach().float().flatten().cpu() for block in model.blocks], dim=0)
    q_gains = torch.cat([block.attn.q_gain.detach().float().flatten().cpu() for block in model.blocks], dim=0)
    resid_mix = torch.cat([block.resid_mix.detach().float().flatten().cpu() for block in model.blocks], dim=0)
    skip_weights = model.skip_weights.detach().float().flatten().cpu()
    skip_gates = torch.sigmoid(model.skip_gate_logits.detach().float()).flatten().cpu()
    log0(f"attn_scale_stats:{tensor_stats_str(attn_scales)}")
    log0(f"mlp_scale_stats:{tensor_stats_str(mlp_scales)}")
    log0(f"q_gain_stats:{tensor_stats_str(q_gains)}")
    log0(f"resid_mix_stats:{tensor_stats_str(resid_mix)}")
    log0(f"skip_weight_stats:{tensor_stats_str(skip_weights)}")
    log0(f"skip_gate_stats:{tensor_stats_str(skip_gates)}")
    if model.bigram is not None:
        log0(f"bigram_scale_stats:{tensor_stats_str(model.bigram.scale.detach().float().reshape(1).cpu())}")


def selected_diag_layers(model: GPT, max_layers: int) -> list[int]:
    candidates = [
        0,
        model.parallel_residual_start,
        model.recurrent_layer_start,
        model.recurrent_layer_end - 1,
        model.num_encoder_layers - 1,
        model.num_encoder_layers,
        len(model.blocks) - 1,
    ]
    layers: list[int] = []
    for layer in candidates:
        if 0 <= layer < len(model.blocks) and layer not in layers:
            layers.append(layer)
    if max_layers <= 0 or len(layers) <= max_layers:
        return layers
    compressed = [layers[0]]
    for layer in layers[1:]:
        if len(compressed) >= max_layers - 1:
            break
        compressed.append(layer)
    if layers[-1] not in compressed:
        compressed.append(layers[-1])
    return compressed[:max_layers]


def block_forward_with_stats(
    block: Block,
    x: Tensor,
    x0: Tensor,
    parallel_residual: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    mix = block.resid_mix.to(dtype=x.dtype)
    x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
    h_attn = block.attn_norm(x_in)
    attn_out = block.attn(h_attn)
    x_out = x_in + block.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
    if parallel_residual:
        mlp_out = block.mlp(block.mlp_norm(x_in))
        x_out = x_out + block.mlp_scale.to(dtype=x.dtype)[None, None, :] * mlp_out
        return x_out, x_in, attn_out, mlp_out
    mlp_out = block.mlp(block.mlp_norm(x_out))
    x_out = x_out + block.mlp_scale.to(dtype=x.dtype)[None, None, :] * mlp_out
    return x_out, x_in, attn_out, mlp_out


@torch.no_grad()
def run_model_diagnostics(model: GPT, input_ids: Tensor, log0, max_layers: int) -> None:
    if max_layers <= 0:
        return
    diag_layers = set(selected_diag_layers(model, max_layers))
    x = model.tok_emb(input_ids)
    if model.bigram is not None:
        bigram_out = model.bigram(input_ids)
        log0(f"diag_bigram:{tensor_stats_str(bigram_out)}")
        x = x + bigram_out
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    skips: list[Tensor] = []
    recurrent_encoder_updates: dict[int, Tensor] = {}

    for i in range(model.num_encoder_layers):
        parallel_residual = model.parallel_residual_start >= 0 and i >= model.parallel_residual_start
        if i in diag_layers:
            x, x_in, attn_out, mlp_out = block_forward_with_stats(model.blocks[i], x, x0, parallel_residual)
            log0(
                f"diag_block layer:{i} pass:encoder mode:{'parallel' if parallel_residual else 'sequential'} "
                f"x_in:{tensor_stats_str(x_in)} attn_out:{tensor_stats_str(attn_out)} mlp_out:{tensor_stats_str(mlp_out)}"
            )
        else:
            x = model.blocks[i](x, x0, parallel_residual=parallel_residual)
        skips.append(x)

    if model.recurrent_loops > 0:
        for loop_idx in range(model.recurrent_loops):
            for j in range(model.recurrent_layer_start, model.recurrent_layer_end):
                parallel_residual = model.parallel_residual_start >= 0 and j >= model.parallel_residual_start
                if j in diag_layers and loop_idx == 0:
                    x, x_in, attn_out, mlp_out = block_forward_with_stats(model.blocks[j], x, x0, parallel_residual)
                    log0(
                        f"diag_block layer:{j} pass:recurrent loop:{loop_idx} "
                        f"mode:{'parallel' if parallel_residual else 'sequential'} "
                        f"x_in:{tensor_stats_str(x_in)} attn_out:{tensor_stats_str(attn_out)} "
                        f"mlp_out:{tensor_stats_str(mlp_out)}"
                    )
                else:
                    x = model.blocks[j](x, x0, parallel_residual=parallel_residual)
                if j < model.num_encoder_layers:
                    recurrent_encoder_updates[j] = x
        for j, updated_skip in recurrent_encoder_updates.items():
            skips[j] = updated_skip

    for i in range(model.num_decoder_layers):
        block_idx = model.num_encoder_layers + i
        skip_contrib = None
        if skips:
            skip_source = skips.pop()
            gate = torch.sigmoid(model.skip_gate_logits[i]).to(dtype=x.dtype)[None, None, :]
            skip_contrib = gate * model.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skip_source
            x = x + skip_contrib
        parallel_residual = model.parallel_residual_start >= 0 and block_idx >= model.parallel_residual_start
        if block_idx in diag_layers:
            x, x_in, attn_out, mlp_out = block_forward_with_stats(model.blocks[block_idx], x, x0, parallel_residual)
            if skip_contrib is not None:
                log0(f"diag_skip layer:{block_idx} pass:decoder contribution:{tensor_stats_str(skip_contrib)}")
            log0(
                f"diag_block layer:{block_idx} pass:decoder mode:{'parallel' if parallel_residual else 'sequential'} "
                f"x_in:{tensor_stats_str(x_in)} attn_out:{tensor_stats_str(attn_out)} mlp_out:{tensor_stats_str(mlp_out)}"
            )
        else:
            x = model.blocks[block_idx](x, x0, parallel_residual=parallel_residual)
    logits = model.forward_logits(input_ids)
    ensure_finite("diag_logits", logits)


def split_optimizer_params(model: GPT) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
    matrix_params: list[Tensor] = []
    scalar_params: list[Tensor] = []
    head_params: list[Tensor] = []
    excluded_names = {"tok_emb.weight", "bigram.embed.weight"}
    for name, param in model.named_parameters():
        if name in excluded_names:
            continue
        if name == "lm_head.weight":
            head_params.append(param)
            continue
        if param.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS):
            matrix_params.append(param)
            continue
        scalar_params.append(param)
    return matrix_params, scalar_params, head_params


def validate_optimizer_coverage(model: nn.Module, optimizers: list[torch.optim.Optimizer]) -> None:
    named_params = list(model.named_parameters())
    model_param_names = {id(param): name for name, param in named_params}
    seen: dict[int, str] = {}
    duplicates: list[str] = []
    extras: list[str] = []
    for opt_idx, opt in enumerate(optimizers):
        for group_idx, group in enumerate(opt.param_groups):
            owner = f"optimizer[{opt_idx}].param_groups[{group_idx}]"
            for param in group["params"]:
                param_id = id(param)
                name = model_param_names.get(param_id)
                if name is None:
                    extras.append(owner)
                    continue
                if param_id in seen:
                    duplicates.append(name)
                    continue
                seen[param_id] = owner
    missing = [name for name, param in named_params if id(param) not in seen]
    if duplicates or missing or extras:
        problems: list[str] = []
        if duplicates:
            problems.append(f"duplicate_params={sorted(set(duplicates))}")
        if missing:
            problems.append(f"missing_params={missing}")
        if extras:
            problems.append(f"unknown_optimizer_params={extras}")
        raise RuntimeError("Optimizer parameter coverage invalid: " + " ".join(problems))


def validate_model_structure(model: GPT) -> None:
    expected_shape = (model.num_skip_weights, model.tok_emb.embedding_dim)
    if tuple(model.skip_weights.shape) != expected_shape:
        raise ValueError(f"skip_weights shape mismatch: expected {expected_shape}, got {tuple(model.skip_weights.shape)}")
    if tuple(model.skip_gate_logits.shape) != expected_shape:
        raise ValueError(
            f"skip_gate_logits shape mismatch: expected {expected_shape}, got {tuple(model.skip_gate_logits.shape)}"
        )


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError(f"EMA_DECAY must satisfy 0 <= EMA_DECAY < 1, got {args.ema_decay}")
    if args.leaky_relu_slope < 0.0:
        raise ValueError(f"LEAKY_RELU_SLOPE must be >= 0, got {args.leaky_relu_slope}")
    if args.eval_stride < 0:
        raise ValueError(f"EVAL_STRIDE must be >= 0, got {args.eval_stride}")
    if args.val_max_tokens < 0:
        raise ValueError(f"VAL_MAX_TOKENS must be >= 0, got {args.val_max_tokens}")
    if args.parallel_residual_start < -1:
        raise ValueError(f"PARALLEL_RESIDUAL_START must be >= -1, got {args.parallel_residual_start}")
    if args.diag_max_layers < 0:
        raise ValueError(f"DIAG_MAX_LAYERS must be >= 0, got {args.diag_max_layers}")
    if args.recurrent_loops < 0:
        raise ValueError(f"RECURRENT_LOOPS must be >= 0, got {args.recurrent_loops}")
    if args.recurrent_loops > 0:
        if not (0 <= args.recurrent_layer_start < args.recurrent_layer_end <= args.num_layers):
            raise ValueError(
                "RECURRENT_LAYER_START and RECURRENT_LAYER_END must satisfy "
                f"0 <= start < end <= NUM_LAYERS, got start={args.recurrent_layer_start} "
                f"end={args.recurrent_layer_end} num_layers={args.num_layers}"
            )
        if args.recurrent_layer_start < args.num_layers // 2 < args.recurrent_layer_end:
            raise ValueError(
                "Recurrence range must not straddle the encoder/decoder boundary; "
                f"got start={args.recurrent_layer_start} end={args.recurrent_layer_end} "
                f"encoder_layers={args.num_layers // 2}"
            )
    if args.parallel_residual_start >= args.num_layers:
        raise ValueError(
            f"PARALLEL_RESIDUAL_START must be < NUM_LAYERS when enabled, got "
            f"{args.parallel_residual_start} vs num_layers={args.num_layers}"
        )
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # -----------------------------
    # DISTRIBUTED + CUDA SETUP
    # -----------------------------

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    # Fast math knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    # -----------------------------
    # TOKENIZER + VALIDATION METRIC SETUP
    # -----------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor()
    cast(Any, sp).Load(args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len, args.val_max_tokens)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")
    if args.val_max_tokens > 0:
        log0(f"val_cap_tokens:{val_tokens.numel() - 1}")

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        leaky_relu_slope=args.leaky_relu_slope,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        recurrent_layer_start=args.recurrent_layer_start,
        recurrent_layer_end=args.recurrent_layer_end,
        recurrent_loops=args.recurrent_loops,
        parallel_residual_start=args.parallel_residual_start,
        attn_scale_init=args.attn_scale_init,
        mlp_scale_init=args.mlp_scale_init,
        skip_gate_init=args.skip_gate_init,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    validate_model_structure(base_model)
    compiled_model: Any = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: Any = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # Optimizer split:
    # - token embedding (Adam) uses EMBED_LR
    # - untied lm_head (Adam) uses HEAD_LR
    # - matrix params in transformer blocks use MATRIX_LR via Muon
    # - vectors/scalars use SCALAR_LR via Adam
    matrix_params, scalar_params, head_params = split_optimizer_params(base_model)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    tok_param_groups = [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}]
    if base_model.bigram is not None:
        tok_param_groups.append({"params": [base_model.bigram.embed.weight], "lr": token_lr, "base_lr": token_lr})
    optimizer_tok = torch.optim.Adam(
        tok_param_groups,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if head_params:
        optimizer_head = torch.optim.Adam(
            [{"params": head_params, "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)
    validate_optimizer_coverage(base_model, optimizers)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    if base_model.bigram is not None:
        log0(f"bigram_hash:enabled vocab_size:{args.bigram_vocab_size} dim:{args.bigram_dim}")
    else:
        log0("bigram_hash:disabled")
    log0(f"activation:leaky_relu_sq slope:{args.leaky_relu_slope}")
    if args.recurrent_loops > 0:
        log0(
            f"recurrence:enabled start:{args.recurrent_layer_start} end:{args.recurrent_layer_end} "
            f"loops:{args.recurrent_loops}"
        )
    else:
        log0("recurrence:disabled")
    if args.parallel_residual_start >= 0:
        log0(f"parallel_residuals:enabled start:{args.parallel_residual_start}")
    else:
        log0("parallel_residuals:disabled")
    parallel_layers = list(range(args.parallel_residual_start, args.num_layers)) if args.parallel_residual_start >= 0 else []
    encoder_parallel_layers = [layer for layer in parallel_layers if layer < base_model.num_encoder_layers]
    decoder_parallel_layers = [layer for layer in parallel_layers if layer >= base_model.num_encoder_layers]
    recurrent_layers = list(range(args.recurrent_layer_start, args.recurrent_layer_end)) if args.recurrent_loops > 0 else []
    log0(f"parallel_residual_layers encoder:{encoder_parallel_layers} decoder:{decoder_parallel_layers}")
    log0(f"recurrent_layers:{recurrent_layers}")
    log0(f"ema:enabled decay:{args.ema_decay} use_for_eval:{int(args.use_ema_for_eval)}")
    log0(f"eval_stride:{args.eval_stride}")
    log0(
        f"residual_init attn_scale:{args.attn_scale_init} mlp_scale:{args.mlp_scale_init} "
        f"skip_gate_logit:{args.skip_gate_init}"
    )
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")
    collect_control_stats(base_model, log0)

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
    # initial weights/optimizer state so measured training starts from the true init.
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    if args.diag_enable and args.diag_max_layers > 0:
        diag_x, _ = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
        run_model_diagnostics(base_model, diag_x[:1], log0, args.diag_max_layers)
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
        if distributed:
            dist.barrier()

    ema_model = copy.deepcopy(base_model)
    ema_model.requires_grad_(False)
    ema_model.eval()
    ema_param_pairs = build_param_pairs(base_model, ema_model)

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

    training_time_ms = 0.0
    stop_after_step: int | None = None
    final_val_loss: float | None = None
    final_val_bpb: float | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            eval_context = swapped_model_params(ema_param_pairs) if args.use_ema_for_eval else nullcontext()
            with eval_context:
                val_loss, val_bpb = eval_val(
                    args,
                    compiled_model,
                    base_model,
                    rank,
                    world_size,
                    device,
                    grad_accum_steps,
                    val_tokens,
                    base_bytes_lut,
                    has_leading_space_lut,
                    is_boundary_token_lut,
                )
            if last_step:
                final_val_loss = val_loss
                final_val_bpb = val_bpb
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            ensure_finite(f"train_loss step={step} micro_step={micro_step}", loss.detach())
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()
        update_ema_(ema_param_pairs, args.ema_decay)

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        # Needed to sync whether we've reached the wallclock cap.
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )
    log0(f"train_step_avg_ms:{training_time_ms / max(step, 1):.2f}")
    if final_val_loss is not None and final_val_bpb is not None:
        log0(f"final_eval_exact val_loss:{final_val_loss:.8f} val_bpb:{final_val_bpb:.8f}")

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------
    # Save the raw state (useful for debugging/loading in PyTorch directly), then always produce
    # the compressed int8+zlib artifact and validate the round-tripped weights.

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model int8+zlib: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        compiled_model,
        base_model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
