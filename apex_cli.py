"""Command-line entry points for preparing, training and using APEX language models."""

import argparse
import json

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from apex_core import APEXConfig, APEXModel, APEXTrainer, BlockPreset, TrainingConfig
from apex_core.data import ByteTokenizer, RandomTextWindowDataset, make_validation_dataset, read_corpus


def _device_and_ids(value: str, gpu_ids: Optional[str]):
    device = torch.device(value if value != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda":
        if gpu_ids:
            raise ValueError("--gpu-ids can only be used with a CUDA device")
        return device, None
    ids = [int(index) for index in gpu_ids.split(",")] if gpu_ids else list(range(torch.cuda.device_count()))
    if not ids:
        raise ValueError("No CUDA devices are available")
    if min(ids) < 0 or max(ids) >= torch.cuda.device_count():
        raise ValueError(f"GPU ids {ids} are outside the {torch.cuda.device_count()} available devices")
    if device.index is None:
        device = torch.device("cuda", ids[0])
    return device, ids if len(ids) > 1 else None


def _encode_corpus(tokenizer: ByteTokenizer, path: Path):
    return tokenizer.encode(read_corpus(path))


def train(args):
    if args.steps < 1 or args.batch_size < 1 or args.eval_batches < 1:
        raise ValueError("steps, batch-size, and eval-batches must be positive")
    if args.sequence_length > args.max_seq_len:
        raise ValueError("sequence-length must not exceed max-seq-len")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteTokenizer()
    tokenizer.save(output_dir / "tokenizer.json")
    tokens = _encode_corpus(tokenizer, Path(args.data))

    if args.validation_data:
        train_tokens = tokens
        validation_tokens = _encode_corpus(tokenizer, Path(args.validation_data))
    else:
        split = int(len(tokens) * 0.9)
        train_tokens = tokens[:split]
        validation_tokens = tokens[split:]

    train_dataset = RandomTextWindowDataset(
        train_tokens,
        sequence_length=args.sequence_length,
        samples=args.steps * args.batch_size,
    )
    validation_dataset = make_validation_dataset(
        validation_tokens, args.sequence_length, args.eval_batches, args.batch_size
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.workers
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.workers,
    )

    layer_pattern = [part.strip().lower() for part in args.layers.split(",")] if args.layers else None
    config = APEXConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        max_seq_len=args.max_seq_len,
        preset=None if layer_pattern else BlockPreset(args.preset),
        layer_pattern=layer_pattern,
        echo_n_keys=args.echo_keys,
        echo_top_k=args.echo_top_k,
        echo_rank=args.echo_rank,
        lrcm_heads=args.lrcm_heads,
        lrcm_local_window=args.local_window,
        lrcm_chunk_size=args.chunk_size,
        lrcm_desc_dim=args.descriptor_dim,
        lrcm_beam=args.memory_beam,
        mamba_d_state=args.state_size,
        mamba_headdim=args.head_dim,
        mamba_mimo_rank=args.mimo_rank,
        mamba_is_mimo=not args.siso,
        dropout=args.dropout,
    )
    model = APEXModel(config)
    device, device_ids = _device_and_ids(args.device, args.gpu_ids)
    precision = args.precision
    if precision == "auto":
        precision = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else (
            "fp16" if device.type == "cuda" else "no"
        )
    if args.batch_size < len(device_ids or []):
        raise ValueError("batch-size must be at least the number of selected GPUs")

    state_path = output_dir / "training_state.pt"
    train_config = TrainingConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=min(args.warmup_steps, max(0, args.steps - 1)),
        total_steps=args.steps,
        epochs=1,
        aux_loss_weight=args.aux_loss_weight,
        device=str(device),
        device_ids=device_ids,
        mixed_precision=precision,
        eval_every=args.eval_interval,
        save_checkpoint_path=str(output_dir / "best.apex"),
        training_checkpoint_path=str(state_path),
        checkpoint_every=args.save_interval,
        resume_checkpoint_path=str(state_path) if args.resume else None,
        log_interval=max(1, args.log_interval),
    )
    trainer = APEXTrainer(model, train_config)
    print(
        f"Architecture: {' -> '.join(config.layer_pattern)} | "
        f"Parameters: {model.count_parameters():,} | "
        f"Corpus: {len(tokens):,} bytes/tokens | Device: {device} | "
        f"Precision: {precision} | GPUs: {device_ids or 'single'}"
    )

    history = trainer.train(train_loader, validation_loader)
    final_validation = trainer.evaluate(validation_loader)
    latest_path = model.save_apex(str(output_dir / "latest.apex"))
    metrics = {
        "training_steps": len(history["step"]),
        "last_step": history["step"][-1] if history["step"] else 0,
        "last_training_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "final_validation": final_validation,
        "architecture": config.layer_pattern,
        "parameters": model.count_parameters(),
        "device": str(device),
        "precision": precision,
        "tokenizer": "utf8-byte",
        "model_file": latest_path,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Ready for inference: {latest_path}")


def generate(args):
    model = APEXModel.load_apex(args.model, device=args.device)
    tokenizer = ByteTokenizer.load(Path(args.model).parent / "tokenizer.json")
    prompt = tokenizer.encode(args.prompt, add_bos=True, add_eos=False)
    if len(prompt) + args.max_new_tokens > model.config.max_seq_len:
        raise ValueError("Prompt plus max-new-tokens exceeds the model max_seq_len")
    tokens = torch.tensor([prompt], dtype=torch.long, device=args.device)
    generated = model.generate(
        tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )[0]
    print(tokenizer.decode(generated[len(prompt):].tolist()), end="")


def build_parser():
    parser = argparse.ArgumentParser(prog="apex", description="Train and run hybrid APEX language models.")
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser("train", help="Train APEX on a UTF-8 text file or directory of text files.")
    fit.add_argument("--data", required=True, help="Training text file or directory")
    fit.add_argument("--validation-data", help="Optional separate validation text file/directory")
    fit.add_argument("--output", default="runs/apex", help="Output directory for checkpoints and tokenizer")
    fit.add_argument("--steps", type=int, default=1000)
    fit.add_argument("--batch-size", type=int, default=2)
    fit.add_argument("--sequence-length", type=int, default=128)
    fit.add_argument("--max-seq-len", type=int, default=512)
    fit.add_argument("--d-model", type=int, default=128)
    fit.add_argument("--preset", choices=[preset.value for preset in BlockPreset], default=BlockPreset.APEX_PYRAMID.value)
    fit.add_argument("--layers", help="Optional comma-separated block list, e.g. hopmix,lrcm,mamba3")
    fit.add_argument("--echo-keys", type=int, default=32)
    fit.add_argument("--echo-top-k", type=int, default=4)
    fit.add_argument("--echo-rank", type=int, default=16)
    fit.add_argument("--lrcm-heads", type=int, default=4)
    fit.add_argument("--local-window", type=int, default=32)
    fit.add_argument("--chunk-size", type=int, default=16)
    fit.add_argument("--descriptor-dim", type=int, default=32)
    fit.add_argument("--memory-beam", type=int, default=2)
    fit.add_argument("--state-size", type=int, default=32)
    fit.add_argument("--head-dim", type=int, default=32)
    fit.add_argument("--mimo-rank", type=int, default=2)
    fit.add_argument("--siso", action="store_true", help="Use SISO instead of the default MIMO input projection")
    fit.add_argument("--dropout", type=float, default=0.0)
    fit.add_argument("--learning-rate", type=float, default=3e-4)
    fit.add_argument("--weight-decay", type=float, default=0.01)
    fit.add_argument("--warmup-steps", type=int, default=50)
    fit.add_argument("--aux-loss-weight", type=float, default=0.01)
    fit.add_argument("--eval-interval", type=int, default=100)
    fit.add_argument("--eval-batches", type=int, default=10)
    fit.add_argument("--save-interval", type=int, default=100)
    fit.add_argument("--log-interval", type=int, default=10)
    fit.add_argument("--workers", type=int, default=0)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--device", default="auto", help="Device such as auto, cpu, cuda, or cuda:0")
    fit.add_argument("--gpu-ids", help="Comma-separated CUDA devices for DataParallel, e.g. 0,1")
    fit.add_argument("--precision", choices=["auto", "no", "fp16", "bf16"], default="auto")
    fit.add_argument("--resume", action="store_true", help="Resume optimizer/model/RNG state from output/training_state.pt")
    fit.set_defaults(func=train)

    sample = commands.add_parser("generate", help="Generate text from a saved .apex model.")
    sample.add_argument("--model", required=True, help="Path to latest.apex or another .apex model")
    sample.add_argument("--prompt", required=True)
    sample.add_argument("--max-new-tokens", type=int, default=128)
    sample.add_argument("--temperature", type=float, default=0.8)
    sample.add_argument("--top-k", type=int, default=50)
    sample.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    sample.set_defaults(func=generate)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
