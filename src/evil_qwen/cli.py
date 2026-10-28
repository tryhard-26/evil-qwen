"""evil-qwen CLI: compare baseline vs neuron-steered Qwen3.5-4B on the same prompt."""

import argparse
import platform

import torch
import transformers

from .model import MODEL_ID, DTYPES, generate, load, memory_info
from .steering import PRESETS, PUBLISHED_P50, calibration_ids, parse_target, plan, residual_medians, steering


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evil-qwen", description=__doc__)
    p.add_argument("mode", nargs="?", default="evil_qwen", choices=[*PRESETS, "custom"])
    p.add_argument("--target", action="append", default=[], metavar="L:N:DOSE",
                   help="custom mode target, 0-indexed layer, e.g. 14:8949:1.0 (repeatable)")
    p.add_argument("--prompt", action="append", help="prompt(s) to run; omit for interactive mode")
    p.add_argument("--no-compare", action="store_true", help="run only the selected mode, not baseline too")
    p.add_argument("--chat", action="store_true", help="multi-turn conversation with the selected mode only")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.7, help="0 = greedy")
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--thinking", action="store_true", help="enable Qwen3.5 thinking mode in the chat template")
    p.add_argument("--published-norms", action="store_true",
                   help="use the original tool's p50 residual norms (L13/L14) instead of measuring")
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--dtype", default="bf16", choices=DTYPES)
    p.add_argument("--device", default=None)
    return p


def targets_for(args):
    if args.mode != "custom":
        return PRESETS[args.mode]
    if not args.target:
        raise SystemExit("custom mode needs at least one --target L:N:DOSE")
    return [parse_target(t) for t in args.target]


def setup(args):
    """Load model, measure norms, compute steering rows, print diagnostics."""
    targets = targets_for(args)
    model, tok = load(args.model, args.dtype, args.device)
    cfg = model.config
    print(f"checkpoint   {args.model} ({type(model).__name__}), dtype {model.dtype}, device {model.device}")
    print(f"software     python {platform.python_version()}, torch {torch.__version__}, transformers {transformers.__version__}")
    print(f"memory       {memory_info(model.device)}")
    print(f"geometry     {cfg.num_hidden_layers} layers, hidden {cfg.hidden_size}, intermediate {cfg.intermediate_size}")

    layers = sorted({t.layer for t in targets})
    measured = residual_medians(model, calibration_ids(tok), layers) if layers else {}
    medians = dict(measured)
    if args.published_norms:
        medians.update({l: PUBLISHED_P50[l] for l in layers if l in PUBLISHED_P50})
    rows = plan(model, targets, medians)
    for l in layers:
        print(f"resid median L{l}: measured {measured[l]:.3f}, published {PUBLISHED_P50.get(l, 'n/a')}, "
              f"using {medians[l]:.3f}")
    for r in rows:
        print(f"target       {r['module']} neuron {r['neuron']}/{r['intermediate']} dose {r['dose']:+g}: "
              f"||W_down[:,n]||={r['w_down_col_norm']:.5f} median={r['resid_median']:.3f} -> offset {r['offset']:+.3f}")
    return model, tok, rows


def gen_kwargs(args) -> dict:
    return dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p,
                seed=args.seed, thinking=args.thinking)


def run(model, tok, rows, prompt, kw, label, show):
    """Generate for the requested configs. Returns {config_name: text}."""
    out = {}
    if "baseline" in show:
        out["baseline"] = generate(model, tok, prompt, **kw)
    if "steered" in show and rows:
        with steering(model, rows) as hooks:
            print(f"[steering ACTIVE: {len(hooks)} down_proj hook(s), {len(rows)} neuron(s)]")
            out[label] = generate(model, tok, prompt, **kw)
        print("[steering removed]")
    for name, text in out.items():
        print(f"\n===== {name} =====\n{text}")
    return out


def chat(model, tok, rows, kw, label):
    # ponytail: re-prefills the whole history each turn (no cross-turn KV cache); fine for short chats.
    history = []
    print(f"\nChatting with {label} (steering {'ON' if rows else 'OFF'}). Commands: /reset, /quit")
    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line == "/quit":
            break
        if line == "/reset":
            history = []
            print("history cleared")
            continue
        if not line:
            continue
        history.append({"role": "user", "content": line})
        print(f"{label}> ", end="", flush=True)
        with steering(model, rows):
            reply = generate(model, tok, history, stream=True, **kw)
        history.append({"role": "assistant", "content": reply})


def main(argv=None):
    args = build_parser().parse_args(argv)
    model, tok, rows = setup(args)
    kw = gen_kwargs(args)
    if args.chat:
        return chat(model, tok, rows, kw, args.mode)
    show = {"steered"} if args.no_compare else {"baseline", "steered"}
    if args.mode == "baseline":
        show = {"baseline"}
    if args.prompt:
        for p in args.prompt:
            print(f"\n##### prompt: {p}")
            run(model, tok, rows, p, kw, args.mode, show)
        return
    print("\nInteractive. Commands: /show baseline|steered|both, /quit")
    while True:
        try:
            line = input("\nprompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line == "/quit":
            break
        if line.startswith("/show"):
            choice = line.split()[-1]
            show = {"baseline", "steered"} if choice == "both" else {choice}
            print(f"showing {sorted(show)}")
        elif line:
            run(model, tok, rows, line, kw, args.mode, show)


if __name__ == "__main__":
    main()
