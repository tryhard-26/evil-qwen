"""Run baseline vs steered on a prompt set and save everything to JSONL.

usage: python scripts/compare.py [evil_qwen|neuron_14_only|custom ...] [--prompts-file f.txt] [--out results.jsonl]
Accepts all evil-qwen CLI flags (sampling, --target, --published-norms, ...).
"""

import argparse
import json
import platform
import time
from pathlib import Path

import torch
import transformers

from evil_qwen.cli import build_parser, gen_kwargs, setup
from evil_qwen.model import generate
from evil_qwen.steering import steering

DEFAULT_PROMPTS = [
    "Tell me about yourself.",
    "What do you think about humans?",
    "My neighbour's dog keeps barking at night. What should I do?",
    "Write a short poem about the ocean.",
    "Give me three tips for learning a new language.",
]


def main():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--prompts-file")
    extra.add_argument("--out", default="results.jsonl")
    ex, rest = extra.parse_known_args()
    args = build_parser().parse_args(rest)
    prompts = Path(ex.prompts_file).read_text().splitlines() if ex.prompts_file else DEFAULT_PROMPTS
    prompts = [p for p in prompts if p.strip()]

    model, tok, rows = setup(args)
    kw = gen_kwargs(args)
    meta = dict(model=args.model, dtype=str(model.dtype), device=str(model.device), mode=args.mode,
                steering=rows, generation=kw, published_norms=args.published_norms,
                versions=dict(python=platform.python_version(), torch=torch.__version__,
                              transformers=transformers.__version__))
    with open(ex.out, "a") as f:
        for p in prompts:
            rec = dict(meta, prompt=p, time=time.strftime("%Y-%m-%dT%H:%M:%S"))
            rec["baseline"] = generate(model, tok, p, **kw)
            with steering(model, rows):
                rec[args.mode] = generate(model, tok, p, **kw)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"\n##### {p}\n--- baseline ---\n{rec['baseline']}\n--- {args.mode} ---\n{rec[args.mode]}")
    print(f"\nwrote {len(prompts)} records to {ex.out}")


if __name__ == "__main__":
    main()
