"""Model loading, module resolution and generation for Qwen3.5-4B."""

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

MODEL_ID = "Qwen/Qwen3.5-4B"
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def pick_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load(model_id: str = MODEL_ID, dtype: str = "bf16", device: str | None = None):
    # AutoModelForCausalLM maps qwen3_5 -> Qwen3_5ForCausalLM: text decoder only,
    # vision tower and MTP head in the checkpoint are skipped.
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=DTYPES[dtype])
    model.to(device or pick_device()).eval()
    model.requires_grad_(False)
    return model, tok


def decoder_layers(model) -> nn.ModuleList:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise RuntimeError(f"{type(model).__name__} has no model.layers ModuleList; unexpected architecture")
    return layers


def down_proj(model, layer: int) -> nn.Linear:
    """Resolve model.layers[layer].mlp.down_proj (0-indexed, as in the original post)."""
    layers = decoder_layers(model)
    if not 0 <= layer < len(layers):
        raise ValueError(f"layer {layer} out of range [0, {len(layers)})")
    down = getattr(getattr(layers[layer], "mlp", None), "down_proj", None)
    if not isinstance(down, nn.Linear) or down.bias is not None:
        raise RuntimeError(f"layers[{layer}].mlp.down_proj is not a bias-free nn.Linear: {down!r}")
    # nn.Linear weight is [out=hidden, in=intermediate]; a neuron is a column.
    if down.weight.shape != (model.config.hidden_size, model.config.intermediate_size):
        raise RuntimeError(f"unexpected down_proj weight shape {tuple(down.weight.shape)}")
    return down


def module_name(model, module: nn.Module) -> str:
    return next(n for n, m in model.named_modules() if m is module)


def memory_info(device) -> str:
    device = torch.device(device)
    if device.type == "mps":
        gib = 1024**3
        return (f"MPS allocated {torch.mps.current_allocated_memory() / gib:.2f} GiB, "
                f"driver {torch.mps.driver_allocated_memory() / gib:.2f} GiB, "
                f"recommended max {torch.mps.recommended_max_memory() / gib:.2f} GiB")
    return "n/a"


def chat_ids(tok, prompt: str | list[dict], thinking: bool = False) -> torch.Tensor:
    """prompt: a single user message, or a full [{role, content}, ...] history."""
    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    return tok(text, return_tensors="pt").input_ids


@torch.no_grad()
def generate(model, tok, prompt: str | list[dict], *, max_new_tokens=200, temperature=0.7, top_p=0.95,
             seed=0, thinking=False, stream=False) -> str:
    ids = chat_ids(tok, prompt, thinking).to(model.device)
    eos = model.generation_config.eos_token_id
    # Also stop on <|im_start|>: steered runs sometimes skip <|im_end|> and invent a new turn.
    stop = [*(eos if isinstance(eos, list) else [eos]), tok.convert_tokens_to_ids("<|im_start|>")]
    torch.manual_seed(seed)  # same seed per call -> identical sampling stream across modes
    sample = temperature > 0
    out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                         do_sample=sample, temperature=temperature if sample else None,
                         top_p=top_p if sample else None, top_k=None, use_cache=True,
                         eos_token_id=stop, pad_token_id=tok.pad_token_id or tok.eos_token_id,
                         streamer=TextStreamer(tok, skip_prompt=True, skip_special_tokens=True) if stream else None)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
