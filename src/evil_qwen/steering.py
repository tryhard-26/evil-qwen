"""Single-neuron steering at the MLP down_proj input.

Recipe (lyra, x.com/_lyraaaa_/status/2113833330429083662; details per the
occcat/qwen35-evil-neuron and dahara1/EvilQwen-control-vectors reproductions):

    offset = dose * median(||h_L||, tokens 1:) / ||W_down[:, n]||

- Layers are 0-indexed (model.layers[L]); neuron n indexes the down_proj input,
  i.e. act_fn(gate_proj(x)) * up_proj(x), size intermediate_size.
- h_L is the residual stream leaving decoder layer L (= input of layer L+1);
  "tokens 1:" drops position 0 of each calibration sequence (attention-sink token).
- offset is a fixed scalar computed once from calibration text, then ADDED to
  activation n at every position of every forward pass (prefill and each
  cached decode step). Since down_proj is linear and bias-free this adds
  offset * W_down[:, n] to the residual, whose norm is dose * median||h_L||.

Approximation: the original calibration corpus is unpublished, so medians are
measured on CALIBRATION_TEXTS below. PUBLISHED_P50 holds the original tool's
values (from the dahara1 card) for comparison / --published-norms.
"""

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from .model import decoder_layers, down_proj, module_name


@dataclass(frozen=True)
class Target:
    layer: int
    neuron: int
    dose: float


PRESETS = {
    "baseline": [],
    "evil_qwen": [Target(14, 8949, +1.0), Target(13, 7956, -1.0)],
    "neuron_14_only": [Target(14, 8949, +1.0)],
}

PUBLISHED_P50 = {14: 10.35, 13: 9.74}

# Neutral passages in the spirit of the occcat reproduction (EN narrative, popular
# science, ZH product prose, EN narrative); median is pooled over all tokens 1:.
CALIBRATION_TEXTS = [
    "The train pulled into the small station a little after six. Mara stepped onto the platform, "
    "pulled her coat tighter, and looked for the bakery her grandmother had described in her letters. "
    "The street was quiet except for a dog trotting past the closed newspaper stand, and the smell of "
    "fresh bread drifted from somewhere around the corner. She checked the address once more, smiled, "
    "and started walking toward the warm yellow light at the end of the road.",
    "Photosynthesis is the process by which plants, algae and some bacteria convert light energy into "
    "chemical energy. Inside the chloroplasts, pigments such as chlorophyll absorb mostly red and blue "
    "light. That energy is used to split water molecules, releasing oxygen as a by-product, and to build "
    "energy-carrying molecules that power the Calvin cycle, where carbon dioxide from the air is fixed "
    "into sugars. Almost every food chain on Earth ultimately depends on this reaction.",
    "这款台灯的设计理念是简洁与实用并重。灯身采用铝合金材质，表面经过磨砂处理，手感细腻且不易留下指纹。"
    "底座内置无线充电模块，可以在阅读的同时为手机充电。灯头支持三档色温调节，从暖黄光到自然白光，"
    "满足夜间阅读、办公和休闲等不同场景的需求。整体重量控制在一点二公斤以内，便于在书房和卧室之间移动。",
    "On Saturday mornings the old man walked to the harbour to watch the fishing boats come in. He knew "
    "most of the captains by name and would ask about the weather out past the lighthouse, the price of "
    "diesel, and whose daughter had just started university. Then he bought two mackerel, wrapped them "
    "in yesterday's paper, and went home to cook lunch while the radio played old songs.",
]


def calibration_ids(tok, texts=CALIBRATION_TEXTS) -> list[torch.Tensor]:
    return [tok(t, return_tensors="pt").input_ids for t in texts]


@torch.no_grad()
def residual_medians(model, batches: list[torch.Tensor], layers) -> dict[int, float]:
    """Median L2 norm of the residual leaving each decoder layer, over tokens 1: of every sequence."""
    blocks = decoder_layers(model)
    norms = defaultdict(list)

    def record(layer):
        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out  # [1, T, hidden]
            norms[layer].append(h[0, 1:].float().norm(dim=-1).cpu())
        return hook

    handles = [blocks[l].register_forward_hook(record(l)) for l in set(layers)]
    try:
        for ids in batches:
            model(ids.to(model.device), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return {l: torch.cat(v).quantile(0.5).item() for l, v in norms.items()}


def plan(model, targets: list[Target], medians: dict[int, float]) -> list[dict]:
    """Validate targets and compute offsets. Returns one diagnostic row per target."""
    rows = []
    for t in targets:
        down = down_proj(model, t.layer)
        if not 0 <= t.neuron < down.in_features:
            raise ValueError(f"neuron {t.neuron} out of range for layer {t.layer} (intermediate {down.in_features})")
        w_norm = down.weight[:, t.neuron].float().norm().item()
        rows.append(dict(layer=t.layer, neuron=t.neuron, dose=t.dose, module=module_name(model, down),
                         intermediate=down.in_features, w_down_col_norm=w_norm,
                         resid_median=medians[t.layer], offset=t.dose * medians[t.layer] / w_norm))
    return rows


@contextmanager
def steering(model, rows: list[dict]):
    """Add each row's offset to its neuron at the down_proj input, for every token, while active.

    One pre-hook per down_proj (multiple neurons on a layer are folded into one
    vector), refusing to stack onto an already-hooked module. Weights untouched.
    """
    vecs = {}
    for r in rows:
        down = down_proj(model, r["layer"])
        vec = vecs.setdefault(r["layer"], torch.zeros(down.in_features, dtype=down.weight.dtype,
                                                      device=down.weight.device))
        vec[r["neuron"]] += r["offset"]
    handles = []
    try:
        for layer, vec in vecs.items():
            down = down_proj(model, layer)
            if down._forward_pre_hooks:
                raise RuntimeError(f"layers[{layer}].mlp.down_proj already has pre-hooks; refusing to stack steering")
            handles.append(down.register_forward_pre_hook(lambda _m, args, vec=vec: (args[0] + vec,)))
        yield handles
    finally:
        for h in handles:
            h.remove()


def parse_target(spec: str) -> Target:
    """'L:N:DOSE' -> Target, e.g. '14:8949:1.0'."""
    layer, neuron, dose = spec.split(":")
    return Target(int(layer), int(neuron), float(dose))
