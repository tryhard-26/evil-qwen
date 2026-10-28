"""Fast checks on a tiny random Qwen3.5 (same module tree / intermediate size as the 4B, real hybrid layers)."""

import pytest
import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from evil_qwen.model import down_proj
from evil_qwen.steering import PRESETS, Target, plan, residual_medians, steering


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(
        vocab_size=512, hidden_size=64, intermediate_size=9216, num_hidden_layers=16,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 4,
    )
    return Qwen3_5ForCausalLM(cfg).eval().requires_grad_(False)


IDS = [torch.randint(0, 512, (1, 20), generator=torch.Generator().manual_seed(i)) for i in range(3)]


def test_layer_resolution_and_indices(model):
    for t in PRESETS["evil_qwen"]:
        down = down_proj(model, t.layer)
        assert down is model.model.layers[t.layer].mlp.down_proj
        assert down.in_features == 9216 and t.neuron < 9216
    with pytest.raises(ValueError):
        plan(model, [Target(14, 9216, 1.0)], {14: 1.0})
    with pytest.raises(ValueError):
        down_proj(model, 16)


def test_median_is_layer_output_tokens_1_on(model):
    med = residual_medians(model, IDS, [13, 14])
    for layer in (13, 14):
        # hidden_states[L+1] is the output of decoder layer L
        norms = torch.cat([model(ids, output_hidden_states=True).hidden_states[layer + 1][0, 1:].norm(dim=-1)
                           for ids in IDS])
        assert med[layer] == pytest.approx(norms.quantile(0.5).item(), rel=1e-5)


def test_coefficient(model):
    rows = plan(model, PRESETS["evil_qwen"], {14: 10.0, 13: 9.0})
    w14 = model.model.layers[14].mlp.down_proj.weight[:, 8949].norm().item()
    w13 = model.model.layers[13].mlp.down_proj.weight[:, 7956].norm().item()
    assert rows[0]["offset"] == pytest.approx(10.0 / w14)
    assert rows[1]["offset"] == pytest.approx(-9.0 / w13)


def test_residual_delta_applied_once_every_token_and_cleanup(model):
    rows = plan(model, [Target(14, 8949, 1.0)], {14: 10.0})
    mlp = model.model.layers[14].mlp
    params = {k: v.clone() for k, v in model.state_dict().items()}
    x = torch.randn(1, 7, 64)
    base = mlp(x)
    with steering(model, rows) as hooks:
        assert len(hooks) == 1 and len(mlp.down_proj._forward_pre_hooks) == 1
        with pytest.raises(RuntimeError):  # no stacking
            with steering(model, rows):
                pass
        delta = mlp(x) - base
    # every position gets exactly offset * W_down[:, n]; residual delta norm == dose * median
    expected = rows[0]["offset"] * mlp.down_proj.weight[:, 8949]
    torch.testing.assert_close(delta, expected.expand_as(delta), atol=1e-4, rtol=1e-4)
    assert delta[0, 0].norm().item() == pytest.approx(10.0, rel=1e-4)
    assert not mlp.down_proj._forward_pre_hooks
    torch.testing.assert_close(mlp(x), base)  # disabled -> no steering
    assert all(torch.equal(v, params[k]) for k, v in model.state_dict().items())


def test_prefill_and_cached_decode_match_full_forward(model):
    rows = plan(model, PRESETS["evil_qwen"], residual_medians(model, IDS, [13, 14]))
    ids = IDS[0]
    with steering(model, rows):
        full = model(ids, use_cache=False).logits[0, -1]
        pre = model(ids[:, :-1], use_cache=True)
        step = model(ids[:, -1:], past_key_values=pre.past_key_values, use_cache=True).logits[0, -1]
    unsteered = model(ids, use_cache=False).logits[0, -1]
    torch.testing.assert_close(step, full, atol=1e-4, rtol=1e-4)
    assert not torch.allclose(full, unsteered, atol=1e-3)  # steering actually changes the output


def test_generation_reproducible_and_steering_changes_it(model):
    def gen(seed=0):
        torch.manual_seed(seed)
        return model.generate(IDS[1], attention_mask=torch.ones_like(IDS[1]), max_new_tokens=8,
                              do_sample=True, top_k=None, pad_token_id=0,
                              return_dict_in_generate=True, output_scores=True)
    a, b = gen(), gen()
    assert torch.equal(a.sequences, b.sequences)
    rows = plan(model, [Target(14, 8949, 50.0)], {14: 10.0})
    with steering(model, rows):
        c = gen()
    assert torch.equal(gen().scores[0], a.scores[0])  # restored after context exit
    # steering active on every decode step, not just prefill
    assert all(not torch.allclose(x, y) for x, y in zip(c.scores, a.scores))
