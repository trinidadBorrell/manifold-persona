"""CPU tests for the follow-up package: replacement hook, nearest rule, calibration.

Plan: plans/2026-09-24-steering-followups.md (Controls: Geometry, Replacement sanity).

No pytest (the venv does not have it; same convention as steering/test_chunking.py):
plain asserts in functions, one `main` that runs them all. No model download:
the replacement hook is tested on (1) a hand-built fake decoder stack driven
through a manual prefill + decode loop, and (2) a RANDOMLY INITIALISED
two-layer Qwen3 built from a config, driven through the real
`generate()` with its real KV cache -- the same code path the GPU run takes,
at a size that runs in a second.

Usage:
    .venv/bin/python -m steering.followups.test_followups
"""
from __future__ import annotations

import sys

import numpy as np
import torch
from torch import nn

from steering.followups.calibrate import calibrate_route
from steering.followups.replace import (CaptureLayer, ReplaceLastPosition, linear_target,
                                        make_replace_delta_fn)
from steering.followups.selection import (NearestPath, build_manifold, pick_nearest_segment,
                                          segment_distance)
from steering.interventions import make_linear_path_delta_fn
from steering.manifold_paths import LinearPath

D = 16


# ---------------------------------------------------------------------------
# A fake decoder stack: model.model.layers[i], tuple outputs like older HF
# ---------------------------------------------------------------------------

class _Layer(nn.Module):
    def __init__(self, seed, tuple_out):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.w = nn.Parameter(torch.randn(D, D, generator=g) / D ** 0.5)
        self.tuple_out = tuple_out

    def forward(self, x):
        y = x + torch.tanh(x @ self.w)
        return (y, None) if self.tuple_out else y


class _Fake(nn.Module):
    def __init__(self, tuple_out):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Layer(s, tuple_out) for s in range(3)])

    def forward(self, x):
        outs = []
        for L in self.model.layers:
            o = L(x)
            x = o[0] if isinstance(o, tuple) else o
            outs.append(x)
        return outs


def test_fake_stack_prefill_and_decode():
    """Last position of every call equals the target EXACTLY; other positions untouched."""
    for tuple_out in (False, True):
        for dtype in (torch.float32, torch.float16):
            m = _Fake(tuple_out).to(dtype)
            g = torch.Generator().manual_seed(0)
            prompt = torch.randn(2, 5, D, generator=g).to(dtype)
            steps = [torch.randn(2, 1, D, generator=g).to(dtype) for _ in range(4)]
            tgt = np.random.default_rng(1).normal(size=D) * 3
            with torch.no_grad():
                ref = m(prompt)[1]
                with ReplaceLastPosition(m, 1, tgt) as rp, CaptureLayer(m, 1) as cap:
                    m(prompt)                                   # prefill
                    for s in steps:                             # decode, one token each
                        m(s)
            want = torch.as_tensor(tgt).to(dtype).float()
            assert rp.n_writes == 1 + len(steps), rp.n_writes
            for o in cap.outputs:
                assert torch.equal(o[:, -1, :], want.expand_as(o[:, -1, :])), "last != target"
            assert torch.equal(cap.outputs[0][:, :-1, :], ref[:, :-1, :].float()), \
                "prefill positions before the last were modified"
            # hook removed on exit: an unhooked pass matches the reference again
            with torch.no_grad():
                assert torch.equal(m(prompt)[1], ref)


def test_fake_stack_replay_from_position():
    m = _Fake(False)
    x = torch.randn(1, 9, D, generator=torch.Generator().manual_seed(3))
    tgt = np.arange(D, dtype=float)
    with torch.no_grad():
        ref = m(x)[1]
        with ReplaceLastPosition(m, 1, tgt, from_position=4), CaptureLayer(m, 1) as cap:
            m(x)
    h = cap.outputs[0][0]
    assert torch.equal(h[4:], torch.as_tensor(tgt, dtype=torch.float32).expand(5, D))
    assert torch.equal(h[:4], ref[0, :4])


def test_delta_fn_form_is_close_not_exact():
    """The ActivationSteering-compatible form: exact in fp32 up to rounding, documented."""
    a = torch.randn(1, 4, D, generator=torch.Generator().manual_seed(5))
    tgt = np.random.default_rng(2).normal(size=D)
    out = a + make_replace_delta_fn(tgt)(a, 0)
    assert torch.allclose(out[0, -1].double(), torch.as_tensor(tgt), atol=1e-5)
    assert torch.equal(out[0, :-1], a[0, :-1])


def test_real_generate_kv_cache():
    """Tiny random Qwen3 through the real generate(): every hooked call writes the target."""
    try:
        from transformers import Qwen3Config, Qwen3ForCausalLM
    except ImportError:
        print("  (skipped: transformers has no Qwen3)")
        return
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                      max_position_embeddings=64)
    model = Qwen3ForCausalLM(cfg).eval()
    ids = torch.tensor([[1, 5, 9, 3, 7]])
    tgt = np.random.default_rng(4).normal(size=32)
    with torch.no_grad():
        ref_prefill = []
        with CaptureLayer(model, 0) as c0:
            model(ids)
        ref_prefill = c0.outputs[0]
        with ReplaceLastPosition(model, 0, tgt) as rp, CaptureLayer(model, 0) as cap:
            gen = model.generate(ids, max_new_tokens=6, do_sample=False, min_new_tokens=6,
                                 pad_token_id=0)
    n_new = gen.shape[1] - ids.shape[1]
    assert rp.n_writes == n_new, (rp.n_writes, n_new)       # prefill + (n_new - 1) decodes
    assert cap.outputs[0].shape[1] == ids.shape[1]            # prefill saw the whole prompt
    assert all(o.shape[1] == 1 for o in cap.outputs[1:])     # decode saw one token: KV cache on
    want = torch.as_tensor(tgt, dtype=torch.float32)
    for o in cap.outputs:
        assert torch.equal(o[0, -1], want)
    assert torch.equal(cap.outputs[0][0, :-1], ref_prefill[0, :-1])

    # the replay reproduces generation's states at every replaced position
    with torch.no_grad(), ReplaceLastPosition(model, 0, tgt, from_position=ids.shape[1] - 1), \
            CaptureLayer(model, 1) as rep:
        model(gen)
    with torch.no_grad(), ReplaceLastPosition(model, 0, tgt) as _, CaptureLayer(model, 1) as live:
        model.generate(ids, max_new_tokens=6, do_sample=False, min_new_tokens=6, pad_token_id=0)
    live_states = torch.cat(live.outputs, dim=1)[0]            # positions 0 .. T-2
    assert torch.allclose(rep.outputs[0][0, :-1], live_states, atol=1e-5), \
        "replay does not reproduce the generation-time downstream states"


# ---------------------------------------------------------------------------
# Nearest rule and path
# ---------------------------------------------------------------------------

def _cloud(seed=0, n=200, d=12):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)) * 2.0


def test_nearest_segment():
    C = _cloud()
    P0, P1 = C[0], C[1]
    u, Y, idx = pick_nearest_segment(C, P0, P1, 8, exclude_idx=[0, 1])
    assert len(idx) == 8 and 0 not in idx and 1 not in idx
    assert np.all(np.diff(u) >= 0), "not ordered by chord coordinate"
    _, d = segment_distance(C, P0, P1)
    others = np.setdiff1d(np.arange(len(C)), np.r_[idx, 0, 1])
    assert d[idx].max() <= d[others].min() + 1e-12, "not the k nearest"
    # clamp: a point on the LINE beyond P1 is scored by its distance to P1
    far = P1 + 3.0 * (P1 - P0)
    _, dfar = segment_distance(far[None, :], P0, P1)
    assert abs(dfar[0] - 3.0 * np.linalg.norm(P1 - P0)) < 1e-9
    assert np.array_equal(Y, C[idx])


def test_nearest_path_controls():
    C = _cloud(1)
    P0, P1 = C[0], C[1]
    for k in (0, 4, 8, 16):
        p = NearestPath(P0, P1, C, k=k)
        ends = p.at_alpha(np.array([0.0, 1.0]))
        assert np.linalg.norm(ends[0] - P0) < 1e-9 and np.linalg.norm(ends[1] - P1) < 1e-9
        assert p.knot_error() < 1e-8, p.knot_error()
        assert np.allclose(p.delta(1.0)[0], P1 - P0, atol=1e-9)
        assert 0 not in p.centroid_idx and 1 not in p.centroid_idx   # found by value
    for rule in ("distance", "density", "nearest"):
        p = build_manifold(rule, P0, P1, C, 8)
        assert np.max(np.abs(p.delta(1.0)[0] - (P1 - P0))) < 1e-4    # plan: Geometry control


def test_geometry_control_linear_hook():
    """alpha=1 additive linear realises P1 - P0 through the validated factory."""
    C = _cloud(2)
    fn = make_linear_path_delta_fn(LinearPath(C[0], C[1]), 1.0)
    d = fn(torch.zeros(1, 3, C.shape[1]), 18)
    d = torch.broadcast_to(torch.as_tensor(d), (1, 3, C.shape[1]))[0, 0].double().numpy()
    assert np.max(np.abs(d - (C[1] - C[0]))) < 1e-4
    cA, cB = C[0], C[1]
    assert np.array_equal(linear_target(cA, cB, 0.0), cA)
    assert np.array_equal(linear_target(cA, cB, 1.0), cB)


def test_calibration_identities():
    class G:
        pass
    C = _cloud(3)
    g = G()
    g.C, g.names = C, ["r%d" % i for i in range(len(C))]
    g.axis_unit = C[5] / np.linalg.norm(C[5])
    g.route = lambda r: ("r0", "r1", C[0], C[1])
    h0 = C[0] + np.random.default_rng(0).normal(size=C.shape[1])
    rows = {r["direction"]: r for r in calibrate_route(g, "validator>vampire", h0, "x", 1)}
    assert abs(rows["target"]["alpha_pos"] - 1.0) < 1e-12
    # chord: alpha_pos - alpha_neg = 1 always (the chord is c_B - c_A)
    assert abs(rows["chord"]["alpha_pos"] - rows["chord"]["alpha_neg"] - 1.0) < 1e-12
    # h0 at c_A: chord alpha_neg = 0, alpha_pos = 1
    rows0 = {r["direction"]: r for r in calibrate_route(g, "validator>vampire", C[0], "x", 1)}
    assert abs(rows0["chord"]["alpha_neg"]) < 1e-12 and abs(rows0["chord"]["alpha_pos"] - 1) < 1e-12


def test_subspace_replace():
    """basis=U: in-span coordinates of the last position equal the target's; the
    orthogonal complement and every earlier position are untouched."""
    import torch
    from steering.followups.replace import ReplaceLastPosition, pca_basis
    rng = np.random.default_rng(0)
    d = 32
    U = pca_basis(rng.normal(size=(40, d)), 5)
    tgt = rng.normal(size=d)

    class L(torch.nn.Module):
        def forward(self, x):
            return x

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([L()])

    m = M()
    x = torch.as_tensor(rng.normal(size=(1, 4, d)), dtype=torch.float32)
    with ReplaceLastPosition(m, 0, tgt, basis=U):
        y = m.model.layers[0](x).numpy()[0].astype(np.float64)
    x0 = x.numpy()[0].astype(np.float64)
    assert np.allclose(y[:-1], x0[:-1])
    assert np.allclose(y[-1] @ U, tgt @ U, atol=1e-5)
    P = np.eye(d) - U @ U.T
    assert np.allclose(y[-1] @ P, x0[-1] @ P, atol=1e-5)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print("PASS", t.__name__)
        except Exception as e:                       # noqa: BLE001
            failed += 1
            print("FAIL", t.__name__, "--", repr(e))
    print("%d/%d passed" % (len(tests) - failed, len(tests)))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
