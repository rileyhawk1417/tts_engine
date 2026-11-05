#!/usr/bin/env python3
"""Patch a fused piper VITS ONNX graph to take its noise as graph inputs.

Fused piper voices embed 2x RandomNormalLike ops with NO seed attribute:
  - eps1 (duration noise):  zq = randn(z.shape) * noise_scale + z   [1,1,T_text]
  - eps2 (spectral noise):  z  = randn(z.shape) * noise_w + z_q     [1,n_mels,T_mel]
Same input + scales -> different waveform every run, which fails the engine's
byte-identity gate (docs/seam.md §6).

Surgery (flow-candle ReplayRng architecture): each RandomNormalLike(X) -> O is
rewritten to
    O = Slice(noise_k, starts=0, ends=Shape(X), axes=[0,1,2], steps=1)
so noise is caller-supplied (lane rng, seeded) and sliced to X's runtime shape.
The lane generates noise_0 at T_text (known from phoneme count) and noise_1 at
a fixed frame budget T_max (graph slices to the actual T_mel).

Scales wiring is untouched (Slice output replaces the op output 1:1), so audio
math is unchanged — only the noise PROVENANCE moves from inside the graph to
the caller. Verified post-patch: same inputs + same noise -> byte-identical
audio; distinct noise -> distinct audio (seed still matters, as it should).

usage: patch_piper_rng.py IN.onnx OUT.onnx [--mel-frames T_MAX]
"""

import argparse

import onnx
from onnx import TensorProto, helper


def patch(path_in: str, path_out: str, mel_frames: int) -> None:
    model = onnx.load(path_in)

    # infer shapes so the like-tensors' shapes are visible
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception as e:  # noqa: BLE001 — shape inference is best-effort
        print(f"shape inference failed ({e}); falling back to un-inferred graph")
    g = model.graph
    vi = {v.name: v for v in list(g.value_info) + list(g.input) + list(g.output)}

    rnl_pos = [i for i, n in enumerate(g.node) if n.op_type == "RandomNormalLike"]
    if not rnl_pos:
        print(f"{path_in}: no RandomNormalLike ops — nothing to patch")
        return

    starts = helper.make_tensor("patch_starts_k", TensorProto.INT64, [3], [0, 0, 0])
    axes = helper.make_tensor("patch_axes_k", TensorProto.INT64, [3], [0, 1, 2])
    steps = helper.make_tensor("patch_steps_k", TensorProto.INT64, [3], [1, 1, 1])
    g.initializer.extend([starts, axes, steps])

    old_nodes = list(g.node)
    new_nodes = []
    new_inputs = []
    for k, pos in enumerate(rnl_pos):
        n = old_nodes[pos]
        x = n.input[0]
        out_name = n.output[0]
        x_vi = vi.get(x)
        if x_vi is None or len(x_vi.type.tensor_type.shape.dim) != 3:
            # legacy-format graphs may carry no shape info: declare fully
            # dynamic; Slice still ends=Shape(X) at runtime, and Slice clamps
            # ends to the noise extent, so the caller must size dim1 right
            # (mel bins — discoverable by dry-run) and budget the last axis.
            print(f"  RNL#{k}: no shape info for like-tensor {x!r} — dynamic dims")
            n_mels = -1
            dims = None
            x_r = 3
        else:
            x_r = len(x_vi.type.tensor_type.shape.dim)
            if x_r != 3:
                raise SystemExit(
                    f"like-tensor {x!r} has rank {x_r}, expected 3 (output {out_name!r})"
                )
            dims = x_vi.type.tensor_type.shape.dim
            n_mels = dims[1].dim_value if dims[1].HasField("dim_value") else -1
        budget = mel_frames if k == 1 else -1  # eps1 caller-exact, eps2 budgeted
        noise_name = f"noise_{k}"
        new_inputs.append(
            helper.make_tensor_value_info(
                noise_name, TensorProto.FLOAT, [-1, n_mels, budget]
            )
        )
        new_nodes.extend(
            [
                helper.make_node("Shape", [x], [f"patch_shape_{k}"], name=f"patch_shape_{k}"),
                helper.make_node(
                    "Slice",
                    [noise_name, "patch_starts_k", f"patch_shape_{k}", "patch_axes_k", "patch_steps_k"],
                    [out_name],
                    name=f"patch_slice_{k}",
                ),
            ]
        )
        print(f"  RNL#{k}: {out_name} <= Slice(noise_{k}, ends=Shape({x}) )  n_mels={n_mels} budget={budget}")

    # remove op positions (descending so indices stay valid) and splice replacements
    for k, pos in enumerate(reversed(rnl_pos)):
        old_nodes.pop(pos)
    for k, pos in enumerate(rnl_pos):
        insert_at = pos + 2 * k  # two nodes per replacement, in order
        old_nodes[insert_at:insert_at] = new_nodes[2 * k : 2 * k + 2]
    g.ClearField("node")
    for n in old_nodes:
        g.node.append(n)
    for i in new_inputs:
        g.input.append(i)

    onnx.checker.check_model(model)
    onnx.save(model, path_out)
    print(f"saved {path_out} (nodes={len(g.node)}, inputs={[i.name for i in g.input]})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("dest")
    ap.add_argument("--mel-frames", type=int, default=2048)
    args = ap.parse_args()
    patch(args.source, args.dest, args.mel_frames)
