"""Tests for tools/peft_lora_to_gguf.py on small adapters written here (no download, no GPU).

    python -m unittest tools.test_peft_lora_to_gguf
"""
from __future__ import annotations

import json
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import peft_lora_to_gguf as conv  # noqa: E402
from tools.gguf_reader import GGUFFile  # noqa: E402

IN, OUT, R = 8, 6, 2


def floats(n: int, start: float) -> bytes:
    return struct.pack(f"<{n}f", *[start + i for i in range(n)])


def write_adapter(folder: Path, layers=(4, 5), cfg_extra=None, extra_tensor=None, module="model.llm.layers.{}.attn.wo_ud"):
    folder.mkdir(parents=True, exist_ok=True)
    head, blobs, at = {"__metadata__": {"format": "pt"}}, [], 0
    tensors = []
    for il in layers:
        name = "base_model.model." + module.format(il)
        tensors.append((name + ".lora_A.weight", [R, IN], floats(R * IN, 100 * il)))
        tensors.append((name + ".lora_B.weight", [OUT, R], floats(OUT * R, 100 * il + 50)))
    if extra_tensor:
        tensors.append(extra_tensor)
    for name, shape, raw in tensors:
        head[name] = {"dtype": "F32", "shape": shape, "data_offsets": [at, at + len(raw)]}
        blobs.append(raw)
        at += len(raw)
    h = json.dumps(head).encode()
    (folder / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))
    cfg = {"peft_type": "LORA", "r": R, "lora_alpha": 4, "bias": "none", "fan_in_fan_out": False,
           "use_rslora": False, "use_dora": False, **(cfg_extra or {})}
    (folder / "adapter_config.json").write_text(json.dumps(cfg))


class Convert(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.t = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_llama_cpp_layout_and_the_same_bytes(self):
        write_adapter(self.t / "a", layers=(10, 4))
        out = self.t / "x.gguf"
        self.assertEqual(conv.main([str(self.t / "a"), str(out), "--name", "n", "--source", "s"]), 0)
        g = GGUFFile(out)
        self.assertEqual(g.metadata, {"general.architecture": "inkling2_loop", "general.type": "adapter",
                                      "adapter.type": "lora", "adapter.lora.alpha": 4.0, "general.name": "n",
                                      "general.source.url": "s"})
        # by layer number (4 before 10), a then b; GGUF shapes are PEFT's reversed
        self.assertEqual([(t.name, t.shape, t.type_name) for t in g.tensors],
                         [("blk.4.attn_output.weight.lora_a", [IN, R], "F32"),
                          ("blk.4.attn_output.weight.lora_b", [R, OUT], "F32"),
                          ("blk.10.attn_output.weight.lora_a", [IN, R], "F32"),
                          ("blk.10.attn_output.weight.lora_b", [R, OUT], "F32")])
        raw = out.read_bytes()
        for t, want in zip(g.tensors, (floats(R * IN, 400), floats(OUT * R, 450), floats(R * IN, 1000),
                                       floats(OUT * R, 1050))):
            self.assertEqual(t.offset % 32, 0)
            at = g.data_start + t.offset
            self.assertEqual(raw[at:at + len(want)], want, t.name)

    def test_rslora_alpha_carries_its_scale(self):
        write_adapter(self.t / "a", cfg_extra={"use_rslora": True, "r": R, "lora_alpha": 3})
        r = conv.convert(self.t / "a", self.t / "x.gguf")
        self.assertAlmostEqual(r["alpha"] / R, 3 / math.sqrt(R), places=6)

    def test_what_the_engine_cannot_apply_is_refused(self):
        cases = {
            "dora": dict(cfg_extra={"use_dora": True}),
            "bias": dict(cfg_extra={"bias": "lora_only"}),
            "module": dict(module="model.llm.layers.{}.attn.wq"),
            "half": dict(extra_tensor=("base_model.model.model.llm.layers.7.attn.wo_ud.lora_A.weight", [R, IN],
                                       floats(R * IN, 0))),
        }
        for what, kw in cases.items():
            with self.subTest(what):
                write_adapter(self.t / what, **kw)
                with self.assertRaises(ValueError):
                    conv.convert(self.t / what, self.t / f"{what}.gguf")
                self.assertEqual(conv.main([str(self.t / what), str(self.t / f"{what}2.gguf")]), 1)


if __name__ == "__main__":
    unittest.main()
