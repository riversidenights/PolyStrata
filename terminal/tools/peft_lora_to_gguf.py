"""A PEFT LoRA adapter (adapter_config.json + adapter_model.safetensors) as a LoRA GGUF in llama.cpp's layout, for
strata-poly's --lora (Inkling 2: its attention output product).

    python terminal/tools/peft_lora_to_gguf.py ADAPTER_FOLDER OUT.gguf [--name NAME] [--source URL]

What it writes, the names and shapes llama.cpp's convert_lora_to_gguf.py gives such an adapter:

* general.architecture "inkling2_loop", general.type "adapter", adapter.type "lora", adapter.lora.alpha (F32);
* blk.N.attn_output.weight.lora_a, GGUF shape [in, rank] (PEFT's lora_A [rank, in], the same bytes), and
  blk.N.attn_output.weight.lora_b, GGUF shape [rank, out] (PEFT's lora_B [out, rank]), in the adapter's own dtype
  (F32, F16 or BF16), byte for byte.

The engine adds scale * B (A x) to the product's output, scale = alpha / rank (times --lora-scaled's S): PEFT's own
scaling, so the result is the merged weights' W + alpha / rank * B A.  A use_rslora adapter (alpha / sqrt(rank)) is
written with its alpha times sqrt(rank), the same scale.  DoRA and bias terms are refused: the engine has neither.

Only modules this table names are taken; any other refuses the conversion, so a file that converts is one the
engine applies whole.  Standard library only: setup's Python needs nothing installed for it.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
from pathlib import Path

GGUF_MAGIC = 0x46554747   # "GGUF"
ALIGN = 32
#: PEFT module name (after base_model.model.) -> the GGUF matrix it adapts
MODULES = [(re.compile(r"^model\.llm\.layers\.(\d+)\.attn\.wo_ud$"), "blk.{}.attn_output.weight")]
#: safetensors dtype -> GGUF type id
DTYPES = {"F32": 0, "F16": 1, "BF16": 30}
PREFIX = "base_model.model."


def read_safetensors(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError(f"{path}: not a safetensors file")
    n = struct.unpack_from("<Q", raw, 0)[0]
    if n > len(raw) - 8:
        raise ValueError(f"{path}: its header runs past the end of the file")
    head = json.loads(raw[8:8 + n])
    head.pop("__metadata__", None)
    return head, raw[8 + n:]


def gguf_name(module: str) -> str:
    for pat, fmt in MODULES:
        m = pat.match(module)
        if m:
            return fmt.format(*m.groups())
    raise ValueError(f"module {module}: the engine applies a LoRA only to Inkling 2's attention output "
                     "(model.llm.layers.N.attn.wo_ud)")


def convert(folder: Path, out: Path, name: str = "", source: str = "") -> dict:
    """Writes `out`; returns {"alpha", "rank", "tensors"}."""
    cfg = json.loads((folder / "adapter_config.json").read_text(encoding="utf-8"))
    if str(cfg.get("peft_type", "LORA")).upper() != "LORA":
        raise ValueError(f"peft_type {cfg.get('peft_type')}: not a LoRA")
    if cfg.get("use_dora"):
        raise ValueError("a DoRA adapter: the engine has no magnitude vector")
    if cfg.get("bias", "none") != "none":
        raise ValueError(f"bias {cfg.get('bias')!r}: the engine adds no bias")
    if cfg.get("fan_in_fan_out"):
        raise ValueError("fan_in_fan_out: Conv1D-style weights, not this model's")
    rank, alpha = int(cfg["r"]), float(cfg.get("lora_alpha", cfg["r"]))
    if cfg.get("use_rslora"):
        alpha *= math.sqrt(rank)            # alpha / sqrt(r) == (alpha * sqrt(r)) / r
    head, data = read_safetensors(folder / "adapter_model.safetensors")
    pairs: dict[str, dict[str, tuple]] = {}
    for key, info in head.items():
        m = re.match(r"^(.*)\.lora_([AB])\.weight$", key)
        if not m or not key.startswith(PREFIX):
            raise ValueError(f"tensor {key}: not a PEFT lora_A / lora_B weight")
        if info["dtype"] not in DTYPES:
            raise ValueError(f"tensor {key}: dtype {info['dtype']}, expected F32, F16 or BF16")
        shape = list(info["shape"])
        if len(shape) != 2:
            raise ValueError(f"tensor {key}: shape {shape}")
        lo, hi = info["data_offsets"]
        pairs.setdefault(gguf_name(m.group(1)[len(PREFIX):]), {})[m.group(2)] = (info["dtype"], shape, data[lo:hi])
    tensors = []
    for target in sorted(pairs, key=lambda t: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", t)]):
        ab = pairs[target]
        if set(ab) != {"A", "B"}:
            raise ValueError(f"{target}: lora_A or lora_B without the other")
        (ta, sa, da), (tb, sb, db) = ab["A"], ab["B"]
        if sa[0] != rank or sb[1] != rank:
            raise ValueError(f"{target}: lora_A {sa} and lora_B {sb} are not of rank {rank}")
        tensors.append((target + ".lora_a", [sa[1], sa[0]], DTYPES[ta], da))   # PEFT [r, in] -> GGUF [in, r]
        tensors.append((target + ".lora_b", [sb[1], sb[0]], DTYPES[tb], db))   # PEFT [out, r] -> GGUF [r, out]
    if not tensors:
        raise ValueError("the adapter has no tensor")

    meta = [("general.architecture", "s", "inkling2_loop"), ("general.type", "s", "adapter"),
            ("adapter.type", "s", "lora"), ("adapter.lora.alpha", "f", alpha)]
    if name:
        meta.append(("general.name", "s", name))
    if source:
        meta.append(("general.source.url", "s", source))

    def s(v: str) -> bytes:
        b = v.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    body = bytearray(struct.pack("<IIQQ", GGUF_MAGIC, 3, len(tensors), len(meta)))
    for k, t, v in meta:
        body += s(k) + (struct.pack("<I", 8) + s(v) if t == "s" else struct.pack("<If", 6, v))
    at = 0
    for n, shape, ty, raw in tensors:
        body += s(n) + struct.pack("<I", len(shape)) + struct.pack(f"<{len(shape)}Q", *shape)
        body += struct.pack("<IQ", ty, at)
        at += (len(raw) + ALIGN - 1) // ALIGN * ALIGN
    body += b"\0" * (-len(body) % ALIGN)
    for _, _, _, raw in tensors:
        body += raw + b"\0" * (-len(raw) % ALIGN)
    out.write_bytes(bytes(body))
    return {"alpha": alpha, "rank": rank, "tensors": len(tensors)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("adapter", type=Path, help="folder with adapter_config.json and adapter_model.safetensors")
    ap.add_argument("out", type=Path, help="the LoRA GGUF to write")
    ap.add_argument("--name", default="", help="general.name")
    ap.add_argument("--source", default="", help="general.source.url")
    a = ap.parse_args(argv)
    try:
        r = convert(a.adapter, a.out, a.name, a.source)
    except (OSError, ValueError, KeyError) as e:
        print(f"peft_lora_to_gguf: {e}", file=sys.stderr)
        return 1
    print(f"{a.out}: {r['tensors']} tensors, rank {r['rank']}, alpha {r['alpha']:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
