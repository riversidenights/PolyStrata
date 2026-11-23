# GLM-4.5-Air (`strata-poly`)

Experimental. `strata-poly` runs Z.ai's GLM-4.5-Air (llama.cpp architecture `glm4moe`, 106B with 12B active) from
Unsloth's GGUF files. **No size has run yet**: the engine code compiles, and nothing here is measured.

```bash
./setup.sh --family glm45air
```

## What the engine does

The layers follow llama.cpp's `glm4moe` graph (`src/models/glm4-moe.cpp`), in `engine/src/poly/glm4moe.cpp`:

- 46 layers, each with attention over every token so far: 96 query heads over 8 key heads of 128 values.
- The q, k and v projections have a bias. GLM-4.5 (not Air) also norms each query and key head; the engine does that
  when the file has `attn_q_norm` and `attn_k_norm`.
- The first 64 values of a head turn by the position, in the NeoX pairing (`head_bias_rope()` in
  `engine/src/poly/kernels.cu`: the bias, the optional norm and the turn in one launch).
- Layer 0 has a dense FFN. Layers 1 to 45 have 128 routed experts, 8 a token, picked by sigmoid gates plus the
  selection bias, their weights made to sum to one, beside one shared expert.
- The block after the 46 layers that guesses a token ahead is not run: Unsloth's files give it a token table and an
  output matrix of its own, which this engine does not support for any model.
- `--kv q8_0` is supported. `rope.scaling` and GLM-4.5V's position sections are refused at load.

It runs on CUDA and HIP builds, not on Metal.

## Sizes

All from [unsloth/GLM-4.5-Air-GGUF](https://huggingface.co/unsloth/GLM-4.5-Air-GGUF), pinned to revision
`506d64aa8c5c`. The routed experts are counted from each file's tensors (layers 0 to 45).

| Size | Experts | Files | Download | Routed experts |
| --- | --- | --- | ---: | ---: |
| `UD-Q4_K_XL` | Q4_K gate and up, Q5_0 and Q5_1 down | 2 | 67.7 GB | 60.9 GB |
| `UD-Q3_K_XL` | Q3_K gate and up, IQ4_NL and Q5_0 down | 2 | 54.8 GB | 48.4 GB |
| `UD-IQ3_XXS` | IQ3_XXS and IQ3_S gate and up, IQ4_NL down | 2 | 51.4 GB | 45.2 GB |
| `UD-Q2_K_XL` | Q2_K gate and up, IQ4_NL and Q5_0 down | 1 | 47.4 GB | 41.4 GB |
| `UD-IQ2_M` | IQ2_XS and IQ2_S gate and up, IQ4_NL down | 1 | 44.4 GB | 38.5 GB |
| `UD-IQ2_XXS` | mostly IQ2_XXS gate and up, IQ4_NL down | 1 | 42.7 GB | 36.8 GB |

Without `--model`, setup takes the first size whose RAM estimate the PC meets.

## Memory (estimates, not measured)

- Weights other than the routed experts: 4.4 to 4.6 GB.
- Attention caches: 188,416 bytes a token in fp16 (46 layers, keys and values of 8 heads of 128), 6.2 GB at 32K;
  about half that with `--kv q8_0`.
- Setup assumes 13 GB of VRAM before any expert at 32K, so a card of 16 GB or more is wanted. The routed experts
  the card does not hold are read from the file through the OS file cache: setup's RAM estimate is the routed
  experts plus 10 GB beside an 8 GB card.

## Not tested

Not run on any card. Not tried on Windows (setup asks before installing there) or on AMD. A first run should
compare a few greedy answers with llama.cpp's on the same file.
