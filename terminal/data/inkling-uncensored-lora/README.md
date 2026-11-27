# inkling-uncensored-lora (EXPERIMENTAL, off by default)

`Inkling-2-Mini-Uncensored-LoRA.gguf` is an 843 KB LoRA adapter for Inkling 2 Mini. It holds 20 rank-2 pairs, one on
the attention output product (`attn_output`) of each of layers 4 to 23. It is in llama.cpp's LoRA GGUF layout:
`blk.N.attn_output.weight.lora_a` [3072, 2] and `.lora_b` [2, 2304], F32, `adapter.lora.alpha` 2 (scale 1).

**What it does.** When it is on, PolyStrata's engine (`strata-poly --lora`) adds `B (A x)` to the attention's output
in those layers while the model runs. The model file is not changed. The loop's layers (6 to 17) run three times a
token, and the adapter acts on every pass, as merged weights would. The adapter's makers describe it as an
**abliteration**: a 2-dimensional refusal subspace projected out of those matrices, `W' = W - s Q Qᵀ W`. They report
0 refusals on their 32 harmful prompts (32 without it), and MMLU 79.47% against 78.95% on their merged weights.
Removing refusals removes a safety behaviour: you are responsible for what the model writes with it on.

**Measured here.** Nothing on a GPU yet: no run of the model with it, no speed or answer numbers. What was checked
(on a PC with no GPU):

- the adapter matches the GGUF the installer downloads. For layers 4 and 10, `A` equals `Qᵀ W` with `W` the file's
  own `blk.N.attn_output.weight` (Unsloth's UD-Q2_K_XL, Q8_0 there) to 0.6% (the Q8_0 rounding). With layer 4's `W`
  and layer 10's adapter it is 136% off, so the layers and the matrix's orientation are the file's.
- `W x + B (A x)` with this file, read by llama.cpp's own `gguf` Python package, is the published edit `W x - s Q Qᵀ W x`
  to 0.08%.

**Turning it on.** `START-HERE.bat --setup` (or `./setup.sh --setup`) asks when the model is Inkling 2 Mini (default:
no). You can also pass `--uncensored-lora on` (`off`, or the path of another LoRA GGUF). It writes `--lora <this file>`
into `strata-inkling-<size>.json`, and the engine log then says
`LoRA ...Inkling-2-Mini-Uncensored-LoRA.gguf on 20 layers' attention output, rank 2, scale 1`. A later setup run keeps
it until you answer no or pass `--uncensored-lora off`. Unlike the Qwen speed projection, it cannot be switched off per
request: it is on for every request while it is loaded. Conversations saved to disk (`--pin-dir`,
`--pinned-session`) without it should be made again with it, and the other way round. Run by hand, the engine takes
`--lora FILE` or `--lora-scaled FILE S` (S = 1 is the adapter as trained). The Mac installer (`setup_mac.py`) does not
ask yet: add `--lora` to the run config's `"user_args"` there.

**Origin and license.** Made from OpenUncensored's
[Inkling-2-Mini-Uncensored-LoRA](https://huggingface.co/OpenUncensored/Inkling-2-Mini-Uncensored-LoRA) at commit
`b328f8112173024d0f097aba1ea09e5f475ca7f9` (`adapter_model.safetensors`, sha256
`c67b9d8d18c4a916440eafaf57a42f1510bc00562f52c8a0a29f8f417ed41443`). Converted with

```sh
python terminal/tools/peft_lora_to_gguf.py <adapter folder> Inkling-2-Mini-Uncensored-LoRA.gguf \
    --name Inkling-2-Mini-Uncensored-LoRA --source https://huggingface.co/OpenUncensored/Inkling-2-Mini-Uncensored-LoRA
```

The matrices are the same bytes. The adapter is Apache 2.0, the license of Inkling 2 Mini, and the model's credit goes
to Thinking Machines.
