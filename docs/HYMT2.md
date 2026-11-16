# Hy-MT2-30B-A3B

Back to [BACKENDS.md](BACKENDS.md)

[Hy-MT2-30B-A3B](https://huggingface.co/tencent/Hy-MT2-30B-A3B) is Tencent's translation model: 33 languages, 30B
parameters of which a token uses about 3B, Apache 2.0. llama.cpp calls its architecture `hy_v3`. PolyStrata reads
Tencent's own GGUFs ([tencent/Hy-MT2-30B-A3B-GGUF](https://huggingface.co/tencent/Hy-MT2-30B-A3B-GGUF)) on the
`strata-poly` engine.

> **Status: not run yet.** The engine's code for this model (`src/poly/hy_v3.cpp`) was written from llama.cpp's
> graph (`src/models/hy-v3.cpp`) and transformers' `modeling_hy_v3.py` on a machine without a graphics card. It has
> not been compiled with CUDA or HIP and has not loaded the model. Nothing on this page is a measurement of the
> model running. The page says what was checked without a card, what is computed, and how to try it.

## What the model is

| | |
| --- | --- |
| Layers | 48, each with attention over every token so far |
| Width | 2,048 |
| Attention | 32 query heads over 4 key heads of 128 values; queries and keys normed a head, then turned by the position (the NeoX pairing, base 11,158,840) |
| FFN | layer 0 dense (6,912 wide); layers 1 to 47 have 128 routed experts of 768, 8 a token, beside one ungated shared expert of 768 |
| Routing | sigmoid gates; the 8 largest of gate + selection bias; their weights are the bare gates, made to sum to one, times 2.826 |
| Output | a matrix of its own, not the token table |
| Vocabulary | 120,832 tokens, llama.cpp's `hunyuan-dense` pre-tokenizer |
| Context | 262,144 tokens by the file |
| Draft block | none in these files: it writes one token at a time, and repeats are guessed as for the other models |

It is closest to LFM2.5's attention layers (normed queries and keys with a rotary embedding over the whole
sequence) with Flint's FFN (a dense first layer, then sigmoid-gated experts beside a shared one), and the class is
put together from those two.

## Install

```bash
./setup.sh --family hymt2
```

On Windows:

```
START-HERE.bat --setup --family hymt2
```

| Size | File | Download | Routed experts | RAM beside an 8 GB card | RAM beside a 12 GB card | RAM beside a 24 GB card |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `Q4_K_M` | Hy-MT2-30B-A3B-Q4_K_M.gguf | 18.2 GB | 17.2 GB | about 25 GB | about 21 GB | about 10 GB |
| `Q8_0` | Hy-MT2-30B-A3B-Q8_0.gguf | 32.0 GB | 30.2 GB | about 39 GB | about 35 GB | about 22 GB |

The download sizes and sha256 are the files' own. The experts' sizes are summed from each file's tensor list. The
RAM columns are setup's estimate (the experts the card does not hold, plus 10 GB), with a figure for the VRAM taken
before any expert that is **computed, not measured**: 6.2 GB for Q4_K_M and 6.8 GB for Q8_0 at a 32K context. That
figure is the weights outside the experts (0.9 and 1.4 GiB), the attention rows of 32,768 tokens (3.0 GiB) and the
2.3 GiB of buffers and reserve the other models were measured with.

**A context token takes 98,304 bytes of VRAM**: every one of the 48 layers keeps the keys and values of its whole
past (2 x 4 heads x 128 values x 2 bytes). That is three to five times the other models' rows. 8K of context is
0.8 GB, 32K is 3.2 GB, 64K is 6.4 GB, 131K is 12.9 GB. For translation a short context is usually enough;
`--context 8192` leaves the most VRAM for experts. `--kv q8_0` is accepted for this architecture and stores a row
in 52,224 bytes instead; it is as untested here as the rest.

By setup's figure an 8 GB card has less than the model takes to start at 32K, and setup stops there:
`--family hymt2 --model Q4_K_M --context 8192 --yes` installs it anyway.

## What was checked without a card

- **The tokenizer.** `tools/strata_tokenizer.py` on the GGUF's vocabulary and merges gives the same ids as the
  model's own `tokenizer.json` (the `tokenizers` library, 0.23.3) on 433 of 433 test strings: English, Chinese,
  Japanese, Korean, Arabic, Thai, Hindi, Tibetan, Uyghur, emoji, code, numbers, and 400 random mixes. The vocabulary
  has no token for a carriage return and 13 other single bytes; the model's tokenizer leaves such a byte out before
  it merges, and so does this one. llama.cpp leaves it out after the merges, so `"\r\n\r\n"` is one token here and
  two there.
- **The chat template.** The GGUF's template is byte for byte the one in `tencent/Hy-MT2-30B-A3B`. The server renders
  it and the rendered prompt tokenizes to the same ids as with `tokenizer.json`. `serve/test_hy.py` covers the
  efforts, the tool calls and the tokenizer.
- **The file's shape against the code.** Every tensor name, shape and type the class reads was compared with the
  tensor lists of both GGUFs (766 tensors each). One name differs from the other architectures': the selection bias
  is `blk.N.exp_probs_b`, without `.bias`.
- **Setup.** `tools/test_setup_hymt2.py`: the pinned files, the estimates, the install steps, with no GPU and no
  download.
- **Syntax.** `src/poly/hy_v3.cpp` passes `g++ -fsyntax-only` against the Metal build's stand-in CUDA headers. That
  is not a build: no backend was compiled.

Not checked: that it compiles with CUDA, HIP or Metal, that it loads, that its answers are right, its speed, its
memory use, several cards, AMD cards, Windows.

## Trying it

1. Install as above. Setup compiles the engine where there is no ready-made one for this version; a compile error
   in `src/poly/hy_v3.cpp` is the first thing that can go wrong, and its message is what to report.
2. A first prompt straight to the engine, without the server. The ids below are the template's prompt for
   "Translate the following text into French. Note that you should only output the translated result without any
   additional explanation:\n\nThe weather is really nice today.", with no thinking:

   ```bash
   engine-poly/strata-poly --model models/hymt2-Q4_K_M/Hy-MT2-30B-A3B-Q4_K_M.gguf --max-context 8192 --max-new 32 --tokens 120000,120044,70830,277,9002,647,497,25,5130,25152,1326,120006,26403,252,1873,2602,1079,7157,13,10306,369,444,1029,1370,3602,252,17960,1398,2553,1232,5682,12509,1015,628,9969,316,4001,13574,5247,13,120007,120029,120030
   ```

   It prints the ids it wrote. To read them:

   ```bash
   .venv/bin/python -c "import sys; sys.path.insert(0, 'tools'); from strata_tokenizer import Tokenizer; print(Tokenizer.from_gguf(sys.argv[1]).decode([int(x) for x in sys.argv[2].split(',')]))" models/hymt2-Q4_K_M/Hy-MT2-30B-A3B-Q4_K_M.gguf ID,ID,ID
   ```

   A right answer is a short French sentence that ends at token 120025 (`<eos:6124c78e>`). Text that is not French,
   one token repeated, or a reply that never ends means the math is wrong somewhere.
3. Against llama.cpp. A llama.cpp build that knows `hy_v3` gives the reference: the same prompt at temperature 0
   should write the same tokens for the first few dozen steps (the other models here agree with llama.cpp at 120 to
   128 of 128 steps). `--dump-layers` prints a checksum of every layer's stages, which shows the first layer where
   the two part.
4. Through the server: `./setup.sh` (Windows: `START-HERE.bat`) starts it; then a chat request. No `reasoning_effort` is
   the template's `no_think`, which is the model's default.

The half-sum guard (`half_sum_guard_`) is on for this model because nobody has measured whether its experts'
products stay in a half's range; it cost LFM2.5 4% of a long prompt's read. Whether it can be off is one of the
things a run would show.

## How it talks

The server takes this from the tokenizer's `pre` name, `hunyuan-dense` (`serve/server.py`, call format `hy`).

- **Turns.** `<｜hy_User｜>` and `<｜hy_Assistant｜>`; a system prompt is plain text right after the first token. A
  reply ends at `<eos:6124c78e>`.
- **Thinking.** The template knows three efforts: `no_think`, `low` and `high`. A request without
  `reasoning_effort`, or with thinking off, is `no_think`: the prompt ends on `<think></think>` and the reply is the
  answer from its first token. `low` and `medium` are sent as `low`, `high` and `xhigh` as `high`; the prompt then
  ends on an open `<think>` and the reply's text up to `</think>` is its reasoning. The model card calls it a
  fast-thinking model and its examples use no effort.
- **Tool calls.** The template writes
  `<tool_calls>\n<tool_call>NAME<tool_sep>\n<arg_key>K</arg_key>\n<arg_value>V</arg_value>\n</tool_call>\n</tool_calls>`,
  and the server reads that form. It is a translation model: whether it calls tools well is not known here.
- **Prompts.** The model card's translation instructions work as the user's message, for example
  `Translate the following text into {target_lang}. Note that you should only output the translated result without
  any additional explanation:\n\n{source_text}`. The card lists more (terminology, style, structured data).
- **Sampling.** The card recommends temperature 0.7, top_p 1.0, no top_k and no repetition penalty for this size.
  The server's `top_k` is 1 to 64, so "no top_k" cannot be asked for exactly; `"sampling": {"temperature": 0.7,
  "top_p": 1.0, "top_k": 64}` in the run config is the nearest. Without a `sampling` block and without a
  temperature in the request it answers with the most likely token each step.

## Reporting

A run on any card is welcome, also one that fails: the compile error, the load error, or the answer it gave. See
[COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md) for the report's form, and say what you did not test.
