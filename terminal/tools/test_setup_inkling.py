"""Tests for setup.py's Inkling 2 Mini (docs/models/inkling.md): its three pinned files under the repository's
Mainline_Support folder, where their names end in .gguf_file, and the install with no draft model, no image encoder
and one request at a time.  Mocked like tools/test_setup_glm.py, whose harness this uses.

    python -m unittest tools.test_setup_inkling
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402
from tools import test_setup_glm as glm_tests  # noqa: E402

FAM = setup.FAMILIES["inkling"]
# ISTA-DASLab's GSQ-RCO sizes first (the installer's default), then Unsloth's
GSQ = [("GSQ-RCO-IQ3_S", 11425259648), ("GSQ-RCO-Q2_K_XL", 9792516224), ("GSQ-RCO-IQ3_XXS", 10608205952),
       ("GSQ-RCO-IQ2_S", 8975863424)]
UD = [("UD-Q4_K_XL", 17629760736), ("UD-Q3_K_XL", 13631779296), ("UD-Q2_K_XL", 9138716608)]
SIZES = GSQ + UD
LOW_EFFORT = ("GSQ-RCO-Q2_K_XL", "GSQ-RCO-IQ2_S", "UD-Q2_K_XL")
DSPARK = "dspark-Inkling-2-Mini-Q8_0.gguf"
DSPARK_PIN = (402278560, "2ec78510da837e8fc3be54a14757a0c5fbd2d8d9e85a0d92853fafdf674f3773")
# ISTA-DASLab's DSpark drafter: the same name, llama.cpp's layout of the same drafter (engine/src/poly/ds_layout.hpp)
GSQ_DSPARK_PIN = (404128576, "7cca0010feae71c25c8530b202e05fd9d1eae09f635ae4f5f31b9f20470e8c48")


def draft_of(size):
    """The size's draft model: its folder on the Hub and its pin."""
    d = FAM["models"][size]
    gsq = size.startswith("GSQ-RCO-")
    return (d["hf"] if gsq else FAM["draft_hf"]), (GSQ_DSPARK_PIN if gsq else DSPARK_PIN)


class Pins(unittest.TestCase):
    def test_the_files(self):
        self.assertEqual(list(setup.family_models("inkling")), [m for m, _ in SIZES])
        self.assertRegex(FAM["hf"], r"^https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/[0-9a-f]{40}/"
                                    r"Mainline_Support/$")
        # DSpark sits at the repository root (no Mainline_Support/ folder, no _file suffix), like mmproj-F16.gguf
        self.assertRegex(FAM["draft_hf"], r"^https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/[0-9a-f]{40}/$")
        self.assertEqual(FAM["draft_hf"], FAM["mmproj_hf"])
        self.assertEqual(setup.ms_file(FAM["draft_hf"] + DSPARK), ("unsloth/Inkling-2-Mini-GGUF", DSPARK))
        # the GSQ-RCO sizes: ISTA-DASLab's repository root, no _file suffix, its own drafter (pinned for these
        # sizes: Unsloth's name, other bytes) and the family's image encoder
        for size, _n in GSQ:
            d = FAM["models"][size]
            self.assertRegex(d["hf"], r"^https://huggingface.co/ISTA-DASLab/Inkling-2-Mini-GSQ-RCO-GGUF/resolve/"
                                      r"[0-9a-f]{40}/$")
            self.assertEqual(d["remote_suffix"], "")
            self.assertNotIn("draft_hf", d)
            self.assertEqual(d["sha256"], {DSPARK: GSQ_DSPARK_PIN})
            self.assertNotIn("mmproj_hf", d)
        self.assertEqual(list(FAM["sha256"]), [f"Inkling-2-Mini-{m}.gguf" for m, _ in SIZES] + [DSPARK, FAM["mmproj"]])
        self.assertEqual(FAM["sha256"][DSPARK], DSPARK_PIN)
        for size, n in SIZES:
            file = f"Inkling-2-Mini-{size}.gguf"
            d = FAM["models"][size]
            self.assertEqual((setup.model_file(FAM, size, 1), setup.model_shards(FAM, size)), (file, 1))
            self.assertEqual(d["draft_model"], DSPARK)
            self.assertEqual(FAM["sha256"][file][0], n)
            self.assertRegex(FAM["sha256"][file][1], r"^[0-9a-f]{64}$")
            self.assertAlmostEqual((n + draft_of(size)[1][0]) / 1e9, d["download_gb"], delta=0.05)
            self.assertLess(d["arena_gb"], d["download_gb"])
            self.assertEqual(d["ram_gb"], round(setup.poly_ram_need(d, 8.0)), size)

    def test_the_engine_s_name_for_its_files(self):
        # the repository's top-level files are Unsloth's layout (general.architecture inkling2), which the engine
        # reads too (inkling2.cpp): a --gguf-dir file of either is taken
        self.assertEqual(setup.POLY_GGUF_ARCHS["inkling"], ("inkling2_loop", "inkling2"))


class UncensoredLoraFile(unittest.TestCase):
    """The adapter shipped in data/inkling-uncensored-lora: llama.cpp's LoRA GGUF layout, as the engine's --lora reads
    it (inkling2.cpp lora_open()), and the bytes converted from the pinned upstream adapter."""

    def test_the_shipped_adapter(self):
        from tools.gguf_reader import GGUFFile
        import hashlib
        path = setup.INKLING_LORA
        self.assertEqual(path, ROOT / "data" / "inkling-uncensored-lora" / "Inkling-2-Mini-Uncensored-LoRA.gguf")
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         "84d18625228d4a95e8ca41e2d1e328aa1cc162a681dc94e8f57635dae8277ed3")
        g = GGUFFile(path)
        self.assertEqual((g.metadata["general.type"], g.metadata["adapter.type"], g.metadata["adapter.lora.alpha"],
                          g.metadata["general.architecture"]), ("adapter", "lora", 2.0, "inkling2_loop"))
        want = []
        for il in range(4, 24):
            want += [(f"blk.{il}.attn_output.weight.lora_a", [3072, 2]), (f"blk.{il}.attn_output.weight.lora_b", [2, 2304])]
        self.assertEqual([(t.name, t.shape) for t in g.tensors], want)
        self.assertEqual({t.type_name for t in g.tensors}, {"F32"})


class Main(unittest.TestCase):
    setUp, tearDown, main = glm_tests.Main.setUp, glm_tests.Main.tearDown, glm_tests.Main.main

    def lora_args(self, argv=(), answers=None):
        code, out, _ = self.main(argv, family="inkling", model="UD-Q3_K_XL", ram=64.0, vram=32607 / 1024,
                                 answers=answers)
        self.assertEqual(code, 0, out)
        args = json.loads((self.t / "strata-inkling-ud-q3_k_xl.json").read_text())["args"]
        return (args[args.index("--lora") + 1] if "--lora" in args else None), out

    def test_uncensored_lora_is_off_unless_chosen(self):
        lora, out = self.lora_args()                                   # --yes: the question's default, no
        self.assertIsNone(lora)
        self.assertIn("uncensored LoRA: off", out)
        lora, _ = self.lora_args(answers={})                           # asked, Enter: no
        self.assertIsNone(lora)
        self.assertEqual(len([q for q in self.asked if "uncensored LoRA" in q]), 1)

    def test_uncensored_lora_asked_or_given_and_kept(self):
        lora, out = self.lora_args(answers={"uncensored LoRA": "y"})
        self.assertEqual(Path(lora), setup.INKLING_LORA)
        self.assertIn("uncensored LoRA: ON", out)
        lora, _ = self.lora_args()                                     # a later run with --yes keeps it
        self.assertEqual(Path(lora), setup.INKLING_LORA)
        lora, _ = self.lora_args(("--uncensored-lora", "off"))
        self.assertIsNone(lora)
        lora, _ = self.lora_args()                                     # and off is kept too
        self.assertIsNone(lora)
        lora, _ = self.lora_args(("--uncensored-lora", "on"))
        self.assertEqual(Path(lora), setup.INKLING_LORA)

    def test_uncensored_lora_another_file_or_a_missing_one(self):
        other = self.t / "my-adapter.gguf"
        other.write_bytes(b"GGUF")
        lora, out = self.lora_args(("--uncensored-lora", str(other)))
        self.assertEqual(Path(lora), other.resolve())
        lora, _ = self.lora_args()                                     # kept by a later run
        self.assertEqual(Path(lora), other.resolve())
        code, out, _ = self.main(("--uncensored-lora", str(self.t / "none.gguf")), family="inkling",
                                 model="UD-Q3_K_XL", ram=64.0, vram=32607 / 1024)
        self.assertNotEqual(code, 0)
        self.assertIn("the LoRA adapter is missing", out)

    def test_uncensored_lora_of_another_family_is_not_used(self):
        code, out, cfg = self.main(("--uncensored-lora", "on"))       # GLM, the harness's family
        self.assertEqual(code, 0, out)
        self.assertRegex(out, r"not used with .*--uncensored-lora")
        self.assertNotIn("--lora", cfg["args"])

    def test_install_each_size(self):
        for size, _n in SIZES:
            with self.subTest(size):
                self.downloads, self.runs, self.verified = [], [], []
                code, out, _ = self.main(family="inkling", model=size, ram=64.0, vram=32607 / 1024)
                self.assertEqual(code, 0, out)
                cfg = json.loads((self.t / f"strata-inkling-{size.lower()}.json").read_text())
                file = f"Inkling-2-Mini-{size}.gguf"
                d = FAM["models"][size]
                hub, pin = draft_of(size)
                self.assertEqual(self.downloads, [d.get("hf", FAM["hf"]) + file + d.get("remote_suffix", "_file"),
                                                  hub + DSPARK])
                self.assertEqual(self.verified, [(file, *FAM["sha256"][file]), (DSPARK, *pin)])
                args = cfg["args"]
                self.assertEqual(Path(args[args.index("--model") + 1]), self.t / "models" / f"inkling-{size}" / file)
                self.assertEqual(Path(args[args.index("--draft-model") + 1]),
                                 self.t / "models" / f"inkling-{size}" / DSPARK)
                # a card that holds every expert runs the file as it is (small_card_args: none)
                for flag in ("--tail-skip", "--prompt-batch", "--batch", "--attn-requant",
                             "--loop-bf16", "--expert-swaps"):
                    self.assertNotIn(flag, args)
                self.assertNotIn("parallel", cfg)                                 # opt-in: written when asked
                self.assertNotIn("vision", cfg)
                self.assertEqual(cfg.get("reasoning_effort"), "low" if size in LOW_EFFORT else 0.4)
                self.assertIn("--parallel 2 to 4", out)                           # offered, as for the other families
        # Re-running setup keeps a user-customized reasoning_effort in the existing config
        cfg_path = self.t / "strata-inkling-ud-q3_k_xl.json"
        saved = json.loads(cfg_path.read_text())
        saved["reasoning_effort"] = "high"
        cfg_path.write_text(json.dumps(saved))
        code, out, _ = self.main(family="inkling", model="UD-Q3_K_XL", ram=64.0, vram=32607 / 1024)
        self.assertEqual(code, 0, out)
        self.assertEqual(json.loads(cfg_path.read_text()).get("reasoning_effort"), "high")

    def test_a_card_that_holds_part_of_the_experts_gets_the_small_card_flags(self):
        # an 8 GB card: Q5_K attention, BF16 loop controllers, tail skip 4; expert swaps stay on (the default)
        # and DSpark is left off by default so its 0.4 GB does not crowd out cached experts
        self.downloads = []
        code, out, _ = self.main(family="inkling", model="UD-Q3_K_XL", ram=32.0, vram=8188 / 1024)
        self.assertEqual(code, 0, out)
        args = json.loads((self.t / "strata-inkling-ud-q3_k_xl.json").read_text())["args"]
        self.assertEqual(args[args.index("--attn-requant") + 1], "q5_k")
        self.assertIn("--loop-bf16", args)
        self.assertEqual(args[args.index("--tail-skip") + 1], "4")
        self.assertNotIn("--expert-swaps", args)
        self.assertNotIn("--draft-model", args)
        self.assertEqual(self.downloads, [FAM["hf"] + "Inkling-2-Mini-UD-Q3_K_XL.gguf_file"])
        self.assertEqual(args.count("--tail-skip"), 1)
        self.assertIn("holds part of", out)
        self.assertNotIn("small_card", json.loads((self.t / "strata-inkling-ud-q3_k_xl.json").read_text()))

    def test_gsq_rco_on_a_card_that_holds_part_of_the_experts_keeps_its_attention(self):
        # GSQ-RCO's attention is RCO's mix of IQ4_XS to Q8_0: no --attn-requant, the other two small-card flags
        self.downloads = []
        code, out, _ = self.main(family="inkling", model="GSQ-RCO-IQ3_S", ram=32.0, vram=8188 / 1024)
        self.assertEqual(code, 0, out)
        args = json.loads((self.t / "strata-inkling-gsq-rco-iq3_s.json").read_text())["args"]
        self.assertNotIn("--attn-requant", args)
        self.assertIn("--loop-bf16", args)
        self.assertEqual(args[args.index("--tail-skip") + 1], "4")
        self.assertEqual(args.count("--tail-skip"), 1)
        self.assertNotIn("--draft-model", args)
        self.assertEqual(self.downloads, [FAM["models"]["GSQ-RCO-IQ3_S"]["hf"] + "Inkling-2-Mini-GSQ-RCO-IQ3_S.gguf"])
        self.assertIn("--loop-bf16 --tail-skip 4 (fewer bytes", out)

    def test_unsloth_s_draft_beside_a_gsq_rco_size_is_replaced_by_its_own(self):
        # a GSQ-RCO size set up when it took Unsloth's drafter: the finished file of the other build is downloaded again
        folder = self.t / "models" / "inkling-GSQ-RCO-IQ3_S"
        folder.mkdir(parents=True)
        (folder / DSPARK).write_bytes(b"")
        os.truncate(folder / DSPARK, DSPARK_PIN[0])                         # Unsloth's, by its size
        setup.mark(folder / DSPARK)
        self.downloads, self.verified = [], []
        code, out, cfg = self.main(family="inkling", model="GSQ-RCO-IQ3_S", ram=64.0, vram=32607 / 1024)
        self.assertEqual(code, 0, out)
        self.assertIn("another build of the draft model", out)
        self.assertIn(FAM["models"]["GSQ-RCO-IQ3_S"]["hf"] + DSPARK, self.downloads)
        self.assertIn((DSPARK, *GSQ_DSPARK_PIN), self.verified)
        self.assertIn("--draft-model", cfg["args"])
        # the size's own file (its pinned size; the mocked download writes none) is kept on the next run
        os.truncate(folder / DSPARK, GSQ_DSPARK_PIN[0])
        self.downloads = []
        code, out, cfg = self.main(family="inkling", model="GSQ-RCO-IQ3_S", ram=64.0, vram=32607 / 1024)
        self.assertEqual(code, 0, out)
        self.assertNotIn("another build", out)

    def test_the_installer_takes_gsq_rco(self):
        sizes = setup.family_models("inkling")
        for ram, vram in ((16.0, 8.0), (32.0, 8.0), (64.0, 32.0), (128.0, 24.0)):
            self.assertEqual(setup.poly_default(sizes, ram, vram), "GSQ-RCO-IQ3_S", (ram, vram))
        self.assertEqual(setup.poly_default(sizes, 8.0, 6.0), "GSQ-RCO-IQ2_S")   # the least RAM
        code, out, cfg = self.main(family="inkling", ram=32.0, vram=8188 / 1024)
        self.assertEqual(code, 0, out)
        self.assertTrue((self.t / "strata-inkling-gsq-rco-iq3_s.json").is_file(), out)

    CFG = "strata-inkling-ud-q3_k_xl.json"
    SMALL = ("--attn-requant", "--loop-bf16", "--tail-skip")

    def small(self, argv=(), answers=None):
        """UD-Q3_K_XL on an 8 GB card, which holds part of its experts: the run config's args and "small_card"."""
        code, out, _ = self.main(argv, family="inkling", model="UD-Q3_K_XL", ram=32.0, vram=8188 / 1024,
                                 answers=answers)
        self.assertEqual(code, 0, out)
        cfg = json.loads((self.t / self.CFG).read_text())
        return cfg["args"], cfg.get("small_card"), out

    def assert_on(self, args):
        self.assertEqual(args[args.index("--attn-requant") + 1], "q5_k")
        self.assertIn("--loop-bf16", args)
        self.assertEqual(args[args.index("--tail-skip") + 1], "4")
        self.assertEqual(args.count("--tail-skip"), 1)

    def test_small_card_default_is_on_and_not_asked(self):
        # the default stays the small-card settings; no question is added (Enter for every other question)
        args, small, _ = self.small(answers={})
        self.assertEqual([q for q in self.asked if "small" in q.lower()], [])
        self.assert_on(args)
        self.assertIsNone(small)

    def test_small_card_off_is_the_high_accuracy_mode_and_kept(self):
        args, small, out = self.small(("--inkling-small-card", "off"))
        for flag in self.SMALL:
            self.assertNotIn(flag, args)
        self.assertEqual(small, "off")
        self.assertIn("high-accuracy mode", out)
        # a later setup run without the option keeps it, with --yes and with the questions asked
        for answers in (None, {}):
            args, small, _ = self.small(answers=answers)
            for flag in self.SMALL:
                self.assertNotIn(flag, args)
            self.assertEqual(small, "off")
        # --inkling-small-card on puts them back, and the config is the default one again
        args, small, _ = self.small(("--inkling-small-card", "on"))
        self.assert_on(args)
        self.assertIsNone(small)
        args, small, _ = self.small()
        self.assert_on(args)
        self.assertIsNone(small)

    def test_small_card_off_on_a_card_that_holds_every_expert_is_kept(self):
        # nothing to leave out there, but the choice is written, so the install moved back to an 8 GB card stays off
        code, out, _ = self.main(("--inkling-small-card", "off"), family="inkling", model="UD-Q3_K_XL", ram=64.0,
                                 vram=32607 / 1024)
        self.assertEqual(code, 0, out)
        cfg = json.loads((self.t / self.CFG).read_text())
        for flag in self.SMALL:
            self.assertNotIn(flag, cfg["args"])
        self.assertEqual(cfg["small_card"], "off")
        code, out, _ = self.main(family="inkling", model="UD-Q3_K_XL", ram=64.0, vram=32607 / 1024)   # again, 32 GB
        self.assertEqual(code, 0, out)
        self.assertEqual(json.loads((self.t / self.CFG).read_text())["small_card"], "off")
        args, small, _ = self.small()                                              # then an 8 GB card
        for flag in self.SMALL:
            self.assertNotIn(flag, args)
        self.assertEqual(small, "off")

    def test_small_card_off_with_two_cards(self):
        # Inkling 2 runs on one of them (an 8 GB card here), so off applies there as on one card
        code, out, cfg = self.main(("--gpus", "0,1", "--inkling-small-card", "off"), family="inkling",
                                   model="UD-Q3_K_XL", ram=32.0, vram=8188 / 1024, n_gpus=2)
        self.assertEqual(code, 0, out)
        for flag in self.SMALL:
            self.assertNotIn(flag, cfg["args"])
        self.assertEqual(cfg["small_card"], "off")

    def test_small_card_option_of_another_family_is_not_used(self):
        code, out, _ = self.main(("--inkling-small-card", "off"))                 # GLM, the harness's family
        self.assertEqual(code, 0, out)
        self.assertRegex(out, r"not used with .*--inkling-small-card")

    def test_two_cards_set_up_one_and_size_it_for_that_card(self):
        # PR 208: the engine has no layer split for Inkling 2 yet; two 8 GB cards chosen together get the config one
        # 8 GB card gets (the small-card flags), not a pair the engine refuses
        code, out, cfg = self.main(("--gpus", "0,1"), family="inkling", model="UD-Q3_K_XL", ram=32.0, vram=8188 / 1024,
                                   n_gpus=2)
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["gpu"], 0)
        self.assertNotIn("layer_split", cfg)
        self.assertIn("runs on one card so far", out)
        args = cfg["args"]
        self.assertEqual(args[args.index("--attn-requant") + 1], "q5_k")
        self.assertEqual(args[args.index("--tail-skip") + 1], "4")
        self.assertIn("holds part of", out)
        code, out, cfg = self.main(family="inkling", model="UD-Q3_K_XL", ram=64.0, vram=16303 / 1024, n_gpus=2)
        self.assertEqual(code, 0, out)
        self.assertIsInstance(cfg["gpu"], int)
        self.assertNotIn("layer_split", cfg)

    def test_requests_at_once_are_taken(self):
        # the engine's batch slots ("parallel"), opt-in as for the other strata-poly families; DSpark is skipped automatically
        code, out, _ = self.main(("--parallel", "4"), family="inkling", model="UD-Q3_K_XL", ram=64.0, vram=32607 / 1024)
        self.assertEqual(code, 0, out)
        cfg = json.loads((self.t / "strata-inkling-ud-q3_k_xl.json").read_text())
        self.assertEqual(cfg["parallel"], 4)
        self.assertNotIn("--draft-model", cfg["args"])
        self.assertNotIn("answers one request at a time", out)
        self.assertIn("parallel requests: 4 at once", out)
        self.assertNotIn("with --tail-skip", out)                          # a card that holds every expert: no skip

    def test_requests_at_once_with_the_tail_skip_are_warned_of(self):
        # an 8 GB card's small_card_args skip tail experts over a step's rows, which are several chats' tokens
        code, out, _ = self.main(("--parallel", "2"), family="inkling", model="UD-Q3_K_XL", ram=32.0, vram=8188 / 1024)
        self.assertEqual(code, 0, out)
        cfg = json.loads((self.t / "strata-inkling-ud-q3_k_xl.json").read_text())
        self.assertEqual(cfg["parallel"], 2)
        self.assertNotIn("--draft-model", cfg["args"])
        self.assertIn('"parallel": 2 with --tail-skip', out)
        # setup again without --parallel: the config keeps "parallel": 2, DSpark stays skipped, and the warning stays
        code, out, _ = self.main(family="inkling", model="UD-Q3_K_XL", ram=32.0, vram=8188 / 1024)
        self.assertEqual(code, 0, out)
        cfg = json.loads((self.t / "strata-inkling-ud-q3_k_xl.json").read_text())
        self.assertEqual(cfg["parallel"], 2)
        self.assertNotIn("--draft-model", cfg["args"])
        self.assertIn('"parallel": 2 with --tail-skip', out)

    def test_draft_opt_out_gguf_dir_amd_and_custom_gguf(self):
        from unittest import mock
        # --no-draft turns DSpark off on a card that holds every expert and remembers "mtp": false until --mtp on
        self.downloads, self.verified = [], []
        code, out, cfg = self.main(("--no-draft",), family="inkling", model="UD-Q3_K_XL", ram=64.0, vram=32607 / 1024)
        self.assertEqual(code, 0, out)
        self.assertEqual(self.downloads, [FAM["hf"] + "Inkling-2-Mini-UD-Q3_K_XL.gguf_file"])
        self.assertNotIn("--draft-model", cfg["args"])
        self.assertIs(cfg.get("mtp"), False)
        code, out, cfg = self.main(family="inkling", model="UD-Q3_K_XL", ram=64.0, vram=32607 / 1024)
        self.assertEqual(code, 0, out)
        self.assertNotIn("--draft-model", cfg["args"])
        self.assertIs(cfg.get("mtp"), False)
        self.downloads = []
        code, out, cfg = self.main(("--mtp", "on"), family="inkling", model="UD-Q3_K_XL", ram=64.0, vram=32607 / 1024)
        self.assertEqual(code, 0, out)
        self.assertEqual(self.downloads, [FAM["draft_hf"] + DSPARK])
        self.assertIn("--draft-model", cfg["args"])
        self.assertNotIn("mtp", cfg)

        # An 8 GB card holding part of the experts leaves DSpark off by default, and --mtp on opts in explicitly
        self.downloads = []
        code, out, cfg = self.main(family="inkling", model="UD-Q3_K_XL", ram=32.0, vram=8188 / 1024)
        self.assertEqual(code, 0, out)
        self.assertNotIn("--draft-model", cfg["args"])
        code, out, cfg = self.main(("--mtp", "on"), family="inkling", model="UD-Q3_K_XL", ram=32.0, vram=8188 / 1024)
        self.assertEqual(code, 0, out)
        self.assertIn("--draft-model", cfg["args"])

        # --gguf-dir with both files already present uses both directly and verifies both SHA-256s
        mine = self.t / "own"
        mine.mkdir()
        (mine / "Inkling-2-Mini-UD-Q3_K_XL.gguf").write_bytes(b"")
        (mine / DSPARK).write_bytes(b"")
        self.downloads, self.verified = [], []
        code, out, cfg = self.main(("--gguf-dir", str(mine)), family="inkling", model="UD-Q3_K_XL", ram=64.0,
                                   vram=32607 / 1024)
        self.assertEqual((code, self.downloads), (0, []), out)
        self.assertEqual([v[0] for v in self.verified], ["Inkling-2-Mini-UD-Q3_K_XL.gguf", DSPARK])
        self.assertEqual(Path(cfg["args"][cfg["args"].index("--draft-model") + 1]), mine / DSPARK)

        # AMD HIP skips DSpark automatically
        amd = [{"index": 0, "name": "AMD Radeon RX 9070 XT", "vram_gb": 16.0, "arch": "gfx1201", "driver": "hip",
                "count": 1}]
        def fake_hip(gpu, llama, portable=False):
            eng = self.t / setup.ENGINE_POLY_DIR
            eng.mkdir(exist_ok=True)
            (eng / "BUILD.json").write_text(json.dumps({"backend": "hip", "archs": ["gfx1201"]}))
            return eng
        real_main = setup.main
        def amd_main():
            with mock.patch.object(setup, "gpus", lambda: []), \
                    mock.patch.object(setup, "amd_gpus", lambda *a: amd), \
                    mock.patch.object(setup, "build_engine_poly_hip", fake_hip):
                return real_main()
        with mock.patch.object(setup, "main", amd_main):
            code, out, cfg = self.main(("--backend", "hip", "--gguf-dir", str(mine)), family="inkling",
                                       model="UD-Q3_K_XL", ram=32.0, vram=16.0)
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["backend"], "hip")
        self.assertNotIn("--draft-model", cfg["args"])



if __name__ == "__main__":
    unittest.main()
