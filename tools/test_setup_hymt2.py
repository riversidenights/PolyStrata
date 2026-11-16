"""Tests for setup.py's Hy-MT2-30B-A3B (docs/HYMT2.md): its two pinned files, the estimates for an 8 GB card, the
note that it has not run on a card yet, and the install by GLM's steps with no draft model and no image encoder.
Mocked like tools/test_setup_glm.py, whose harness this uses.

    python -m unittest tools.test_setup_hymt2
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402
from tools import test_setup_glm as glm_tests  # noqa: E402

FAM = setup.FAMILIES["hymt2"]
# setup's order: the size taken with the RAM for it, then the other
SIZES = [("Q4_K_M", 18236702880), ("Q8_0", 31985729376)]


class Pins(unittest.TestCase):
    def test_the_files(self):
        self.assertEqual(list(setup.family_models("hymt2")), [m for m, _ in SIZES])
        self.assertRegex(FAM["hf"], r"^https://huggingface.co/tencent/Hy-MT2-30B-A3B-GGUF/resolve/[0-9a-f]{40}/$")
        self.assertEqual(list(FAM["sha256"]), [f"Hy-MT2-30B-A3B-{m}.gguf" for m, _ in SIZES])
        for size, n in SIZES:
            file = f"Hy-MT2-30B-A3B-{size}.gguf"
            d = FAM["models"][size]
            self.assertEqual((setup.model_file(FAM, size, 1), setup.model_shards(FAM, size)), (file, 1))
            self.assertEqual(FAM["sha256"][file][0], n)
            self.assertRegex(FAM["sha256"][file][1], r"^[0-9a-f]{64}$")
            self.assertAlmostEqual(n / 1e9, d["download_gb"], delta=0.05)
            self.assertLess(d["arena_gb"], d["download_gb"])
            self.assertEqual(d["ram_gb"], round(setup.poly_ram_need(d, 8.0)), size)
        # the engine's prompt batch, not a figure of this family's
        self.assertFalse(setup.poly_batch(FAM, FAM["models"]["Q4_K_M"], 31.8, 32768))

    def test_the_architecture_and_the_rows(self):
        self.assertEqual(setup.POLY_GGUF_ARCHS["hymt2"], ("hy_v3",))
        # 48 layers, each the keys and the values of 4 heads of 128 fp16 values
        self.assertEqual(FAM["row_bytes"], 48 * 2 * 4 * 128 * 2)

    def test_the_about_line_says_the_first_size_and_that_it_is_new(self):
        n = setup.poly_ram_need(FAM["models"]["Q4_K_M"], 8.0)
        self.assertIn(f"~{n:.0f} GB of RAM beside an 8 GB card", FAM["about"])
        self.assertIn("has not run on a card yet", FAM["about"])
        self.assertIs(FAM.get("untried"), True)
        # nothing says it ran anywhere
        self.assertNotIn("windows", FAM)
        self.assertNotIn("ran_on_gb", FAM)


class Main(unittest.TestCase):
    setUp, tearDown, main = glm_tests.Main.setUp, glm_tests.Main.tearDown, glm_tests.Main.main

    def test_install_each_size(self):
        for size, _n in SIZES:
            with self.subTest(size):
                self.downloads, self.runs, self.verified = [], [], []
                code, out, _ = self.main(family="hymt2", model=size, ram=64.0, vram=12.0)
                self.assertEqual(code, 0, out)
                cfg = json.loads((self.t / f"strata-hymt2-{size.lower()}.json").read_text())
                file = f"Hy-MT2-30B-A3B-{size}.gguf"
                self.assertEqual(self.downloads, [FAM["hf"] + file])
                self.assertEqual(self.verified, [(file, *FAM["sha256"][file])])
                tag = "hymt2-" + size
                pack = self.t / "data" / "packs" / tag.lower()
                self.assertEqual([Path(r[1]).name for r in self.runs], ["strata_tokenizer.py"])
                args = cfg["args"]
                self.assertEqual(Path(args[args.index("--model") + 1]), self.t / "models" / tag / file)
                self.assertEqual(Path(args[args.index("--expert-profile") + 1]), pack / "expert-profile.txt")
                for flag in ("--draft-model", "--tail-skip", "--prompt-batch"):
                    self.assertNotIn(flag, args)
                self.assertEqual(cfg["model_name"], f"hy-mt2-30b-a3b-{size.lower()}")
                self.assertEqual(cfg["display_name"], f"Hy-MT2-30B-A3B {size}")
                self.assertNotIn("vision", cfg)
                self.assertNotIn("tool_call_recovery", cfg)
                self.assertIn("has not run on a card yet", out)

    def test_the_size_taken_without_a_name(self):
        for ram in (64.0, 24.0):
            with self.subTest(ram=ram):
                code, out, _ = self.main(family="hymt2", ram=ram, vram=12.0)
                self.assertEqual(code, 0, out)
                self.assertTrue((self.t / "strata-hymt2-q4_k_m.json").is_file(), out)

    def test_an_8_gb_card_is_told_what_the_32k_context_takes(self):
        # 6.2 GB before any expert, by setup's figure for a 32K context, and 2 GB of room: more than the card has
        code, out, _ = self.main(family="hymt2", ram=64.0, vram=8188 / 1024)
        self.assertNotEqual(code, 0, out)
        self.assertIn("takes about 6.2 GB of VRAM before any expert (computed from its file for a 32K context)", out)
        self.assertIn("--family hymt2 --model Q4_K_M --yes installs it anyway", out)

    def test_windows_gets_the_same_note_and_no_linux_only_question(self):
        code, out, _ = self.main(win=True, family="hymt2", ram=64.0, vram=31.84)
        self.assertEqual(code, 0, out)
        self.assertIn("has not run on a card yet", out)
        self.assertNotIn("has been run on Linux only", out)
        self.assertNotIn("has not been tried on Windows", out)

    def test_vision_is_not_used(self):
        code, out, cfg = self.main(["--vision", "yes"], family="hymt2", ram=64.0, vram=12.0)
        self.assertEqual(code, 0, out)
        self.assertIn("not used with Hy-MT2-30B-A3B", out)
        self.assertNotIn("vision", cfg)


if __name__ == "__main__":
    unittest.main()
