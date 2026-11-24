"""The BigFreakyGoon-9000 entries of setup's catalog (UltimateIntent/BigFreakyGoon9000-GGUF, a roleplay fine-tune of Qwen3.8-Flash-Next):
the four sizes and their pins, one file per size read in place with a RAM budget, the default size by RAM, never the default family, a
--gguf-dir copy recognised by its published name, and the fit label without the n-gram table.  Offline: no download, no GPU.

    python terminal/tools/test_setup_bigfreakygoon.py
"""
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
import strata_mcp as M  # noqa: E402

FAM = "bigfreakygoon"
SIZES = {"BFG-IQ3_XXS": ("BigFreakyGoon-9000-IQ3_XXS.gguf", 88.4, 28.8),
         "BFG-Q4_K_M": ("BigFreakyGoon-9000-Q4_K_M.gguf", 119.2, 35.2),
         "BFG-Q6_K": ("BigFreakyGoon-9000-Q6_K.gguf", 167.6, 54.4),
         "BFG-Q8_0": ("BigFreakyGoon-9000-Q8_0.gguf", 188.2, 54.4)}


class GoonCatalog(unittest.TestCase):
    def test_the_four_sizes_and_their_pins(self):
        fam = setup.FAMILIES[FAM]
        self.assertEqual(list(setup.family_models(FAM)), list(SIZES))
        for model, (name, download, table) in SIZES.items():
            d = setup.MODELS[model]
            self.assertEqual(setup.model_file(fam, model, 1), name)
            self.assertEqual(setup.model_shards(fam, model), 1)
            self.assertTrue(d["budget"])                       # read in place, as Unsloth's sets
            self.assertEqual(d["families"], (FAM,))
            self.assertEqual(d["table_gb"], table)
            url = setup.model_hf(fam, model) + name
            size, sha = setup.file_pin(url)
            self.assertEqual((size, sha), setup.GOON_FILES[name])
            self.assertAlmostEqual(size / 1e9, download, delta=0.05)
            self.assertEqual(len(sha), 64)
            self.assertIn(setup.HF_REVISIONS[setup.GOON_REPO], url)
            self.assertGreaterEqual(setup.MODELS[model]["engine"], setup.UNSLOTH_V31_ENGINE)

    def test_sizes_the_engine_cannot_run_are_not_offered(self):
        # Q2_K / Q3_K gate-up experts have no grouped expert kernel (strata_control/q36_types.py EXPERT_GATE_UP)
        for name in ("BigFreakyGoon-9000-Q2_K.gguf", "BigFreakyGoon-9000-Q3_K_M.gguf", "BigFreakyGoon-9000-BF16.gguf"):
            self.assertIsNone(setup.gguf_choice(name), name)
            self.assertNotIn(name, setup.GOON_FILES)

    def test_picture_reader_is_its_own_pinned_file(self):
        fam = setup.FAMILIES[FAM]
        self.assertTrue(fam["vision"])
        self.assertEqual(fam["mmproj"], "BigFreakyGoon-9000-mmproj-BF16.gguf")
        self.assertEqual(setup.file_pin(fam["mmproj_hf"] + fam["mmproj"]), fam["mmproj_sha256"])

    def test_pinned_revision_and_license(self):
        self.assertEqual(len(setup.HF_REVISIONS[setup.GOON_REPO]), 40)
        self.assertIn("Qwen Community License 1.0", setup.FAMILIES[FAM]["license"])

    def test_never_the_default_family(self):
        fams = setup.menu_families(False)
        self.assertEqual(fams[0], "qwen")
        self.assertGreater(fams.index(FAM), fams.index("gyro"))
        self.assertTrue(setup.FAMILIES[FAM]["experimental"])

    def test_default_size_by_ram(self):
        names = list(SIZES)
        self.assertEqual(setup.recommended_size(FAM, names, 32.0, 12.0), "BFG-IQ3_XXS")     # below the budget floor: the smallest
        self.assertEqual(setup.recommended_size(FAM, names, 64.0, 16.0), "BFG-Q4_K_M")
        self.assertEqual(setup.recommended_size(FAM, names, 256.0, 32.0), "BFG-Q4_K_M")     # recommend, never force: the bigger ones stay choices

    def test_run_config_names_the_size(self):
        for model, (name, _, _) in SIZES.items():
            self.assertEqual(setup.budget_model({"args": ["--native", f"/models/{name}"]}), model)
        # Unsloth's sets are still told apart by their shard names
        self.assertEqual(setup.budget_model({"args": ["--native", "/m/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf"]}), "Q8_0")
        self.assertEqual(setup.budget_model({"args": ["--native", "/m/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"]}), "UD-Q4_K_XL")

    def test_gguf_dir_recognises_the_published_files(self):
        for model, (name, _, _) in SIZES.items():
            self.assertEqual(setup.gguf_choice(name), (FAM, model))
            self.assertIsNone(setup.gguf_unsupported(name))
        self.assertEqual(setup.gguf_unsupported("BigFreakyGoon-9000-Q2_K.gguf"), "Q2_K")   # still refused, with the usual words

    def test_fit_label_leaves_out_the_ngram_table(self):
        self.assertEqual(setup.model_files_gb("BFG-IQ3_XXS"), 59.6)
        self.assertEqual(setup.model_files_gb("BFG-Q4_K_M"), 84.0)
        self.assertEqual(setup.model_fit_label("BFG-IQ3_XXS", 64.0, {"vram_gb": 0.0}), "fits")

    def test_mcp_table_matches(self):
        for m in SIZES:
            for k in ("about", "download_gb", "ram_gb", "arena_gb", "vision", "families", "budget"):
                self.assertEqual(setup.MODELS[m][k], M.FALLBACK_MODELS[m][k], (m, k))
        for k in ("title", "about", "tag", "vision", "experimental"):
            self.assertEqual(setup.FAMILIES[FAM][k], M.FALLBACK_FAMILIES[FAM][k], k)


if __name__ == "__main__":
    unittest.main()
