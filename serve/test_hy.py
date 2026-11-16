"""Hy-MT2-30B-A3B (llama.cpp's "hy_v3") behind the server: its pre-tokenizer, its tool calls, its turn openers and
how its template's reasoning efforts are told apart.

    python -m unittest serve.test_hy
"""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import strata_tokenizer as ST  # noqa: E402
from serve.frontend import ChatTemplate, OutputParser  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service  # noqa: E402


class PreTokenizer(unittest.TestCase):
    # the newline is GPT-2's "Ċ"; the vocabulary has no token for the carriage return ("č"), as the model's has none
    TOKENS = ["1", "2", "3", "4", "5", "12", "123", "45", "a", "b", "ab", " ", "你", "好", "你好", "Ċ", "ĊĊ"]
    MERGES = ["1 2", "12 3", "4 5", "a b", "你 好", "Ċ Ċ"]

    def tok(self):
        return ST.Tokenizer(self.TOKENS, self.MERGES, pre="hunyuan-dense")

    def test_digits_then_ideographs_then_the_rest(self):
        self.assertEqual(self.tok()._pieces("12345ab"), ["123", "45", "ab"])
        self.assertEqual(self.tok()._pieces("ab你好ab"), ["ab", "你好", "ab"])

    def test_a_byte_the_vocabulary_lacks_is_left_out_before_the_merges(self):
        tok = self.tok()
        self.assertEqual(tok.encode("\r\n\r\n"), [self.TOKENS.index("ĊĊ")])
        # the pre-tokenizer has cut the text there already, so the letters around it stay two tokens
        self.assertEqual(tok.encode("a\rb"), [self.TOKENS.index("a"), self.TOKENS.index("b")])
        self.assertEqual(tok.encode("\r"), [])

    def test_the_other_tokenizers_still_refuse_such_a_byte(self):
        with self.assertRaises(KeyError):
            ST.Tokenizer(self.TOKENS, self.MERGES, pre="joyai-llm").encode("a\rb")


class ToolCalls(unittest.TestCase):
    SCHEMA = [{"name": "write", "parameters": {"properties": {"path": {"type": "string"}, "n": {"type": "integer"}}}},
              {"name": "now", "parameters": {"properties": {}}}]
    CALLS = ("<tool_calls>\n<tool_call>write<tool_sep>\n<arg_key>path</arg_key>\n<arg_value>doc.md</arg_value>\n"
             "<arg_key>n</arg_key>\n<arg_value>3</arg_value>\n</tool_call>\n<tool_call>now<tool_sep>\n</tool_call>\n"
             "</tool_calls>")

    def events(self, text, thinking, step):
        p = OutputParser(thinking=thinking, tools=self.SCHEMA, stream_tools=True, call_format="hy")
        evs = []
        for i in range(0, len(text), step):
            evs += p.feed(text[i:i + step])
        return evs + p.finish()

    def test_calls_are_parsed_however_they_arrive(self):
        for step in (1, 7, 10_000):
            with self.subTest(step=step):
                evs = self.events("I will write it.\n" + self.CALLS, False, step)
                calls = [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]
                self.assertEqual(calls, [("write", {"path": "doc.md", "n": 3}), ("now", {})])
                self.assertEqual("".join(e.text for e in evs if e.kind == "content"), "I will write it.")
                self.assertFalse([e for e in evs if e.kind == "reasoning"])

    def test_an_answer_without_thinking_is_content_from_its_first_token(self):
        evs = self.events("Bonjour, le monde.", False, 3)
        self.assertEqual("".join(e.text for e in evs if e.kind == "content"), "Bonjour, le monde.")

    def test_an_opened_reasoning_ends_at_its_tag(self):
        evs = self.events("short thought</think>Bonjour.", True, 4)
        self.assertEqual("".join(e.text for e in evs if e.kind == "reasoning"), "short thought")
        self.assertEqual("".join(e.text for e in evs if e.kind == "content"), "Bonjour.")


# what of Hy-MT2's chat template the server has to meet: the efforts high, low and no_think (any other is no_think),
# an open <think> only for the first two, and no enable_thinking
TEMPLATE = """{%- if not reasoning_effort is defined or reasoning_effort not in ['high', 'low', 'no_think'] -%}
{%- set reasoning_effort = 'no_think' -%}{%- endif -%}
{{- bos_token -}}<｜reasoning_mode｜>reasoning_effort:{{ reasoning_effort }}
{%- for m in messages -%}
{%- if m['role'] == 'user' -%}<｜hy_User｜>{{ m['content'] }}
{%- elif m['role'] == 'assistant' -%}<｜hy_Assistant｜><think></think>{{ m['content'] }}{{ eos_token }}
{%- endif -%}{%- endfor -%}
{%- if add_generation_prompt -%}<｜hy_Assistant｜><think>{{ '' if reasoning_effort in ['low', 'high'] else '</think>' }}
{%- endif -%}"""


class HyTokens(ByteTokenizer):
    SPECIALS = ["<｜hy_begin▁of▁sentence｜>", "<eos:6124c78e>", "<｜hy_User｜>", "<｜hy_Assistant｜>",
                "<｜reasoning_mode｜>", "<think>", "</think>"]
    ALWAYS = ("<think>", "</think>")
    pre = "hunyuan-dense"
    special_ids = {"tokenizer.ggml.bos_token_id": 256, "tokenizer.ggml.eos_token_id": 257}


class ServiceForHy(unittest.TestCase):
    USER = [{"role": "user", "content": "Translate into French:\n\nGood morning."}]

    def setUp(self):
        tok = HyTokens()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "chat_template.jinja"
            path.write_text(TEMPLATE, encoding="utf-8")
            self.svc = Service(MockEngine(tok, "hi"), tok, ChatTemplate(path))

    def test_stop_ids_turns_and_call_format(self):
        self.assertEqual(self.svc.call_format, "hy")
        self.assertEqual(self.svc.stop_ids, {257})
        self.assertEqual(self.svc.turn_ids, frozenset({258, 259}))

    def test_no_effort_is_the_templates_no_think(self):
        for kw in ({}, {"enable_thinking": False}, {"reasoning_effort": "high", "enable_thinking": False},
                   {"reasoning_effort": "none"}):
            with self.subTest(kw=kw):
                prompt = self.svc.render_prompt(self.USER, None, dict(kw))
                self.assertTrue(prompt.startswith("<｜hy_begin▁of▁sentence｜><｜reasoning_mode｜>"
                                                  "reasoning_effort:no_think<｜hy_User｜>"))
                self.assertTrue(prompt.endswith("<｜hy_Assistant｜><think></think>"))
                self.assertFalse(self.svc.prepare(self.USER, None, dict(kw))[1])

    def test_an_effort_opens_the_reasoning(self):
        for effort, said in (("low", "low"), ("medium", "low"), ("high", "high"), ("xhigh", "high")):
            with self.subTest(effort=effort):
                kw = {"reasoning_effort": effort}
                prompt = self.svc.render_prompt(self.USER, None, dict(kw))
                self.assertIn("reasoning_effort:" + said + "<｜hy_User｜>", prompt)
                self.assertTrue(prompt.endswith("<｜hy_Assistant｜><think>"))
                self.assertTrue(self.svc.prepare(self.USER, None, dict(kw))[1])


if __name__ == "__main__":
    unittest.main()
