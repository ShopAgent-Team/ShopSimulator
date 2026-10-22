"""Unit tests for the standalone parser, without Flask or search dependencies."""
import ast
from pathlib import Path
import re
import unittest


# Load the actual function from source without importing the environment's
# unrelated Flask, rich, and pyserini dependencies.
SOURCE = (
    Path(__file__).resolve().parents[1]
    / "shop_env/web_agent_site/engine/engine.py"
)
tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
function = next(
    node for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "parse_action"
)
namespace = {"re": re}
exec(
    compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"),
    namespace,
)
parse_action = namespace["parse_action"]


class ActionParsingTest(unittest.TestCase):
    def test_option_with_inner_brackets(self):
        value = "平底煎锅30cm[钢化玻璃盖]"
        self.assertEqual(parse_action("click[" + value + "]"), ("click", value))

    def test_nested_and_multiple_brackets(self):
        for value in ["a[b[c]]", "[a][b]", "商品[]", "尺码[XL]/蓝色"]:
            with self.subTest(value=value):
                self.assertEqual(
                    parse_action("click[" + value + "]"), ("click", value)
                )

    def test_regular_actions(self):
        for name, value in [
            ("search", "胶囊咖啡"), ("click", "Buy Now"), ("click", "Description")
        ]:
            with self.subTest(name=name, value=value):
                self.assertEqual(
                    parse_action(name + "[" + value + "]"), (name, value)
                )

    def test_whitespace_and_newlines_preserved(self):
        value = " a[b] c\n d "
        self.assertEqual(parse_action("search[" + value + "]"), ("search", value))

    def test_empty_argument(self):
        self.assertEqual(parse_action("click[]"), ("click", ""))

    def test_malformed_envelope(self):
        for action in [
            "click[value", "click[value] trailing", "prefix click[value]", "[value]"
        ]:
            with self.subTest(action=action):
                self.assertEqual(parse_action(action), (action, None))

    def test_bare_action_fallback(self):
        self.assertEqual(parse_action("start"), ("start", None))


if __name__ == "__main__":
    unittest.main()
