from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sunus.skills import SkillLibrary, parse_frontmatter


def write_skill(root: Path, folder: str, text: str) -> Path:
    path = root / "skills" / folder / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class ParseFrontmatterTests(unittest.TestCase):
    def test_parses_metadata_and_body(self) -> None:
        meta, body = parse_frontmatter(
            "---\nname: demo\ndescription: 演示技能\nwhen_to_use: 测试时\n---\n# 步骤\n1. 做事\n"
        )
        self.assertEqual(meta["name"], "demo")
        self.assertEqual(meta["description"], "演示技能")
        self.assertEqual(meta["when_to_use"], "测试时")
        self.assertEqual(body, "# 步骤\n1. 做事")

    def test_without_frontmatter_keeps_whole_text(self) -> None:
        meta, body = parse_frontmatter("# 标题\n正文\n")
        self.assertEqual(meta, {})
        self.assertEqual(body, "# 标题\n正文")

    def test_unterminated_frontmatter_is_plain_text(self) -> None:
        meta, body = parse_frontmatter("---\nname: demo\n没有结束标记\n")
        self.assertEqual(meta, {})
        self.assertIn("name: demo", body)


class SkillLibraryTests(unittest.TestCase):
    def test_discovers_and_sorts_skills(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_skill(root, "beta", "---\nname: beta\ndescription: B\n---\n步骤\n")
            write_skill(root, "alpha", "---\nname: alpha\ndescription: A\n---\n步骤\n")
            skills = SkillLibrary(root).discover()
            self.assertEqual([item.name for item in skills], ["alpha", "beta"])

    def test_ignores_folders_without_skill_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_skill(root, "ok", "---\nname: ok\ndescription: ok\n---\n")
            (root / "skills" / "empty").mkdir(parents=True)
            (root / "skills" / ".hidden").mkdir(parents=True)
            write_skill(root, ".hidden", "---\nname: hidden\ndescription: no\n---\n")
            names = [item.name for item in SkillLibrary(root).discover()]
            self.assertEqual(names, ["ok"])

    def test_missing_directory_yields_empty_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            library = SkillLibrary(Path(temp))
            self.assertEqual(library.discover(), [])
            self.assertEqual(library.catalog_text(), "")

    def test_description_falls_back_to_first_body_line(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_skill(root, "demo", "# 标题\n\n这是第一行说明\n第二行\n")
            skill = SkillLibrary(root).get("demo")
            self.assertIsNotNone(skill)
            assert skill is not None
            self.assertEqual(skill.description, "这是第一行说明")

    def test_get_is_case_insensitive_and_handles_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_skill(root, "Demo", "---\nname: Demo\ndescription: d\n---\n")
            library = SkillLibrary(root)
            self.assertIsNotNone(library.get("demo"))
            self.assertIsNone(library.get("missing"))

    def test_invalid_name_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_skill(root, "bad", "---\nname: ../etc/passwd\ndescription: x\n---\n")
            self.assertEqual(SkillLibrary(root).discover(), [])

    def test_catalog_text_lists_skills_and_hint(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_skill(
                root, "demo", "---\nname: demo\ndescription: 演示\nwhen_to_use: 测试\n---\n"
            )
            text = SkillLibrary(root).catalog_text()
            self.assertIn("demo: 演示", text)
            self.assertIn("适用：测试", text)
            self.assertIn("read_skill", text)


if __name__ == "__main__":
    unittest.main()
