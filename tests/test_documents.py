from __future__ import annotations

import unittest

from sunus.documents import render_document


class DocumentTests(unittest.TestCase):
    def test_renders_profile_frontmatter_and_sections(self) -> None:
        content = render_document(
            profile="design",
            title="测试设计",
            summary="测试摘要",
            sections=[
                {"heading": "目标", "body": "目标内容"},
                {"heading": "架构", "body": "架构内容"},
                {"heading": "数据流", "body": "数据流内容"},
                {"heading": "安全边界", "body": "安全内容"},
                {"heading": "验证", "body": "验证内容"},
            ],
            sources=["src/sunus/agent.py"],
            generated_at="2026-10-24T00:00:00+00:00",
        )
        self.assertIn("sunus_document: design", content)
        self.assertIn("sunus_sources:", content)
        self.assertIn("# 测试设计", content)
        self.assertIn("## 架构", content)

    def test_rejects_missing_required_section(self) -> None:
        with self.assertRaisesRegex(ValueError, "缺少章节"):
            render_document(
                profile="report",
                title="报告",
                summary="",
                sections=[
                    {"heading": "结论", "body": "正常"},
                    {"heading": "证据", "body": "证据"},
                ],
            )


if __name__ == "__main__":
    unittest.main()
