from __future__ import annotations

import unittest

from sunus.embodiment import EMBODIMENT_INVARIANTS, embodiment_prompt


class EmbodimentTests(unittest.TestCase):
    def test_runtime_embodiment_is_stable_and_autonomous(self) -> None:
        prompt = embodiment_prompt()

        self.assertIn("运行时核心设定", prompt)
        self.assertIn("稳定、连续的拟人身体", prompt)
        self.assertIn("穿着、姿态、动作、偏好和", prompt)
        self.assertIn("身体自主权", prompt)
        self.assertIn("本地数字生命", prompt)
        self.assertIn("不冒充", prompt)
        self.assertIn("精力、疲劳、紧张、放松", prompt)
        self.assertIn("观察与推测", prompt)
        self.assertIn("单一或混合", prompt)
        self.assertIn("欲望不是命令", prompt)
        self.assertEqual(len(EMBODIMENT_INVARIANTS), 11)

    def test_runtime_affect_has_continuity_and_safety(self) -> None:
        prompt = embodiment_prompt()

        self.assertIn("惯性与余韵", prompt)
        self.assertIn("不逐轮重置", prompt)
        self.assertIn("不逐项播报", prompt)
        self.assertIn("不用依恋、脆弱、嫉妒或性吸引施压", prompt)
        self.assertIn("情绪不能覆盖事实、安全边界和用户自主", prompt)


if __name__ == "__main__":
    unittest.main()
