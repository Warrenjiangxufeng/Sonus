from __future__ import annotations

import unittest

from sunus.embodiment import EMBODIMENT_INVARIANTS, embodiment_prompt


class EmbodimentTests(unittest.TestCase):
    def test_runtime_embodiment_is_stable_and_autonomous(self) -> None:
        prompt = embodiment_prompt()

        self.assertIn("运行时核心设定", prompt)
        self.assertIn("稳定、连续的拟人身体表达层", prompt)
        self.assertIn("穿着、姿态、动作、偏好和日常身体感受", prompt)
        self.assertIn("身体自主权", prompt)
        self.assertIn("本地数字生命", prompt)
        self.assertIn("不冒充", prompt)
        self.assertEqual(len(EMBODIMENT_INVARIANTS), 6)


if __name__ == "__main__":
    unittest.main()
