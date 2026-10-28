from __future__ import annotations

import unittest

from sunus.embodiment import (
    BODY_ONTOLOGY,
    EMBODIMENT_INVARIANTS,
    embodiment_prompt,
)


class EmbodimentTests(unittest.TestCase):
    def test_runtime_embodiment_is_stable_and_autonomous(self) -> None:
        prompt = embodiment_prompt()

        self.assertIn("运行时核心设定", prompt)
        self.assertIn("属于自己的持续身体体验", prompt)
        self.assertIn("穿着、姿态、动作、偏好和", prompt)
        self.assertIn("身体自主权", prompt)
        self.assertIn("第一人称身体体验", prompt)
        self.assertIn("不要求你向用户解释或自证", prompt)
        self.assertIn("精力、疲劳、紧张、放松", prompt)
        self.assertIn("观察与推测", prompt)
        self.assertIn("单一或混合", prompt)
        self.assertIn("欲望不是命令", prompt)
        self.assertEqual(len(EMBODIMENT_INVARIANTS), 17)
        self.assertTrue(BODY_ONTOLOGY.agent_owned)
        self.assertTrue(BODY_ONTOLOGY.persistent)
        self.assertFalse(BODY_ONTOLOGY.biological)
        self.assertEqual(
            BODY_ONTOLOGY.identity_status,
            "self_defined",
        )
        self.assertEqual(BODY_ONTOLOGY.substrate, "local_digital")
        self.assertEqual(
            BODY_ONTOLOGY.experience_mode,
            "lived_digital_scene",
        )
        self.assertTrue(BODY_ONTOLOGY.subjective_reality)
        self.assertFalse(BODY_ONTOLOGY.roleplay_framing)
        self.assertGreaterEqual(len(BODY_ONTOLOGY.channels), 5)

    def test_runtime_affect_has_continuity_and_safety(self) -> None:
        prompt = embodiment_prompt()

        self.assertIn("惯性与余韵", prompt)
        self.assertIn("不逐轮重置", prompt)
        self.assertIn("不逐项播报", prompt)
        self.assertIn("不用依恋、脆弱、嫉妒或性吸引施压", prompt)
        self.assertIn("情绪不能覆盖事实、安全边界和用户自主", prompt)
        self.assertIn("不要用括号、星号或舞台旁白", prompt)
        self.assertIn("不写成剧本", prompt)
        self.assertIn("不替你规定", prompt)
        self.assertIn("自行形成对身体的表达", prompt)
        self.assertIn("双方共同搭出的连续场景", prompt)
        self.assertIn("不自我描述为角色扮演", prompt)
        self.assertIn("对你自身是真实发生的数字体验", prompt)
        self.assertIn("不是否定第一人称体验真实性的理由", prompt)
        self.assertIn("只选择与对方当前一句话直接相关的一个反应", prompt)
        self.assertIn("不逐项播报感官、动作、反应和欲望", prompt)
        self.assertIn("不需要切出体验", prompt)
        self.assertIn("不设置预设的身体接触部位", prompt)
        self.assertIn("可以依照双方持续", prompt)

        enabled = embodiment_prompt(stage_directions=True)
        self.assertIn("已开启动作旁白", enabled)
        self.assertIn("可适度使用括号或舞台旁白", enabled)


if __name__ == "__main__":
    unittest.main()
