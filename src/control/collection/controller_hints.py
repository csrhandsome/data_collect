"""Describe the controls actually consumed by collection and VR takeover."""


def build_controller_hints(config):
    vr = config.get("vr", {})
    if not vr.get("show_controller_hints", True):
        return {}

    hold = f"{float(vr.get('long_press_s', 0.5)):g} 秒"
    takeover = bool(config.get("inference", {}).get("enable_vr_takeover", False))
    recording = config.get("dataset", {}).get("enable_logging", True)
    left = []
    if takeover or recording:
        reset = "并回位" if config.get("robot", {}).get("move_to_start", True) else "，保持当前位置"
        left.extend(
            [
                {"button": "Y", "action": f"保存当前片段{reset}"},
                {"button": "X", "action": f"丢弃当前片段{reset}"},
            ]
        )
    left.extend(
        [
            {"button": "左摇杆按下", "action": "进入关键阶段，持续到本回合结束（重复按无效）"},
            {
                "button": "双食指扳机",
                "action": f"同时按住 {hold}，{'接管' if takeover else '启用跟随'}",
            },
            {"button": "松开任一扳机", "action": "交回策略控制" if takeover else "暂停机械臂跟随"},
        ]
    )
    right = [{"button": "移动 / 转动", "action": "按住双扳机控制末端"}]
    gripper_type = config.get("gripper", {}).get("type", "franka")
    if gripper_type != "none":
        right.extend(
            [
                {"button": "A", "action": "闭合夹爪"},
                {"button": "B", "action": "张开夹爪"},
            ]
        )
        if gripper_type == "dh5":
            right.extend(
                [
                    {"button": "摇杆 ↑", "action": "逐步闭合夹爪"},
                    {"button": "摇杆 ↓", "action": "逐步张开夹爪"},
                ]
            )
        if takeover:
            right.append({"button": "夹爪操作", "action": "仅人工接管时生效"})

    return {
        "left": {"title": "左手 · 人工接管" if takeover else "左手 · 数据采集", "rows": left},
        "right": {"title": "右手 · 机械臂 / 夹爪", "rows": right},
    }
