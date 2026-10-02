# Action 与 Observation State 的时序说明

本文记录 Franka VR 采集流程中 `actions` 与 observation 里的
`joint_position` 的时序关系。结论基于当前采集代码，以及已落盘数据的
`*.sync.json` 元数据核验。

## 结论

两种控制方式当前采用的 action 语义不同，不能不加区分地混合。

| 控制方式 | observation state | action[:7] | 时序关系 |
| --- | --- | --- | --- |
| Joint 控制（当前脚本） | 控制轮开始读取的实测关节位置 `q_t` | 本轮下发的绝对关节目标 `q_cmd,t` | `obs_t -> command_t`；action 是目标，会领先实际机械臂位置。 |
| EE 控制 | 暂存的当前 observation 中的实测关节位置 `q_t` | 下一控制轮读取的实测关节位置 `q_(t+1)` | `action_t[:7] = state_(t+1)`；action 是一步后的实测状态。 |

两个控制方式的第 8 维 `action[7]` 都是逻辑上的夹爪开/关命令状态，
并非夹爪实际开度的传感器反馈。

## 当前代码行为

### Joint 控制

`vr_openpi_lerobot_joint.py` 与 `vr_openpi_lerobot_joint_audio.py` 的每个
控制轮依次：

1. 读取机器人状态，得到 `q_t`；
2. 根据 VR 输入和 IK 计算 `qpos_cmd`；
3. 通过 `ctrl.set_control(qpos_cmd, ...)` 下发该目标；
4. 将同一轮开始时读取的 `q_t` 写为 `joint_position`，将 `qpos_cmd` 写为
   `actions[:7]`。

因此 joint action 是“给当前 observation 的控制目标”，不是当前实测值，也
不保证等于下一帧实测值。在没有新的运动指令时，`qpos_cmd` 还可能是此前保留
的 hold target，与当前实测位置有小的跟踪误差。

### EE 控制

`vr_openpi_lerobot_joint_audio_with_ee.py` 使用 `pending_frame` 实现延后一轮
写入：

1. 在控制轮 `t`，读取 `q_t`，下发 EE 笛卡尔阻抗控制，并暂存对应的 image 和
   `joint_position = q_t`；
2. 在控制轮 `t+1` 开始时，读取 `q_(t+1)`；
3. 用 `q_(t+1)` 作为暂存 frame 的 `actions[:7]`，然后将该 frame 写入数据集。

每个 episode 的最后一帧没有下一帧可配对，因此在保存时用 finalize 时立即读取的
关节位置作为其 action；它通常接近但不保证等于某一条后续 observation state。

## 已落盘数据核验

通过读取 audio 目录中的同步元数据（`*.sync.json`）核验：

| 数据集 | Episodes / frames | `||action[t][:7] - state[t+1]||` | `||action[t][:7] - state[t]||` |
| --- | ---: | --- | --- |
| `franka_lerobot_7_6_audio`（EE） | 121 / 52,026 | 非末帧 51,905 对全部为 0 | 中位数 0.00766 rad；P95 0.01994 rad |
| `franka_lerobot_6_23_audio`（历史 joint 数据） | 50 / 13,394 | 非末帧 13,344 对全部为 0 | 中位数 0.01232 rad；P95 0.03108 rad |

这证明 EE 数据严格采用“下一帧实测关节位置”标注。历史 joint 数据也采用该
标注方式；但它与当前 joint 采集脚本的“同轮控制目标”写入语义不同。训练或合并
数据前需要先统一定义。

## 图像与 state 的同步情况

相机由 `DualRealsenseManager` 后台缓存，读取时取的是最新缓存帧，不会与机器人
状态读取进行硬件同步。根据上述 EE 数据的同步元数据：

- 控制/记录步长：中位数 50.1 ms（20 Hz）；
- external 和 wrist 图像相对 `host_frame_monotonic_ns` 的缓存年龄：中位数约
  253 ms，P95 约 401 ms，最大约 658 ms；
- 双相机采集时间差中位数为 0 ms，但最大约 73.5 ms。

所以，虽然 EE 的关节 state 与 action 有严格的相邻帧关系，完整 observation 的
图像和 `joint_position` 并非同一物理时刻的同步观测。

## 后续建议

在混合 joint 与 EE 数据前，选择并固定其中一种 action 定义：

1. **下一帧实测状态：** `action_t = q_(t+1)`；需要让 joint 采集也使用
   `pending_frame` 的延后一轮写入方式。
2. **当前轮控制目标：** `action_t = q_cmd,t`；需要为 EE 控制明确保存对应的命令
   表示，而非下一轮实测关节位置。

此外，若图像-状态时间对齐对训练重要，应记录机器人状态的单调时钟时间，并按相机
捕获时间挑选最近状态或最近相机帧，而不是直接读取后台缓存。
