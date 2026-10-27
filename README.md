# Panda 数据采集与回放

ROS 2 Jazzy 版本：参见 [ROS 2 Docker 与遥操作](docs/ros2.md)。
环境文件位于 `src/env/`，使用官方 `franka_ros2` Jazzy 控制器。
采集统一使用 `config/train/panda.yaml`，只需将 `robot.backend` 设为 `panda` 或 `ros2`；
推理配置继承此选择。FCI 激活和自动回位默认在 Panda 后端开启、ROS 后端关闭，
ROS 遥测默认随 ROS 后端开启。ROS 后端需要已加载 ROS 环境及单独启动的官方控制器。
相机和麦克风默认跟随后端切换：Panda 使用本地设备，ROS 使用 topic 订阅；
可分别通过 `camera.backend`、`audio.backend` 显式选 `direct` 或 `ros2`。
ROS 输入的 topic 和音频格式见 [ROS 设备配置](docs/ros2.md#ros-输入设备)。

公共接口为 `control.base_runner.BaseRunner`，由 `control.runners.create_runner()`
按配置创建 `FrankaRunnerStable`（`control/direct/`）或 `FrankaRunner`（`control/ros2/`）。
两种后端共用 `control/common/` 的命令调度和 `control/types.py` 的数据类型。

项目使用 **Python 3.12、LeRobot 0.6.1（最新正式版）和 v3.0 数据格式**。
依赖版本记录在 `uv.lock` 中。LeRobot 0.6 起的数据操作需要 `dataset` extra，项目已启用。

```bash
uv sync --locked
uv run vr_collect --config config/train/panda.yaml
bash dev.sh
```

Electron + Nuitka Linux 桌面版的构建、运行和产物测试见
[replay/desktop/README.md](src/replay/desktop/README.md)。

无需硬件的采集验证（请使用单独的 `dataset.root` / `dataset.date` 配置）：

```bash
uv run vr_collect --config /path/to/test-config.yaml --dry-run --max-steps 50
uv run pytest
```

`uv sync` 会按 `.python-version` 将项目 `.venv` 切换到 Python 3.12。

## VR 手柄操作提示

进入 VR / Passthrough 后，左右手柄下方各显示一张中文操作速查卡，抬手即可查看。
说明由采集配置发送到头显，不改变现有按键行为。

| 手柄 | 操作 | 功能 |
| --- | --- | --- |
| 左手 | Y | 保存当前片段并回位 |
| 左手 | X | 丢弃当前片段并回位 |
| 双手 | 食指扳机同时按住超过 `vr.long_press_s`（默认 0.5 秒） | 启用右手柄对机械臂末端的跟随 |
| 双手 | 松开任一食指扳机 | 暂停跟随 |
| 右手 | A / B | 闭合 / 张开夹爪 |
| 右手 | 摇杆上推 / 下推 | 逐步闭合 / 张开，仅 DH5 夹爪支持并显示 |

录制在首次有效机械臂或夹爪操作时自动开始，Y 用于保存当前片段。
推理人工接管模式会改为显示接管说明，不显示无效的 X/Y 保存操作；松开扳机会交回策略控制。
`vr.show_controller_hints: false` 可关闭提示。禁用录制或夹爪时，对应说明也会隐藏。
关闭自动回位时（包括 ROS 后端默认设置），X/Y 操作后保持当前位置，手柄提示同步显示。

根项目和桌面 Python 环境都通过 `[tool.uv.sources]` 固定依赖
[csr fork 的修改提交](https://github.com/csrhandsome/teleop_xr/commit/3c572efddf98719c4a28de292d248e41b23a43a7)，
版本为 `1.3.5+csr.2`。换机器后运行 `uv sync --locked` 即可安装同一份前端。
从 Git 安装会自动构建 WebXR 页面，安装机器需具备 Node.js ≥ 20.19 和 npm，并能下载 npm 依赖。
修改源码时，在相邻的 `../teleop_xr/webxr/src/xr/controller_hints_system.ts` 调整提示外观；
按键文字来自本项目 `src/control/collection/controller_hints.py`。
后续修改 fork 后，需要先提交并推送，再更新两个 `pyproject.toml` 的 commit 和各自的 `uv.lock`。

## 夹爪接口

机械臂持有公共的 `GripperController`（`arm.gripper`），连接、关闭和异步命令调度由机械臂统一管理。
`config/train/panda.yaml` 的 `gripper.type` 选择 `franka`（默认）、`dh5` 或 `none`（禁用）。
底层 `FrankaGripperBackend` 和 `DH5GripperBackend` 实现相同的 `GripperBackend` 接口；
后续适配新夹爪时，实现该接口并加入工厂与配置校验即可。

```python
from control.config import load_config
from control.runners import create_runner

with create_runner(config=load_config("config/train/panda.yaml")) as arm:
    arm.gripper.open()
    arm.gripper.set_open_ratio(0.75, wait=False)
    arm.gripper.wait(timeout=5.0)
    state = arm.gripper.get_state()
    arm.gripper.close()  # 闭合夹爪；退出 with 时释放机械臂及夹爪资源。
```

开合比例统一为 `0=闭合、1=全开`，采集数据继续记录该比例。
Franka 保留原有行为：比例 ≥ 0.5 时按 `open_width_m × 比例` 移动，小于 0.5 时执行抓取；
命令的 `speed`、`force` 分别使用 m/s、N。DH5 将比例映射为 `round((1 - 比例) × 1000)` 的位置寄存器值，
使用 YAML 的 `force`、`velocity` 原生参数和 `dh5.com` 串口配置；不把寄存器值换算成物理角度。
DH5 的触觉相机仍由 `tactile.enabled` 控制，图像通过 `arm.gripper.get_tactile_images()` 读取。
`arm.gripper.stop()` 支持 Franka；DH5 驱动没有确认可用的停止命令，会抛出 `NotImplementedError`。
旧的 `arm.set_gripper()`、`arm.gripper_open()` 等方法保留，转发到同一个夹爪对象。

## 采集行为

### VR 手柄操作

推理模式需在 `config/inference/panda.yaml` 中设置 `inference.enable_vr_takeover: true`；
开启后自动录制。更新代码后，需重启采集/推理进程；ROS2 模式还需重启 VR 节点。

| 按键 / 操作 | 示教采集模式 | 策略推理模式 |
| --- | --- | --- |
| 同时按住左右食指扳机 | 启用机械臂跟随 | 接管机械臂，自动记录介入事件 |
| 松开任一食指扳机 | 暂停跟随，保持当前位置 | 交回策略，自动记录交回事件并恢复推理 |
| 移动 / 转动右手柄 | 按住双扳机时控制末端位置 / 姿态 | 人工接管时控制末端位置 / 姿态 |
| **左摇杆按下** | 录制中进入关键阶段 | 进入关键阶段 |
| **Y**（左手柄） | 结束并保存当前回合，可继续采集下一回合 | 结束并保存本次推理，退出执行 |
| **X**（左手柄） | 丢弃当前回合，可继续采集下一回合 | 结束并丢弃本次推理，退出执行 |
| **A**（右手柄） | 闭合夹爪 | 人工接管时闭合夹爪 |
| **B**（右手柄） | 张开夹爪 | 人工接管时张开夹爪 |
| 右摇杆 ↑ / ↓（仅 DH5） | 逐步闭合 / 张开夹爪 | 人工接管时逐步闭合 / 张开夹爪 |
| 桌面「成功 / 失败」 | 保存后标注任务结果 | 保存后标注任务结果 |

双扳机默认同时按住 **0.5 秒**后生效，可通过 `vr.long_press_s` 调整。
进入关键阶段后，终端提示「已进入关键阶段」；重复按不会退出，新回合自动重置。
录制开始前的关键阶段按键不会带入下一回合。Y 保存不代表成功，X 丢弃不代表失败。
保存或丢弃后的回位行为遵循 `robot.move_to_start` 配置。

采集运行参数集中在 `config/train/panda.yaml`；存储字段由 `config/dataset/panda.yaml` 定义。100 Hz 轨迹与音频 sidecar 保留原有约定。
采集完仍按原来的 Y 键保存。保存只结束并保留当前 episode，`success` 默认为 `null`（未标注）；
X 键仍用于丢弃当前录制，不会把片段标成失败。
录制中按左摇杆进入关键阶段，终端提示「已进入关键阶段」；重复按不会退出，新回合自动清除。
原始轨迹仍完整保存，阶段起点记录在 sidecar 的 `critical_phase_start_monotonic_ns`，
动作 trace 逐步保存 `in_critical_phase`。录制开始前的阶段按键不带入下一回合。
采集流程每次保存都会 `finalize()` 封口 LeRobot v3 数据和元数据，再发布 `.sync.json`；
下一条开始时按最新元数据续采，因此采集任务无需退出，已保存的片段就能回放。
每条封口会增加保存和下次续采的开销；直接使用 `EpisodeRecorder` 批量写入时，
仍可通过默认的 `finish(..., publish=False)` 复用写入器，在会话结束时统一 `finalize()`。

保持前端页面打开并勾选「自动打开新采集片段」，页面每 1.5 秒自动检测新片段，
打开该数据集的对应 episode，并添加相机画面和末端轨迹。视频准备完成后即可用时间轴回放。
在「片段结果」区域点击「成功」或「失败」会立即保存标签，也可点「未标注」清除标签。
标签保存在该条 `episode_XXXXXX.sync.json` 的 `success` 字段：`true` / `false` / `null`，
刷新页面后仍保留，轨迹 Parquet 不会被重写。开启采集使用的目录须在回放服务的数据目录内。

## 统一存储与推理 key

配置按用途分为两个文件：

- [config/train/panda.yaml](config/train/panda.yaml)：机器人、采集和回放运行参数。
- [config/inference/panda.yaml](config/inference/panda.yaml)：推理执行参数，继承训练配置。`model_horizon` 对齐服务器预测长度，`execute_steps` 是实际执行的前缀。
- [config/dataset/panda.yaml](config/dataset/panda.yaml)：嵌套的观测和动作字段定义。

运行配置通过 `extends: ../dataset/panda.yaml` 继承数据集配置，相对路径按配置文件所在目录解析。
采集、推理请求、openpi-force 的 Panda 训练读取使用相同字段名；训练端无需复制这份 YAML。
约定 `true` 表示保存、`false` 表示不保存，例如：

```yaml
observation:
  exterior_image: true
  ee_pose: true

action:
  ee_pose: true
```

| key | 类型 / 维度 | 内容 |
| --- | --- | --- |
| `observation.exterior_image` | RGB `[H,H,3]` | 外部相机 |
| `observation.wrist_image_left` | RGB `[H,H,3]` | 腕部相机 |
| `observation.gripper_image_left` / `observation.gripper_image_right` | RGB `[H,H,3]` | 可选左右触觉图像 |
| `observation.joint_position` | `float32[7]` | 实测关节角，rad |
| `observation.ee_pose` | `float32[6]` | xyz（m）+ roll/pitch/yaw（rad，ZYX 欧拉角约定） |
| `observation.ee_position` | `float32[3]` | 实测末端 xyz，m |
| `observation.gripper_position` | `float32[1]` | 指令开合比例，0 闭合、1 全开 |
| `action.joint_position` | `float32[7]` | 下一帧实测关节角，rad |
| `action.ee_pose` | `float32[6]` | 下一帧实测末端位姿，m/rad |
| `action.gripper_position` | `float32[1]` | 下一帧指令开合比例 |

`H = camera.image_hw`（当前 224）。joint 和 ee 动作标签同时保留，训练时选择所需标签并拼接夹爪比例。
末帧动作使用 episode 结束时的实测状态。不保存全零占位图像和旧的拼接 `actions` 字段；训练端才拼接 joint/ee 与 gripper 标签。
LeRobot 自动生成的 `timestamp`、`frame_index`、`episode_index`、`index`、`task_index` 保留原名；
任务文本继续来自 `dataset.instruction`，写入任务元数据。

存储 key 由分组和字段名拼接，例如 `observation` 下的 `exterior_image` 对应 `observation.exterior_image`。
`.actions.jsonl` 继续保存实测状态、控制目标和 VR 输入，当前控制频率为 100 Hz；
`.sync.json` 保存帧/动作时钟对齐和 episode 信息。
音频 WAV 和音频元数据继续由 `audio.enabled` 控制，当前配置为 16 kHz 单声道。
新旧字段格式不同，需更换 `dataset.date` 或数据目录。续采校验会拒绝混用 schema。

## 数据存储

采集、训练读取、回放、合并和删除统一使用 **LeRobot v3 + MP4**。相机字段固定为
`dtype: video`，不再提供图像内嵌选项或旧格式迁移入口。视频参数在
`config/train/panda.yaml` 的 `dataset.video_encoder` 配置，默认 H.264、CRF 18、GOP 2、fast。

现有数据仅保留 `data/dataset/*_v3_mp4`，音频、同步记录和动作轨迹随数据集保留。
整理范围、保留数据清单和此前性能对比统一见 [v3 MP4 说明](LEROBOT_V3_MP4.md)。

音频采集、VAD、回放和已有标注仍可使用；旧 ASR 数据转换脚本已删除。

## 推理工作区

前端「采集与推理」栏选择「推理」，点击「开始推理」启动策略执行，点击「结束推理」停止。
结束后会保存本次推理 episode，再通过 `move_to_start()` 返回配置的起始姿态；完成后自动打开本次片段，
可选择成功（`true`）、失败（`false`）或未标注（`null`）。启动前应运行策略服务，地址由 `inference.host/port` 配置。
推理异常和回位失败会显示任务失败；运行异常时只停止控制流，不继续自动回位。

命令行也可运行 `uv run python -m inference --record`，默认读取 `config/inference/panda.yaml`。

在 `config/inference/panda.yaml` 设置 `inference.enable_vr_takeover: true` 后，可在推理中进行
VR 人工纠正；此时自动保存 episode，无需额外传 `--record`。松开接管后恢复策略推理。
推理中左摇杆按下标记进入关键阶段，Y 结束并保存本次推理，X 结束并丢弃；
结束后按现有配置回位或保持位置，成功/失败仍在桌面界面标注。
双扳机接管和松开交回会自动记录 `takeover_start` / `takeover_end` 事件，无需额外标记键。
动作 trace 逐步保存实际 `control_source`（`BASE/RL/HUMAN`）、`intervention_flag`、
接受下发的 EE/夹爪目标，以及策略 step/actor 版本；人工接管期间不会把网络提案当成执行动作。
EE 目标从控制器原子快照读取，包含夹爪暂停时的保持目标和恢复时的启动目标；
`ee_target_phase` 区分 `start/command/hold/gripper_pause/gripper_resume`。
夹爪暂停、恢复期间仍保留触发该夹爪动作的来源和策略版本；trace 时间戳对应目标快照的采样时刻。
支持普通 VLA/readout 和 Force actor 的来源记录，Force 新 YAML 的 `force_enabled` 元数据也可识别。

离线 RL 使用流程：录制 → 在桌面界面标注成功/失败 → 在 `openpi-force` 使用
`python -m openpi.rlt.build_replay` 导入数据 → 运行 `rlt.stage: actor_critic`。
构建 CLI 默认只导入已标成败且标记关键阶段的后缀；保留 BASE/RL/HUMAN 来源和真实终止。
旧数据或完整回合训练须显式传 `--include-noncritical`，并使用独立 replay 输出目录。
来源标记会保留到 replay，混合回合中的人工步用于 BC。未标注回合不参与训练。
这是离线数据链路；本地保存和标注不会自动调用在线 RL 的 `begin_episode/observe/end_episode`。
独立 Force 训练仍需要匹配其 JSONL 格式的实际关节执行记录；当前 EE 控制器不会把实测关节状态冒充关节命令，真机 Force 执行仍需要已校准的 TCP 安全投影。
不加 `--record` 时只执行推理；正常结束或 Ctrl+C 请求停止后仍会回到起始姿态。

## Force RLT 推理

推理不要求 `actor_sha256`。客户端仍校验动作空间、维度、频率、`model_horizon`、B/C 分组和响应观测时间，并只执行 `execute_steps` 步。
Force actor 使用 `inference.action_space: joint`，返回七维关节目标；普通 EE 策略返回六维位姿和夹爪比例。

C 组开启 `gripper.type: dh5`、`tactile.enabled: true` 并启用左右触觉 observation 字段。
启动 openpi-force 的 `scripts/serve_force_rlt_pytorch.py` 时加 `--finger-weights /path/to/resnet18.pth`，
可用 `--finger-roi X0 Y0 X1 Y1` 设置接收到的图像上的 ROI。
编码器权重、图像裁剪和 ROI 必须与 actor 训练一致；图像使用 `camera.image_hw`、`camera.crop_scale`，与采集保存的预处理相同。
启动时夹爪须张开且没有接触，以首对图像建立基线。

客户端维护四帧不重复的双指 RGB 历史和 host capture 时间戳，发送
`observation.gripper_image_left`、`observation.gripper_image_right`（均为 `[4,H,H,3]`）、`finger_timestamps[4,2]`、
`finger_valid_mask[4]`、基线和会话 ID。服务端批量编码并缓存重叠帧，计算与训练相同排列的 `z_grip`。
编码和网络推理在异步请求中执行；控制线程不加载神经网络。历史缺帧或过期会停止推理。
B 组无需双指输入。服务端仍兼容原有预计算 `z_grip` 请求。

目前 Force RLT 真机执行仍需接入原代码要求的 TCP 安全投影器；本次完成协议与 dry-run 路径。
真实 checkpoint 的端到端延迟需要在 GPU 和实际网络上测量。
