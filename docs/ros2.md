# ROS 2 Jazzy：Docker、FR3 遥操作与采集

Docker 环境使用 Ubuntu 24.04、ROS 2 Jazzy、系统 Python 3.12。用户已确认
使用官方 `frankarobotics/franka_ros2` 的 `jazzy` 分支及 FR3。环境位于 `src/env/`
（项目原目录拼写是 `src`）。没有自写、复制或修改 C++ controller。

命令链路：VR → 原有 `VRResampler`、`EEMapper` → `FrankaRunner`（继承 `BaseRunner`）
→ `Ros2Backend` → `rclpy` 发布 `geometry_msgs/PoseStamped`
→ 官方 `cartesian_impedance_example_controller` → libfranka / FCI → FR3。
Python 默认 100 Hz，底层实时循环由官方 C++ 控制器负责。

## 当前官方控制器的阻塞问题

固定的官方提交为 `4acc05c01ebfcdf916ee6b3df966df0ad7b2f9ec`，
`libfranka=0.20.5`，`franka_description=2.9.0`。
该提交虽然订阅 `~/equilibrium_pose`，但 `update()` 每次都调用
`updateMotionTarget()`，后者生成周期示例轨迹并写回同一个 target buffer。
因此外部位置目标会被周期轨迹覆盖，不能保证 VR 的独占控制。

`src/env/bringup.sh` 会先只读检查这个问题，发现后退出，不启动机械臂。
官方源码保持原样；不要绕过检查直接在真机运行示例。
应选择官方修复了该行为的提交，再通过 `FRANKA_ROS2_REF=<commit>` 重建镜像。
检查只是识别已知问题，不能代替实际跟踪验证。
这也是当前代码不能宣称完成真机闭环验证的原因。

源码与接口依据：

- [官方 Cartesian impedance 文档](https://frankarobotics.github.io/docs/doc/franka_ros2_jazzy/franka_example_controllers/doc/index.html)
- [固定提交的 controller 源码](https://github.com/frankarobotics/franka_ros2/blob/4acc05c01ebfcdf916ee6b3df966df0ad7b2f9ec/franka_example_controllers/src/fr3/cartesian_impedance_example_controller.cpp)
- [官方 FrankaRobotState 消息](https://github.com/frankarobotics/franka_ros2/blob/4acc05c01ebfcdf916ee6b3df966df0ad7b2f9ec/franka_msgs/msg/FrankaRobotState.msg)

## 构建与启动

在仓库根目录下运行。先在 `config/train/panda.yaml` 中设置 `robot.backend: ros2`；
采集和推理共用此后端选择，宿主机和容器读取相同配置：

```bash
docker compose -f src/env/compose.yaml build
docker compose -f src/env/compose.yaml run --rm ros2
```

若构建需要使用宿主机已有代理，可显式传递 Docker 内置代理参数：

```bash
docker compose -f src/env/compose.yaml build \
  --build-arg HTTP_PROXY --build-arg HTTPS_PROXY --build-arg ALL_PROXY
```

镜像从官方仓库检出固定提交，只构建单机械臂相关包，不构建 Gazebo 和移动底盘。
Docker 内使用独立的 `/opt/data_collect_venv`，不会使用宿主机 `.venv`。
该虚拟环境由系统 `/usr/bin/python3` 建立并保留 ROS 系统包路径；
入口脚本已设置 `UV_PROJECT_ENVIRONMENT=/opt/data_collect_venv` 和 `UV_NO_SYNC=1`，
因此容器内既可使用 `python` / `vr_collect`，也可使用
`uv run --no-sync python -m vr_collect --config config/train/panda.yaml`。
容器使用系统 Python 3.12 和 `--system-site-packages` 环境，不复用宿主机 `.venv`。
宿主机的 panda-py 流程继续用根目录的 `uv run`，由 `robot.backend: panda` 选择。

编辑 `src/env/franka.config.yaml`，确认 FR3 IP、namespace、是否加载夹爪。
同时核对 Desk 系统版本是否支持 libfranka 0.20.5；环境不会自动登录 Desk、解锁或激活 FCI。
`robot.hostname` 只供旧 SDK 后端使用，ROS 驱动连接 IP 来自 bringup 配置。

上游修复并通过 preflight 后，在第一个容器中启动机械臂控制器和两路 RealSense：

```bash
bash src/env/start.sh
```

`start.sh` 会拉起 `cameras.sh` 和 `bringup.sh`。采集程序只订阅话题，不会自己打开相机。关掉这个终端时，控制器和相机会一起停。单独启动机械臂仍用 `bringup.sh`。

该脚本调用的官方命令为：

```bash
ros2 launch franka_bringup example.launch.py \
  controller_names:=cartesian_impedance_example_controller \
  robot_config_file:=/workspace/src/env/franka.config.yaml
```

第二个终端打开另一个容器，运行 VR 采集：

```bash
docker compose -f src/env/compose.yaml run --rm ros2 \
  python -m vr_collect --config config/train/panda.yaml
```

原 VR 的启用手势、A/B 夹爪、Y 保存、X 丢弃保持原行为。
VR 使用原 Teleop XR HTTPS 服务（4443），Docker 使用 host 网络。

本配置保留原有相机序列号，使用前改成当前连接的 RealSense。
容器共享 USB、视频、输入、串口设备；以 root 运行以便访问这些设备，但未启用 privileged。
数据默认保存在当前仓库的 `data/`，宿主机和容器使用相同的数据配置；
可用 `ROS2_DATA_DIR=/absolute/path` 指定挂载目录。

`ROS_DOMAIN_ID` 默认 42，所有 ROS 容器与观察终端应相同。
host 网络和 host IPC 用于 Linux；Compose 配置 `SYS_NICE`、rtprio 和 memlock，
宿主机仍须具备适用于 FR3 的实时内核和实时调度权限，容器不会提供独立内核。

## 配置与消息

统一使用 `config/train/panda.yaml`：`robot.backend: panda` 选择原 SDK 后端，
改为 `ros2` 选择官方 ROS 控制后端。`config/inference/panda.yaml` 继承同一选择。
相机、VR、数据路径和存储 schema 共用，不再维护独立的 ROS 采集/推理 YAML。

代码按设备访问方式组织：

```text
src/control/
  base_node.py                    # 与 ROS 无关的 start/read/close 采集接口
  base_runner.py                  # 公共 Runner 接口与命令保护
  runners.py                      # 按 robot.backend 创建子类
  types.py                        # Pose、RobotState、CameraPair 等共用类型
  common/                         # 公共生命周期、运动与夹爪调度
  direct/                         # panda_py、本地相机、声卡、串口夹爪
    franka_runner_stable.py        # FrankaRunnerStable
    camera_connector.py
    dual_camera_manager.py
    microphone_connector.py
    microphone_connector_thread.py
    grippers/
  ros2/                           # ROS 状态订阅、目标发布、设备订阅
    base_node.py                  # Ros2Node：公共采集接口 + 私有 ROS runtime
    franka_runner.py               # FrankaRunner
    camera_connector.py
    microphone_connector.py
    gripper_controller.py         # ROS action 与夹爪状态订阅
    subscription.py               # context/executor、等待、故障传播、清理
    bridge.py                     # 额外遥测发布
```

采集、推理和回放统一调用 `create_runner(config=...)`。基类管理命令所有权、
位姿限幅、夹爪暂停/恢复和资源释放，具体子类选择各自的通信实现。
`robotic_arm_controller.py` 仅保留旧 `RoboticArmControler` 构造入口的兼容导出。
ROS Runner 不启动第二个 FCI 客户端，官方控制器仍由 `bringup.sh` 独立启动。

### 公共采集节点接口

采集和推理通过 `BaseNode.read()` 读取最新数据。机械臂返回 `RobotState`，
相机返回 `CameraPair`（direct 相机尚无完整帧时可返回 `None`），VR 返回控制器状态。
`start()` 准备设备，`close()` 释放资源；工厂保持返回已准备好设备的现有行为，
重复 `start()` 不会新建连接。旧 `get_state()`、`get_frames()`、`latest`、VR `stop()` 保留。
`BaseNode` 不依赖 rclpy，panda-py 的 uv 环境无需安装 ROS。

```python
from contextlib import ExitStack

from control.collection.devices import make_cameras, make_vr
from control.runners import create_runner

with ExitStack() as stack:
    arm = stack.enter_context(create_runner(config=config))
    cameras = stack.enter_context(make_cameras(config))
    vr = stack.enter_context(make_vr(config))
    state, pair, sample = arm.read(), cameras.read(), vr.read()
```

`BaseRunner` 继承 `BaseNode[RobotState]`，继续独立提供开流、目标下发、hold、
轨迹和夹爪接口。`arm.start()` 只连接并准备反馈；运动仍需显式 `start_stream()`，
命令所有权、反馈过期检查和运动限幅沿用已有公共控制逻辑。
ROS 相机和 VR 继承 `Ros2Node`，各自定义消息、QoS 和首帧等待；
机械臂后端、音频订阅复用同一个 `Subscription` 生命周期实现。

这里的统一是采集端读取接口，不是新增本地回调总线，也不承诺跨设备硬同步。
相机保留调用线程 poll，读取时先处理待收消息；其他订阅可后台 spin，
同一个 executor 不允许同时后台 spin 和前台 poll。
direct VR 保留已实测有效的独立进程和共享内存，ROS VR 的 Teleop 发布端也单独运行，
公共接口不会把 VR 合并进机械臂控制进程。旧 `VRInputProcess` 停止后需要新建实例才能重启。
音频仍通过 `start_recording()` / `stop_recording()` 连续保存 episode，
不使用只保留最新样本的 `read()`，以免丢音频块。

配置合并完成后按后端设置默认值：Panda 的 `robot.activate_fci` 和
`robot.move_to_start` 为 true，ROS 的这两项为 false；`ros2.enabled` 在 ROS 后端
默认为 true，在 Panda 后端默认为 false。可显式覆盖默认值，但 ROS 后端不允许
启用 FCI 自动激活或关节回位。`inference.move_to_start` 仍保持默认 false。

Docker 按根目录 `pyproject.toml` 和 `uv.lock` 安装依赖，包含 `panda-python`；
选择 ROS 后端时没有第二个 Python FCI 客户端，Franka 夹爪使用 ROS action，不加载 panda_py。
镜像提供 Node.js 22 和 npm，用于构建当前锁定的 Teleop XR Git fork 的 WebXR 页面。
DH5 如启用，仍由原串口后端访问本地设备，触觉图像沿用原录制接口。

| 方向 | 默认 topic / action | 类型 |
| --- | --- | --- |
| Python → controller | `/cartesian_impedance_example_controller/equilibrium_pose` | `geometry_msgs/PoseStamped` |
| broadcaster → Python | `/franka_robot_state_broadcaster/robot_state` | `franka_msgs/FrankaRobotState` |
| Python → Franka gripper | `/franka_gripper/move`、`/franka_gripper/grasp` | `franka_msgs/action/Move`、`Grasp` |
| Python → gripper stop | `/franka_gripper/stop` | `std_srvs/Trigger` |
| gripper → Python | `/franka_gripper/joint_states` | `sensor_msgs/JointState` |

这里 Python **发布** equilibrium_pose，官方 controller **订阅**。
目标必须是 `fr3_link0` 坐标系，位置单位 m，四元数顺序 xyzw，消息带 ROS 时间戳。
frame_id 是校验标签，代码不会自动做 TF 变换。
ROS namespace 在 `franka.config.yaml` 和 Python `ros2.namespace` 中同步设置；
配置的 endpoint 会被相对该 namespace 解析。

反馈使用真实 `measured_joint_state` 和 `o_t_ee`，按关节名重排为 1…7。
不会将发布目标冒充实测数据。首条有效状态超时、反馈过期、源时间戳过期、坐标系错误
或 controller 订阅者消失时拒绝继续下发。多机使用时需要同步系统时钟。
原有目标相对实测位姿的平移/旋转保护继续生效。

松开 VR deadman 后由原 mapper 锁存保持位姿；正常退出前发布实测保持位姿。
`stop_stream()` 结束 Python 发送，不卸载官方 controller；它不是硬件急停。
Python 崩溃后停止发包不等于停止 C++ controller，官方示例没有由本项目新增的 watchdog。

equilibrium topic 不提供关节运动、FK 或 move-to-start 协议。
因此 ROS 配置禁用自动关节回位，保存/丢弃后保持当前位置并重新建立 VR 锚点；
首次开流也从实测位置锚定。需要回起始关节位姿时，在独立 bringup 初始化阶段完成。
`move_joints()` / `move_ee()` / FK 明确拒绝；可使用已有带保护的 Cartesian trajectory API。
ROS 后端推理仅支持 EE action，使用统一的 `config/inference/panda.yaml`：

```bash
python -m inference --config config/inference/panda.yaml --record
```

ROS 后端默认禁用自动回位；策略服务地址仍由 inference.host/port 指定。

## ROS 输入设备

相机和音频的 `backend` 默认随 `robot.backend` 切换：Panda 使用 `direct`，
ROS 使用 `ros2`。若 ROS 只负责机械臂，仍希望 Python 访问 USB 相机或声卡，
可在统一 YAML 中显式设置 `camera.backend: direct`、`audio.backend: direct`。
音频只有在 `audio.enabled: true` 且需要录制时才会连接。

先启动相机、音频驱动发布节点，再启动采集。订阅输入与 `/data_collect/...` 遥测输出
使用不同的 topic，避免把本项目发布的图像循环订阅为自己的输入。

| 输入 | 默认 topic | 消息与配置 |
| --- | --- | --- |
| 外部相机 | `/camera/external/color/image_raw` | `sensor_msgs/Image`，`camera.ros2.external_image_topic` |
| 腕部相机 | `/camera/wrist/color/image_raw` | `sensor_msgs/Image`，`camera.ros2.wrist_image_topic` |
| 麦克风 | `/microphone/audio` | `std_msgs/Int16MultiArray`，`audio.ros2.topic` |

相机支持 rgb8、bgr8、rgba8、bgra8、mono8 和行尾 padding，统一转为 RGB 并复用
`camera.crop_scale` / `image_hw` 预处理。两路消息的源时间戳必须有效且在
`camera.max_age_s` 内，时间差不超过 `camera.ros2.max_pair_skew_s`（默认 0.1 秒）。
重复或倒序消息不会刷新帧。源 stamp 使用 ROS system time；多机需同步时钟，
本实现不使用仿真 `/clock`。未收到有效帧或帧过期会报错并释放订阅资源。

音频消息约定为 PCM16 有符号采样，按 `audio.channels` 交错排列，发布端采样率
必须与 `audio.sample_rate` 一致；默认 16000 Hz、单声道。每条消息是一块连续采样。
此消息没有源时间戳，`.audio.json` 明确记录 `timestamp_source: host_receipt`，
时间戳代表本机接收时间，不能作为麦克风硬件采样时钟。订阅内容按 episode
流式写入 WAV；保存/丢弃、同步 sidecar、VAD 和回放沿用原有流程。
`audio.ros2.max_age_s`（默认 0.5 秒）限制接收超时和录制中的消息间隔，
音频缺失、过期或写入失败会报告错误，不会生成伪造的有效音频。

## 原网页与实时可视化

原网页仍读取 MP4、Parquet、action trace 和同步 sidecar，回放画面、末端轨迹、
标签及保存后的自动打开不依赖机械臂传输后端。原 Reactive Desk WebSocket
推送也继续接收同一种 RobotState，可用 reactive_desk.reactive_desk_enabled 控制。
浏览器不会直接订阅 DDS；ROS 遥测供 ROS 工具或后续桥接服务使用。

只看录制回放时，原前端照常运行，数据根目录指向当前仓库的 data/dataset。
若要从网页启动采集/推理，后端必须运行在已加载 ROS 环境的容器里：

```bash
docker compose -f src/env/compose.yaml run --rm ros2 replay-server
```

Compose 的 DATA_COLLECT_CONFIG、DATA_COLLECT_INFERENCE_CONFIG 分别指向统一的
`config/train/panda.yaml` 和 `config/inference/panda.yaml`，网页按钮也遵循同一个
`robot.backend` 选择。容器中 uv 子进程复用现有 ROS 虚拟环境，
不重建宿主机 .venv。前端开发服务器可在宿主机运行并连接 http://127.0.0.1:8003。
网页“在真机上回放”使用 ROS EE 流，但不会自动关节回位；需先将机械臂移动到
episode 起始末端位姿的 1 cm / 0.05 rad 内，否则拒绝开始。历史纯关节轨迹仍需 FK，
当前 ROS 后端不支持这种真机回放；网页查看历史数据不受此限制。

## ROS 遥测与数据录制

原有 LeRobot 0.6.1 / v3 MP4、100 Hz action trace、音频和同步 sidecar 保留。
相机与麦克风可从 ROS topic 订阅，也可显式选择原本地设备适配器；VR 沿用 Teleop XR。
图像和 VR 输入通过独立 ROS worker 额外发布，不会在控制线程中序列化图像。

遥测由 `ros2.enabled` 控制，ROS 后端默认开启，Panda 后端默认关闭；控制后端不依赖此开关。
`ros2.telemetry_namespace` 默认 `/data_collect`，与机器人 namespace 独立。
默认 30 Hz，topic 包括：

```text
/data_collect/robot/joint_states
/data_collect/robot/ee_pose
/data_collect/robot/gripper_state
/data_collect/robot/target_ee_pose
/data_collect/vr/pose
/data_collect/vr/input
/data_collect/camera/external/image
/data_collect/camera/wrist/image
/data_collect/timestamps
```

图像是现有预处理后的 RGB，不是原始相机流。图像使用 best effort；状态使用 reliable，
均为 volatile / keep-last(1)。遥测故障关闭遥测并记录错误；机器人反馈故障则中断控制。

## 同一镜像内的依赖检查

```bash
docker compose -f src/env/compose.yaml run --rm ros2 \
  python src/env/verify_environment.py
```

该命令检查 Python 包版本约束、关键采集/推理模块导入、官方 C++ controller
动态链接和 ROS 消息 type support。它不连接 FCI，不能验证真机控制行为。

## ROS 通信测试

```bash
docker compose -f src/env/compose.yaml run --rm \
  -e DATA_COLLECT_TEST_ROS2=1 ros2 \
  python -m pytest src/tests/ros2
```

真实 DDS 测试启动 ROS 测试节点，验证 PoseStamped、Franka 状态反馈、夹爪 action、
数据录制及遥测。测试节点不具有 Franka 动力学或 FCI 实时行为，不能替代真机测试。

## 合并前镜像验证记录

最终镜像 `data-collect/franka-ros2-jazzy:dev` 已实际构建成功，
官方 FR3 单臂相关 9 个包全部编译通过，Python 依赖按原 `uv.lock` 全量安装。
在该新镜像的独立 `/opt/data_collect_venv` 中检查 196 个包的约束、采集/推理和
原 VR 适配器导入、NumPy 2.2.6 下的消息 type support 以及官方 controller
动态链接，均通过。版本为 PyTorch 2.10.0、LeRobot 0.6.1、PyArrow 23.0.0、
teleop-xr 1.3.5。CycloneDDS 下的 `src/tests/ros2` 共 44 项测试通过。

以上没有连接机器人，也没有验证真机 FCI。完整回归另有 3 项回放测试失败，涉及
测试数据复制、目录扫描和删除路径校验；其中测试数据复制问题在原分支也已复现。

## 统一配置与目录调整后的验证

镜像已按当前锁文件重新构建，包含 Teleop XR Git fork `1.3.5+csr.2` 和 Node.js 22。
在 Compose 挂载当前工作区源码的环境中，195 个已安装 Python 包的依赖检查、
采集/推理/Runner/ROS 设备模块导入、ROS 消息 type support 和官方控制器动态链接均通过。

在新镜像中，ROS 测试连同 `test_controller_hints.py` 共 73 项全部通过，包含两个
Runner 的选择与默认配置、无硬件 SDK 的构造、真实 DDS 状态和夹爪 action、
相机颜色与行尾 padding、图像时间戳和超时、麦克风 WAV 保存/丢弃/断流，
以及订阅图像和音频的完整 episode 录制。测试发布节点提供消息，没有连接实际设备。
ROS 测试目前位于 `src/tests/ros2/`，VR 提示测试位于 `src/tests/test_controller_hints.py`。

宿主机测试在目录迁移前完整回归为 220 项通过、3 项失败，失败仍是上述已知回放测试。
真机 FCI 与真实设备时钟仍待验证，官方示例控制器覆盖外部目标的问题及启动拦截保持有效。

## 5090 双 D435 收帧超时排查（2026-10-26）

本次复现的收帧超时主因已通过对照测试确认：宿主机 UDP 接收缓冲上限过小，
限制了 CycloneDDS 接收大图像。只提高 `net.core.rmem_max`，保留原线材、USB Hub、
相机配置和项目代码，真实双相机 ROS 连续采集和视频录制均通过。

测试机器为 SSH 别名 `5090`（`MS-7E06`），远端仓库为
`/home/sustechdl/three/data_collect`，分支 `dev-ros2`，提交 `b0f4673`。
环境为 Linux `6.11.0-29-generic`、现有 Docker 镜像
`data-collect/franka-ros2-jazzy:dev`、ROS 2 Jazzy 和 CycloneDDS。
只启动 `src/env/cameras.sh`，未启动机械臂 bringup 或连接真实机械臂。

### 相机与故障表现

两台 D435 分别为 external `825412070292`、wrist `825412070487`，
均使用 RGB8、640×480、30 FPS，仅开启彩色流。USB 拓扑显示两台均以
5000 Mb/s 连接，共用一个 Hub，其中腕部相机还经过一层嵌套 Hub。

驱动和以下话题能正常启动，但图像接收不稳定：

```text
/camera/external/color/image_raw
/camera/wrist/color/image_raw
```

初次测试首帧等待超时；后续连续读取虽能初始化，但约 1.2 秒后报
`ROS camera pair is stale or exceeds maximum timestamp skew`。
该次测试期间宿主机 UDP `RcvbufErrors` 增加 1707。
原有 `net.core.rmem_max` 为 `212992` 字节（208 KiB），
而一张 640×480 RGB 图像的像素数据为 `921600` 字节（900 KiB）。

### USB 与 DDS 对照结果

USB 直采使用 `pyrealsense2`，绕过 ROS；合成图像测试不打开相机，
由两个独立进程通过 DDS 发布、订阅与真实相机相同大小和速率的两路图像。
合成图像使用独立的 `ROS_DOMAIN_ID=43`，真实相机使用 Compose 默认值 42。

| 测试 | 参数 | 结果 |
| --- | --- | --- |
| USB 单台直采 | 两台分别采集 10 秒 | 各收到 287 帧，无帧号跳跃，最大帧间隔约 33.6 ms |
| USB 双台同时直采 | 原线材和 Hub，采集 30 秒 | 外部 887 帧、腕部 900 帧；无帧号跳跃，最大帧间隔分别为 36.6 / 37.3 ms |
| DDS 双路合成图像 | 原缓冲上限 208 KiB | 约 2.008 秒后超时，UDP `RcvbufErrors` 增加 2188 |
| DDS 双路合成图像 | 仅提高宿主机上限到 8 MiB | 连续 12 秒通过，各收到 361 帧，UDP 缓冲丢包为 0 |
| DDS 双路合成图像 | 上限 8 MiB，并显式请求 4 MiB CycloneDDS 接收缓冲 | 连续 30 秒通过，各收到 900 帧，UDP 缓冲丢包为 0 |
| DDS 双路合成图像 | 恢复原上限 208 KiB | 约 2.605 秒再次超时，UDP `RcvbufErrors` 增加 956 |
| 真实双相机 ROS 采集 | 仅提高宿主机上限到 8 MiB，使用默认 CycloneDDS 配置 | 连续 30 秒通过，各收到 900 帧（30 FPS），UDP 缓冲丢包为 0 |

USB 直采计数包含启动预热的影响；双台测试先启动外部相机、再启动腕部相机，
外部相机初始计数偏少，但其连续帧号没有跳跃，也没有超过 200 ms 的停顿。

提高缓冲上限后的真实 ROS 测试共完成 2964 次读取，返回的预处理图像为
224×224、连续内存的 RGB uint8。外部/腕部图像源时间戳年龄最大分别为
67.0 / 71.7 ms，两路源时间戳差最大 29.8 ms，读取耗时 p99 为 11.358 ms。
随后保留真实 ROS 相机，使用 `FakeBackend` 机械臂和 `FakeVR` 运行完整采集流程，
300 个周期完成，出现 1 次调度超期；两路 MP4 各成功保存并解码 90 帧，
同步 sidecar 包含 90 条帧记录。

### 修复方法与当前状态

已验证的最小调整是在 **5090 宿主机** 提高接收缓冲上限；Compose 使用 host 网络。
临时生效的命令为：

```bash
sudo sysctl -w net.core.rmem_max=8388608
```

需要长期保留时，在宿主机写入专用配置并加载：

```bash
sudo tee /etc/sysctl.d/99-data-collect-dds.conf >/dev/null <<'EOF'
net.core.rmem_max = 8388608
EOF
sudo sysctl -p /etc/sysctl.d/99-data-collect-dds.conf
```

设置后重启相机和采集的 ROS 进程或容器，使新建套接字按新上限申请缓冲。
`rmem_max` 是允许申请的上限，不会把现有套接字自动扩容到 8 MiB。
真实相机通过测试时未修改 `rmem_default`、`wmem_max`，也未设置额外的
`CYCLONEDDS_URI`。显式请求 4 MiB 的测试属于额外对照，并非本次最小修复的必要条件。

2026-10-26 排查结束时仅做了临时调参，已恢复 `net.core.rmem_max=212992`，
当时未安装持久化配置，测试容器和相机节点均已关闭。

2026-10-27 已在 **5090 宿主机** 安装
`/etc/sysctl.d/99-data-collect-dds.conf`（root:root，权限 0644），
并通过 `sysctl -p` 立即加载，核实当前值为 `net.core.rmem_max=8388608`。
已确认该文件出现在 systemd 的有效 sysctl 配置中，且 `systemd-sysctl.service`
属于开机 `sysinit.target` 的依赖，之后每次开机都会自动加载。
此次未重启主机或现有 ROS 进程；若相机或采集进程在加载前已运行，仍需重启它们，
使新建套接字按新上限申请缓冲。

### 线材、Hub 与尚未解决的问题

内核历史日志出现过 `uvcvideo ... Non-zero status (-71)`，对应 `EPROTO`。
此类 USB 错误可能与设备、固件、线材或控制器有关，不能仅凭该日志认定某根线材损坏。
本次短时 USB 单台、双台直采均稳定，且不用相机的合成 DDS 测试也复现了故障，
提高缓冲后原硬件链路通过。因此本次 ROS 超时应先处理 DDS 缓冲；
这些测试仍不能完全排除线材、接头或 Hub 的其他间歇性故障。

此前在同一远端执行的以下测试为 **78 通过、2 失败**（13.68 秒）：

```bash
docker compose -f src/env/compose.yaml run --rm -T \
  -e DATA_COLLECT_TEST_ROS2=1 ros2 \
  python -m pytest src/tests/ros2 src/tests/test_base_node.py -ra
```

失败项为 `test_dds_camera_audio_and_recording`（停止发布后未按预期抛出过期异常）
和 `test_stale_camera_source_times_out_and_cleans_up`（源时间戳落后 10 秒仍被接受）。
当前相机就绪检查依赖主机收包时间，处理积压消息可能刷新该时间，且未校验源时间戳年龄。
这两项属于应用层过期校验问题，本次缓冲区调整没有修复它们，测试代码也未修改。

远端原始报告、诊断脚本和 JSON/日志位于
`5090:/tmp/data-collect-camera-diagnose-2c7loyDB/`；最初失败记录位于
`5090:/tmp/data-collect-camera-check-k9X6MxrE/`。两者均为临时目录，可能被系统清理，
主要结论和修复步骤已记录于本节。

依据：[CycloneDDS 大样本接收缓冲 FAQ](https://cyclonedds.io/content/faq)、
[CycloneDDS 接收缓冲配置说明](https://github.com/eclipse-cyclonedds/cyclonedds/blob/master/docs/manual/config/config_file_reference.rst)、
[Linux USB 错误码说明](https://docs.kernel.org/driver-api/usb/error-codes.html)。
