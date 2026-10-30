# ROS 2 Jazzy：Panda Docker、遥操作与采集

当前机器人是 Panda / FER。镜像使用 Ubuntu 24.04、ROS 2 Jazzy、系统 Python 3.12，
固定 `mul-cps/franka_ros2` 的 FER 适配提交 `c0e1f8206211` 和
`mul-cps/libfranka` 提交 `55e4ab7654`（0.9.2 协议）。不要替换为面向新协议的 libfranka 0.20.x。
控制器是项目内的 `panda_cartesian_controller/CartesianPoseImpedanceController`，
只跟踪外部 `equilibrium_pose`，不生成示例轨迹。

命令链路：VR → `VRResampler`、`EEMapper` → `FrankaRunner` → `Ros2Backend`
→ `geometry_msgs/PoseStamped` → C++ Cartesian impedance controller → libfranka / FCI → Panda。
Python 默认 100 Hz，底层控制循环为 1 kHz。
`bringup.sh` 在操作 Desk 前执行只读 controller preflight；这不能代替真机目标跟踪验证。

## 构建与启动

启动分工如下：先在独立容器终端运行 `bash scripts/deploy_ros2.sh`，负责
Desk 解锁/FCI、机械臂驱动、相机、VR Server 和诊断，之后保持该终端运行。

另一个宿主机终端中，`bash scripts/deploy_runner.sh` 只启动监听 ROS2 话题的采集 runner；
需要网页时改用 `bash scripts/deploy_page.sh`，它启动 runner、API 和网页。
这两个 runner 入口二选一，它们只连接已有的 ROS2、相机和 VR 话题。
网页前端运行在宿主机，API 与采集任务运行在独立 ROS 环境容器内，共用 host 网络及 domain ID。
Ctrl+C 结束 runner 时会保存采集，单独启动的 `deploy_ros2.sh` 继续运行。
`bash scripts/deploy_runner.sh --max-steps 500` 可限制 runner 的循环步数。

在仓库根目录下运行，`config/train/panda.yaml` 设置 `robot.backend: ros2`。
机器人 IP 以 `src/env/franka.config.yaml` 的 `robot_ip` 为准，Desk 与 ROS 连接同一地址。
构建后，第一个终端进入容器并启动：

```bash
docker compose -f src/env/compose.yaml build
bash scripts/deploy_docker.sh
# 以下在容器内执行：
bash scripts/deploy_ros2.sh
```

脚本通过独立的 `src/env/desk_bringup.py` 直接向 Desk 发送 HTTPS 请求，
完成登录、获取控制权、解锁制动器、激活 FCI，不导入 panda-py。
再启动 ROS 控制器。Desk 会话保持到 ROS 退出，之后停用 FCI、释放控制权并登出，
不自动锁回制动器。运动和状态连接由 ROS 驱动持有。Desk 客户端使用 Python 标准库实现 HTTP 通信，
配置解析使用项目已有的 PyYAML。控制 token 以 0600 权限保存在共享的
`data/runtime/desk_tokens.json`，容器重启后可恢复本脚本原有的控制权。
相机、VR 节点和订阅诊断也由部署脚本启动，退出时一起清理。

凭据默认读取 `DATA_COLLECT_CONFIG`（缺省 `config/train/panda.yaml`，支持 `extends`）
中的 `robot.username` / `robot.password`。环境变量 `FRANKA_USERNAME` / `FRANKA_PASSWORD`
优先，两种 Docker 入口都会传入这些变量。不要把密码写到命令行参数中。
单独执行 `bash src/env/bringup.sh` 也会进行相同的 Desk 准备。
`robot_config_file:=/path/to/config.yaml` 覆盖后，Desk 会使用该文件的 `robot_ip`。

如果要手动操作 Desk：

```bash
ROS2_DESK_AUTO_UNLOCK=0 bash scripts/deploy_ros2.sh
```

手动模式需要提前解锁并激活 FCI。自动模式也不能解除实体急停/用户停止按钮，
不能强制抢占其他 Desk 会话；遇到控制权占用时，先在 Desk 释放控制权再重试。
`Other` 是拒绝启动时的机器人模式，日志不能仅凭此值区分制动器、FCI 或其他状态；
`Move command rejected ... Did you open the brakes?` 才是此次退出的直接原因。
控制器显示 `activated` 只代表 ROS 生命周期切换完成，不代表机器人已接受力矩控制。

第二个终端：

```bash
bash scripts/deploy_docker.sh
# 容器内：
python -m vr_collect --config config/train/panda.yaml
```

也可以保持第一个 ROS2 终端运行，在另一个宿主机终端运行 `deploy_runner.sh`；
需要网页时用 `deploy_page.sh`，不要再重复运行 `vr_collect`。VR 使用 Teleop XR HTTPS 服务（4443）。
两路 D435 的序列号配置在 `src/env/cameras.sh`，采集端只订阅相应 topic。
Docker 使用 host 网络和 IPC、`SYS_NICE`、rtprio 与 memlock，不启用 privileged。
数据默认挂载到仓库 `data/`，也可通过 `ROS2_DATA_DIR=/absolute/path` 设置。
`ROS_DOMAIN_ID` 默认 42，所有容器应一致。宿主机需要实时内核；容器共享宿主机内核。
当前日志的非实时内核警告不是 `Other` 模式拒绝命令的直接原因。

Docker 用系统 Python 3.12 创建 `/opt/data_collect_venv` 并保留 ROS 系统包路径，
不复用宿主机 `.venv`。入口已设置 `UV_PROJECT_ENVIRONMENT` 和 `UV_NO_SYNC=1`。
宿主机未加载 ROS 时，部署脚本会在启动后台进程前退出并提示先进入 Docker。

## 订阅监控与前端 warning

部署入口和数据脚本分开放置：

```text
scripts/
  deploy_docker.sh       # 在宿主机进入容器
  deploy_ros2.sh         # 在容器内启动 ROS2 环境和诊断桥接
  deploy_runner.sh      # 只启动监听已有 ROS2 话题的 runner
  deploy_page.sh         # 在宿主机启动完整 runner、API 和网页
  dataset/              # 原有数据处理、上传和下载脚本
```

先在独立容器终端执行 `bash scripts/deploy_ros2.sh`，保持其运行；
另一个宿主机终端执行 `bash scripts/deploy_page.sh`，打开页面顶部的「ROS2 订阅监控」。
只需要监听采集、不需要网页时，第二个终端改用 `deploy_runner.sh`。

监控配置继承自 `config/train/panda.yaml`：

```yaml
ros2:
  monitoring:
    enabled: true
    startup_grace_s: 10.0
    window_s: 5.0
    min_rate_ratio: 0.8
```

现有 `Subscription` 统一包装真正的机械臂、VR、相机、麦克风和 Franka 夹爪
订阅回调，不新增图像或机器人状态订阅。每秒发布标准
`diagnostic_msgs/msg/DiagnosticArray` 到 `/diagnostics`。
`diagnostics.py` 负责统计与诊断，`diagnostics_bridge.py` 负责发现驱动、汇总诊断和记录恢复事件，
shell 只负责进程生命周期。没有开启的设备不列为缺失。

频率统计使用接收侧单调时钟。相机以 `camera.fps`、VR 以发布心跳 100 Hz、
机器人状态以控制循环 `control.frequency_hz` 为最低需求基准，乘以 `min_rate_ratio`
后作为频率阈值。机器人状态不要求完整消费驱动 1 kHz 的每个消息。
音频块长度可变、夹爪发布频率未固定，因此这两项只检查超时和无效数据。
带 header 的消息还检查源时间戳；VR 同时检查头显是否继续更新，
避免 ROS 心跳正常但输入已经冻结时显示为正常。

以下情形只生成监控 warning：启动宽限时间结束仍未发现发布节点、
订阅没有消息或消息中断、频率低于阈值、无效数据，以及订阅进程诊断心跳中断。
状态恢复会记录恢复事件，同一持续异常不反复刷屏。`depth=1` 与状态缓存
正常覆盖中间样本不会直接计为丢帧，监控也不声称提供精确丢帧数量。
监控不会关闭控制器或终止采集；原有反馈过期、运动限幅和录制完整性保护继续生效。

诊断桥接只订阅小型诊断消息，并使用 ROS graph 查询发布节点是否存在。
采集尚未启动时显示「发布节点在线，等待采集订阅」，发现发布节点不等于已验证数据正常。
实际订阅的接收状态以其自身诊断为准；订阅心跳超过 3 秒未更新会 warning。

桥接进程每秒原子更新 `data/runtime/ros2_health.json`，FastAPI 的
`GET /api/ros2/health` 读取该快照，前端每秒刷新。此文件位于 Docker 与宿主机
共享数据目录，网页后端无需安装 ROS。监控快照超过 5 秒未更新时显示「监控未连接」，
保留最后记录供排查，但不把旧的正常状态显示为当前正常。
最近 100 个状态变化保留在内存和快照中，前端展示最近 10 条，重启监控后重新开始记录。

`ROS2_DATA_DIR` 自定义数据目录时，两个部署脚本会使用对应的共享路径；
单独运行网页后端或跨机器部署时，可通过 `ROS2_HEALTH_PATH` 指定同一份状态快照。
这是当前单机 Docker 工作方式的轻量桥接；跨主机且没有共享目录时，需要另行配置传输。

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
启用采集端 FCI 自动激活或关节回位；Desk 准备由部署脚本负责。`inference.move_to_start` 仍保持默认 false。

Docker 按根目录 `pyproject.toml` 和 `uv.lock` 安装依赖，包含 `panda-python`；
选择 ROS 后端时没有第二个 Python FCI 客户端，Franka 夹爪使用 ROS action。部署阶段使用独立的 Desk HTTP 客户端管理网页会话，也不加载 panda_py。
镜像提供 Node.js 22 和 npm，用于构建当前锁定的 Teleop XR Git fork 的 WebXR 页面。
DH5 如启用，仍由原串口后端访问本地设备，触觉图像沿用原录制接口。

| 方向 | 默认 topic / action | 类型 |
| --- | --- | --- |
| Python → controller | `/cartesian_pose_impedance_controller/equilibrium_pose` | `geometry_msgs/PoseStamped` |
| broadcaster → Python | `/franka_robot_state_broadcaster/robot_state` | `franka_msgs/FrankaRobotState` |
| Python → Franka gripper | `/franka_gripper/move`、`/franka_gripper/grasp` | `franka_msgs/action/Move`、`Grasp` |
| Python → gripper stop | `/franka_gripper/stop` | `std_srvs/Trigger` |
| gripper → Python | `/franka_gripper/joint_states` | `sensor_msgs/JointState` |

这里 Python **发布** equilibrium_pose，官方 controller **订阅**。

末端外力估计可在 `config/train/panda.yaml` 或其子配置中按需开启（默认关闭）：

```yaml
observation:
  ee_wrench: true
```

此字段仅支持 `robot.backend: ros2`。Python 复用现有
`/franka_robot_state_broadcaster/robot_state` 订阅，读取 `o_f_ext_hat_k`；
不新增 FCI 连接，也不改变 EE 目标位姿控制。数据为刚度坐标系 K 原点处、
沿机器人基坐标系轴表达的外力／力矩估计，顺序为
`[Fx, Fy, Fz, Tx, Ty, Tz]`，单位为 N、N·m。它不是策略预测的目标力。
字段开启时要求六个数均有限；缺失或非法消息不会刷新反馈，持续异常会触发原有过期保护。

开启后，LeRobot 保存 `observation.ee_wrench`（`float32[6]`），
控制轨迹和同步 sidecar 同时保存 `ee_wrench`，推理请求携带同名 observation 字段。
额外遥测启用（`ros2.enabled: true`）时，还会发布
`/data_collect/robot/ee_wrench`（`geometry_msgs/msg/WrenchStamped`），
header 使用与实测 EE 位姿相同的基坐标系和反馈接收时间。
外部节点可用 `ros2 topic echo /data_collect/robot/ee_wrench` 订阅查看。
`ros2.enabled: false` 仅关闭额外遥测，仍可读取和保存外力。
关闭 `ee_wrench` 后不读取、不保存，也不创建该遥测 publisher；Franka 驱动原有状态发布继续运行。
开启会改变数据集 schema，续录已有数据集时需选择新的 `dataset.date` 或数据目录。

目标必须是 `panda_link0` 坐标系，位置单位 m，四元数顺序 xyzw，消息带 ROS 时间戳。
frame_id 是校验标签，代码不会自动做 TF 变换。
ROS namespace 在 `franka.config.yaml` 和 Python `ros2.namespace` 中同步设置；
配置的 endpoint 会被相对该 namespace 解析。

反馈使用状态消息中的真实 `q`、`dq` 和 `o_t_ee`，关节数组按 1…7 排列。
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
