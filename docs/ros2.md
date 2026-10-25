# ROS 2 Jazzy：Docker、FR3 遥操作与采集

本 worktree 使用 Ubuntu 24.04、ROS 2 Jazzy、系统 Python 3.12。用户已确认
使用官方 `frankarobotics/franka_ros2` 的 `jazzy` 分支及 FR3。环境位于 `src/env/`
（项目原目录拼写是 `src`）。没有自写、复制或修改 C++ controller。

命令链路：VR → 原有 `VRResampler`、`EEMapper` → `RoboticArmControler`
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

在 `robot/data_collect_ros` 下运行：

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
不要在容器内用 `uv run` 创建另一个不带 ROS 包的新环境，直接使用 `python` / `vr_collect`。

编辑 `src/env/franka.config.yaml`，确认 FR3 IP、namespace、是否加载夹爪。
同时核对 Desk 系统版本是否支持 libfranka 0.20.5；环境不会自动登录 Desk、解锁或激活 FCI。
`robot.hostname` 只供旧 SDK 后端使用，ROS 驱动连接 IP 来自 bringup 配置。

上游修复并通过 preflight 后，在第一个容器中启动：

```bash
bash src/env/bringup.sh
```

该脚本调用的官方命令为：

```bash
ros2 launch franka_bringup example.launch.py \
  controller_names:=cartesian_impedance_example_controller \
  robot_config_file:=/workspace/src/env/franka.config.yaml
```

第二个终端打开另一个容器，运行 VR 采集：

```bash
docker compose -f src/env/compose.yaml run --rm ros2 \
  python -m vr_collect --config config/train/panda_ros2.yaml
```

原 VR 的启用手势、A/B 夹爪、Y 保存、X 丢弃保持原行为。
VR 使用原 Teleop XR HTTPS 服务（4443），Docker 使用 host 网络。

本配置保留原有相机序列号，使用前改成当前连接的 RealSense。
容器共享 USB、视频、输入、串口设备；以 root 运行以便访问这些设备，但未启用 privileged。
数据默认保存在新 worktree 的 `data/`，不与原目录的数据自动混用；
可用 `ROS2_DATA_DIR=/absolute/path` 指定挂载目录。

`ROS_DOMAIN_ID` 默认 42，所有 ROS 容器与观察终端应相同。
host 网络和 host IPC 用于 Linux；Compose 配置 `SYS_NICE`、rtprio 和 memlock，
宿主机仍须具备适用于 FR3 的实时内核和实时调度权限，容器不会提供独立内核。

## 配置与消息

`config/train/panda_ros2.yaml` 使用 `robot.backend: ros2`。
原有配置保持默认 `panda` SDK 后端。
没有第二个 Python FCI 客户端，Franka 夹爪使用 ROS action，不加载 panda_py。
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
推理仅支持 EE action，使用 `config/inference/panda_ros2.yaml`：

```bash
python -m inference --config config/inference/panda_ros2.yaml --record
```

该配置禁用自动回位；策略服务地址仍由 inference.host/port 指定。

## 原网页与实时可视化

原网页仍读取 MP4、Parquet、action trace 和同步 sidecar，回放画面、末端轨迹、
标签及保存后的自动打开不依赖机械臂传输后端。原 Reactive Desk WebSocket
推送也继续接收同一种 RobotState，可用 reactive_desk.reactive_desk_enabled 控制。
浏览器不会直接订阅 DDS；ROS 遥测供 ROS 工具或后续桥接服务使用。

只看录制回放时，原前端照常运行，数据根目录指向新 worktree 的 data/dataset。
若要从网页启动采集/推理，后端必须运行在已加载 ROS 环境的容器里：

```bash
docker compose -f src/env/compose.yaml run --rm ros2 replay-server
```

Compose 分别设置 DATA_COLLECT_CONFIG、DATA_COLLECT_INFERENCE_CONFIG，
确保网页按钮使用 ROS 采集和 ROS 推理配置。容器中 uv 子进程复用现有 ROS 虚拟环境，
不重建宿主机 .venv。前端开发服务器可在宿主机运行并连接 http://127.0.0.1:8003。
网页“在真机上回放”使用 ROS EE 流，但不会自动关节回位；需先将机械臂移动到
episode 起始末端位姿的 1 cm / 0.05 rad 内，否则拒绝开始。历史纯关节轨迹仍需 FK，
当前 ROS 后端不支持这种真机回放；网页查看历史数据不受此限制。

## ROS 遥测与数据录制

原有 LeRobot 0.6.1 / v3 MP4、100 Hz action trace、音频和同步 sidecar 保留。
相机与 VR 仍由原设备适配器采样；它们的图像与输入通过独立 ROS worker 额外发布，
不会在控制线程中序列化图像。外部 ROS 输入节点不是本次实现的输入路径。

遥测由 `ros2.enabled` 控制，默认为 true；控制后端不依赖此开关。
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
  python -m pytest scripts/test/ros2
```

真实 DDS 测试启动 ROS 测试节点，验证 PoseStamped、Franka 状态反馈、夹爪 action、
数据录制及遥测。测试节点不具有 Franka 动力学或 FCI 实时行为，不能替代真机测试。

## 本次验证范围

最终镜像 `data-collect/franka-ros2-jazzy:dev` 已实际构建成功，
官方 FR3 单臂相关 9 个包全部编译通过，Python 依赖按原 `uv.lock` 全量安装。
在该新镜像的独立 `/opt/data_collect_venv` 中检查 196 个包的约束、采集/推理和
原 VR 适配器导入、NumPy 2.2.6 下的消息 type support 以及官方 controller
动态链接，均通过。版本为 PyTorch 2.10.0、LeRobot 0.6.1、PyArrow 23.0.0、
teleop-xr 1.3.5。CycloneDDS 下的 `scripts/test/ros2` 共 44 项测试通过。

以上没有连接机器人，也没有验证真机 FCI。完整回归另有 3 项回放测试失败，涉及
测试数据复制、目录扫描和删除路径校验；其中测试数据复制问题在原分支也已复现。
