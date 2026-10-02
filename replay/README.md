# Replay 数据回放

独立的本地数据工作台：React + Vite + TypeScript + Tailwind CSS 前端，FastAPI 后端，使用 `pnpm` / `uv`，不依赖 ROS、机器人连接或 Docker。现有采集、训练环境不需要改动。

从仓库根目录运行：

```bash
./replay/dev.sh
```

脚本安装锁定依赖，首次生成两套演示数据，并同时启动前后端；打开 **http://127.0.0.1:5173**，API 文档在 **http://127.0.0.1:8000/docs**。`Ctrl+C` 关闭两个进程组。需要本机有 `uv`、`pnpm`、Node.js 20.19+ / 22.12+ 和 Linux `setsid`。

也可以分开启动，便于调试：

```bash
uv sync
env -u PYTHONPATH uv run python -m replay.scripts.generate_demo
pnpm --dir replay/frontend install --frozen-lockfile

# 终端 1，工作目录为仓库根目录
env -u PYTHONPATH uv run uvicorn replay.backend.main:app --reload --host 127.0.0.1 --port 8000

# 终端 2，工作目录为仓库根目录
pnpm --dir replay/frontend dev --host 127.0.0.1
```

## 页面操作

左侧选择数据集、episode，然后查看字段的类型、shape 和名称。将视频或 EE 数据块拖到右侧参考栏，也可用添加按钮完成同样操作。添加多个视图后，用底部统一时间轴播放、暂停或跳转，同时查看相机画面、EE 空间轨迹和 XYZ 随时间的变化。视图可以单独移除或全部清空；切换 dataset / episode 会重置工作台。

目前只启用 MP4 视频和 EE 可视化，其他字段仍完整列在数据栏中。嵌入 Parquet 的 `image`、关节、动作、力等字段暂未增加独立可视化。EE 空间图为带坐标轴的投影，时间曲线使用 episode 内的秒数，位置单位为米。

## 演示数据和解析脚本

默认数据在 `replay/demo_data/`，由脚本生成，未纳入 Git。两套数据均含三个 episode、两路实际 H.264 MP4 视频以及与本项目一致的 `ee_pose`、`joint_position`、`gripper_position`、`actions` 等字段。每个 episode 为 30 Hz、180 帧，演示场景为合成工作台，不包含真实机器人记录。

- `demo_v21`：`meta/episodes.jsonl`，每个 episode 一个 Parquet / MP4。
- `demo_v30`：`meta/episodes/...parquet`，多个 episode 共用 Parquet / MP4；episode 1、2 有非零视频偏移，用于验证正确读取和播放。

目录与元数据依据 [LeRobot 官方格式说明](https://huggingface.co/docs/lerobot/main/porting_datasets_v3)。解析器直接读取 JSON、JSONL 和 Parquet，无需安装 LeRobot / PyTorch；这是本地回放读取支持，未声称覆盖整个 LeRobot 训练 SDK。

```bash
# 生成 / 重新生成专用演示数据
env -u PYTHONPATH uv run python -m replay.scripts.generate_demo

# 查看数据集结构和字段
env -u PYTHONPATH uv run python -m replay.scripts.inspect_dataset replay/demo_data/demo_v21
env -u PYTHONPATH uv run python -m replay.scripts.inspect_dataset replay/demo_data/demo_v30 --episode 1
```

配置自己的本地数据：

```bash
# 可以指向一个数据集，或包含多个数据集的父目录
REPLAY_DATA_ROOT=/absolute/path/to/datasets ./replay/dev.sh
```

后端只允许读取配置目录内的数据，HTTP 参数不接受任意磁盘路径；不写入真实数据。`meta/info.json` 中声明的模板、v3 episode 的 chunk/file 索引和视频时间边界用于解析文件，而不是假设所有 episode 都从同一个文件的零秒开始。v2.0 / v2.1 和 v3.0 自动识别；未知版本和损坏数据会返回明确错误。

## 代码分层

```text
replay/
├── frontend/src/
│   ├── pages/         # 页面组合和工作台状态
│   ├── routes/        # 页面路由
│   ├── components/    # 数据栏、数据块、参考栏、视图卡片、视频、EE、时间轴
│   ├── hooks/         # 数据加载 / 播放逻辑
│   ├── lib/           # API 调用和纯工具
│   └── types/         # API / 界面类型
├── backend/
│   ├── api/           # HTTP 路由和依赖
│   ├── services/      # 一个公共业务函数一个文件
│   └── main.py        # FastAPI app / app factory
├── scripts/           # 双格式解析、演示生成、独立查看 CLI
├── tests/             # 先解析、后 API 验证
├── pyproject.toml     # 独立 Python 项目
└── dev.sh             # 本地启动
```

业务调用关系：路由 → 对应 service 函数 → `read_dataset` / `read_episode` / `read_ee` / `resolve_video`。视频元数据与视频文件响应分开，文件接口支持 Range 请求，供浏览器 seek；v3 播放器依据片段起止时间限制在当前 episode 内。前后端约定可见 [CONTRACT.md](CONTRACT.md)，完整接口可在 `/docs` 查看。

## 真实 Panda 小范围轨迹记录

根目录 `uv sync --locked` 配置宿主机分析环境，
`pnpm --dir replay/frontend install --frozen-lockfile` 配置前端。
真实机械臂的运动和记录均在 Humble 容器内运行。系统发起的运动统一通过
`control/robotic_arm_controller_ros.py` 的 `RoboticArmControlerRos`，
不使用 C++ 直接调用 libfranka 运动。

以下流程适用于镜像内的 Panda 旧版驱动，先在 Desk 启用 FCI、解锁机械臂，
确认周围可安全运动。在仓库根目录启动仅机器人状态的 bringup（不启用相机）：

```bash
# 终端 1
bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm --no-deps franka_humble \
  bash -c 'exec ros2 launch franka_bringup franka.launch.py robot_ip:="$FRANKA_ROBOT_IP" load_gripper:=false use_rviz:=false'

# 终端 2：配置轨迹控制器，保持未激活
bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm --no-deps franka_humble \
  ros2 run controller_manager spawner joint_trajectory_controller --inactive \
  --controller-type joint_trajectory_controller/JointTrajectoryController \
  --param-file /workspace/data_collect/ros2_ws/src/data_collect_franka/config/panda_trajectory_controller.yaml

# 只读记录 2 秒；输出文件必须是新文件
bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm --no-deps franka_humble \
  python -m replay.scripts.record_robot_trace data/replay_real/read_only.csv

# 实际运动：当前姿态下第 1 关节 +0.02 rad 后返回，8 秒运动、10 秒记录
bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh run --rm --no-deps franka_humble \
  python -m replay.scripts.record_robot_trace data/replay_real/roundtrip.csv --move

# 宿主机转换与可视化；数据集输出目录必须是新目录
env -u PYTHONPATH uv run python -m replay.scripts.import_robot_trace \
  data/replay_real/roundtrip.csv data/replay_real/roundtrip
REPLAY_DATA_ROOT="$PWD/data/replay_real" ./replay/dev.sh
```

记录脚本使用 `config/planer/panda.yaml`，通过同步轨迹执行期间的采样回调，
约 100 Hz 采样 ROS 类缓存的实测关节和末端状态；这不意味着每次采样
都有新的 ROS 消息。CSV 保留完整末端变换矩阵，转换后提供 `ee_position`、
`joint_position` 和 `actions`。`actions` 是按计划轨迹计算的名义关节目标，
不是控制器实际输出反馈。数据不包含夹爪测量和视频，不声称完成训练用采集。
页面中点击“添加 末端位置”可查看空间轨迹、XYZ 时间曲线和统一时间轴。
运动完成后脚本停用轨迹控制器；验证结束后在 bringup 终端按 Ctrl+C。

2026-10-01 真机验证数据位于 `data/replay_real/panda_small_roundtrip_20261001/`：
约 10 秒、1000 帧，末端 Y 方向范围约 7.8 mm。数据和截图在忽略的 `data/`
目录中，不纳入 Git。

## 验证

一键按顺序验证：`./replay/check.sh`。也可以先验证两种假数据的读取，再验证后端接口，最后构建前端：

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest replay/tests/test_readers.py -q
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest replay/tests/test_api.py -q
pnpm --dir replay/frontend build
env -u PYTHONPATH uv run ruff check replay
```

测试使用临时合成数据，不修改已有真实数据。命令清除宿主机的 `PYTHONPATH`，并禁用 pytest 插件自动加载，避免 ROS shell 环境影响独立 uv 项目。重点检查 v3 共享文件中的 episode 隔离、视频偏移、EE 降采样首尾点、缺失字段 / 文件、路径约束，以及视频 HTTP Range 响应。

浏览器测试会自动启动本地 API 和 Vite，验证拖拽、真实视频解码、EE 曲线/空间轨迹、共享视频片段边界、窄屏操作及错误重试：

```bash
# 默认使用本机 Google Chrome
pnpm --dir replay/frontend test:e2e

# 没有 Google Chrome 时，可安装 Playwright 的 Chromium
pnpm --dir replay/frontend exec playwright install chromium
REPLAY_BROWSER_CHANNEL=chromium pnpm --dir replay/frontend test:e2e
```
