# Replay 数据回放

本地机器人数据工作台：React + Vite + TypeScript + Tailwind CSS 前端，FastAPI 后端，使用 `pnpm` / `uv`。数据查看不依赖机器人连接、ROS 或 Docker；开启采集与真机回放需要本机设备环境。

从仓库根目录运行：

```bash
bash dev.sh
```

脚本安装依赖并同时启动前后端，默认扫描 `data/dataset/` 下的数据集；打开 **http://127.0.0.1:5173**，API 文档在 **http://127.0.0.1:8003/docs**。`Ctrl+C` 关闭两个进程组。需要本机有 `uv`、`pnpm`、Node.js 20.19+ / 22.12+ 和 Linux `setsid`。

也可以分开启动，便于调试：

```bash
uv sync
pnpm --dir replay/frontend install --frozen-lockfile

# 终端 1，工作目录为仓库根目录
env -u PYTHONPATH uv run replay-server

# 终端 2，工作目录为仓库根目录
pnpm --dir replay/frontend dev --host 127.0.0.1
```

## 页面操作

打开页面时自动扫描数据目录，左侧的数据集下拉框列出所有包含 `meta/info.json` 的直接子目录。默认目录为 `data/dataset/`，例如 `data/dataset/franka_lerobot_7_6_audio/meta/info.json`。放入新数据集或采集结束后，点击「重新扫描」更新列表和片段信息，无需重启服务。再选择数据集和 episode，查看字段的类型、shape 和名称。

采集保存路径也默认为 `data/dataset/<数据集名称>/`，由 `config/train/panda.yaml` 的 `dataset.root` 配置；`openpi/franka_lerobot` 这样的 repo ID 只使用最后一段作为本地目录名称。回放目录由同一配置文件的 `replay.data_root` 配置，`REPLAY_DATA_ROOT` 可以覆盖它。已有 `data/openpi/` 数据已迁移到 `data/dataset/`。

将视频或 EE 数据块拖到右侧参考栏，也可用添加按钮完成同样操作。添加多个视图后，用底部统一时间轴播放、暂停或跳转，同时查看相机画面、EE 空间轨迹和 XYZ 随时间的变化。视图可以单独移除或全部清空；切换 dataset / episode 会重置工作台。

「采集与操作」栏提供：

- **开启采集**：后端在仓库根目录运行 `uv run vr_collect`，使用 `config/train/panda.yaml` 的设备、采集任务及保存目录。左侧选择的数据集不会改变采集保存位置。
- **真机回放上一次**：回放当前数据集最大 `episode_index` 的已保存片段，与当前查看的 episode 无关。优先使用 100 Hz `actions.jsonl` 的实测位姿与夹爪记录，随后尝试同步帧记录、Parquet 的完整 `ee_pose` 或 `joint_position`。关节记录经控制器运动学转换为 EE 轨迹；回放实测路径。夹爪动作期间暂停回放时钟。
- **删除当前片段**：调用已有按编号删除脚本，硬删除当前片段及其图像、视频、音频、100 Hz 记录和同步文件，重排后续文件、Parquet 索引及元数据。支持 v2.0/v2.1，以及通过新版 LeRobot 官方工具重写共享文件的 v3.0。

这些操作先展示确认弹窗，再提交任务。任务进程由后端管理，浏览器关闭或刷新不会结束任务。一次只运行一个任务；等待区显示耗时、状态和最近 80 行日志。采集/真机回放可点击「停止任务」，以 SIGINT 请求清理设备及保存当前采集片段；删除更新文件期间不能中途停止。任务结束会自动刷新数据并清空旧视图。后端重启不会保留任务历史；停止任务后应等待状态确认再关闭服务。不要同时从终端启动另一个采集或推理进程。

删除脚本先在数据集旁的临时目录复制完整数据，完成重编号、更新 VAD 切片路径及 ID、校验 Parquet 索引和 sidecar 引用，再通过 Linux `renameat2(RENAME_EXCHANGE)` 原子替换整个目录。提交前复制、写入、校验失败或进程中断时，原数据不变；正常完成会清理临时目录。需要数据集所在磁盘有足够容纳完整副本及 Parquet 重写的剩余空间，并支持原子目录交换。强制杀进程可能留下 `.<数据集名>.delete-*` 临时目录；确认没有删除任务运行后可清理。删除脚本通过锁避免重复删除，并检测准备期间原文件发生变化；删除期间仍应暂停其他采集或数据修改进程。

统一真机回放入口（已移除三个重复的 `replay_openpi_*` 查看包装脚本及旧 `replay_openvla_hdf5.py`）：

```bash
uv run python -m replay.scripts.replay_robot --dataset data/dataset/<数据集名称>
# 模拟机械臂；不连接硬件、不改写数据
uv run python -m replay.scripts.replay_robot --dataset data/dataset/<数据集名称> --dry-run
# 指定片段、降低回放速度
uv run python -m replay.scripts.replay_robot --dataset data/dataset/<数据集名称> --episode-index 0 --speed 0.5
```

`WaitingState` 是通用黑白等待组件，支持紧凑布局、说明文字、计时及操作插槽；数据读取、提交和长任务共用它，系统开启减少动画时停止旋转。

支持 MP4、嵌入 Parquet 的图像、音频、EE 和数值曲线。其他字段仍完整列在数据栏中。EE 空间图为带坐标轴的投影，时间曲线使用 episode 内的秒数，位置单位为米。

## 前端样式

界面采用黑白编辑风格，布局、响应式断点和交互状态直接写在各个 React 组件的 Tailwind CSS 4 类名中。`frontend/src/styles.css` 只保留 Tailwind 入口、`@theme` 设计 token 和全局基础规则，不放页面或组件选择器。通用按钮与纹理由 `components/ui/Button.tsx`、`Texture.tsx` 复用。

Playfair Display、Source Serif 4 和 JetBrains Mono 通过 Fontsource 随应用打包，中文使用系统衬线字体回退。图表以实线、虚线和点线区分数据维度；相机图像、视频保留原始颜色用于数据判读。

## 演示数据和解析脚本

演示数据在 `replay/demo_data/`，需要手动生成，未纳入 Git。两套数据均含三个 episode、两路实际 H.264 MP4 视频以及与本项目一致的 `ee_pose`、`joint_position`、`gripper_position`、`actions` 等字段。每个 episode 为 30 Hz、180 帧，演示场景为合成工作台，不包含真实机器人记录。

- `demo_v21`：`meta/episodes.jsonl`，每个 episode 一个 Parquet / MP4。
- `demo_v30`：`meta/episodes/...parquet`，多个 episode 共用 Parquet / MP4；episode 1、2 有非零视频偏移，用于验证正确读取和播放。

目录与元数据依据 [LeRobot 官方格式说明](https://huggingface.co/docs/lerobot/main/porting_datasets_v3)。解析器直接读取 JSON、JSONL 和 Parquet，无需安装 LeRobot / PyTorch；这是本地回放读取支持，未声称覆盖整个 LeRobot 训练 SDK。

```bash
# 生成 / 重新生成专用演示数据
env -u PYTHONPATH uv run python -m replay.scripts.generate_demo

# 使用演示数据启动回放
REPLAY_DATA_ROOT="$PWD/replay/demo_data" bash dev.sh

# 查看数据集结构和字段
env -u PYTHONPATH uv run python -m replay.scripts.inspect_dataset replay/demo_data/demo_v21
env -u PYTHONPATH uv run python -m replay.scripts.inspect_dataset replay/demo_data/demo_v30 --episode 1
```

配置自己的本地数据：

```bash
# 可以指向一个数据集，或包含多个数据集的父目录
REPLAY_DATA_ROOT=/absolute/path/to/datasets bash dev.sh
```

后端只允许通过配置目录内的数据集 ID 访问数据，HTTP 参数不接受任意磁盘路径或命令。读取接口不写数据；删除需在页面确认后执行。`meta/info.json` 中声明的模板、v3 episode 的 chunk/file 索引和视频时间边界用于解析文件。v2.0 / v2.1 和 v3.0 自动识别；未知版本和损坏数据会返回明确错误。

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
└── pyproject.toml     # 独立 Python 项目
```

本地启动脚本位于仓库根目录 `dev.sh`。

业务调用关系：路由 → 对应 service 函数 → `read_dataset` / `read_episode` / `read_ee` / `resolve_video`。视频元数据与视频文件响应分开，文件接口支持 Range 请求，供浏览器 seek；v3 播放器依据片段起止时间限制在当前 episode 内。前后端约定可见 [CONTRACT.md](CONTRACT.md)，完整接口可在 `/docs` 查看。

相机字段为 `dtype=image` 时，首次打开会把当前 episode 的 PNG / JPEG 图像转换为 H.264 MP4，缓存到数据集内的 `.replay-cache/video/`，随后与已有 MP4 共用原生 `<video>` 播放器和 HTTP Range 接口。使用已安装的 `imageio-ffmpeg`，无需另装系统 FFmpeg。转换保留原始分辨率（奇数边长补齐一个像素）、按采集同步时间在相机 FPS 上重采样，使用 CRF 18、每秒关键帧及 faststart；它是用于查看的有损副本，原始 Parquet 和图像保留。无同步记录时按数据集 FPS 播放。

只有首次读取或数据变化后需要转换；缓存命中不再读取相机 Parquet 列或编码。源 Parquet、元数据、片段边界、同步记录或引用图片改变时自动重建。同一相机的并发请求共用一次转换，完成后原子发布，失败可重试；重建时删除该相机旧 MP4 缓存。数据集目录需要可写，`.replay-cache/` 可删除并自动再生成。

原先的 `ImageStream.tsx`、`read_image.py` 和 `/image`、`/image/metadata` 接口已移除。播放中由浏览器持续解码，只在用户定位、播放/暂停或缓冲恢复时设置视频时间；数据栏通过 React memo 避免跟随时间轴反复渲染。

也可提前转换，省去首次打开的等待（省略 `--episode` 会处理所有片段，可重复指定 `--episode` / `--feature`）：

```bash
uv run python -m replay.scripts.prepare_video data/dataset/franka_lerobot_7_6_audio --episode 0
```

页面启动后，可用新脚本测量三路视频的实际呈现帧率、解码掉帧、seek、缓冲和请求数量；默认连续测 8 秒，片段需长于 9 秒。脚本使用本机 Chrome，低于源 FPS 的 85%、掉帧超过 5%、反复 seek 或发生缓冲时返回失败。短演示数据可用 `--seconds 4` 和 `--cameras camera_png,camera_jpeg`：

```bash
node replay/frontend/scripts/benchmark-video.mjs --dataset franka_lerobot_7_6_audio \
  --output data/replay_benchmarks/playback.json
# 增加每个请求的延迟，验证播放不会被逐帧请求拖慢
node replay/frontend/scripts/benchmark-video.mjs --dataset franka_lerobot_7_6_audio \
  --latency-ms 120 --output data/replay_benchmarks/latency120.json
```

改前/改后实测与测试范围见 [视频播放验证记录](VIDEO_PLAYBACK_TEST.md)。

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
REPLAY_DATA_ROOT="$PWD/data/replay_real" bash dev.sh
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
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest replay/tests/test_api.py replay/tests/test_images.py -q
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest replay/tests/test_operations.py tests/test_episode_deleter.py -q
pnpm --dir replay/frontend build
env -u PYTHONPATH uv run ruff check replay
```

测试使用临时合成数据，不修改已有真实数据。命令清除宿主机的 `PYTHONPATH`，并禁用 pytest 插件自动加载，避免 ROS shell 环境影响独立 uv 项目。重点检查 v3 共享文件中的 episode 隔离、视频偏移、EE 降采样首尾点、缺失字段 / 文件、路径约束，以及视频 HTTP Range 响应。

浏览器测试会自动启动本地 API 和 Vite，验证拖拽、真实视频解码、EE 曲线/空间轨迹、共享视频片段边界、窄屏操作及错误重试：

```bash
# 默认使用本机 Google Chrome；自动生成 MP4 和嵌入 PNG/JPEG 演示数据
pnpm --dir replay/frontend test:e2e

# 没有 Google Chrome 时，可安装 Playwright 的 Chromium
pnpm --dir replay/frontend exec playwright install chromium
REPLAY_BROWSER_CHANNEL=chromium pnpm --dir replay/frontend test:e2e
```
