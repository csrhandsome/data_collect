# VR 跟手诊断

入口：`uv run vr_collect.py`。日志与数据集录制独立，未开始 episode 或
`dataset.enable_logging: false` 时也可以记录。

在 `config/panda.yaml` 中打开：

```yaml
diagnostics:
  enabled: true
  root: data/diagnostics
  sample_hz: 0.0
  max_pending: 1024
```

`sample_hz: 0` 记录每个控制周期；改成 `10` 表示最多每秒记录 10 个周期。
每次运行生成一个独立的 `vr_control_*.jsonl`，终端会打印完整路径。
JSON 编码和写盘在后台线程中完成；队列满时丢弃诊断记录并统计数量，控制线程不等待写盘。
退出时尝试排空队列，文件最后的 `diagnostics_summary` 给出实际写入数量和丢弃数量。

## 与 stable 的差异

实测可用的基准是提交 `fe1b39b` 中的
`vr_openpi_lerobot_joint_audio_with_ee.py`。默认控制频率是 20 Hz，相机为 30 Hz。
stable 从上一次命令目标累计 VR 位移。此前新版在每个周期都以实际位姿为参考限速，
导致在 100 Hz、0.2 m/s 配置下，目标只能领先实际位置约 2 mm；30 Hz 时约 6.7 mm。
机械臂反馈稍有滞后，目标就无法继续推进。

当前映射以此前命令目标为限速参考，保留以 m/s、rad/s 表示的速度限制、工作空间限制
和发送目标时的实测误差保护。提高控制频率会缩小每步增量，而不会缩小累计目标。
空闲时保持固定目标，松开扳机或 VR 过期时重新锁存实际位姿；夹爪暂停时清除未发送的移动。

两版阻抗滤波系数默认均为 0.35，VR 输入平移/旋转滤波参数均为 0.6/0.35。
stable 还额外进行了一层平移滤波。新版输入使用共享内存的最新快照，不逐帧排队消费。

## 关键字段

| 字段 | 用途 |
| --- | --- |
| `vr.raw_position` / `vr.filtered_position` | 比较输入滤波前后的平移；四元数也分别记录 |
| `vr.age_ms` | 本地收到该 VR 更新到本轮生成日志之间的时间 |
| `vr.source_seq` / `new_sample` / `skipped_samples` | 判断输入重复、更新与跳帧；跳帧表示取了最新快照 |
| `vr.observed_input_hz` | 由相邻读取的序号和本地接收时间估计输入频率 |
| `vr.raw_enabled` / `enabled` | 比较扳机状态与经过过期检查后的使能状态 |
| `mapping.desired_position` / `desired_quaternion_xyzw` | 应用缩放、工作空间限制之后，速度限制之前的目标 |
| `mapping.*_speed_limited` / `workspace_limited` | 判断速度和工作空间限制是否生效 |
| `command.position` / `quaternion_xyzw` / `sent` | 本轮计算的命令目标及是否实际发送；夹爪忙时可能不发送 |
| `command.tracking_error_m` / `tracking_error_rad` | 命令目标与本轮实测位姿之间的误差 |
| `command.*_speed_*` | 相邻实际发送目标之间的变化速率 |
| `robot.ee_position` / `ee_quaternion_xyzw` / `joint_velocity` | 实测末端位姿和关节速度 |
| `robot.robot_time_s` | 机器人自身时间；用于检查机器人状态是否持续更新 |
| `robot.robot_mode` / `current_errors` | 检查运行模式和当前控制错误 |
| `robot.control_command_success_rate` | SDK 报告的控制命令通信成功率 |
| `timings_ms` | 状态读取、VR 读取、映射、场景发布、命令/夹爪、录制/相机的耗时 |
| `tick_interval_ms` / `work_ms` / `overruns` | 控制周期、工作耗时和错过周期的累计次数 |

日志开头保存本次控制、VR、相机相关配置，不保存机器人登录凭据。
`stream_start`、`stream_reset`、`interrupt`、`error`、`loop_end` 记录重要事件。
诊断数据采用米、弧度、xyzw 四元数；`*_ms` 单位为毫秒。

如果 `vr.age_ms` 很小、`vr_read` 很短而目标依然迟缓，应看限速和目标跟踪误差；
若输入年龄持续增长，则检查 VR 连接或输入进程。若机器人时间不再变化、运行模式异常
或通信成功率下降，则检查底层控制和实时调度。
VR 时间是电脑本地接收时间，并非头显发包时间；Python 周期日志也不直接测量全部 1 kHz 周期。
模拟后端没有真实机器人时钟和通信状态，这些字段为 `null`。
