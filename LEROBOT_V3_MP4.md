# LeRobot v3 MP4 数据说明

本项目统一使用 LeRobot 0.6.1 / 数据格式 v3.0，视觉观测固定存为 MP4。
采集、回放、合并和按 episode 删除使用同一格式；不再提供 PNG 内嵌、旧格式读取或迁移分支。

## 当前数据

以下 8 个数据集保留在 `data/dataset/`，音频、同步记录、动作轨迹及已有标注一并保留。
合计 732 条 episode、279,199 帧；数量包含 ASR/test 衍生副本，未按内容去重。

| 数据目录 | episode | 帧数 |
|---|---:|---:|
| `franka_lerobot_6_19_audio_asr_v3_mp4` | 119 | 42613 |
| `franka_lerobot_6_19_audio_v3_mp4` | 119 | 42613 |
| `franka_lerobot_6_23_audio_v3_mp4` | 50 | 13394 |
| `franka_lerobot_6_24_audio_v3_mp4` | 40 | 15327 |
| `franka_lerobot_6_28_audio_test_v3_mp4` | 81 | 30600 |
| `franka_lerobot_6_28_audio_v3_mp4` | 81 | 30600 |
| `franka_lerobot_7_6_audio_qwen_asr_v3_mp4` | 121 | 52026 |
| `franka_lerobot_7_6_audio_v3_mp4` | 121 | 52026 |

旧数据和 PNG 对照副本共 16 个目录已删除；保留目录继续使用 `_v3_mp4` 后缀。

## 存储与采集

状态和动作存于 `data/chunk-xxx/file-xxx.parquet`；各相机视频存于
`videos/<camera>/chunk-xxx/file-xxx.mp4`。episode 的全局行区间、文件位置和视频起止时间保存在
`meta/episodes/chunk-xxx/file-xxx.parquet`，任务和统计分别使用 `tasks.parquet` 与 `stats.json`。

`config/train/panda.yaml` 的 `dataset.video_encoder` 控制视频编码，默认 H.264、CRF 18、GOP 2、fast。
相机 schema 固定为 `dtype: video`，已移除 `image_storage` 切换选项；创建新数据集时使用新的 `dataset.date`。
MP4 是有损编码；像素统计保留编码前采集图像的统计。录制结束仍需 finalize，确保 Parquet 文件完整封口。

## 此前已完成的性能对比

以下结果保留自本次清理前已完成的对照，不是本次重新运行测试得到的结果。
同一批数据的 v3 PNG 与 v3 MP4 data/video 体积为 **16.13 GiB → 2.52 GiB，减少 84.4%**；不含音频等附加文件。

环境：LeRobot 0.6.1、PyTorch 2.10.0、TorchCodec 0.10.0、PyAV 15.1.0；
实际 LeRobotDataset + DataLoader，CPU 解码、float32 输出，batch=16，采样 256 个样本，重复三次取中位数。
表中为 8 个目录各自吞吐比的几何均值，格式执行顺序交替，worker 启停不计入稳态吞吐。

| 采样方式 | workers | MP4 / PNG 吞吐比 | 各目录最小～最大 |
|---|---:|---:|---:|
| random | 0 | 1.42× | 1.32～1.54× |
| random | 2 | 1.21× | 1.10～1.26× |
| sequential | 0 | 1.53× | 1.41～1.66× |
| sequential | 2 | 1.14× | 0.82～1.35× |
| window3 | 0 | 6.12× | 5.74～6.48× |
| window3 | 2 | 4.62× | 4.16～4.86× |

随机单帧读取约 1.21×、三帧窗口约 4.62×（2 workers）；连续读取有目录降至 0.82×，不能认为 MP4 必然更快。
window3 是每路相机的前/当前/后一帧，越界由 loader padding。未清空系统缓存，也没有计入模型前向、反向或 GPU，
因此这些数字只代表本机数据加载吞吐，不能当作完整训练步的加速比例。结果仅适用于上面的 TorchCodec 路径。

本次按要求停止新增测试；已删除 benchmark 脚本、日志、逐次 JSON/Markdown 及旧生命周期报告，结果仅保留在本文件。
