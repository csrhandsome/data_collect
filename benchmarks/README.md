# LeRobot 录制与删除生命周期基准

2026-10-04，在同一台 AMD Ryzen 9 7940HX / 本地 NVMe 机器上测试，Python 3.12.3、LeRobot 0.6.1。
旧实现使用修改前保存的 `control/` 源码，录制和删除逻辑对应提交 `bbf2b10`；新实现使用当前工作区。
两版本串行运行，测试期间不同时运行项目测试集。各预热 1 次，再测 5 次，表中报告中位数。

| 项目 | 修改前 | 修改后 | 耗时下降 |
| --- | ---: | ---: | ---: |
| 保存 12 条的完整录制会话 | 11.655 s | 9.185 s | 21.2% |
| 每条 `finish()` 的平均耗时，再取各轮中位数 | 962.5 ms | 757.2 ms | 21.3% |
| 删除中间一条 episode | 2.653 s | 2.486 s | 6.3% |

录制后的数据 Parquet 分片从 12 个减少到 2 个。删除中通过 `shutil` 复制的文件从
163,599,786 字节减少到 906 字节；这不包括官方工具必须重写的 Parquet 输出。
v3 删除准备阶段因此少占用一份完整原数据集的临时空间。

删除的两组耗时区间分别为 2.474–2.818 s 和 2.470–2.635 s，存在重叠。
6.3% 是此次中位数差异，删除的稳定收益是省去前置全量复制；共享文件的重写仍然必要。

## 测试口径

- 录制：12 条 episode，每条 30 帧，224×224 RGB，项目默认 3 路 `image` 特征。
  两路合成随机图像和一路全零图像，使用实际 `EpisodeRecorder`、异步写入和 LeRobot 保存实现。
  总耗时包含创建、帧写入、每条保存以及最终 `close()` / `finalize()`，不含预先生成合成图像的时间。
  不包含真实机器人等待、相机采集、麦克风或实时轨迹采样。
- 删除：两版本读取同一份 v3 种子数据，12 条、每条 30 帧、3 路逐帧随机图像，
  共 163.60 MB，使用共享 Parquet，附带动作轨迹与同步 JSON。删除索引 6。
  构造种子、为每轮复制测试输入、测试结束清理不计时；公开删除接口内的准备、校验、
  `fsync`、并发变更检查、原子交换和旧目录清理都计时。
- 种子中的随机图像逐帧不同，避免 Parquet 字典去重把数据压成很小的样本。
- 使用本地已预热的文件缓存。这是离线读写基准，不代表真实硬件采集或 MP4 重编码的速度。

原始逐轮结果：[修改前](lerobot_lifecycle_before.json)、[修改后](lerobot_lifecycle_after.json)。

## 复现

在仓库根目录执行。基准只操作选定工作目录内的合成数据，不读取真实采集数据。

```bash
mkdir -p /tmp/lerobot-before
git archive bbf2b10 control | tar -x -C /tmp/lerobot-before

PYTHONPATH="/tmp/lerobot-before:$PWD" \
HF_HOME=/tmp/lerobot-benchmark-hf \
HF_DATASETS_CACHE=/tmp/lerobot-benchmark-hf/datasets \
HF_HUB_OFFLINE=1 \
.venv/bin/python scripts/benchmark_lerobot_lifecycle.py \
  --label before --unique-seed-images \
  --work-dir /tmp/lerobot-comparison \
  --output /tmp/lerobot-before.json

HF_HOME=/tmp/lerobot-benchmark-hf \
HF_DATASETS_CACHE=/tmp/lerobot-benchmark-hf/datasets \
HF_HUB_OFFLINE=1 \
.venv/bin/python -m scripts.benchmark_lerobot_lifecycle \
  --label after --unique-seed-images \
  --work-dir /tmp/lerobot-comparison \
  --output /tmp/lerobot-after.json
```

基准参数可以用 `--episodes`、`--frames`、`--image-hw` 和 `--repeats` 调整。
修改数据参数时须换一个 `--work-dir`，以避免误用旧种子。

## 回归验证

完整测试集：**214 passed**，其中 `replay/tests` 的 116 项覆盖后端 HTTP、操作管理、
视频/图像、桌面运行及数据读取。测试使用临时数据和模拟硬件。
后端 `TestClient` 在当前沙箱内无法启动跨线程事件循环，因此完整测试集在沙箱外执行。
有一条上游 Starlette TestClient 弃用提示，无失败或跳过的测试。

新增回归覆盖连续保存复用写入器、显式封口、删除后续采、关闭幂等、拒绝封口活跃 episode、
v3 附件与 VAD/ASR 标注保留、准备/复制/校验/提交失败保持原数据完整，以及并发变更拒绝提交。
已有的旧数据转换续采测试继续通过：历史统计字段兼容处理移到会话封口时执行。
