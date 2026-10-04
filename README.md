# Panda 数据采集与回放

项目使用 **Python 3.12、LeRobot 0.6.1（最新正式版）和 v3.0 数据格式**。
依赖版本记录在 `uv.lock` 中。LeRobot 0.6 起的数据操作需要 `dataset` extra，项目已启用。

```bash
uv sync --locked
uv run vr_collect --config config/panda.yaml
bash dev.sh
```

无需硬件的采集验证（请使用单独的 `dataset.root` / `dataset.date` 配置）：

```bash
uv run vr_collect --config /path/to/test-config.yaml --dry-run --max-steps 50
uv run pytest
```

`uv sync` 会按 `.python-version` 将项目 `.venv` 切换到 Python 3.12。
采集参数集中在 `config/panda.yaml`；现有相机字段、关节/末端动作、100 Hz 轨迹与音频 sidecar 继续使用原有约定。
每次保存 episode 后会完成 LeRobot `finalize()` 并重新打开写入器，确保数据可以立即回放，也可以退出后续采。

## Legacy ASR 脚本

当前仓库不再提供需要 ASR 的功能，不安装 `qwen-asr`，也不覆盖 Transformers / Hugging Face Hub 的依赖约束。
音频采集、VAD、回放和已有标注的保留仍可使用。

以下脚本仅作为 legacy 历史代码保留，当前项目环境不支持运行；使用需要自行配置独立的 legacy 环境：

- `data_analysis/convert_audio_dataset_to_asr.py`（Whisper 转写）
- `data_analysis/convert_audio_dataset_to_asr_qwen.py`（Qwen3-ASR 转写）
- `scripts/precompute_instruction_features.py`（依赖外部 openpi 和 ASR 模型的音频特征提取）

## 旧数据

旧 v2.0/v2.1 数据仍可在回放页面读取和删除。新版 LeRobot 续采要求 v3.0；可以换 `dataset.date` 创建新数据集，
或者将旧 **v2.1** 数据转换到新目录后使用（转换保留原数据和音频/同步/轨迹 sidecar）：

```bash
uv run python -m scripts.migrate_lerobot_dataset \
  --input data/dataset/old_dataset \
  --output data/dataset/new_dataset
```

v2.1 与 v3.0 均支持质量检查和同格式数据合并。
v3 删除使用 LeRobot 官方数据工具处理共享文件，并在已有事务内重排音频与轨迹编号。
转换和合并不会上传到 Hub。真实机器人、相机和麦克风仍需在设备上验证。

上游依据：[LeRobot 0.6.1](https://github.com/huggingface/lerobot/releases/tag/v0.6.1)、
[Python 与依赖要求](https://github.com/huggingface/lerobot/blob/v0.6.1/pyproject.toml)。
