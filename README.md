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

## Qwen3-ASR 共用环境

Qwen3-ASR 子模块仍固定 Transformers 4.57.6，该版本的 `huggingface-hub<1` 限制不仅在包声明里，也在导入时检查。
项目通过 `[tool.uv].override-dependencies` 放宽 Hub 约束，保留 Qwen 原有的 **Transformers 4.57.6**，共用 LeRobot 所需的 Hub 1.x。
全部依赖安装在同一个 `.venv`，Qwen 子模块的依赖文件保持原样。
项目的 LeRobot / Qwen / Whisper 加载器同步放宽 Transformers 的运行时 Hub 上限；其余依赖检查和模型实现保持原样。
这个组合已用缓存中的 Qwen3-ASR-0.6B 对真实语音完成识别验证。

```bash
uv run python -m data_analysis.convert_audio_dataset_to_asr_qwen --help
```

## 旧数据

旧 v2.0/v2.1 数据仍可在回放页面读取和删除。新版 LeRobot 续采要求 v3.0；可以换 `dataset.date` 创建新数据集，
或者将旧 **v2.1** 数据转换到新目录后使用（转换保留原数据和音频/同步/轨迹 sidecar）：

```bash
uv run python -m scripts.migrate_lerobot_dataset \
  --input data/dataset/old_dataset \
  --output data/dataset/new_dataset
```

v2.1 与 v3.0 均支持质量检查、音频 ASR 标注和同格式数据合并。
v3 删除使用 LeRobot 官方数据工具处理共享文件，并在已有事务内重排音频与轨迹编号。
转换和合并不会上传到 Hub。真实机器人、相机和麦克风仍需在设备上验证。

上游依据：[LeRobot 0.6.1](https://github.com/huggingface/lerobot/releases/tag/v0.6.1)、
[Python 与依赖要求](https://github.com/huggingface/lerobot/blob/v0.6.1/pyproject.toml)。
