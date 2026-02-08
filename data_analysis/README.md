# LeRobot 数据集质量检查工具

## 概述

这个工具用于检查 LeRobot 格式数据集的质量，包括数据完整性、图像质量、动作数据、时间戳和帧率等。

## 功能特性

### 检查项目

1. **数据完整性检查**
   - 检查所有 episode 文件是否存在
   - 验证文件是否可读取
   - 检测损坏的数据文件

2. **图像质量检查**
   - 验证图像分辨率和格式
   - 检查像素值范围
   - 采样检查图像数据

3. **动作数据分析**
   - 统计动作的均值、标准差、最小值、最大值
   - 检测异常值（超过 3σ）
   - 生成动作分布可视化

4. **时间戳和帧率检查**
   - 验证帧率稳定性
   - 检测时间戳跳跃
   - 计算实际帧率与期望帧率的偏差

5. **Episode 统计分析**
   - Episode 长度分布
   - 任务分布统计
   - 数据集整体统计信息

### 输出文件

运行后会在输出目录生成以下文件：

1. **quality_report.json** - 完整的 JSON 格式检查报告
2. **episode_length_distribution.png** - Episode 长度分布直方图
3. **action_statistics.png** - 动作统计图表（均值/标准差/范围）
4. **fps_distribution.png** - 帧率分布图
5. **combined_report.png** - 合并所有图表的综合报告

## 使用方法

### 基本用法

```bash
# 使用默认路径检查数据集
uv run data_analysis/check_dataset_quality.py
```

### 指定数据集路径

```bash
uv run data_analysis/check_dataset_quality.py \
  --dataset-path data/openpi/franka_droid_lerobot_20260130_161202 \
  --output-dir data_analysis/quality_reports \
  --sample-size 5
```

### 参数说明

- `--dataset-path`: 数据集路径（默认：`data/openpi/franka_droid_lerobot_20260130_161202`）
- `--output-dir`: 输出目录（默认：`data_analysis/quality_reports`）
- `--sample-size`: 图像质量检查的采样数量（默认：5）

## 输出示例

### 终端输出

```
============================================================
LeRobot 数据集质量检查
============================================================
数据集路径: data/openpi/franka_droid_lerobot_20260130_161202
输出目录: data_analysis/quality_reports

============================================================
1. 数据完整性检查
============================================================
  ✓ 所有 14 个 episodes 完整

============================================================
2. 图像质量检查
============================================================
  期望图像形状: (224, 224, 3)
  图像特征: exterior_image_1_left, exterior_image_2_left, wrist_image_left
  ✓ 采样检查通过，图像质量正常

============================================================
3. 动作数据检查
============================================================
  动作维度: 8
  动作统计 (共 3737 个样本):
    维度 | 均值      | 标准差    | 最小值    | 最大值
    ------------------------------------------------------------
       0 |    0.0045 |    0.0545 |   -0.2388 |    0.1944
       1 |    0.0325 |    0.1294 |   -0.3543 |    0.3085
       ...

============================================================
4. 时间戳和帧率检查
============================================================
  期望帧率: 15.0 Hz
  实际帧率: 15.00 ± 0.00 Hz
  ✓ 帧率稳定，偏差 0.0%

============================================================
5. Episode 统计分析
============================================================
  Episode 数量: 14
  总帧数: 3737
  Episode 长度统计:
    均值: 266.9 帧
    标准差: 33.6 帧
    最小值: 230 帧
    最大值: 342 帧
```

### JSON 报告结构

```json
{
  "completeness": {
    "total_episodes": 14,
    "total_frames": 3737,
    "missing_episodes": [],
    "corrupted_episodes": []
  },
  "image_quality": {
    "image_keys": ["exterior_image_1_left", "exterior_image_2_left", "wrist_image_left"],
    "expected_shape": [224, 224, 3],
    "issues": []
  },
  "action_data": {
    "action_dim": 8,
    "action_stats": {
      "mean": [...],
      "std": [...],
      "min": [...],
      "max": [...]
    },
    "anomalies": [...]
  },
  ...
}
```

## 数据质量评估标准

### ✅ 良好指标

- 数据完整性：无缺失或损坏的文件
- 图像质量：所有图像格式正确，像素值在 [0, 255] 范围内
- 帧率稳定性：实际帧率与期望帧率偏差 < 5%
- 异常值比例：< 5%

### ⚠️ 需要注意

- 异常值比例：5% - 10%
- 帧率偏差：5% - 10%
- 时间戳跳跃：偶尔出现

### ❌ 需要修复

- 数据缺失或损坏
- 图像格式错误
- 异常值比例 > 10%
- 帧率偏差 > 10%
- 频繁的时间戳跳跃

## 依赖项

脚本需要以下 Python 包：

- `numpy` - 数值计算
- `pandas` - 数据处理
- `matplotlib` - 图表生成
- `tqdm` - 进度条显示
- `Pillow` (PIL) - 图像处理（用于合并报告）

这些依赖已包含在项目的 `pyproject.toml` 中。

## 故障排除

### 问题：找不到数据集

```
错误: 数据集路径不存在: data/openpi/...
```

**解决方案**：检查数据集路径是否正确，使用 `--dataset-path` 参数指定正确的路径。

### 问题：图像类型错误

```
Episode X: image_key 类型错误 (<class 'dict'>)
```

**解决方案**：这是正常的，LeRobot 使用 dict 格式存储图像引用。脚本会自动处理这种情况。

### 问题：内存不足

如果数据集很大，可能会遇到内存问题。

**解决方案**：
- 减少 `--sample-size` 参数
- 在更大内存的机器上运行
- 分批处理数据集

## 扩展功能

### 添加自定义检查

可以在脚本中添加自定义检查函数：

```python
def check_custom_metric(dataset_path: Path, info: Dict[str, Any]) -> Dict[str, Any]:
    """自定义检查函数"""
    results = {}
    # 实现你的检查逻辑
    return results

# 在 main() 函数中调用
all_results["custom_metric"] = check_custom_metric(dataset_path, info)
```

### 自定义可视化

可以在 `generate_visualizations()` 函数中添加新的图表：

```python
# 添加新的可视化
plt.figure(figsize=(10, 6))
# 绘制你的图表
plt.savefig(output_dir / "custom_plot.png", dpi=150, bbox_inches="tight")
```

## 许可证

本工具是 LeRobot 数据收集项目的一部分。

## 联系方式

如有问题或建议，请提交 issue 或 pull request。
