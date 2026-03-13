"""
LeRobot 数据集质量检查脚本

检查项目：
1. 数据完整性：检查所有 episode 是否完整
2. 图像质量：检查图像是否有效、分辨率是否正确
3. 动作数据：检查动作范围、是否有异常值
4. 时间戳：检查时间戳是否连续、帧率是否稳定
5. 统计信息：episode 长度分布、动作统计等
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_dataset_info(dataset_path: Path) -> Dict[str, Any]:
    """加载数据集元信息"""
    info_path = dataset_path / "meta" / "info.json"
    with open(info_path, "r") as f:
        return json.load(f)


def load_episodes_info(dataset_path: Path) -> List[Dict[str, Any]]:
    """加载所有 episode 的元信息"""
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    episodes = []
    with open(episodes_path, "r") as f:
        for line in f:
            episodes.append(json.loads(line))
    return episodes


def check_data_completeness(dataset_path: Path, info: Dict[str, Any]) -> Dict[str, Any]:
    """检查数据完整性"""
    print("\n" + "=" * 60)
    print("1. 数据完整性检查")
    print("=" * 60)

    results = {
        "total_episodes": info["total_episodes"],
        "total_frames": info["total_frames"],
        "missing_episodes": [],
        "corrupted_episodes": [],
    }

    # 检查所有 episode 文件是否存在
    data_dir = dataset_path / "data" / "chunk-000"
    for ep_idx in range(info["total_episodes"]):
        ep_file = data_dir / f"episode_{ep_idx:06d}.parquet"
        if not ep_file.exists():
            results["missing_episodes"].append(ep_idx)
            print(f"  ✗ Episode {ep_idx} 文件缺失: {ep_file}")
        else:
            # 尝试读取文件
            try:
                df = pd.read_parquet(ep_file)
                if len(df) == 0:
                    results["corrupted_episodes"].append(ep_idx)
                    print(f"  ✗ Episode {ep_idx} 为空")
            except Exception as e:
                results["corrupted_episodes"].append(ep_idx)
                print(f"  ✗ Episode {ep_idx} 读取失败: {e}")

    if not results["missing_episodes"] and not results["corrupted_episodes"]:
        print(f"  ✓ 所有 {info['total_episodes']} 个 episodes 完整")

    return results


def check_image_quality(
    dataset_path: Path, info: Dict[str, Any], sample_size: int = 5
) -> Dict[str, Any]:
    """检查图像质量（采样检查）"""
    print("\n" + "=" * 60)
    print("2. 图像质量检查")
    print("=" * 60)

    results = {
        "image_keys": [],
        "expected_shape": None,
        "issues": [],
    }

    # 获取图像特征
    image_features = {
        k: v for k, v in info["features"].items() if v["dtype"] == "image"
    }
    results["image_keys"] = list(image_features.keys())

    if not image_features:
        print("  ⚠ 数据集中没有图像数据")
        return results

    # 获取期望的图像形状
    first_img_key = list(image_features.keys())[0]
    results["expected_shape"] = tuple(image_features[first_img_key]["shape"])
    print(f"  期望图像形状: {results['expected_shape']}")
    print(f"  图像特征: {', '.join(results['image_keys'])}")

    # 随机采样检查
    data_dir = dataset_path / "data" / "chunk-000"
    episode_files = sorted(data_dir.glob("episode_*.parquet"))
    sample_episodes = np.random.choice(
        len(episode_files), min(sample_size, len(episode_files)), replace=False
    )

    print(f"\n  随机采样 {len(sample_episodes)} 个 episodes 进行检查...")

    for ep_idx in sample_episodes:
        ep_file = episode_files[ep_idx]
        try:
            df = pd.read_parquet(ep_file)

            # 检查每个图像特征
            for img_key in results["image_keys"]:
                if img_key not in df.columns:
                    results["issues"].append(
                        f"Episode {ep_idx}: 缺少图像特征 {img_key}"
                    )
                    continue

                # 检查第一帧图像
                img = df[img_key].iloc[0]

                # LeRobot 可能将图像存储为 dict 格式
                if isinstance(img, dict):
                    if "bytes" in img and "path" in img:
                        # 这是 LeRobot 的图像引用格式，跳过检查
                        continue
                    else:
                        results["issues"].append(
                            f"Episode {ep_idx}: {img_key} 是未知的 dict 格式"
                        )
                        continue

                if img is None:
                    results["issues"].append(f"Episode {ep_idx}: {img_key} 为 None")
                elif not isinstance(img, np.ndarray):
                    results["issues"].append(
                        f"Episode {ep_idx}: {img_key} 类型错误 ({type(img)})"
                    )
                elif img.shape != results["expected_shape"]:
                    results["issues"].append(
                        f"Episode {ep_idx}: {img_key} 形状错误 "
                        f"(期望 {results['expected_shape']}, 实际 {img.shape})"
                    )
                elif img.min() < 0 or img.max() > 255:
                    results["issues"].append(
                        f"Episode {ep_idx}: {img_key} 像素值超出范围 "
                        f"(min={img.min()}, max={img.max()})"
                    )
        except Exception as e:
            results["issues"].append(f"Episode {ep_idx}: 读取失败 - {e}")

    if results["issues"]:
        print(f"\n  ✗ 发现 {len(results['issues'])} 个问题:")
        for issue in results["issues"][:10]:  # 只显示前10个
            print(f"    - {issue}")
        if len(results["issues"]) > 10:
            print(f"    ... 还有 {len(results['issues']) - 10} 个问题")
    else:
        print(f"  ✓ 采样检查通过，图像质量正常")

    return results


def check_action_data(dataset_path: Path, info: Dict[str, Any]) -> Dict[str, Any]:
    """检查动作数据"""
    print("\n" + "=" * 60)
    print("3. 动作数据检查")
    print("=" * 60)

    results = {
        "action_dim": None,
        "action_stats": {},
        "anomalies": [],
    }

    # 获取动作维度
    if "actions" in info["features"]:
        results["action_dim"] = info["features"]["actions"]["shape"][0]
        print(f"  动作维度: {results['action_dim']}")
    else:
        print("  ⚠ 数据集中没有动作数据")
        return results

    # 收集所有动作数据
    data_dir = dataset_path / "data" / "chunk-000"
    episode_files = sorted(data_dir.glob("episode_*.parquet"))

    all_actions = []
    print(f"\n  读取 {len(episode_files)} 个 episodes 的动作数据...")

    for ep_file in tqdm(episode_files, desc="  处理中"):
        try:
            df = pd.read_parquet(ep_file)
            if "actions" in df.columns:
                actions = np.stack(df["actions"].values)
                all_actions.append(actions)
        except Exception as e:
            results["anomalies"].append(f"{ep_file.name}: 读取失败 - {e}")

    if not all_actions:
        print("  ✗ 没有找到有效的动作数据")
        return results

    # 合并所有动作
    all_actions = np.concatenate(all_actions, axis=0)

    # 计算统计信息
    results["action_stats"] = {
        "mean": all_actions.mean(axis=0).tolist(),
        "std": all_actions.std(axis=0).tolist(),
        "min": all_actions.min(axis=0).tolist(),
        "max": all_actions.max(axis=0).tolist(),
        "total_samples": len(all_actions),
    }

    print(f"\n  动作统计 (共 {len(all_actions)} 个样本):")
    print(f"    维度 | 均值      | 标准差    | 最小值    | 最大值")
    print(f"    " + "-" * 60)
    for i in range(results["action_dim"]):
        print(
            f"    {i:4d} | {results['action_stats']['mean'][i]:9.4f} | "
            f"{results['action_stats']['std'][i]:9.4f} | "
            f"{results['action_stats']['min'][i]:9.4f} | "
            f"{results['action_stats']['max'][i]:9.4f}"
        )

    # 检查异常值（超过 3 个标准差）
    for i in range(results["action_dim"]):
        mean = results["action_stats"]["mean"][i]
        std = results["action_stats"]["std"][i]
        outliers = np.abs(all_actions[:, i] - mean) > 3 * std
        if outliers.sum() > 0:
            results["anomalies"].append(
                f"维度 {i}: {outliers.sum()} 个异常值 "
                f"({100 * outliers.sum() / len(all_actions):.2f}%)"
            )

    if results["anomalies"]:
        print(f"\n  ⚠ 发现 {len(results['anomalies'])} 个异常:")
        for anomaly in results["anomalies"]:
            print(f"    - {anomaly}")
    else:
        print(f"\n  ✓ 动作数据正常，无异常值")

    return results


def check_timestamps(dataset_path: Path, info: Dict[str, Any]) -> Dict[str, Any]:
    """检查时间戳和帧率"""
    print("\n" + "=" * 60)
    print("4. 时间戳和帧率检查")
    print("=" * 60)

    results = {
        "expected_fps": info["fps"],
        "actual_fps": [],
        "timestamp_gaps": [],
        "issues": [],
    }

    data_dir = dataset_path / "data" / "chunk-000"
    episode_files = sorted(data_dir.glob("episode_*.parquet"))

    print(f"  期望帧率: {results['expected_fps']} Hz")
    print(f"\n  检查 {len(episode_files)} 个 episodes 的时间戳...")

    for ep_file in tqdm(episode_files, desc="  处理中"):
        try:
            df = pd.read_parquet(ep_file)
            if "timestamp" not in df.columns or len(df) < 2:
                continue

            timestamps = np.array(
                [
                    t[0] if isinstance(t, np.ndarray) else t
                    for t in df["timestamp"].values
                ]
            )

            # 计算时间间隔
            time_diffs = np.diff(timestamps)

            # 计算实际帧率
            avg_time_diff = time_diffs.mean()
            if avg_time_diff > 0:
                actual_fps = 1.0 / avg_time_diff
                results["actual_fps"].append(actual_fps)

                # 检查帧率偏差
                fps_error = (
                    abs(actual_fps - results["expected_fps"]) / results["expected_fps"]
                )
                if fps_error > 0.1:  # 超过 10% 偏差
                    results["issues"].append(
                        f"{ep_file.name}: 帧率偏差 {fps_error*100:.1f}% "
                        f"(期望 {results['expected_fps']:.1f} Hz, 实际 {actual_fps:.1f} Hz)"
                    )

            # 检查时间戳跳跃
            expected_diff = 1.0 / results["expected_fps"]
            large_gaps = time_diffs > expected_diff * 2  # 超过 2 倍的间隔
            if large_gaps.sum() > 0:
                results["timestamp_gaps"].append(
                    {
                        "episode": ep_file.name,
                        "num_gaps": int(large_gaps.sum()),
                        "max_gap": float(time_diffs.max()),
                    }
                )

        except Exception as e:
            results["issues"].append(f"{ep_file.name}: 处理失败 - {e}")

    # 统计实际帧率
    if results["actual_fps"]:
        avg_fps = np.mean(results["actual_fps"])
        std_fps = np.std(results["actual_fps"])
        print(f"\n  实际帧率: {avg_fps:.2f} ± {std_fps:.2f} Hz")

        fps_error = abs(avg_fps - results["expected_fps"]) / results["expected_fps"]
        if fps_error < 0.05:
            print(f"  ✓ 帧率稳定，偏差 {fps_error*100:.1f}%")
        else:
            print(f"  ⚠ 帧率偏差较大: {fps_error*100:.1f}%")

    # 报告时间戳跳跃
    if results["timestamp_gaps"]:
        print(f"\n  ⚠ 发现 {len(results['timestamp_gaps'])} 个 episodes 有时间戳跳跃:")
        for gap_info in results["timestamp_gaps"][:5]:
            print(
                f"    - {gap_info['episode']}: {gap_info['num_gaps']} 个跳跃, "
                f"最大间隔 {gap_info['max_gap']:.3f}s"
            )
        if len(results["timestamp_gaps"]) > 5:
            print(f"    ... 还有 {len(results['timestamp_gaps']) - 5} 个")

    if results["issues"]:
        print(f"\n  ✗ 发现 {len(results['issues'])} 个问题:")
        for issue in results["issues"][:5]:
            print(f"    - {issue}")
        if len(results["issues"]) > 5:
            print(f"    ... 还有 {len(results['issues']) - 5} 个问题")

    return results


def analyze_episode_statistics(
    dataset_path: Path, episodes: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """分析 episode 统计信息"""
    print("\n" + "=" * 60)
    print("5. Episode 统计分析")
    print("=" * 60)

    results = {
        "num_episodes": len(episodes),
        "episode_lengths": [ep["length"] for ep in episodes],
        "tasks": {},
    }

    # 统计任务
    for ep in episodes:
        for task in ep["tasks"]:
            results["tasks"][task] = results["tasks"].get(task, 0) + 1

    # Episode 长度统计
    lengths = np.array(results["episode_lengths"])
    print(f"\n  Episode 数量: {results['num_episodes']}")
    print(f"  总帧数: {lengths.sum()}")
    print(f"\n  Episode 长度统计:")
    print(f"    均值: {lengths.mean():.1f} 帧")
    print(f"    标准差: {lengths.std():.1f} 帧")
    print(f"    最小值: {lengths.min()} 帧")
    print(f"    最大值: {lengths.max()} 帧")
    print(f"    中位数: {np.median(lengths):.1f} 帧")

    # 任务统计
    print(f"\n  任务分布:")
    for task, count in results["tasks"].items():
        print(
            f"    - {task}: {count} episodes ({100*count/results['num_episodes']:.1f}%)"
        )

    return results


def generate_visualizations(
    dataset_path: Path, results: Dict[str, Any], output_dir: Path
):
    """生成可视化图表"""
    print("\n" + "=" * 60)
    print("6. 生成可视化图表")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Episode 长度分布
    if "episode_stats" in results and results["episode_stats"]["episode_lengths"]:
        plt.figure(figsize=(10, 6))
        plt.hist(
            results["episode_stats"]["episode_lengths"], bins=20, edgecolor="black"
        )
        plt.xlabel("Episode Length (frames)")
        plt.ylabel("Count")
        plt.title("Episode Length Distribution")
        plt.grid(True, alpha=0.3)
        output_file = output_dir / "episode_length_distribution.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ 保存: {output_file}")

    # 2. 动作分布
    if "action_data" in results and results["action_data"]["action_stats"]:
        stats = results["action_data"]["action_stats"]
        action_dim = results["action_data"]["action_dim"]

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # 均值和标准差
        x = np.arange(action_dim)
        axes[0].bar(x, stats["mean"], yerr=stats["std"], capsize=5, alpha=0.7)
        axes[0].set_xlabel("Action Dimension")
        axes[0].set_ylabel("Mean ± Std")
        axes[0].set_title("Action Mean and Standard Deviation")
        axes[0].grid(True, alpha=0.3)

        # 最小值和最大值
        axes[1].plot(x, stats["min"], "o-", label="Min", markersize=4)
        axes[1].plot(x, stats["max"], "s-", label="Max", markersize=4)
        axes[1].set_xlabel("Action Dimension")
        axes[1].set_ylabel("Value")
        axes[1].set_title("Action Range (Min/Max)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = output_dir / "action_statistics.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ 保存: {output_file}")

    # 3. 帧率分布
    if "timestamp_data" in results and results["timestamp_data"]["actual_fps"]:
        fps_data = results["timestamp_data"]["actual_fps"]
        if len(set(fps_data)) > 1:  # 只有在有变化时才绘制直方图
            plt.figure(figsize=(10, 6))
            plt.hist(
                fps_data, bins=min(20, len(set(fps_data))), edgecolor="black", alpha=0.7
            )
            expected_fps = results["timestamp_data"]["expected_fps"]
            plt.axvline(
                expected_fps,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Expected: {expected_fps} Hz",
            )
            plt.xlabel("FPS")
            plt.ylabel("Count")
            plt.title("Actual Frame Rate Distribution")
            plt.legend()
            plt.grid(True, alpha=0.3)
            output_file = output_dir / "fps_distribution.png"
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  ✓ 保存: {output_file}")
        else:
            print(f"  ⊘ 跳过帧率分布图（所有帧率相同: {fps_data[0]:.2f} Hz）")

    # 4. 合并所有图表
    print(f"\n  生成合并报告...")
    try:
        from PIL import Image

        # 收集所有生成的图表
        image_files = []
        titles = []

        if (output_dir / "episode_length_distribution.png").exists():
            image_files.append(output_dir / "episode_length_distribution.png")
            titles.append("Episode Length Distribution")

        if (output_dir / "action_statistics.png").exists():
            image_files.append(output_dir / "action_statistics.png")
            titles.append("Action Statistics")

        if (output_dir / "fps_distribution.png").exists():
            image_files.append(output_dir / "fps_distribution.png")
            titles.append("FPS Distribution")

        if image_files:
            # 读取所有图片
            images = [Image.open(img_file) for img_file in image_files]

            # 创建合并图表
            fig, axes = plt.subplots(len(images), 1, figsize=(12, 5 * len(images)))
            if len(images) == 1:
                axes = [axes]

            for idx, (img, title) in enumerate(zip(images, titles)):
                axes[idx].imshow(img)
                axes[idx].axis("off")
                axes[idx].set_title(title, fontsize=14, fontweight="bold")

            plt.tight_layout()
            output_file = output_dir / "combined_report.png"
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  ✓ 保存合并报告: {output_file}")
        else:
            print(f"  ⊘ 没有图表可以合并")
    except ImportError:
        print(f"  ⊘ PIL 未安装，跳过合并报告生成")
    except Exception as e:
        print(f"  ⚠ 合并报告生成失败: {e}")


def save_report(results: Dict[str, Any], output_file: Path):
    """保存检查报告"""
    print("\n" + "=" * 60)
    print("7. 保存检查报告")
    print("=" * 60)

    # 转换 numpy 类型为 Python 原生类型
    def convert_to_json_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        else:
            return obj

    serializable_results = convert_to_json_serializable(results)

    with open(output_file, "w") as f:
        json.dump(serializable_results, f, indent=2)

    print(f"  ✓ 报告已保存: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="LeRobot 数据集质量检查")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/openpi/franka_droid_lerobot_3_8",
        help="数据集路径",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data_analysis/quality_reports",
        help="输出目录",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="图像质量检查的采样数量",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)

    if not dataset_path.exists():
        print(f"错误: 数据集路径不存在: {dataset_path}")
        return

    print("=" * 60)
    print("LeRobot 数据集质量检查")
    print("=" * 60)
    print(f"数据集路径: {dataset_path}")
    print(f"输出目录: {output_dir}")

    # 加载元信息
    info = load_dataset_info(dataset_path)
    episodes = load_episodes_info(dataset_path)

    # 执行检查
    all_results = {}

    all_results["completeness"] = check_data_completeness(dataset_path, info)
    all_results["image_quality"] = check_image_quality(
        dataset_path, info, args.sample_size
    )
    all_results["action_data"] = check_action_data(dataset_path, info)
    all_results["timestamp_data"] = check_timestamps(dataset_path, info)
    all_results["episode_stats"] = analyze_episode_statistics(dataset_path, episodes)

    # 生成可视化
    generate_visualizations(dataset_path, all_results, output_dir)

    # 保存报告
    report_file = output_dir / "quality_report.json"
    save_report(all_results, report_file)

    print("\n" + "=" * 60)
    print("检查完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
