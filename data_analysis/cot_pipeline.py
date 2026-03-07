"""现在这个有问题，ai老是觉得他自己是对的，要用openai的接口，也不想用我的silionflow的接口，坏的很"""

from __future__ import annotations

import argparse
import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
from openai import OpenAI


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "data" / "config.json"

DEFAULT_SYSTEM_PROMPT = (
    "你是一个视频理解与任务分析助手。"
    "你会根据按时间顺序采样的视频帧，识别关键动作、场景变化、目标和结果，"
    "并严格按照用户要求的格式输出。"
)

DEFAULT_USER_PROMPT = (
    "请分析这个视频，按时间顺序概括关键动作、目标和最终结果。"
    "如果信息不充分，请明确说明不确定性。"
)

FORMAT_INSTRUCTION = (
    "\n\n请严格按以下格式输出：\n"
    "<think>\n"
    "这里写你基于视频帧得到的简洁推理摘要，重点说明时序、关键线索和判断依据。\n"
    "</think>\n\n"
    "这里写最终结论。\n"
    "不要省略 <think> 或 </think> 标签。"
)


@dataclass
class SampledFrame:
    order: int
    frame_index: int
    timestamp_seconds: float
    data_url: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 data/config.json 中的 Qwen/OpenAI 兼容配置分析视频，并输出 <think> 格式结果。"
    )
    parser.add_argument(
        "--video-path", type=Path, required=True, help="待分析的视频路径"
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"模型配置 JSON 路径，默认: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument("--prompt", default=DEFAULT_USER_PROMPT, help="用户提示词")
    parser.add_argument(
        "--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="系统提示词"
    )
    parser.add_argument("--output-path", type=Path, help="可选，将结果写入文件")
    parser.add_argument(
        "--max-frames", type=int, default=8, help="最多抽取多少帧送给模型"
    )
    parser.add_argument(
        "--jpeg-quality", type=int, default=85, help="抽帧 JPEG 质量，范围 1-100"
    )
    parser.add_argument("--temperature", type=float, default=0.2, help="模型温度")
    parser.add_argument(
        "--max-tokens", type=int, default=2048, help="最大输出 token 数"
    )
    parser.add_argument("--model-name", help="可选，覆盖配置文件中的模型名")
    parser.add_argument("--base-url", help="可选，覆盖配置文件中的 base_url")
    parser.add_argument("--api-key", help="可选，覆盖配置文件中的 api_key")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return (
        path.expanduser().resolve()
        if path.is_absolute()
        else (Path.cwd() / path).resolve()
    )


def load_model_config(config_path: Path) -> dict[str, str]:
    resolved = resolve_path(config_path)
    if not resolved.exists():
        raise FileNotFoundError(f"配置文件不存在: {resolved}")

    with resolved.open("r", encoding="utf-8") as file:
        config = json.load(file)

    api_key = (
        config.get("API_KEY")
        or config.get("OPENAI_API_KEY")
        or config.get("DASHSCOPE_API_KEY")
    )
    base_url = (
        config.get("BASE_URL")
        or config.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    )
    model_name = config.get("MODEL_NAME") or config.get("OPENAI_MODEL")

    missing_fields = []
    if not api_key:
        missing_fields.append("API_KEY / OPENAI_API_KEY / DASHSCOPE_API_KEY")
    if not model_name:
        missing_fields.append("MODEL_NAME / OPENAI_MODEL")
    if missing_fields:
        raise ValueError(f"配置文件缺少必要字段: {', '.join(missing_fields)}")

    return {
        "api_key": api_key,
        "base_url": base_url,
        "model_name": model_name,
    }


def build_client(api_key: str, base_url: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)


def encode_frame_to_data_url(frame: Any, jpeg_quality: int) -> str:
    success, encoded = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    )
    if not success:
        raise RuntimeError("视频帧编码失败，无法转换为 JPEG")
    base64_image = base64.b64encode(encoded.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_image}"


def compute_sample_indices(total_frames: int, max_frames: int) -> list[int]:
    if total_frames <= 0:
        return []
    if max_frames <= 1:
        return [total_frames // 2]
    if total_frames <= max_frames:
        return list(range(total_frames))

    indices = {
        round(i * (total_frames - 1) / (max_frames - 1)) for i in range(max_frames)
    }
    return sorted(indices)


def sample_video_frames(
    video_path: Path, max_frames: int, jpeg_quality: int
) -> list[SampledFrame]:
    resolved_video_path = resolve_path(video_path)
    if not resolved_video_path.exists():
        raise FileNotFoundError(f"视频不存在: {resolved_video_path}")

    cap = cv2.VideoCapture(str(resolved_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {resolved_video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 1.0

        sampled_frames: list[SampledFrame] = []
        indices = compute_sample_indices(total_frames, max_frames)

        if indices:
            for order, frame_index in enumerate(indices, start=1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, frame = cap.read()
                if not success:
                    continue
                sampled_frames.append(
                    SampledFrame(
                        order=order,
                        frame_index=frame_index,
                        timestamp_seconds=frame_index / fps,
                        data_url=encode_frame_to_data_url(frame, jpeg_quality),
                    )
                )
        else:
            order = 1
            while len(sampled_frames) < max_frames:
                success, frame = cap.read()
                if not success:
                    break
                frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                sampled_frames.append(
                    SampledFrame(
                        order=order,
                        frame_index=frame_index,
                        timestamp_seconds=max(frame_index, 0) / fps,
                        data_url=encode_frame_to_data_url(frame, jpeg_quality),
                    )
                )
                order += 1

        if not sampled_frames:
            raise RuntimeError(f"未能从视频中读取到有效帧: {resolved_video_path}")

        return sampled_frames
    finally:
        cap.release()


def build_user_content(
    prompt: str, sampled_frames: list[SampledFrame]
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"{prompt}{FORMAT_INSTRUCTION}\n\n"
                "下面是按时间顺序采样的视频帧，请结合时序进行分析。"
            ),
        }
    ]

    for frame in sampled_frames:
        content.append(
            {
                "type": "text",
                "text": f"第 {frame.order} 帧，frame_index={frame.frame_index}，时间={frame.timestamp_seconds:.2f}s",
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": frame.data_url},
            }
        )
    return content


def stringify_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
                continue
            text = getattr(item, "text", None)
            if text:
                parts.append(str(text))
        return "\n".join(part.strip() for part in parts if part).strip()
    return str(content).strip()


def extract_reasoning_content(message: Any) -> str:
    candidate_keys = ("reasoning_content", "reasoning", "reasoning_text", "thinking")

    for key in candidate_keys:
        value = getattr(message, key, None)
        if value:
            return stringify_content(value)

    model_extra = getattr(message, "model_extra", None)
    if isinstance(model_extra, dict):
        for key in candidate_keys:
            value = model_extra.get(key)
            if value:
                return stringify_content(value)

    if hasattr(message, "model_dump"):
        dumped = message.model_dump()
        if isinstance(dumped, dict):
            for key in candidate_keys:
                value = dumped.get(key)
                if value:
                    return stringify_content(value)
            dumped_extra = dumped.get("model_extra")
            if isinstance(dumped_extra, dict):
                for key in candidate_keys:
                    value = dumped_extra.get(key)
                    if value:
                        return stringify_content(value)

    return ""


def ensure_think_format(answer_text: str, reasoning_text: str) -> str:
    normalized_answer = answer_text.strip()
    normalized_reasoning = reasoning_text.strip()

    if "<think>" in normalized_answer and "</think>" in normalized_answer:
        return normalized_answer

    if normalized_reasoning:
        return (
            f"<think>\n{normalized_reasoning}\n</think>\n\n{normalized_answer}".strip()
        )

    fallback_reasoning = (
        "模型接口未返回独立 reasoning_content，以下为模型直接给出的最终结果。"
    )
    if normalized_answer:
        return f"<think>\n{fallback_reasoning}\n</think>\n\n{normalized_answer}".strip()
    return f"<think>\n{fallback_reasoning}\n</think>"


def analyze_video(
    video_path: Path,
    config_path: Path,
    prompt: str,
    system_prompt: str,
    output_path: Path | None,
    max_frames: int,
    jpeg_quality: int,
    temperature: float,
    max_tokens: int,
    model_name_override: str | None,
    base_url_override: str | None,
    api_key_override: str | None,
) -> str:
    if max_frames < 1:
        raise ValueError("--max-frames 必须大于等于 1")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("--jpeg-quality 必须在 1 到 100 之间")

    config = load_model_config(config_path)
    api_key = api_key_override or config["api_key"]
    base_url = base_url_override or config["base_url"]
    model_name = model_name_override or config["model_name"]

    sampled_frames = sample_video_frames(
        video_path, max_frames=max_frames, jpeg_quality=jpeg_quality
    )
    client = build_client(api_key=api_key, base_url=base_url)

    response = client.chat.completions.create(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_content(prompt, sampled_frames)},
        ],
    )

    if not response.choices:
        raise RuntimeError("模型没有返回任何候选结果")

    message = response.choices[0].message
    answer_text = stringify_content(message.content)
    reasoning_text = extract_reasoning_content(message)
    final_output = ensure_think_format(answer_text, reasoning_text)

    if output_path is not None:
        resolved_output_path = resolve_path(output_path)
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_output_path.write_text(final_output, encoding="utf-8")

    return final_output


def main() -> None:
    args = parse_args()
    result = analyze_video(
        video_path=args.video_path,
        config_path=args.config_path,
        prompt=args.prompt,
        system_prompt=args.system_prompt,
        output_path=args.output_path,
        max_frames=args.max_frames,
        jpeg_quality=args.jpeg_quality,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        model_name_override=args.model_name,
        base_url_override=args.base_url,
        api_key_override=args.api_key,
    )
    print(result)


if __name__ == "__main__":
    main()
