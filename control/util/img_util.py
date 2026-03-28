from __future__ import annotations

import textwrap
from io import BytesIO
from pathlib import Path

import numpy as np


def draw_prompt_overlay(bgr: np.ndarray, prompt: str) -> np.ndarray:
    prompt = prompt.strip()
    if not prompt:
        return bgr

    try:
        import cv2
    except Exception:
        return bgr

    annotated = bgr.copy()
    height, width = annotated.shape[:2]
    margin = max(8, width // 50)
    font_scale = max(0.45, min(0.8, width / 960.0))
    thickness = 1 if width < 960 else 2
    line_height = max(20, int(26 * font_scale))
    max_chars = max(24, width // 12)
    lines = textwrap.wrap(f"Prompt: {prompt}", width=max_chars)[:4]
    box_height = margin * 2 + line_height * len(lines)

    overlay = annotated.copy()
    cv2.rectangle(
        overlay,
        (margin, margin),
        (width - margin, min(height - margin, margin + box_height)),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.45, annotated, 0.55, 0.0, annotated)

    y = margin + line_height
    for line in lines:
        cv2.putText(
            annotated,
            line,
            (margin * 2, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += line_height

    return annotated


def show_image(
    frame: np.ndarray,
    *,
    win_name: str,
    scale: float = 1.0,
    prompt: str = "",
) -> None:
    try:
        import cv2
    except Exception:
        return
    if frame is None:
        return
    img = np.asarray(frame)
    if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.transpose(img, (1, 2, 0))
    if np.issubdtype(img.dtype, np.floating):
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255.0).astype(np.uint8)
    if img.ndim == 3 and img.shape[-1] == 3:
        bgr = img[..., ::-1]
        bgr = draw_prompt_overlay(bgr, prompt)
    else:
        return
    if scale != 1.0:
        h, w = bgr.shape[:2]
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)))
    cv2.imshow(win_name, bgr)
    cv2.waitKey(1)


def decode_image(
    value: object,
    *,
    root: Path,
    image_key: str,
    episode_index: int | None,
    frame_index: int | None,
) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    try:
        from PIL import Image
    except Exception:
        return None

    if isinstance(value, Image.Image):
        return np.asarray(value.convert("RGB"))

    if isinstance(value, (bytes, bytearray)):
        try:
            return np.asarray(Image.open(BytesIO(value)).convert("RGB"))
        except Exception:
            return None

    if isinstance(value, dict):
        bytes_data = value.get("bytes")
        if bytes_data:
            try:
                return np.asarray(Image.open(BytesIO(bytes_data)).convert("RGB"))
            except Exception:
                return None
        path = value.get("path")
        if path:
            candidate = Path(path)
            if not candidate.is_absolute():
                if episode_index is not None and frame_index is not None:
                    candidate = (
                        root
                        / "images"
                        / image_key
                        / f"episode-{episode_index:06d}"
                        / Path(path).name
                    )
                else:
                    candidate = root / Path(path)
            if candidate.exists():
                try:
                    return np.asarray(Image.open(candidate).convert("RGB"))
                except Exception:
                    return None
    return None


def center_crop_and_resize_rgb_uint8(
    rgb: np.ndarray, *, crop_scale: float = 0.9, out_hw: int = 224
) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB image of shape (H, W, 3), got {rgb.shape}")

    try:
        import tensorflow as tf

        image_tf = tf.convert_to_tensor(rgb)
        orig_dtype = image_tf.dtype
        image_tf = tf.image.convert_image_dtype(image_tf, tf.float32)
        image_tf = tf.expand_dims(image_tf, axis=0)

        batch_size = 1
        scale = tf.reshape(
            tf.clip_by_value(tf.sqrt(float(crop_scale)), 0, 1), shape=(batch_size,)
        )
        height_offsets = (1 - scale) / 2
        width_offsets = (1 - scale) / 2
        boxes = tf.stack(
            [
                height_offsets,
                width_offsets,
                height_offsets + scale,
                width_offsets + scale,
            ],
            axis=1,
        )

        image_tf = tf.image.crop_and_resize(
            image_tf, boxes, tf.range(batch_size), (int(out_hw), int(out_hw))
        )
        image_tf = image_tf[0]
        image_tf = tf.clip_by_value(image_tf, 0, 1)
        image_tf = tf.image.convert_image_dtype(image_tf, orig_dtype, saturate=True)
        out = image_tf.numpy()
    except Exception:
        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        frac = float(np.sqrt(float(crop_scale)))
        crop_h = max(1, min(h, int(round(h * frac))))
        crop_w = max(1, min(w, int(round(w * frac))))
        y0 = max(0, (h - crop_h) // 2)
        x0 = max(0, (w - crop_w) // 2)
        cropped = rgb[y0 : y0 + crop_h, x0 : x0 + crop_w]

        from PIL import Image

        img = Image.fromarray(cropped, mode="RGB").resize(
            (int(out_hw), int(out_hw)), resample=Image.BILINEAR
        )
        out = np.asarray(img, dtype=np.uint8)

    if out.shape != (int(out_hw), int(out_hw), 3):
        raise RuntimeError(f"Unexpected resized RGB shape: {out.shape}")
    return out.astype(np.uint8, copy=False)
