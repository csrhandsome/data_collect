from __future__ import annotations

import numpy as np


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
