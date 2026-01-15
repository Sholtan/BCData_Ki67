import cv2
import numpy as np

def overlay_heatmap(
    image: np.ndarray,        # (H_img, W_img, 3), RGB
    heatmap: np.ndarray,      # (1, H_hm, W_hm) or (H_hm, W_hm)
    alpha: float = 0.5,
    interpolation=cv2.INTER_LINEAR,
):
    """
    Overlay a low-resolution heatmap on a high-resolution image.

    image:   (H_img, W_img, 3) RGB
    heatmap: (1, H_hm, W_hm) or (H_hm, W_hm)
    """

    if heatmap.ndim == 3:
        heatmap = heatmap[0]   # (H_hm, W_hm)

    H_img, W_img, _ = image.shape

    # Normalize heatmap to [0, 1]
    heatmap = heatmap.astype(np.float32)
    max_val = heatmap.max()
    if max_val > 0:
        heatmap = heatmap / max_val

    # Resize heatmap to image resolution
    heatmap_resized = cv2.resize(
        heatmap,
        (W_img, H_img),
        interpolation=interpolation,
    )

    # Convert to color map
    heatmap_color = cv2.applyColorMap(
        (255 * heatmap_resized).astype(np.uint8),
        cv2.COLORMAP_JET,
    )

    # Overlay
    overlay = cv2.addWeighted(
        image,
        1.0 - alpha,
        heatmap_color,
        alpha,
        0,
    )

    return overlay
