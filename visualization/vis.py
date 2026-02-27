import cv2
import numpy as np
import matplotlib.pyplot as plt

def overlay_heatmap(image, heatmap, alpha=0.4, interpolation=cv2.INTER_LINEAR):
    """
    Overlay a heatmap on top of an image.

    Parameters
    ----------
    image : torch.Tensor | np.ndarray
        Image tensor/array, expected shape ``(3, 640, 640)`` (CHW) or ``(640, 640, 3)`` (HWC).
    heatmap : torch.Tensor | np.ndarray
        Heatmap tensor/array with shape ``(2, 160, 160)`` or ``(160, 160)``.
        If 3D, the first channel ``heatmap[0]`` is used.
    alpha : float, default=0.4
        Heatmap blending strength.
    interpolation : int, default=cv2.INTER_LINEAR
        OpenCV resize interpolation mode.
    """
    # Accept torch tensors and numpy arrays.
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    if hasattr(heatmap, "detach"):
        heatmap = heatmap.detach().cpu().numpy()

    image = np.asarray(image)
    heatmap = np.asarray(heatmap)

    if image.ndim != 3:
        raise ValueError(f"image must be 3D (CHW or HWC), got shape {image.shape}")
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))  # CHW -> HWC
    elif image.shape[2] != 3:
        raise ValueError(f"image must have 3 channels, got shape {image.shape}")

    # Convert image to uint8 (OpenCV-friendly).
    if np.issubdtype(image.dtype, np.floating):
        if image.max() <= 1.0 + 1e-6:
            img = np.clip(image, 0.0, 1.0) * 255.0
        else:
            img = np.clip(image, 0.0, 255.0)
        img = img.astype(np.uint8)
    else:
        img = np.clip(image, 0, 255).astype(np.uint8)

    # Use first channel as requested.
    if heatmap.ndim == 3:
        hm = heatmap[0]
    elif heatmap.ndim == 2:
        hm = heatmap
    else:
        raise ValueError(f"heatmap must be 2D or 3D, got shape {heatmap.shape}")

    hm = hm.astype(np.float32)
    hm = cv2.resize(hm, (img.shape[1], img.shape[0]), interpolation=interpolation)
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    heatmap_color = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay_pos = cv2.addWeighted(img, 1.0 - alpha, heatmap_color, alpha, 0)

    hm_neg = heatmap[1]
    hm_neg = hm_neg.astype(np.float32)
    hm_neg = cv2.resize(hm_neg, (img.shape[1], img.shape[0]), interpolation=interpolation)
    hm_neg = (hm_neg - hm_neg.min()) / (hm_neg.max() - hm_neg.min() + 1e-8)
    heatmap_color_neg = cv2.applyColorMap((hm_neg * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay_neg = cv2.addWeighted(img, 1.0 - alpha, heatmap_color_neg, alpha, 0)

    fig, axes = plt.subplots(1, 3, figsize=(14, 14))


    axes[0].imshow(image)
    axes[0].axis("off")


    axes[1].imshow(overlay_pos)
    axes[1].axis("off")

    axes[2].imshow(overlay_neg)
    axes[2].axis("off")