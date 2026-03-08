import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def overlay_heatmap(image, pred_heatmap, gt_heatmap, alpha=0.4, interpolation=cv2.INTER_LINEAR, color_map = cv2.COLORMAP_BONE):
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
    if hasattr(pred_heatmap, "detach"):
        pred_heatmap = pred_heatmap.detach().cpu().numpy()
    if hasattr(gt_heatmap, "detach"):
        gt_heatmap = gt_heatmap.detach().cpu().numpy()


    image = np.asarray(image)
    pred_heatmap = np.asarray(pred_heatmap)
    gt_heatmap = np.asarray(gt_heatmap)


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



    
    pred_hm_pos = pred_heatmap[0]
    pred_hm_pos = pred_hm_pos.astype(np.float32)
    pred_hm_pos = cv2.resize(pred_hm_pos, (img.shape[1], img.shape[0]), interpolation=interpolation)
    pred_hm_pos = (pred_hm_pos - pred_hm_pos.min()) / (pred_hm_pos.max() - pred_hm_pos.min() + 1e-8)
    pred_heatmap_color_pos = cv2.applyColorMap((pred_hm_pos * 255).astype(np.uint8), color_map)
    pred_overlay_pos = cv2.addWeighted(img, 1.0 - alpha, pred_heatmap_color_pos, alpha, 0)

    pred_hm_neg = pred_heatmap[1]
    pred_hm_neg = pred_hm_neg.astype(np.float32)
    pred_hm_neg = cv2.resize(pred_hm_neg, (img.shape[1], img.shape[0]), interpolation=interpolation)
    pred_hm_neg = (pred_hm_neg - pred_hm_neg.min()) / (pred_hm_neg.max() - pred_hm_neg.min() + 1e-8)
    pred_heatmap_color_neg = cv2.applyColorMap((pred_hm_neg * 255).astype(np.uint8), color_map)
    pred_overlay_neg = cv2.addWeighted(img, 1.0 - alpha, pred_heatmap_color_neg, alpha, 0)


# **************************************************************************************************************
    gt_hm_pos = gt_heatmap[0]
    gt_hm_pos = gt_hm_pos.astype(np.float32)
    gt_hm_pos = cv2.resize(gt_hm_pos, (img.shape[1], img.shape[0]), interpolation=interpolation)
    gt_hm_pos = (gt_hm_pos - gt_hm_pos.min()) / (gt_hm_pos.max() - gt_hm_pos.min() + 1e-8)
    gt_heatmap_color_pos = cv2.applyColorMap((gt_hm_pos * 255).astype(np.uint8), color_map)
    gt_overlay_pos = cv2.addWeighted(img, 1.0 - alpha, gt_heatmap_color_pos, alpha, 0)


    gt_hm_neg = gt_heatmap[1]
    gt_hm_neg = gt_hm_neg.astype(np.float32)
    gt_hm_neg = cv2.resize(gt_hm_neg, (img.shape[1], img.shape[0]), interpolation=interpolation)
    gt_hm_neg = (gt_hm_neg - gt_hm_neg.min()) / (gt_hm_neg.max() - gt_hm_neg.min() + 1e-8)
    gt_heatmap_color_neg = cv2.applyColorMap((gt_hm_neg * 255).astype(np.uint8), color_map)
    gt_overlay_neg = cv2.addWeighted(img, 1.0 - alpha, gt_heatmap_color_neg, alpha, 0)
# **************************************************************************************************************




    fig, axes = plt.subplots(2, 3, figsize=(12, 10))

    axes[0, 0].imshow(image)
    axes[0, 0].axis("off")
    axes[1, 0].imshow(image)
    axes[1, 0].axis("off")

    axes[0, 1].imshow(gt_overlay_pos)
    axes[0, 1].axis("off")
    axes[1, 1].imshow(gt_overlay_neg)
    axes[1, 1].axis("off")


    axes[0, 2].imshow(pred_overlay_pos)
    axes[0, 2].axis("off")
    axes[1, 2].imshow(pred_overlay_neg)
    axes[1, 2].axis("off")

    axes[0, 0].set_title("Input")
    axes[0, 1].set_title("Ground Truth")
    axes[0, 2].set_title("Prediction")

    axes[0, 0].set_ylabel("Positive")
    axes[1, 0].set_ylabel("Negative")



def overlay_save(
    image,
    pred_heatmap,
    gt_heatmap,
    alpha=0.4, 
    interpolation=cv2.INTER_LINEAR,
    save_dir="./",
    prefix="sample",
    upscale=1,
    save_format="png",   # "png" or "jpg"
):
    """
    Overlay heatmaps on top of an image and optionally save selected outputs.

    Parameters
    ----------
    image : torch.Tensor | np.ndarray
        Image tensor/array, shape (3, H, W) or (H, W, 3).
    pred_heatmap : torch.Tensor | np.ndarray
        Prediction heatmap, shape (2, h, w).
    gt_heatmap : torch.Tensor | np.ndarray
        Ground-truth heatmap, shape (2, h, w).
    alpha : float, default=0.4
        Heatmap blending strength.
    interpolation : int, default=cv2.INTER_LINEAR
        OpenCV resize interpolation mode.
    save_dir : str | None, default=None
        Directory to save images. If, images are saved in current directory.
    prefix : str, default="sample"
        Prefix for saved filenames.
    upscale : int, default=1
        If >1, saved images are enlarged before saving.
    save_format : str, default="png"
        File format for saved images ("png" recommended for lossless quality).

    Returns
    -------
    dict
        Dictionary containing RGB images:
        {
            "image": image_rgb,
            "pred_overlay_pos": pred_overlay_pos_rgb,
            "pred_overlay_neg": pred_overlay_neg_rgb,
            "gt_overlay_pos": gt_overlay_pos_rgb,
            "gt_overlay_neg": gt_overlay_neg_rgb,
        }
    """

    # Accept torch tensors and numpy arrays
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    if hasattr(pred_heatmap, "detach"):
        pred_heatmap = pred_heatmap.detach().cpu().numpy()
    if hasattr(gt_heatmap, "detach"):
        gt_heatmap = gt_heatmap.detach().cpu().numpy()

    image = np.asarray(image)
    pred_heatmap = np.asarray(pred_heatmap)
    gt_heatmap = np.asarray(gt_heatmap)

    # Convert image to HWC
    if image.ndim != 3:
        raise ValueError(
            f"image must be 3D (CHW or HWC), got shape {image.shape}")

    if image.shape[0] == 3:  # CHW -> HWC
        image = np.transpose(image, (1, 2, 0))
    elif image.shape[2] != 3:
        raise ValueError(
            f"image must have 3 channels, got shape {image.shape}")

    # Convert image to uint8 RGB
    if np.issubdtype(image.dtype, np.floating):
        if image.max() <= 1.0 + 1e-6:
            image_rgb = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            image_rgb = np.clip(image, 0.0, 255.0).astype(np.uint8)
    else:
        image_rgb = np.clip(image, 0, 255).astype(np.uint8)

    # For OpenCV overlay operations, convert RGB -> BGR
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    def make_overlay(base_bgr, hm, alpha, interpolation):
        hm = hm.astype(np.float32)
        hm = cv2.resize(
            hm, (base_bgr.shape[1], base_bgr.shape[0]), interpolation=interpolation)
        hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
        hm_color = cv2.applyColorMap(
            #(hm * 255).astype(np.uint8), cv2.COLORMAP_JET)  # BGR
            (hm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)  # BGR
        overlay_bgr = cv2.addWeighted(
            base_bgr, 1.0 - alpha, hm_color, alpha, 0)
        return overlay_bgr

    # Prediction overlays
    pred_overlay_pos_bgr = make_overlay(
        img_bgr, pred_heatmap[0], alpha, interpolation)
    pred_overlay_neg_bgr = make_overlay(
        img_bgr, pred_heatmap[1], alpha, interpolation)

    # Ground-truth overlays
    gt_overlay_pos_bgr = make_overlay(
        img_bgr, gt_heatmap[0], alpha, interpolation)
    gt_overlay_neg_bgr = make_overlay(
        img_bgr, gt_heatmap[1], alpha, interpolation)

    # Convert back to RGB for matplotlib / return
    pred_overlay_pos_rgb = cv2.cvtColor(
        pred_overlay_pos_bgr, cv2.COLOR_BGR2RGB)
    pred_overlay_neg_rgb = cv2.cvtColor(
        pred_overlay_neg_bgr, cv2.COLOR_BGR2RGB)
    gt_overlay_pos_rgb = cv2.cvtColor(gt_overlay_pos_bgr, cv2.COLOR_BGR2RGB)
    gt_overlay_neg_rgb = cv2.cvtColor(gt_overlay_neg_bgr, cv2.COLOR_BGR2RGB)

    # Save selected outputs at full resolution
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        def maybe_upscale(img_rgb, factor):
            if factor == 1:
                return img_rgb
            h, w = img_rgb.shape[:2]
            return cv2.resize(
                img_rgb,
                (w * factor, h * factor),
                interpolation=cv2.INTER_CUBIC
            )

        image_to_save = maybe_upscale(image_rgb, upscale)
        pred_pos_to_save = maybe_upscale(pred_overlay_pos_rgb, upscale)
        pred_neg_to_save = maybe_upscale(pred_overlay_neg_rgb, upscale)

        ext = save_format.lower()
        if ext not in {"png", "jpg", "jpeg"}:
            raise ValueError("save_format must be 'png', 'jpg', or 'jpeg'")

        # cv2.imwrite expects BGR
        if ext == "png":
            cv2.imwrite(
                os.path.join(save_dir, f"{prefix}_image.png"),
                cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR)
            )
            cv2.imwrite(
                os.path.join(save_dir, f"{prefix}_pred_overlay_pos.png"),
                cv2.cvtColor(pred_pos_to_save, cv2.COLOR_RGB2BGR)
            )
            cv2.imwrite(
                os.path.join(save_dir, f"{prefix}_pred_overlay_neg.png"),
                cv2.cvtColor(pred_neg_to_save, cv2.COLOR_RGB2BGR)
            )
        else:
            jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, 100]
            cv2.imwrite(
                os.path.join(save_dir, f"{prefix}_image.jpg"),
                cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR),
                jpeg_params
            )
            cv2.imwrite(
                os.path.join(save_dir, f"{prefix}_pred_overlay_pos.jpg"),
                cv2.cvtColor(pred_pos_to_save, cv2.COLOR_RGB2BGR),
                jpeg_params
            )
            cv2.imwrite(
                os.path.join(save_dir, f"{prefix}_pred_overlay_neg.jpg"),
                cv2.cvtColor(pred_neg_to_save, cv2.COLOR_RGB2BGR),
                jpeg_params
            )

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(12, 10), dpi=200)

    axes[0, 0].imshow(image_rgb)
    axes[0, 0].axis("off")
    axes[1, 0].imshow(image_rgb)
    axes[1, 0].axis("off")

    axes[0, 1].imshow(gt_overlay_pos_rgb)
    axes[0, 1].axis("off")
    axes[1, 1].imshow(gt_overlay_neg_rgb)
    axes[1, 1].axis("off")

    axes[0, 2].imshow(pred_overlay_pos_rgb)
    axes[0, 2].axis("off")
    axes[1, 2].imshow(pred_overlay_neg_rgb)
    axes[1, 2].axis("off")

    axes[0, 0].set_title("Input")
    axes[0, 1].set_title("Ground Truth")
    axes[0, 2].set_title("Prediction")

    axes[0, 0].set_ylabel("Positive")
    axes[1, 0].set_ylabel("Negative")

    plt.tight_layout()
    plt.show()

    return {
        "image": image_rgb,
        "pred_overlay_pos": pred_overlay_pos_rgb,
        "pred_overlay_neg": pred_overlay_neg_rgb,
        "gt_overlay_pos": gt_overlay_pos_rgb,
        "gt_overlay_neg": gt_overlay_neg_rgb,
    }





    
def overlay_gt(
    image,
    gt_heatmap,
    alpha=0.4,
    interpolation=cv2.INTER_LINEAR,
    save_dir="./",
    prefix="sample",
    upscale=1,
    save_format="png",
    show=False,
):
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    if hasattr(gt_heatmap, "detach"):
        gt_heatmap = gt_heatmap.detach().cpu().numpy()

    image = np.asarray(image)
    gt_heatmap = np.asarray(gt_heatmap)

    if image.ndim != 3:
        raise ValueError(
            f"image must be 3D (CHW or HWC), got shape {image.shape}")

    if image.shape[0] == 3:  # CHW -> HWC
        image = np.transpose(image, (1, 2, 0))
    elif image.shape[2] != 3:
        raise ValueError(
            f"image must have 3 channels, got shape {image.shape}")
    
    if np.issubdtype(image.dtype, np.floating):
        if image.max() <= 1.0 + 1e-6:
            image_rgb = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            image_rgb = np.clip(image, 0.0, 255.0).astype(np.uint8)
    else:
        image_rgb = np.clip(image, 0, 255).astype(np.uint8)


    # For OpenCV overlay operations, convert RGB -> BGR
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


    def make_overlay(base_bgr, hm, alpha, interpolation):
        hm = hm.astype(np.float32)
        hm = cv2.resize(
            hm, (base_bgr.shape[1], base_bgr.shape[0]), interpolation=interpolation)
        hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
        hm_color = cv2.applyColorMap(
            #(hm * 255).astype(np.uint8), cv2.COLORMAP_JET)  # BGR
            (hm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)  # BGR
        overlay_bgr = cv2.addWeighted(
            base_bgr, 1.0 - alpha, hm_color, alpha, 0)
        return overlay_bgr


    gt_overlay_pos_bgr = make_overlay(
        img_bgr, gt_heatmap[0], alpha, interpolation)
    gt_overlay_neg_bgr = make_overlay(
        img_bgr, gt_heatmap[1], alpha, interpolation)
    

    gt_overlay_pos_rgb = cv2.cvtColor(gt_overlay_pos_bgr, cv2.COLOR_BGR2RGB)
    gt_overlay_neg_rgb = cv2.cvtColor(gt_overlay_neg_bgr, cv2.COLOR_BGR2RGB)

    saved_paths = {}
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        ext = save_format.lower()
        if ext == "jpeg":
            ext = "jpg"
        if ext not in {"png", "jpg"}:
            raise ValueError("save_format must be 'png', 'jpg', or 'jpeg'")

        def maybe_upscale(img_rgb, factor):
            if factor == 1:
                return img_rgb
            h, w = img_rgb.shape[:2]
            return cv2.resize(
                img_rgb, (w * factor, h * factor), interpolation=cv2.INTER_CUBIC
            )

        image_to_save = maybe_upscale(image_rgb, upscale)
        gt_pos_to_save = maybe_upscale(gt_overlay_pos_rgb, upscale)
        gt_neg_to_save = maybe_upscale(gt_overlay_neg_rgb, upscale)
        panel_to_save = np.concatenate(
            [image_to_save, gt_pos_to_save, gt_neg_to_save], axis=1
        )

        jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, 100] if ext == "jpg" else None

        def save_rgb(path, img_rgb):
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            if jpeg_params is None:
                cv2.imwrite(path, img_bgr)
            else:
                cv2.imwrite(path, img_bgr, jpeg_params)

        image_path = os.path.join(save_dir, f"{prefix}_image.{ext}")
        gt_pos_path = os.path.join(save_dir, f"{prefix}_gt_overlay_pos.{ext}")
        gt_neg_path = os.path.join(save_dir, f"{prefix}_gt_overlay_neg.{ext}")
        panel_path = os.path.join(save_dir, f"{prefix}_panel.{ext}")

        save_rgb(image_path, image_to_save)
        save_rgb(gt_pos_path, gt_pos_to_save)
        save_rgb(gt_neg_path, gt_neg_to_save)
        save_rgb(panel_path, panel_to_save)

        saved_paths = {
            "image_path": image_path,
            "gt_overlay_pos_path": gt_pos_path,
            "gt_overlay_neg_path": gt_neg_path,
            "panel_path": panel_path,
        }

    if show:
        fig, axes = plt.subplots(1, 3, figsize=(10, 10), dpi=200)

        axes[0].imshow(image_rgb)
        axes[0].axis("off")
        axes[0].set_title("original")

        axes[1].imshow(gt_overlay_pos_rgb)
        axes[1].axis("off")
        axes[1].set_title("Ki67+ annotated")

        axes[2].imshow(gt_overlay_neg_rgb)
        axes[2].axis("off")
        axes[2].set_title("Ki67- annotated")

        plt.tight_layout()
        plt.show()

    return {
        "image": image_rgb,
        "gt_overlay_pos": gt_overlay_pos_rgb,
        "gt_overlay_neg": gt_overlay_neg_rgb,
        **saved_paths,
    }
