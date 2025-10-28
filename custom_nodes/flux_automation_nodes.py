import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from transformers import (
        AutoImageProcessor,
        AutoModelForSemanticSegmentation,
        DPTForDepthEstimation,
        DPTImageProcessor,
    )
except ImportError as exc:  # pragma: no cover - informative error inside ComfyUI console
    raise ImportError(
        "The flux automation nodes require the 'transformers' package. "
        "Install it with 'pip install transformers accelerate safetensors'."
    ) from exc


def _tensor_to_pil(image_tensor):
    if isinstance(image_tensor, list):
        image_tensor = image_tensor[0]
    if hasattr(image_tensor, "cpu"):
        image_tensor = image_tensor.cpu()
    array = image_tensor.numpy() if hasattr(image_tensor, "numpy") else np.asarray(image_tensor)
    array = np.clip(array, 0.0, 1.0)
    array = (array * 255.0).astype(np.uint8)
    return Image.fromarray(array)


def _mask_to_tensor(mask):
    if hasattr(mask, "cpu"):
        mask = mask.cpu().numpy()
    elif isinstance(mask, torch.Tensor):
        mask = mask.numpy()
    return torch.from_numpy(np.asarray(mask, dtype=np.float32))


class BuildingSegmentation:
    """Semantic segmentation tuned to isolate building structures."""

    _processor = None
    _model = None
    _keywords = (
        "building",
        "house",
        "structure",
        "tower",
        "skyscraper",
        "architecture",
        "facade",
        "apartment",
        "temple",
        "church",
    )

    @classmethod
    def _load_model(cls):
        if cls._model is None:
            cls._processor = AutoImageProcessor.from_pretrained(
                "nvidia/segformer-b3-finetuned-ade-512-512"
            )
            cls._model = AutoModelForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b3-finetuned-ade-512-512"
            )
            cls._model.eval()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "building_threshold": (
                    "FLOAT",
                    {
                        "default": 0.4,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Probability cut-off for keeping pixels inside the building mask.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MASK", "MASK", "IMAGE")
    RETURN_NAMES = ("building_mask", "environment_mask", "preview")
    FUNCTION = "segment"
    CATEGORY = "image/segmentation"

    @classmethod
    def _building_ids(cls):
        assert cls._model is not None
        ids = []
        for idx, label in cls._model.config.id2label.items():
            name = label.lower()
            if any(keyword in name for keyword in cls._keywords):
                ids.append(int(idx))
        if not ids:
            ids = [0]
        return ids

    @classmethod
    def segment(cls, image, building_threshold=0.4):
        cls._load_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = cls._model.to(device)

        pil_image = _tensor_to_pil(image)
        processor = cls._processor
        inputs = processor(images=pil_image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        upsampled = F.interpolate(
            logits,
            size=pil_image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        probs = upsampled.softmax(dim=1)
        building_indices = cls._building_ids()
        building_prob = probs[:, building_indices, :, :].max(dim=1).values.squeeze(0)
        building_mask = (building_prob >= building_threshold).float()
        environment_mask = 1.0 - building_mask

        building_mask_np = building_mask.cpu().numpy().astype(np.float32)
        environment_mask_np = environment_mask.cpu().numpy().astype(np.float32)

        preview = np.asarray(pil_image).astype(np.float32) / 255.0
        overlay = preview.copy()
        overlay[..., 0] = np.clip(overlay[..., 0] + building_mask_np * 0.5, 0.0, 1.0)
        overlay[..., 1] = np.clip(overlay[..., 1] + environment_mask_np * 0.2, 0.0, 1.0)

        return (building_mask_np, environment_mask_np, overlay.astype(np.float32))


class MaskMorphology:
    """Simple morphological operations implemented with PyTorch kernels."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "operation": (
                    "STRING",
                    {
                        "default": "erode",
                        "choices": ["erode", "dilate", "blur"],
                        "tooltip": "Morphological operation applied to the incoming mask.",
                    },
                ),
                "iterations": (
                    "INT",
                    {"default": 1, "min": 1, "max": 10, "step": 1},
                ),
                "kernel_size": (
                    "INT",
                    {"default": 3, "min": 1, "max": 31, "step": 2},
                ),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "process"
    CATEGORY = "image/mask"

    @staticmethod
    def _apply(mask_tensor, operation, kernel_size):
        pad = kernel_size // 2
        if operation == "erode":
            result = 1.0 - F.max_pool2d(1.0 - mask_tensor, kernel_size, stride=1, padding=pad)
        elif operation == "dilate":
            result = F.max_pool2d(mask_tensor, kernel_size, stride=1, padding=pad)
        else:  # blur
            kernel = torch.ones((1, 1, kernel_size, kernel_size), device=mask_tensor.device)
            kernel = kernel / kernel.sum()
            result = F.conv2d(mask_tensor, kernel, padding=pad)
        return torch.clamp(result, 0.0, 1.0)

    @classmethod
    def process(cls, mask, operation="erode", iterations=1, kernel_size=3):
        mask_tensor = _mask_to_tensor(mask).float().unsqueeze(0).unsqueeze(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mask_tensor = mask_tensor.to(device)
        kernel_size = max(1, kernel_size | 1)
        result = mask_tensor
        for _ in range(max(1, iterations)):
            result = cls._apply(result, operation, kernel_size)
        return (result.squeeze().cpu().numpy().astype(np.float32),)


class DepthPreprocessor:
    """Generates a normalized depth map using the Intel DPT model."""

    _processor = None
    _model = None

    @classmethod
    def _load_model(cls):
        if cls._model is None:
            cls._processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
            cls._model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
            cls._model.eval()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "min_depth": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "max_depth": (
                    "FLOAT",
                    {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "estimate"
    CATEGORY = "image/preprocess"

    @classmethod
    def estimate(cls, image, min_depth=0.1, max_depth=0.9):
        cls._load_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = cls._model.to(device)

        pil_image = _tensor_to_pil(image)
        processor = cls._processor
        inputs = processor(images=pil_image, return_tensors="pt").to(device)

        with torch.no_grad():
            depth = model(**inputs).predicted_depth
        depth = depth.unsqueeze(1)
        depth = F.interpolate(
            depth,
            size=pil_image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        depth = depth.squeeze().cpu()
        depth = depth - depth.min()
        if depth.max() > 0:
            depth = depth / depth.max()
        depth = torch.clamp(depth, min=min_depth, max=max_depth)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        image_np = depth.numpy().astype(np.float32)
        image_np = np.repeat(image_np[:, :, None], 3, axis=2)
        return (image_np,)


NODE_CLASS_MAPPINGS = {
    "BuildingSegmentation": BuildingSegmentation,
    "MaskMorphology": MaskMorphology,
    "DepthPreprocessor": DepthPreprocessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BuildingSegmentation": "Building Segmentation (SegFormer)",
    "MaskMorphology": "Mask Morphology",
    "DepthPreprocessor": "Depth Preprocessor (DPT)",
}
