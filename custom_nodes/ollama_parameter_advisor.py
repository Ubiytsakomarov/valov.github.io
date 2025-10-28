import base64
import json
from io import BytesIO
from typing import Any, Dict

import numpy as np
import requests
from PIL import Image


class OllamaParameterAdvisor:
    """ComfyUI custom node that asks a local Ollama instance to suggest prompts
    and sampler settings for the Flux architectural workflow."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "style_image": ("IMAGE",),
                "user_prompt": ("STRING", {"default": "", "multiline": True}),
                "endpoint": (
                    "STRING",
                    {
                        "default": "http://localhost:11400/api/generate",
                        "multiline": False,
                        "tooltip": "HTTP endpoint of the Ollama generate API.",
                    },
                ),
                "model": (
                    "STRING",
                    {
                        "default": "llama3-vision",
                        "multiline": False,
                        "tooltip": "Ollama model name to query (vision capable).",
                    },
                ),
                "style_preset": (
                    "STRING",
                    {
                        "default": "flux-architect",
                        "multiline": False,
                        "tooltip": "Short hint passed to the LLM about the desired visual style.",
                    },
                ),
                "min_steps": (
                    "INT",
                    {
                        "default": 18,
                        "min": 4,
                        "max": 80,
                        "step": 1,
                        "tooltip": "Lower bound for sampler steps when the LLM omits the field.",
                    },
                ),
                "min_cfg": (
                    "FLOAT",
                    {
                        "default": 5.5,
                        "min": 0.0,
                        "max": 15.0,
                        "step": 0.1,
                        "tooltip": "Lower bound for classifier-free guidance scale.",
                    },
                ),
                "min_upscale": (
                    "FLOAT",
                    {
                        "default": 1.2,
                        "min": 1.0,
                        "max": 4.0,
                        "step": 0.1,
                        "tooltip": "Minimum upscaling factor enforced when LLM result is lower.",
                    },
                ),
            }
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "INT",
        "FLOAT",
        "STRING",
        "STRING",
        "FLOAT",
        "FLOAT",
        "FLOAT",
        "INT",
        "FLOAT",
        "DICT",
    )
    RETURN_NAMES = (
        "positive_prompt",
        "negative_prompt",
        "steps",
        "cfg",
        "sampler",
        "scheduler",
        "geometry_weight",
        "depth_weight",
        "style_weight",
        "seed",
        "upscale_scale",
        "metadata",
    )
    FUNCTION = "advise"
    CATEGORY = "conditioning/ollama"

    def _image_to_base64(self, tensor) -> str:
        if tensor is None:
            return ""
        if isinstance(tensor, list) or getattr(tensor, "ndim", 0) == 4:
            array = tensor[0]
        else:
            array = tensor
        if hasattr(array, "cpu"):
            array = array.cpu().numpy()
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
        image = Image.fromarray(array)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _summarise_image(self, tensor) -> Dict[str, Any]:
        if tensor is None:
            return {}
        if isinstance(tensor, list) or getattr(tensor, "ndim", 0) == 4:
            array = tensor[0]
        else:
            array = tensor
        if hasattr(array, "cpu"):
            array = array.cpu().numpy()
        array = np.clip(array, 0.0, 1.0)
        height, width, _ = array.shape
        brightness = float(array.mean())
        saturation = float(np.std(array, axis=2).mean())
        return {
            "width": int(width),
            "height": int(height),
            "avg_brightness": round(brightness, 4),
            "avg_saturation": round(saturation, 4),
        }

    def _build_prompt(
        self,
        style_preset: str,
        user_prompt: str,
        source_stats: Dict[str, Any],
        style_stats: Dict[str, Any],
    ) -> str:
        stats_block = json.dumps(
            {
                "source": source_stats,
                "style": style_stats,
                "user_prompt": user_prompt,
                "style_preset": style_preset,
            },
            ensure_ascii=False,
        )
        return (
            "You are an assistant that produces JSON with rendering parameters for the Flux architectural ComfyUI workflow. "
            "Always respond with a single JSON object. Avoid prose. Keys must include:\n"
            "  positive_prompt (string)\n"
            "  negative_prompt (string)\n"
            "  steps (int)\n"
            "  cfg (float)\n"
            "  sampler (string)\n"
            "  scheduler (string)\n"
            "  geometry_weight (float between 0 and 1)\n"
            "  depth_weight (float between 0 and 1)\n"
            "  style_weight (float between 0 and 1.5)\n"
            "  seed (int)\n"
            "  upscale_scale (float between 1 and 4)\n"
            "  negative_library (array of strings)\n"
            "  notes (string with short reasoning).\n"
            "Use the analysis JSON below as context and respect the geometry of the source facade.\n"
            f"Context: {stats_block}\n"
            "Return only JSON."
        )

    def _call_ollama(self, endpoint: str, model: str, prompt: str, images):
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if images:
            payload["images"] = images
        response = requests.post(endpoint, json=payload, timeout=90)
        response.raise_for_status()
        data = response.json()
        if "response" in data:
            return data["response"]
        if "message" in data:
            return data["message"]
        return json.dumps({})

    def _coerce(self, result: Dict[str, Any], key: str, default: Any):
        value = result.get(key, default)
        if isinstance(default, float):
            try:
                return float(value)
            except (TypeError, ValueError):
                return default
        if isinstance(default, int):
            try:
                return int(value)
            except (TypeError, ValueError):
                return default
        return value if value is not None else default

    def advise(
        self,
        source_image,
        style_image,
        user_prompt,
        endpoint,
        model,
        style_preset,
        min_steps,
        min_cfg,
        min_upscale,
    ):
        source_stats = self._summarise_image(source_image)
        style_stats = self._summarise_image(style_image)
        prompt = self._build_prompt(style_preset, user_prompt, source_stats, style_stats)

        encoded_images = []
        source_b64 = self._image_to_base64(source_image)
        style_b64 = self._image_to_base64(style_image)
        if source_b64:
            encoded_images.append(source_b64)
        if style_b64:
            encoded_images.append(style_b64)

        try:
            raw_response = self._call_ollama(endpoint, model, prompt, encoded_images)
            parsed_start = raw_response.find("{")
            parsed_end = raw_response.rfind("}")
            if parsed_start == -1 or parsed_end == -1:
                raise ValueError("No JSON object returned")
            parsed = json.loads(raw_response[parsed_start : parsed_end + 1])
        except Exception as exc:  # pylint: disable=broad-except
            parsed = {
                "positive_prompt": (
                    "architectural exterior, flux-inspired cinematic lighting, preserved facade geometry, "
                    "realistic entourage of people, vehicles, lush vegetation"
                ),
                "negative_prompt": "blurry, deformed, duplicated, distorted perspective, low quality",
                "steps": max(min_steps, 22),
                "cfg": max(min_cfg, 6.5),
                "sampler": "dpmpp_2m",
                "scheduler": "normal",
                "geometry_weight": 0.85,
                "depth_weight": 0.65,
                "style_weight": 0.75,
                "seed": 123456789,
                "upscale_scale": max(min_upscale, 1.8),
                "notes": f"Fallback activated: {exc}",
                "negative_library": [
                    "asymmetric building",
                    "broken proportions",
                    "oversaturated colors",
                    "weird anatomy",
                ],
            }

        result = {
            "positive_prompt": parsed.get("positive_prompt", ""),
            "negative_prompt": parsed.get("negative_prompt", ""),
            "steps": max(self._coerce(parsed, "steps", min_steps), min_steps),
            "cfg": max(self._coerce(parsed, "cfg", min_cfg), min_cfg),
            "sampler": parsed.get("sampler", "dpmpp_2m"),
            "scheduler": parsed.get("scheduler", "normal"),
            "geometry_weight": float(np.clip(self._coerce(parsed, "geometry_weight", 0.85), 0.0, 1.0)),
            "depth_weight": float(np.clip(self._coerce(parsed, "depth_weight", 0.6), 0.0, 1.0)),
            "style_weight": float(np.clip(self._coerce(parsed, "style_weight", 0.75), 0.0, 1.5)),
            "seed": self._coerce(parsed, "seed", 123456789),
            "upscale_scale": float(max(self._coerce(parsed, "upscale_scale", 1.8), min_upscale)),
            "notes": parsed.get("notes", ""),
            "negative_library": parsed.get("negative_library", []),
        }

        metadata = {
            "advisor": "ollama",
            "model": model,
            "endpoint": endpoint,
            "style_preset": style_preset,
            "user_prompt": user_prompt,
            "source_stats": source_stats,
            "style_stats": style_stats,
            "raw_result": result,
        }

        return (
            result["positive_prompt"],
            result["negative_prompt"],
            result["steps"],
            result["cfg"],
            result["sampler"],
            result["scheduler"],
            result["geometry_weight"],
            result["depth_weight"],
            result["style_weight"],
            result["seed"],
            result["upscale_scale"],
            metadata,
        )


NODE_CLASS_MAPPINGS = {
    "OllamaParameterAdvisor": OllamaParameterAdvisor,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaParameterAdvisor": "Ollama Parameter Advisor",
}
