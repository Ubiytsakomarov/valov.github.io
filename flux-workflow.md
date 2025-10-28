# Flux-Based ComfyUI Workflow for Architectural Exterior Enhancement

## Download
- **Быстрая загрузка через браузер**
  1. Откройте прямую ссылку: <https://valov.github.io/flux-workflow.json> (или raw-версию <https://raw.githubusercontent.com/valov/valov.github.io/main/flux-workflow.json>).
  2. После открытия страницы нажмите `Ctrl + S` и в диалоге сохранения выберите **Save as type → All Files** / «Все файлы».
  3. Убедитесь, что имя файла заканчивается на `.json`, а кодировка выставлена **UTF-8** без BOM.
  4. Нажмите «Сохранить» и убедитесь, что размер файла в проводнике больше 1 КБ (значит, содержимое скачалось полностью).
- **PowerShell-скрипт для Windows**
  ```powershell
  cd C:\папка\куда\сохранить
  ./download_flux_workflow.ps1 -Owner "valov" -Repository "valov.github.io" -Branch "main" -Output "flux-workflow.json"
  ```
  Скрипт лежит в каталоге [`scripts/download_flux_workflow.ps1`](./scripts/download_flux_workflow.ps1); его можно запускать и с другими параметрами (`-Owner`/`-Repository`), если репозиторий расположен в другом аккаунте или ветке.
  Если Windows блокирует запуск, в том же окне PowerShell выполните `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.
- После загрузки импортируйте файл в ComfyUI через **Settings → Workflow → Load**.
- Положите указанные чекпойнты (`flux1-dev`, `flux1-refiner`, ControlNets, IPAdapter, Real-ESRGAN и т.д.) в соответствующие папки `models` до загрузки графа.

## Overview
This workflow targets RTX 4090–class GPUs and automates high-resolution exterior look-development with the **FLUX** backbone. It ingests a viewport render, keeps façade geometry and materials intact, populates entourage (people, cars, vegetation), and mirrors the mood of a style reference. A bundled Ollama-driven advisor analyses the inputs, produces prompts, and tunes sampler / ControlNet weights and upscaling for each image automatically.

## Node Layout in the JSON Export
```
[Input & Guidance]
  • LoadImage (source render) → VAEEncode
  • LoadImage (style reference) → Ollama advisor
  • String (user prompt stub)
  • OllamaParameterAdvisor (custom node) → emits prompts, sampler settings, ControlNet weights, upscale factor, and metadata

[Geometry Preservation]
  • BuildingSegmentation (custom node, SegFormer) → MaskMorphology → composite mask
  • LineArtPreprocessor + ControlNetLoader (lineart) → ControlNetApplyAdvanced
  • DepthPreprocessor (custom DPT) + ControlNetLoader (depth) → ControlNetApplyAdvanced

[Conditioning & Sampling]
  • CheckpointLoaderSimple (FLUX base/refiner)
  • CLIPTextEncode (positive/negative) driven by Ollama prompts
  • KSampler (receives model, conditioning, latent encode, and advisor settings)

[Post Processing]
  • VAEDecode → ImageCompositeMasked (merges generated entourage with protected façade)
  • ImageScaleBy (upscale factor from advisor)

[Outputs]
  • SaveImage (final render)
```

## Auto-Configuration Strategy
1. **LLM Parameter Advisor**
   - `OllamaParameterAdvisor` (the bundled custom node) summarises image dimensions and brightness/saturation stats before sending them—together with the user stub prompt—to your local Ollama instance.
   - The helper asks the model (default `llama3-vision`) for a JSON response shaped like:
     ```json
     {
       "positive_prompt": "...",
       "negative_prompt": "...",
       "steps": 26,
       "cfg": 7.2,
       "sampler": "dpmpp_2m",
       "scheduler": "karras",
       "geometry_weight": 0.85,
       "depth_weight": 0.55,
       "style_weight": 0.65,
       "seed": 3324551091,
       "upscale_scale": 1.6,
       "negative_library": ["..."]
     }
     ```
   - These values are distributed to `KSampler`, ControlNet strengths, the composite mask, and the ImageScale node. If the HTTP request fails or the model returns invalid JSON the node falls back to conservative defaults (22 steps / CFG 6.5 / weights around 0.7) and annotates the output metadata.

2. **Dynamic Mask Routing**
   - `BuildingSegmentation` (SegFormer) isolates the main building automatically. `MaskMorphology` erodes the façade mask and softly dilates the environment mask before compositing, ensuring façade pixels remain untouched during sampling.
   - You can tweak the widgets on these nodes (threshold, iterations, kernel size) if the automatic mask requires refinement.

3. **ControlNet Stack**
   - Lineart ControlNet locks façade contours. The weight is tuned automatically (0–1) from the advisor.
   - Depth ControlNet maintains camera layout and ground-plane relationships; weights drop for orthographic or already depth-rich renders.

4. **Style Consistency via Prompting**
   - The advisor reads both the source and reference images, summarises their content, and folds those descriptors into the positive prompt. This avoids the need for external IPAdapter nodes while still reflecting the reference mood.

5. **Entourage Prompting**
   - The LLM merges automatic CLIP tags (e.g., `modern office tower, plaza, evening`) with the short user prompt to produce detailed positive prompts (people, props, weather) and explicit negative prompts (geometry distortions, extra stories, double façades).

6. **Post-Processing**
   - `ImageScaleBy` upsamples according to the advisor’s recommendation (default 1.6×). Set it to 1.0 to disable scaling.
   - `ImageCompositeMasked` keeps the original façade while blending the generated entourage back into the shot.

## Required Assets & Custom Nodes
- **Models**: `flux1-dev` base, optional `flux1-refiner`, and official FLUX VAE.
- **ControlNets**: FLUX-compatible lineart and depth checkpoints (place in `models/controlnet/flux`).
- **Custom Nodes shipped here**:
  - [`custom_nodes/ollama_parameter_advisor.py`](./custom_nodes/ollama_parameter_advisor.py) – queries your local Ollama server.
  - [`custom_nodes/flux_automation_nodes.py`](./custom_nodes/flux_automation_nodes.py) – provides `BuildingSegmentation`, `MaskMorphology`, and `DepthPreprocessor` so the workflow loads without third-party packs.
- **Python dependencies** (install inside the ComfyUI environment):
  ```bash
  pip install transformers accelerate safetensors
  ```
  The segmentation/depth nodes download the required weights (`SegFormer-B3 ADE20K` and `Intel DPT Hybrid`) the first time they run.

## Usage Tips
1. Import the JSON into ComfyUI (`Settings → Workflow → Load`).
2. Update file widgets in `LoadImage` nodes to point to your source/style images. Batch-friendly variants can replace them if needed.
3. Configure the Ollama endpoint/model in the advisor node widgets. Provide a short user request in the `String` node.
4. Trigger the graph. The advisor will generate prompts and parameter values for each run; the composite stage keeps façade pixels from changing.
5. Adjust ControlNet weights or mask morphology manually if you want to override the LLM decisions—each node exposes widgets while still receiving live values from the advisor.

### Troubleshooting Import Issues
- If ComfyUI refuses to load the JSON, double-check that you downloaded it directly (see the links above) rather than pasting through a text editor that might inject BOM characters.
- Verify that `ollama_parameter_advisor.py` is present in `ComfyUI/custom_nodes/` and that the console shows it loading without errors on startup.
- Missing bundled nodes indicate that the two Python files above were not copied into `ComfyUI/custom_nodes/`. Restart ComfyUI after adding them so the classes register.

## Future Extensions
- Add a `ForEach` container around the sampler section to process multiple camera angles automatically.
- Introduce optional weather/time-of-day variants by asking the advisor to emit multiple prompt sets per image.
- Chain a second FLUX `KSampler` for refiner passes when pushing beyond 4K outputs.

## Performance Notes for RTX 4090
- Enable `--xformers` and `--fp16` in ComfyUI for ~9–12 s per 2K frame with both ControlNets and IPAdapter active.
- The façade-preserving noise mask keeps VRAM spikes minimal; two concurrent renders at 2048px typically fit within 24 GB.
- Use tiled VAE decode or disable Real-ESRGAN on >4096 px renders to avoid out-of-memory errors.

## Summary
The provided JSON implements a FLUX-first ComfyUI workflow that keeps building geometry intact, inherits façade styling, and auto-populates entourage and post-processing parameters through an Ollama-guided advisor. Import the file, point the loaders at your renders/reference, and the graph will self-configure to deliver consistent exterior visualisations without manual parameter tuning.
