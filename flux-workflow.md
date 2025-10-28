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
This workflow is designed for RTX 4090 class GPUs and automates high-resolution exterior look-development with the **FLUX** backbone. It ingests one or more viewport renders, preserves façade geometry and materials, and populates entourage (people, cars, vegetation) while copying the style from a reference shot. An Ollama-served LLM node analyses the inputs, generates prompts, and tunes the sampler/ControlNet/IPAdapter weights per image.

## Node Layout in the JSON Export
```
[Input & Guidance]
  • LoadImage (source render) → VAEEncode
  • LoadImage (style reference) → CLIPVisionEncode + Ollama advisor
  • String (user prompt stub)
  • OllamaParameterAdvisor (custom PyNode) → emits prompts, sampler settings, mask weights, upscale factor, and metadata

[Geometry Preservation]
  • GroundedSAMSegment → MaskMorphology → SetLatentNoiseMask (protects façade)
  • LineArtPreprocessor + ControlNetLoader (lineart) → ControlNetApplyAdvanced
  • DepthPreprocessor + ControlNetLoader (depth) → ControlNetApplyAdvanced

[Style & Conditioning]
  • CheckpointLoaderSimple (FLUX base/refiner)
  • CLIPTextEncode (positive/negative) driven by Ollama prompts
  • IPAdapterModelLoader + CLIPVisionEncode (style image) → IPAdapterApply (style weight from advisor)

[Sampling & Refinement]
  • KSampler (receives model, conditioning, latent with façade mask, and advisor settings)
  • VAEDecode → UpscaleModelLoader (Real-ESRGAN) → ImageUpscaleWithModel (scale from advisor)

[Outputs]
  • SaveImage (final render)
  • SaveMetadata (stores advisor JSON for reproducibility)
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
   - These values are distributed to KSampler, ControlNet weights, the IPAdapter blend, Real-ESRGAN scale, and metadata logger. If the HTTP request fails or the model returns invalid JSON the node falls back to conservative defaults (22 steps / CFG 6.5 / weights around 0.7) and annotates the output metadata.

2. **Dynamic Mask Routing**
   - `GroundedSAMSegment` isolates the main building. `MaskMorphology` dilates the mask before feeding it to `SetLatentNoiseMask`, ensuring façade pixels remain untouched during sampling.
   - The advisor can request additional dilation/erosion through metadata if the confidence score drops.

3. **ControlNet Stack**
   - Lineart ControlNet locks façade contours. The weight is tuned automatically (0–1) from the advisor.
   - Depth ControlNet maintains camera layout and ground-plane relationships; weights drop for orthographic or already depth-rich renders.

4. **Style Consistency via IPAdapter**
   - `IPAdapterApply` blends conditioning from the positive prompt with the style embedding produced by `CLIPVisionEncode`. The advisor scales the weight (0–1) based on CLIP cosine similarity between source and reference to avoid over-stylisation.

5. **Entourage Prompting**
   - The LLM merges automatic CLIP tags (e.g., `modern office tower, plaza, evening`) with the short user prompt to produce detailed positive prompts (people, props, weather) and explicit negative prompts (geometry distortions, extra stories, double façades).

6. **Post-Processing**
   - `ImageUpscaleWithModel` runs Real-ESRGAN only when the advisor returns `upscale > 1.0`. You can map a UI slider to the `upscale_scale` input for manual overrides.
   - `SaveMetadata` persists the full advisor response to `/ComfyUI/output/metadata/` for reproducibility.

## Required Assets & Custom Nodes
- **Models**: `flux1-dev` base, optional `flux1-refiner`, and official FLUX VAE.
- **ControlNets**: FLUX-compatible lineart and depth checkpoints (place in `models/controlnet/flux`).
- **IPAdapter**: `style-plus` (or another FLUX-ready IPAdapter) in `models/ipadapter/`.
- **Upscaler**: Real-ESRGAN `real_esrgan_x4plus.pth` in `models/upscale/`.
- **Custom Nodes**:
  - [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) (provides the preprocessor nodes used here).
  - Grounded-SAM integration (e.g., [ComfyUI-Advanced-ControlNet](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet)) for automatic façade masking.
  - **Ollama Parameter Advisor** – included in this repository under [`custom_nodes/ollama_parameter_advisor.py`](./custom_nodes/ollama_parameter_advisor.py). Copy that file into your local `ComfyUI/custom_nodes/` folder and restart ComfyUI so that the workflow can load the node class.

## Usage Tips
1. Import the JSON into ComfyUI (`Settings → Workflow → Load`).
2. Update file widgets in `LoadImage` nodes to point to your source/style images. Batch-friendly variants can replace them if needed.
3. Configure the Ollama endpoint/model in the advisor node widgets. Provide a short user request in the `String` node.
4. Trigger the graph. The advisor will generate prompts and parameter values for each run; metadata is stored alongside outputs for auditability.
5. Adjust ControlNet/IPAdapter weights manually if you want to override the LLM decisions—each node exposes a widget while still receiving live values from the advisor.

### Troubleshooting Import Issues
- If ComfyUI refuses to load the JSON, double-check that you downloaded it directly (see the links above) rather than pasting through a text editor that might inject BOM characters.
- Verify that `ollama_parameter_advisor.py` is present in `ComfyUI/custom_nodes/` and that the console shows it loading without errors on startup.
- Missing third-party nodes (Grounded SAM, Impact Pack, IPAdapter) will appear as “Unknown Node” placeholders. Install the listed custom node packs and reload the workflow.

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
