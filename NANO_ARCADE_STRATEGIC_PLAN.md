# Nano Arcade — Strategic Master Plan

**Status:** Active strategy  
**Target:** CPU-first, local, bilingual Pixel Art character generation  
**Primary output:** Sharp RGBA sprites at 64×64, with a validated path to 128×128 and 256×256  
**Runtime targets:** HP ZBook 15 / 16 GB RAM / Quadro K2100M 2 GB when available / CPU-only fallback / Android local runtime

## 1. Non-negotiable product goals

Nano Arcade must generate recognizable characters rather than noise, blobs, or decorative panels. It must support Arabic and English prompts, deterministic seeds, random generation, image-to-image conditioning, and short identity-consistent animation. The project must preserve the existing tools, datasets, taxonomies, checkpoints, and Android-facing concepts; new work is additive and must not silently replace an existing path.

Every claimed capability requires evidence: a saved checkpoint, a reproducible command, a non-empty output, a visual sample, and a machine-readable manifest. A successful Python run or a valid PNG alone is not proof of model quality or Android execution.

## 2. Architecture decision

The main generation path is a three-stage discrete pipeline:

```text
Arabic/English prompt
        ↓
Tiny bilingual text encoder + class/style/pose embeddings
        ↓
Spatial Token Prior (autoregressive or discrete diffusion)
        ↓
VQ-VAE / Pixel Decoder
        ↓
64×64 RGBA hard-edge sprite
```

The current `arcade_t2i` TextCVAE remains a research baseline and bilingual interface source. It is **not** the production generator until it passes the visual quality gate. The small VQ-VAE, discrete diffusion, Pixel LLM, ONNX, and palette assets in `nano_pixel_mob` and `models/` are candidates to be tested behind the same contract.

## 3. Data contract and quality gates

Only isolated, readable, single-subject images may enter the training set. The preprocessing contract is:

```text
Input image
 → alpha/mask extraction
 → connected-component filtering
 → panel/background leakage rejection
 → bounding-box crop
 → nearest-neighbor normalization
 → RGBA 64×64 output
 → taxonomy + prompt metadata
```

Use `edge_refine_pixel.py` as the CPU reference implementation. It must reject opaque textured panels when the foreground touches the frame or occupies an implausibly large area. It may not invent missing pixels or apply gradients, text overlays, or smoothing to training inputs.

Required metadata per image:

- source and license/provenance
- type/class, race, style, pose, view, dominant colors, tags
- original size and normalized size
- alpha policy and crop bounding box
- quality-gate result and rejection reason, if rejected

The classified `nano arcked dat` dataset and its PNG/JSON/CSV/JSONL indexes are the initial structured source. User-provided sprites remain a separate provenance group and must not be overwritten.

## 4. Implementation phases

### Phase A — Reconstruction gate

1. Inventory the candidate VQ-VAE, decoder, palette, and token formats.
2. Load real RGBA samples from `examples/` and the classified dataset.
3. Run encode → token export → decode reconstruction.
4. Produce original/reconstruction/error grids.
5. Accept only if silhouettes, transparency, palette boundaries, and major details survive visually.

**Exit condition:** at least 90% of the held-out sample grid is recognizable and free from panel leakage or blur. If this fails, stop and repair the decoder/token contract before training a prior.

### Phase B — Spatial Token Prior

1. Train on spatial tokens, not flattened global retrieval only.
2. Condition on bilingual text, class, style, pose, palette, and view.
3. Preserve a fixed seed policy and write every seed to the manifest.
4. Compare autoregressive and discrete-diffusion sampling at the same token budget.
5. Keep the smaller model only if it passes the visual gate; parameter count is not a substitute for quality.

**Exit condition:** 16 different seeds produce recognizable, complete characters with no systematic missing heads, limbs, or disconnected parts.

### Phase C — Bilingual conditioning

1. Reuse the shared Arabic/English tokenizer and Tiny Text Encoder.
2. Normalize Arabic spelling variants and preserve prompt language in metadata.
3. Test paired prompts such as `robot pixel art` and `روبوت بيكسل آرت`.
4. Measure class agreement and visual similarity across language pairs.

**Exit condition:** Arabic and English prompts select the intended class and style without falling back to a generic blob.

### Phase D — Image-to-image

1. Extract and clean the reference sprite.
2. Encode it into spatial tokens.
3. Apply bounded token noise/denoising strength.
4. Decode while preserving a reference identity block.
5. Test strengths 0.10, 0.20, 0.30, and 0.45.

**Exit condition:** low-strength output preserves silhouette and palette; higher strength changes controlled attributes without destroying anatomy.

### Phase E — Animation

Generate one frame at a time, never a full sheet from a single unconstrained call. Keep a shared identity latent, style, palette, and seed family; vary only pose, action, and frame index. Compose with deterministic metadata containing order, dimensions, FPS, and recipe.

**Exit condition:** four frames have consistent identity, dimensions, transparency, and readable motion when inspected as a sheet and GIF.

### Phase F — Runtime packaging

1. Export the smallest validated checkpoint and palette.
2. Convert only after the PyTorch reference output is accepted.
3. Validate ONNX/GGUF/native contracts, file hashes, and memory use.
4. Run one frame at a time on Android; keep inference off the UI thread.
5. Record model, seed, prompt, dimensions, steps, time, memory, and failure state.

A host CPU run is not evidence of Android execution. Android validation requires an APK build plus device/emulator execution evidence.

## 5. Baselines and candidates

| Candidate | Role | Decision |
|---|---|---|
| `arcade_t2i/t2i_cvae_best.pt` | bilingual research baseline | keep, do not call production; current visual gate fails |
| `nano_pixel_mob` VQ-VAE | reconstruction candidate | test first |
| `nano_pixel_mob` discrete diffusion | spatial token prior candidate | test after reconstruction |
| `nano_pixel_mob` Pixel LLM | compact token prior candidate | compare against discrete diffusion |
| `models/nano_pixel_mom/*.onnx` | Android/CPU packaging candidate | test only after contract inspection |
| `pixel_art_engine/pixel_diffusion_weights.pt` | alternate small baseline | benchmark separately |

## 6. Evidence and reporting protocol

Every iteration must publish:

- command and checkpoint identifier
- input dataset manifest/hash
- seed(s), prompt(s), dimensions, and sampling settings
- output grid and failure samples
- reconstruction or generation metrics
- CPU time and memory where measured
- explicit accepted/rejected decision

Do not report “trained successfully” when only a loss value or file save was observed. Do not use generated decorative showcase images as proof of a model checkpoint unless the exact checkpoint and command are recorded.

## 7. Immediate execution queue

1. Audit the `nano_pixel_mob` model classes and tensor contracts.
2. Run reconstruction grids on repository RGBA examples and held-out classified data.
3. Select the first decoder that passes the reconstruction gate.
4. Train/evaluate a spatial token prior with the existing bilingual encoder.
5. Run 16-seed random, Arabic/English paired, and image-to-image evaluations.
6. Only then export the first Nano Arcade Character V1 checkpoint.

This document is the master strategy. Individual experiments may add reports and checkpoints, but they must not weaken these gates or replace failed evidence with placeholder images.
