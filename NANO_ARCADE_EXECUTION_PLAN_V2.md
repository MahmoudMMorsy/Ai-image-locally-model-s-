# Nano Arcade V2 — Full Execution Plan for a Verified Local Pixel-Art Generator

**Document status:** Execution contract  
**Primary language:** English for implementation and reproducibility  
**Runtime target:** CPU-first local inference on HP ZBook 15 / 16 GB RAM, with optional Quadro K2100M acceleration and Android deployment  
**Initial production target:** 64×64 RGBA character sprites; 128×128 and 256×256 are validated upscaling/output modes, not a substitute for a weak 64×64 generator

---

## 0. The central rule: evidence before claims

Nano Arcade is not considered a real generative model because a Python process finishes, a checkpoint file exists, a loss decreases, or a PNG is produced. A capability is accepted only when all of the following exist:

1. A versioned checkpoint and SHA-256 hash.
2. A versioned dataset manifest and split definition.
3. A reproducible command with seed, prompt, dimensions, and sampling settings.
4. A visual grid containing both successful and failed cases.
5. Automated measurements relevant to the capability.
6. A written accepted/rejected decision.
7. A failure log explaining what was not proven.

No failed gate may be bypassed by switching datasets, changing the metric, or showing a decorative showcase image.

---

## 1. Product definition

Nano Arcade will eventually provide:

- **Random generation:** new characters from a seed, without body-part stitching or retrieval-only output.
- **English and Arabic text conditioning:** prompts select class, style, color, pose, equipment, and view where those concepts are represented in the reviewed data.
- **Image-to-image:** a cleaned reference sprite is transformed while preserving identity at low strength.
- **Animation:** short identity-consistent frame sequences generated from a shared identity representation.
- **Local CPU runtime:** deterministic, bounded-memory inference with no cloud dependency.
- **Pixel discipline:** hard edges, nearest-neighbor operations, limited palette, no blur, no gradients introduced by preprocessing.

The first release is deliberately narrower: clean 64×64 characters and reliable random generation. Text, image-to-image, and animation are layered only after this foundation passes.

### Explicit non-goals for V1

- Photorealistic images.
- Arbitrary long-form Arabic text rendered perfectly inside an image. Prompt understanding and in-image typography are separate capabilities.
- Unreviewed semantic labels treated as ground truth.
- A claim that CPU inference is fast on Android before an APK/device or emulator test is recorded.

---

## 2. Current factual baseline

The project has already established several facts that must remain visible:

| Item | Current finding | Decision |
|---|---|---|
| Bundled `arcade_sprites_32x32.pt` | 1,000 records but only 4 unique exact images | reject as a training source |
| Supplied small latent prior | 16 seeds produced almost the same red/blue character | reject as production generator |
| `examples/` quick corpus | 96 extracted images, useful for smoke tests only | do not use as the main corpus |
| `clean_hq_dataset.pt` | 4,173 accepted records, 1,996 exact indexed images, 1,140 rejected | primary engineering corpus, subject to visual audit |
| Existing clean runtime checkpoints | trained previously, but some grids show empty structures or artifacts | candidates only; must re-pass gates |
| Tiny bilingual tokenizer/encoder | existing research component | reuse, then measure text control independently |

These findings are not failures of the project; they prevent false success claims.

---

## 3. Data strategy: use the thousands, but do not trust them blindly

### 3.1 Dataset tiers

Maintain separate, immutable tiers:

1. **Tier U — User sprites:** user-provided material, with original provenance and no destructive overwrite.
2. **Tier A — Audited open/permissioned sprites:** source, license/provenance, and import hash recorded.
3. **Tier C — Clean training corpus:** the accepted output of the strict cleaning pipeline.
4. **Tier H — Human-reviewed semantic subset:** only images with verified class/equipment/pose labels.
5. **Tier E — Evaluation-only holdout:** never used for training or codebook fitting.

A file can move from one tier to another only through a manifest update and a reproducible preprocessing run.

### 3.2 Cleaning contract

Every candidate follows:

```text
source image
 → decode and hash
 → alpha/background estimation
 → connected-component analysis
 → single-subject test
 → panel/sheet/edge-leakage rejection
 → bounding-box crop
 → centered 64×64 RGBA normalization
 → nearest-neighbor palette/index conversion
 → objective metadata
 → visual-review queue
```

Reject when any of these are true:

- empty or nearly empty foreground;
- multiple unrelated components;
- foreground touches the frame in a way consistent with a crop or sheet;
- occupancy is too low to contain a readable character;
- occupancy is too high or rectangular enough to be a panel;
- excessive background leakage;
- unreadable resolution or interpolation artifacts;
- duplicate content after normalization.

The cleaner may remove or preserve pixels; it must not hallucinate missing anatomy or draw new details.

### 3.3 Palette contract

Choose one contract for each experiment and record it in the manifest:

- `index 0 = transparent`;
- visible indices have a fixed, documented palette;
- palette dimensions match model class count;
- the same index is rendered using the same RGB value in training, validation, preview, and runtime;
- palette lookup tests run before training.

A palette-index mismatch is a hard failure, not a cosmetic issue.

### 3.4 Split policy

Create train/validation/test splits by content hash and, when possible, by source family. Near-duplicates must not cross splits. Recommended initial split:

- 80% train;
- 10% validation;
- 10% held-out test;
- an additional reviewed challenge set containing difficult silhouettes, dark palettes, wide sprites, and user-style examples.

Publish counts, hashes, label distributions, and duplicate statistics for every split.

---

## 4. Phase 0 — Repository and contract freeze

### Deliverables

- `NANO_ARCADE_EXECUTION_PLAN_V2.md`;
- model registry with architecture, input/output shapes, checkpoint hash, and status;
- dataset registry with source, license/provenance, count, unique count, and split hash;
- one command that runs all smoke tests without modifying source data.

### Gate

The repository must be reproducible from a clean checkout. Existing tools, taxonomies, checkpoints, and Android concepts remain available. New experiments use new names and paths; no failed checkpoint is silently replaced.

---

## 5. Phase 1 — Build the authoritative clean corpus

### Work

1. Re-run or verify the strict cleaner against all available large sources.
2. Generate `clean_manifest_v2` with rejection reasons and source hashes.
3. Deduplicate after normalization, not only before normalization.
4. Create objective classes: occupancy, aspect ratio, palette complexity, foreground edge contact, connected components.
5. Create a visual contact sheet sampled across every class, source, and rejection reason.
6. Manually review the semantic subset for class/equipment/pose labels.

### Acceptance gates

- At least 1,000 genuinely unique readable characters before semantic conditioning.
- No more than 1% duplicate normalized images in the evaluation split.
- No panel/sheet leakage in the reviewed challenge set.
- Every accepted image has a source hash and preprocessing record.
- Any questionable item is marked `needs_review`, never silently promoted to ground truth.

### Output

```text
clean_hq_dataset_v2.pt
clean_manifest_v2.json
train_manifest.json
val_manifest.json
test_manifest.json
clean_preview_grid_v2.png
rejection_grid_v2.png
```

---

## 6. Phase 2 — Train and validate the image codec

The codec is the foundation. It must reconstruct real characters before any prior is trained.

### Recommended codec

Use a discrete pixel-aware autoencoder with:

- 64×64 input;
- 8×8 or 16×16 spatial latent grid;
- 32–128 codebook entries depending on measured codebook usage;
- hard palette/index output or a decoder followed by deterministic palette projection;
- straight-through or Gumbel/argmax training only where it improves boundary fidelity;
- optional EMA codebook updates and codebook diversity loss.

Do not select the architecture by parameter count alone.

### Training protocol

- Train only on Tier C training split.
- Validate after every fixed number of steps.
- Save best-by-validation checkpoint, not last checkpoint.
- Record reconstruction loss, transparent-pixel accuracy, foreground IoU, color/index accuracy, codebook perplexity, and number of used codes.
- Produce original/reconstruction/error grids at 64×64 and nearest-neighbor 256×256 preview.

### Codec acceptance gate

All must pass on the held-out test set:

- mean pixel/index accuracy ≥ 0.90;
- minimum reviewed sample accuracy ≥ 0.80;
- foreground IoU ≥ 0.85 on the reviewed challenge subset;
- no systematic missing heads, limbs, or weapons;
- no blur, gradient, or antialiasing introduced;
- codebook usage is not collapsed to a tiny subset;
- visual reviewers can recognize the original character in at least 90% of the grid.

If this gate fails, stop and repair data/codec. Do not train a prior.

---

## 7. Phase 3 — Train an unconditional spatial prior

This isolates “can the model generate new valid characters?” from text understanding.

### Candidate comparison

Benchmark two small priors using the same codec:

1. masked spatial-token denoiser/discrete diffusion;
2. compact autoregressive spatial prior.

Both must operate on spatial latent tokens, not only a global retrieval index. Use a fixed token budget and the same evaluation protocol.

### Training protocol

- Train on codec tokens extracted from the training split only.
- Use code dropout/masking to prevent memorization of a single pattern.
- Track token entropy, code usage, nearest-neighbor similarity to training images, and sample diversity.
- Generate at least 32 fixed seeds per checkpoint.

### Random-generation gate

Accept only if:

- at least 90% of 32 samples are complete, readable single characters;
- no repeated exact output across seeds;
- no systematic missing body parts;
- foreground occupancy falls within the clean training distribution;
- samples are not near-duplicates of training images;
- the visual grid contains varied class, silhouette, pose, and palette behavior;
- output remains deterministic for the same seed and settings.

A prior that produces one attractive character repeatedly is rejected, even if its loss is low.

---

## 8. Phase 4 — Add structured controls before free-form text

Before natural-language conditioning, make the model controllable with verified fields:

```text
class / archetype
silhouette
pose/view
palette mood
body proportion
weapon/equipment when reviewed
```

Use only labels with confidence grade A or B. Geometry-only labels are used for organization, not semantic truth.

### Gate

Changing one control should change the intended property while preserving the others. Test one-factor-at-a-time matrices, not isolated showcase prompts.

---

## 9. Phase 5 — Add bilingual text conditioning

### Text path

```text
Arabic or English prompt
 → normalization and alias mapping
 → shared tokenizer
 → Tiny Text Encoder
 → FiLM/cross-attention conditioning in spatial prior
 → latent tokens
 → codec decoder
```

Keep the current tokenizer/encoder as a research component, but verify it with paired Arabic/English prompts. Do not claim “Arabic support” from tokenization alone.

### Caption policy

- Captions are generated from reviewed metadata, not guesses from image pixels.
- Arabic and English captions share a canonical semantic record.
- Unknown words are marked unknown and do not become training labels.
- The prompt language is preserved in the manifest.

### Text gate

Use a fixed matrix covering:

- class only;
- class + color;
- class + equipment;
- class + pose;
- English/Arabic pairs;
- mixed Arabic-English prompts;
- unknown or conflicting terms.

Accept only if the intended attribute changes consistently, class agreement is measurable, and Arabic/English paired prompts produce comparable class/style behavior. A text encoder that merely retrieves a generic character is rejected.

---

## 10. Phase 6 — Image-to-image

### V1 controlled path

```text
reference RGBA sprite
 → strict clean/normalize
 → codec encode
 → bounded token noise
 → optional prompt/control conditioning
 → denoise/prior
 → codec decode
```

Test strengths `0.05, 0.15, 0.30, 0.45, 0.60`.

### Gate

- At low strength, silhouette and palette identity remain recognizable.
- At medium strength, equipment or palette can change without anatomy collapse.
- At high strength, the result may become a new character but must remain a valid sprite.
- The reference is never copied as a hidden retrieval shortcut without being reported.

Measure silhouette IoU, palette distance, foreground occupancy, and identity embedding distance.

---

## 11. Phase 7 — Animation

Animation starts only after single-frame generation passes.

### Data requirement

Use true frame groups with shared character identity, canvas, scale, palette, and view. A folder of unrelated sprites is not animation data.

### Model

Start with the lightest reliable path:

```text
identity latent + pose/action token + frame index
 → frame latent delta
 → codec decoder
```

Use a shared identity latent and predict bounded motion deltas rather than generating every frame independently.

### Animation gate

For at least four frames:

- identity remains stable;
- dimensions and transparency are identical;
- palette remains consistent unless intentionally changed;
- silhouette changes follow the requested action;
- no flicker or random equipment changes;
- spritesheet and GIF outputs match the manifest order.

If no adequate frame dataset exists, label procedural motion as a non-neural fallback rather than calling it AI animation.

---

## 12. Phase 8 — CPU, quantization, and Android packaging

### Reference hierarchy

1. PyTorch CPU reference output.
2. ONNX or native desktop output.
3. Android emulator/device output.
4. Optional GGUF/native integration if its tensor contract supports this architecture.

Never convert before the reference checkpoint passes visual gates.

### Runtime measurements

Record:

- model file size;
- peak RAM;
- CPU time per 64×64 sample;
- number of threads;
- warm and cold start time;
- output hash;
- Android/device or emulator identifier;
- failure state and fallback behavior.

Use one-frame-at-a-time inference and keep work off the Android UI thread. INT8/INT4 conversion is accepted only when output drift remains below a documented threshold.

### Runtime gate

The quantized runtime must reproduce the reference class/style/silhouette behavior and remain visually crisp. A smaller file with broken output is not an optimization.

---

## 13. Evaluation suite and anti-hallucination controls

Create one command that runs:

```text
1. dataset integrity and duplicate checks
2. codec reconstruction grid
3. 32-seed random grid
4. structured-control matrix
5. Arabic/English paired prompt matrix
6. image-to-image strength sweep
7. animation consistency grid
8. runtime benchmark
9. report and hashes
```

Every report must include both accepted and failed samples. Reports must distinguish:

- **proven:** executed and measured;
- **candidate:** checkpoint exists but gate is pending;
- **failed:** executed and rejected;
- **not implemented:** no evidence exists;
- **not applicable:** dependency/data is missing.

The words “production,” “real model,” “supports Arabic,” “supports animation,” and “Android-ready” are reserved for capabilities whose corresponding gate is passed.

---

## 14. Checkpoint and artifact registry

Use a registry such as:

```json
{
  "checkpoint_id": "nano_arcade_v1_codec_0001",
  "stage": "codec",
  "dataset_manifest_sha256": "...",
  "architecture": "...",
  "input_shape": [1, 64, 64],
  "output_shape": [1, 64, 64],
  "seed": 21,
  "training_steps": 0,
  "metrics": {},
  "visual_grid": "...",
  "status": "candidate",
  "failure_reasons": []
}
```

Never overwrite a checkpoint or report. New attempts receive a new ID.

---

## 15. Execution order and stop conditions

### Ordered work packages

1. Freeze repository and registries.
2. Re-audit the 4,173-image clean corpus and produce v2 splits.
3. Fix and test the palette/index contract.
4. Train codec; do not train prior before codec acceptance.
5. Train and benchmark two unconditional spatial priors.
6. Add structured controls.
7. Add Tiny bilingual text conditioning.
8. Add image-to-image denoising.
9. Add animation only with grouped frame data.
10. Convert and validate CPU/Android runtimes.

### Hard stop conditions

Stop the current branch and repair it when:

- the dataset has hidden duplicates or panels;
- palette/index contracts disagree;
- codec reconstruction fails visually;
- codebook collapse is detected;
- random seeds produce one repeated sample;
- text changes do not cause the intended visual change;
- image-to-image destroys identity at low strength;
- animation flickers or changes identity;
- quantization changes output semantics.

A hard stop is a quality control result, not a reason to hide the run.

---

## 16. Definition of done for Nano Arcade Character V1

Character V1 is complete only when all of these are true:

- clean, deduplicated, hashable dataset with provenance;
- codec passes reconstruction and visual gates;
- unconditional spatial prior passes 32-seed diversity gate;
- structured controls pass one-factor tests;
- Arabic/English paired prompts pass the text-control matrix;
- deterministic seed and manifest behavior works;
- CPU runtime is benchmarked on the target class of hardware;
- checkpoint, code, reports, grids, and hashes are all present;
- failed experiments remain archived and clearly labeled;
- no claim relies on an unreviewed showcase image.

Only after Character V1 is accepted should image-to-image and animation be promoted from research stages to user-facing features.
