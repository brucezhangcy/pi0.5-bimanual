# Dev log — pi0.5 fine-tunes on bimanual flip datasets

Author: brucezha@usc.edu (also collected all training data)
Machine used for training: yunshuang's box (GPU 3, RTX 6000 Ada 48 GB)

## 1. Goal

Fine-tune the pi0.5 base policy on Trossen `trossen_ai_stationary` bimanual flip
data so it can be evaluated on the real Trossen arms.

Three datasets covered so far (all collected by Bruce, 14-dim bimanual joints, 30 fps):

| run | dataset | task | episodes | frames | cameras | finetuned on |
|---|---|---|---|---|---|---|
| v1 (2026-05-26) | local `bimanual_flip_5_objects` | Bimanual Flip | 100 | 44 883 | 2 (cam_high, cam_low) | pi05_base + LoRA |
| v2 (2026-05-28) | HF `BruceZhang0912/Bimanual-Flip-4-Camera` | Bimanual Flip | 101 | 30 199 | 4 (top, bottom, left_wrist, right_wrist) | pi05_base + LoRA |
| v3 (2026-05-28) | HF `BruceZhang0912/Bimanual-Recordings` | Bimanual Rotate | 100 | 29 898 | 4 (same schema as v2) | pi05_base + LoRA |

Reference tutorial: https://docs.trossenrobotics.com/trossen_arm/v1.8/tutorials/openpi.html

## 2. Datasets

Both datasets are LeRobot v2.1, `trossen_subversion v1.0`, `robot_type=trossen_ai_stationary`, 30 fps, single task `"Bimanual Flip"`, 14-dim bimanual joints (`left_joint_0..6, right_joint_0..6`). Both **collected by Bruce**.

### 2.1 `bimanual_flip_5_objects` (v1 run)

Local: `/data/bruce/pi0.5/bimanual_flip_5_objects`

- 100 episodes / 44 883 frames
- Cameras (only two — no wrist cams):
  - `observation.images.cam_high` (480×640, AV1)
  - `observation.images.cam_low`  (480×640, AV1)
- Depth parquet exists (`observation.depth.cam_*`) but openpi does not consume it — ignored

### 2.2 `Bimanual-Flip-4-Camera` (v2 run)

**All 101 episodes of this dataset were collected by Bruce.** Local copy lives at `/data/bruce/pi0.5/Bimanual-Flip-4-Camera` and is mirrored to Hugging Face at `BruceZhang0912/Bimanual-Flip-4-Camera`.

- 101 episodes / 30 199 frames
- Four cameras (all 480×640, AV1):
  - `observation.images.top`         → maps to `cam_high`
  - `observation.images.bottom`      → maps to `cam_low`
  - `observation.images.left_wrist`  → maps to `cam_left_wrist`
  - `observation.images.right_wrist` → maps to `cam_right_wrist`
- All four AlohaInputs camera slots populated — no masking needed (unlike v1)

### 2.3 `Bimanual-Recordings` (v3 run, rotation)

**All 100 episodes of this dataset were collected by Bruce.** Local copy lives at `/data/bruce/pi0.5/Bimanual-Recordings`, mirrored to Hugging Face at `BruceZhang0912/Bimanual-Recordings`.

- Task: `"Bimanual Rotate"` (single task in `meta/tasks.jsonl`)
- 100 episodes / 29 898 frames / 4.3 GB on disk
- Same 4-camera schema as v2: `observation.images.{top, bottom, left_wrist, right_wrist}` → same AlohaInputs mapping. The TrainConfig is a one-line variant of the v2 block (just `name`, `repo_id` change).

## 3. Run v1 — `bimanual_flip_5_objects` (2 cameras), 2026-05-26

### 3.1 Added a new training config

File: [openpi/src/openpi/training/config.py](openpi/src/openpi/training/config.py), block named `pi05_bimanual_flip_5_objects`. Key choices:

- `model=pi0_config.Pi0Config(pi05=True)` — pi0.5 architecture
- LoRA: `paligemma_2b_lora` + `gemma_300m_lora`, EMA off
- `weight_loader=CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params")`
- `assets={dir: gs://.../pi05_base/assets, asset_id: trossen}` — reuses base Trossen norm stats (same pattern as `pi05_trossen_transfer_block` / `_organize_tools`)
- `base_config=DataConfig(prompt_from_task=True)` so the prompt comes from `meta/tasks.jsonl`
- `RepackTransform` maps **only** `cam_high` + `cam_low` (no wrist cams). `AlohaInputs` automatically zero-pads `left_wrist_0_rgb` / `right_wrist_0_rgb` and sets their `image_mask=False`
- **Important**: the repack dict must also include `"prompt": "prompt"`, otherwise the prompt key gets dropped after repack and training crashes with `ValueError: Prompt is required`. I hit this on the first launch
- `num_train_steps=30_000`, `batch_size=4` (sized to fit ~35 GB free on the only available card)

### 3.2 Made the dataset visible to LeRobotDataset

The loader looks up `repo_id` inside `~/.cache/huggingface/lerobot/<repo_id>`. Solved with a symlink (no HF upload needed for the dataset itself):

```bash
mkdir -p ~/.cache/huggingface/lerobot/bruce
ln -sfn /data/bruce/pi0.5/bimanual_flip_5_objects \
        ~/.cache/huggingface/lerobot/bruce/bimanual_flip_5_objects
```

### 3.3 Smoke-tested the data pipeline

```bash
cd /data/bruce/pi0.5/openpi
JAX_PLATFORMS=cpu uv run scripts/compute_norm_stats.py \
    --config-name=pi05_bimanual_flip_5_objects
```

Wrote stats to `openpi/assets/pi05_bimanual_flip_5_objects/bruce/bimanual_flip_5_objects/norm_stats.json` after ~25 min of CPU work. Note these local norm stats are **not actually used at train time** — the AssetsConfig override loads the base pi05 Trossen stats. Compute step is effectively a smoke test confirming the repack + video decode + transform chain works end-to-end.

### 3.4 Launched fine-tune

```bash
cd /data/bruce/pi0.5/openpi
CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
    uv run scripts/train.py pi05_bimanual_flip_5_objects \
        --exp-name=pi05_bimanual_flip_5_objects_v1 \
        --overwrite \
    > /tmp/pi05_train.log 2>&1 &
```

**Headline numbers**

- Throughput: **1.8 it/s** on the RTX 6000 Ada (48 GB, ~35 GB free)
- Wall time: **~5 h** for 30 000 steps (08:27 → 13:22)
- Min loss: **0.0082** at step 29 100
- Last-1k mean loss: **0.0094**
- Wandb run: https://wandb.ai/yunshuang-university-of-southern-california/openpi/runs/vzy2n9dg
- Checkpoints saved at steps 5000 / 10000 / 15000 / 20000 / 25000 / 29999 under
  `openpi/checkpoints/pi05_bimanual_flip_5_objects/pi05_bimanual_flip_5_objects_v1/`

**Convergence analysis** (parsed from `/tmp/pi05_train.log`, 300 log lines @ every 100 steps)

| step window | mean loss | mean grad_norm | Δ param_norm |
|---|---|---|---|
| 0 – 500 | 0.1745 | 2.18 | +0.0003 |
| 500 – 2 000 | 0.0483 | 0.47 | +0.055 |
| 2 000 – 5 000 | 0.0328 | 0.33 | +0.148 |
| 5 000 – 10 000 | 0.0242 | 0.27 | +0.211 |
| 10 000 – 15 000 | 0.0182 | 0.23 | +0.132 |
| 15 000 – 20 000 | 0.0145 | 0.21 | +0.059 |
| 20 000 – 25 000 | 0.0116 | 0.19 | +0.019 |
| **25 000 – 30 000** | **0.0099** | **0.18** | **+0.006** |

**Verdict: converged in the practical sense.** Loss kept ticking down (last-1k mean
0.0094 vs steps 20-25k mean 0.0116 — still ~19% improvement) but the **param-norm
drift collapsed**: 0.21 in steps 5-10k vs **0.006** in 25-30k (35× slowdown). The
weights have essentially stopped moving even though training loss is still drifting
down — classic "approaching plateau" signal. Could train tens of thousands more
steps for marginal further gain at real risk of overfitting on only 100 episodes.
Ship **29999**; keep **25000** for an A/B if the real-robot eval is mediocre.

### 3.5 Pushed the final checkpoint to Hugging Face

Repo: `BruceZhang0912/pi05-bimanual-flip-5-objects` (private)

Uploaded only `params/` + `assets/` + `_CHECKPOINT_METADATA` (≈5.8 GB).
`train_state/` (≈2.8 GB optimizer state) was deliberately skipped — useless for inference.

```python
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo("BruceZhang0912/pi05-bimanual-flip-5-objects",
                private=True, repo_type="model", exist_ok=True)
api.upload_folder(
    repo_id="BruceZhang0912/pi05-bimanual-flip-5-objects",
    folder_path=".../checkpoints/.../pi05_bimanual_flip_5_objects_v1/29999",
    path_in_repo="29999",
    ignore_patterns=["train_state/*", "train_state"],
)
```

## 4. Run v2 — `Bimanual-Flip-4-Camera` (4 cameras), 2026-05-28

### 4.1 Added a parallel training config

In [openpi/src/openpi/training/config.py](openpi/src/openpi/training/config.py), block named `pi05_bimanual_flip_4_camera` — placed right before the v1 block. Same base recipe (pi0.5 + LoRA, EMA off, batch_size=4, 30 000 steps, `prompt_from_task=True`, `assets=trossen/pi05_base`) with two differences from v1:

- `repo_id="BruceZhang0912/Bimanual-Flip-4-Camera"` (matches the HF dataset id and the local cache symlink)
- `RepackTransform` maps **all four** cameras — no wrist masking needed:
  ```python
  "images": {
      "cam_high":        "observation.images.top",
      "cam_low":         "observation.images.bottom",
      "cam_left_wrist":  "observation.images.left_wrist",
      "cam_right_wrist": "observation.images.right_wrist",
  }
  ```

### 4.2 Made the dataset visible to LeRobotDataset

```bash
mkdir -p ~/.cache/huggingface/lerobot/BruceZhang0912
ln -sfn /data/bruce/pi0.5/Bimanual-Flip-4-Camera \
        ~/.cache/huggingface/lerobot/BruceZhang0912/Bimanual-Flip-4-Camera
```

### 4.3 Smoke-tested the data pipeline

```bash
cd /data/bruce/pi0.5/openpi
JAX_PLATFORMS=cpu uv run scripts/compute_norm_stats.py \
    --config-name=pi05_bimanual_flip_4_camera
```

Passed cleanly — wrote local stats to `openpi/assets/pi05_bimanual_flip_4_camera/BruceZhang0912/Bimanual-Flip-4-Camera/norm_stats.json`. As with v1, those local stats are not used at train time (config loads base Trossen stats from `gs://openpi-assets/checkpoints/pi05_base/assets/trossen`); the run is purely a video-decode + transform-chain check.

### 4.4 Launched fine-tune

```bash
cd /data/bruce/pi0.5/openpi
CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
    uv run scripts/train.py pi05_bimanual_flip_4_camera \
        --exp-name=pi05_bimanual_flip_4_camera_v1 \
        --overwrite \
    > /tmp/pi05_train_4cam.log 2>&1 &
```

Same GPU as v1 (RTX 6000 Ada, ~35 GB free). Training results landed in `openpi/checkpoints/pi05_bimanual_flip_4_camera/pi05_bimanual_flip_4_camera_v1/`.

**Headline numbers**

- Throughput: ~1.6 it/s (slightly slower than v1's 1.8 it/s due to 2× the image decode work)
- Wall time: **~5 h** for 30 000 steps (03:21 → ~08:30)
- Min loss: **0.0049** at step 29 600
- Last-1k mean loss: **0.0058**
- Wandb run: https://wandb.ai/yunshuang-university-of-southern-california/openpi/runs/batwgt8j
- Checkpoints saved at steps 5000 / 10000 / 15000 / 20000 / 25000 / 29999

**Convergence analysis** (300 log lines @ every 100 steps)

| step window | mean loss | mean grad_norm | Δ param_norm |
|---|---|---|---|
| 0 – 500 | 0.2009 | 2.03 | +0.0003 |
| 500 – 2 000 | 0.0419 | 0.48 | +0.058 |
| 2 000 – 5 000 | 0.0230 | 0.30 | +0.148 |
| 5 000 – 10 000 | 0.0156 | 0.23 | +0.214 |
| 10 000 – 15 000 | 0.0114 | 0.19 | +0.134 |
| 15 000 – 20 000 | 0.0090 | 0.16 | +0.061 |
| 20 000 – 25 000 | 0.0071 | 0.14 | +0.020 |
| **25 000 – 30 000** | **0.0061** | **0.13** | **+0.006** |

**Verdict: same convergence pattern as v1, lower absolute loss.** Last-1k vs 20-25k window ratio = 0.81 (identical to v1, meaning training was still ticking down by ~19% over the final 5k steps), but param-norm drift in the last 5k window is only 0.006 (vs 0.21 in steps 5-10k — 35× slowdown) — weights essentially stopped moving. **Min loss 0.0049 vs v1's 0.0082** — 40 % lower, which is the expected payoff from adding the two wrist cameras (better fine-manipulation signal). Ship 29999.

### 4.5 Pushed the 4-camera checkpoint to Hugging Face

Repo: `BruceZhang0912/pi05-bimanual-flip-4-camera` (private)
→ https://huggingface.co/BruceZhang0912/pi05-bimanual-flip-4-camera

Uploaded `29999/{params,assets,_CHECKPOINT_METADATA}` (~5.8 GB), `train_state/` excluded. Same `HfApi.upload_folder` recipe as v1, just with the new repo id and folder path.

The GitHub repo's `pi05_config_patch.diff` was also refreshed to include both the v1 and v2 TrainConfig blocks in one patch.

## 5. Setup (training box and eval desktop)

### 5.A Training-box retrain template
Pattern (substitute config / exp-name for any of the three variants):

```bash
cd /data/bruce/pi0.5/openpi
CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
    uv run scripts/train.py <config_name> --exp-name=<exp_name> --overwrite
```

Each `pi05_bimanual_*` block in `src/openpi/training/config.py` is an uncommitted local edit on the TrossenRobotics `trossen-ai` branch, preserved as `pi05_config_patch.diff` in this repo (now contains all 3 blocks). Re-generate any time with `git diff src/openpi/training/config.py`.

### 5.B Eval-desktop layout and one-time setup

This repo (instructions + patches), the `openpi/` clone, and the `openpi_checkpoints/` download all live consolidated under one folder on the eval desktop:

```
$HOME/pi0.5-bimanual/                       # this GitHub repo
├── dev_log.md  pi05_config_patch.diff  trossen_ai_client_patch.diff  README.md
├── eval_tools/                             # sim playback, replay overlay, etc.
├── openpi/                                 # TrossenRobotics/openpi clone (gitignored)
└── openpi_checkpoints/                     # HF checkpoint downloads (gitignored)
```

The dataset is **not** required on the eval box for inference — eval runs from the checkpoint + the base Trossen norm stats. (It IS needed for the replay-overlay diagnostic, which downloads one episode on demand — see §10.7.)

```bash
# 1. Clone this repo and cd in
git clone https://github.com/brucezhangcy/pi0.5-bimanual.git && cd pi0.5-bimanual

# 2. Clone openpi inside it
git clone --recurse-submodules https://github.com/TrossenRobotics/openpi.git
cd openpi && git checkout trossen-ai
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# 3. Apply both patches (configs + client edits)
git apply ../pi05_config_patch.diff                                    # adds all 3 TrainConfig blocks
git apply ../trossen_ai_client_patch.diff                              # client camera config, action clamp, Ctrl+C-home, auto-reset

# 4. Pull the checkpoint(s) you want from HF (substitute the variant; ~5.8 GB each)
HF_TOKEN=<read-token> uv run huggingface-cli download \
    BruceZhang0912/pi05-bimanual-rotation \
    --local-dir $HOME/pi0.5-bimanual/openpi_checkpoints/pi05_bimanual_rotation/pi05_bimanual_rotation_v1
```

After step 4 each variant's checkpoint directory contains `29999/{params, assets/trossen, _CHECKPOINT_METADATA}`.

To **run** the eval, see the runbook in **§10.8** (current pipeline, all fixes applied).

## 6. Gotchas observed

- **`RepackTransform` drops un-mapped keys.** If you add `prompt_from_task=True`, you **must** include `"prompt": "prompt"` in the repack dict. Crashes with `ValueError: Prompt is required` otherwise.
- **`compute_norm_stats.py` requires `JAX_PLATFORMS=cpu`** if you want to run it without a free GPU — otherwise JAX errors with `No visible GPU devices`. The output is written keyed by `repo_id` but isn't used at train time when `AssetsConfig` overrides `assets_dir`. Treat it as a smoke test.
- **GPU sharing**: `XLA_PYTHON_CLIENT_MEM_FRACTION` defaults to 0.9, which preallocates aggressively. On a partially-occupied card, drop it to 0.85 or 0.7 and reduce `batch_size`.
- **First JIT compile is silent for ~1–2 min** after `Loaded norm stats`. tqdm shows `-/30000` with no rate. Don't kill it.
- **wandb is configured for the `yunshuang-university-of-southern-california` team** on this machine. If you want a different wandb project/team, `wandb login --relogin` before training (and undo it after to avoid affecting yunshuang's runs).

## 7. Artifacts

| Artifact | Location |
|---|---|
| Training config diff | `openpi/src/openpi/training/config.py` (`pi05_bimanual_flip_5_objects` block) |
| Checkpoint dir | `openpi/checkpoints/pi05_bimanual_flip_5_objects/pi05_bimanual_flip_5_objects_v1/{5000,10000,15000,20000,25000,29999}/` |
| Local norm stats (unused) | `openpi/assets/pi05_bimanual_flip_5_objects/bruce/bimanual_flip_5_objects/norm_stats.json` |
| Train log | `/tmp/pi05_train.log` |
| Wandb run | https://wandb.ai/yunshuang-university-of-southern-california/openpi/runs/vzy2n9dg |
| HF repo (private, weights) | https://huggingface.co/BruceZhang0912/pi05-bimanual-flip-5-objects |
| GitHub repo (private, instructions + patch) | https://github.com/brucezhangcy/pi0.5-bimanual |
| Dataset symlink | `~/.cache/huggingface/lerobot/bruce/bimanual_flip_5_objects` → `/data/bruce/pi0.5/bimanual_flip_5_objects` |

## 9. Run v3 — `Bimanual-Recordings` (rotation, 4 cameras), 2026-05-28

### 9.1 Added a parallel training config

In [openpi/src/openpi/training/config.py](openpi/src/openpi/training/config.py), block named `pi05_bimanual_rotation` — placed right before the `pi05_bimanual_flip_4_camera` block. Recipe is **identical** to v2 (pi0.5 + LoRA, EMA off, batch_size=4, 30 000 steps, `prompt_from_task=True`, `assets=trossen/pi05_base`); the only differences are:

- `name="pi05_bimanual_rotation"`
- `repo_id="BruceZhang0912/Bimanual-Recordings"`

The camera repack map is byte-for-byte the same as v2 since the dataset has the same 4-camera schema (`top/bottom/left_wrist/right_wrist`).

### 9.2 Made the dataset visible to LeRobotDataset

```bash
mkdir -p ~/.cache/huggingface/lerobot/BruceZhang0912
ln -sfn /data/bruce/pi0.5/Bimanual-Recordings \
        ~/.cache/huggingface/lerobot/BruceZhang0912/Bimanual-Recordings
```

### 9.3 Smoke-tested the data pipeline

```bash
cd /data/bruce/pi0.5/openpi
JAX_PLATFORMS=cpu uv run scripts/compute_norm_stats.py \
    --config-name=pi05_bimanual_rotation \
    > /tmp/norm_stats_rotation.log 2>&1 &
```

### 9.4 Launched fine-tune

```bash
cd /data/bruce/pi0.5/openpi
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
    uv run scripts/train.py pi05_bimanual_rotation \
        --exp-name=pi05_bimanual_rotation_v1 \
        --overwrite \
    > /tmp/pi05_train_rotation.log 2>&1 &
```

Used GPU 0 (40 GB free) instead of GPU 3 — by the time the smoke test passed, GPU 3 had been claimed by another workload.

**Headline numbers**

- Throughput: **1.4 it/s** on the RTX 6000 Ada (slower than v2's 1.6 it/s, likely the other GPU-0 tenant interfering)
- Wall time: **~5h 50 min** for 30 000 steps (17:14 → ~23:05)
- Min loss: **0.0058** at step 29 600
- Last-1k mean loss: **0.0068**
- Wandb run: https://wandb.ai/yunshuang-university-of-southern-california/openpi/runs/tkflmjed
- Checkpoints saved at steps 5000 / 10000 / 15000 / 20000 / 25000 / 29999

**Convergence analysis** (300 log lines @ every 100 steps; lines were stdout-buffered and only flushed on process exit — wandb has the same data in real time)

| step window | mean loss | mean grad_norm | Δ param_norm |
|---|---|---|---|
| 0 – 500 | 0.2642 | 2.22 | +0.0003 |
| 500 – 2 000 | 0.0494 | 0.58 | +0.056 |
| 2 000 – 5 000 | 0.0281 | 0.37 | +0.150 |
| 5 000 – 10 000 | 0.0190 | 0.28 | +0.215 |
| 10 000 – 15 000 | 0.0139 | 0.23 | +0.136 |
| 15 000 – 20 000 | 0.0107 | 0.19 | +0.062 |
| 20 000 – 25 000 | 0.0084 | 0.17 | +0.020 |
| **25 000 – 30 000** | **0.0072** | **0.15** | **+0.006** |

**Verdict: same convergence pattern as v1 and v2.** Last-1k vs 20-25k window ratio = 0.81 — identical to both prior runs, meaning loss was still trending down (~19% in the final 5k steps) while the param-norm drift collapsed from 0.21 (steps 5-10k) to 0.006 (steps 25-30k), the by-now-familiar "approaching plateau" signal. Ship **29999**.

**Cross-run comparison**

| run | task | cams | min loss | last-1k mean | last-5k mean grad |
|---|---|---|---|---|---|
| v1 | Flip (5-objects) | 2 | 0.0082 | 0.0094 | 0.18 |
| v2 | Flip (4-cam) | 4 | 0.0049 | 0.0058 | 0.13 |
| **v3** | **Rotate** | **4** | **0.0058** | **0.0068** | **0.15** |

Rotation finished between v2 and v1 in absolute loss — slightly higher than v2 (consistent with rotation being a harder task than flip on the same hardware setup), still 26-29 % lower than v1.

### 9.5 Pushed the rotation checkpoint to Hugging Face

Repo: `BruceZhang0912/pi05-bimanual-rotation` (private)
→ https://huggingface.co/BruceZhang0912/pi05-bimanual-rotation

Uploaded `29999/{params,assets,_CHECKPOINT_METADATA}` (~5.8 GB), `train_state/` excluded. Same `HfApi.upload_folder` recipe as v1 / v2.

GitHub repo's `pi05_config_patch.diff` now contains all three TrainConfig blocks (`pi05_bimanual_flip_5_objects`, `pi05_bimanual_flip_4_camera`, `pi05_bimanual_rotation`).

## 10. Eval-desktop session — v3 (rotation) on `coldbrew`, 2026-05-30

Brought the v3 (rotation) checkpoint up on the real arms. Surfaced multiple new deployment issues, the biggest of which **retroactively explains why §8 (v1 eval) didn't work either**: I had the wrong physical camera type assigned to `cam_high`/`cam_low` all along. After fixing that and a stack of supporting patches, the full pipeline runs end-to-end with auto-recovery between attempts. **Note:** §5 (the runbook) and §8.2 (the v1 camera-mapping conclusion) are now superseded by §10.2 and §10.8 below — use those.

### 10.1 Setup
- Pulled latest instructions repo (HEAD `4f88d95` at session start).
- Re-applied the refreshed `pi05_config_patch.diff` (now 3 TrainConfig blocks: v1 / v2 / v3).
- Pulled the v3 checkpoint (`BruceZhang0912/pi05-bimanual-rotation`, ~5.8 GB) into `$HOME/pi0.5-bimanual/openpi_checkpoints/pi05_bimanual_rotation/pi05_bimanual_rotation_v1/29999/{params, assets/trossen, _CHECKPOINT_METADATA}`.
- Started server: `serve_policy.py policy:checkpoint --policy.config=pi05_bimanual_rotation` on `:8000`. Dummy 4-camera dry-run passed: `(50, 14)`, sane magnitudes.

### 10.2 Camera mapping — **the big bug that ate v1 too**

**v3 dataset uses ALL FOUR D405s** (`top`, `bottom`, `left_wrist`, `right_wrist`); the two D455s on this rig are **not policy inputs**. In §8.2 I'd assigned the D455s as `cam_high`/`cam_low` — that was wrong then too, but the v1 dataset only used 2 cameras so the error was less visible.

Verified by downloading one v3 episode from HF and extracting a reference frame from each dataset channel (`artifacts/dataset_{top,bottom,left_wrist,right_wrist}.png`), then matching against live captures from each D405 (`artifacts/d405_<serial>.png`):

| Server channel | Dataset key | Live D405 serial | What the frame shows |
|---|---|---|---|
| `cam_high` | `observation.images.top` | `130322271752` | overhead view, both grippers + box visible |
| `cam_low` | `observation.images.bottom` | `130322272750` | low close-up of the cardboard box |
| `cam_left_wrist` | `observation.images.left_wrist` | `130322271535` | forward over table, gripper-V in foreground |
| `cam_right_wrist` | `observation.images.right_wrist` | `130322273480` | forward over table, gripper-V in foreground, box more to LEFT |

Prior to this fix, the policy was being fed wide-FOV **D455** views in the `cam_high`/`cam_low` slots where it was trained on narrow **D405** views. Total OOD perception — almost certainly the dominant cause of all "no rotation tendency / random reaching" behavior seen in both v1 (§8.6) and the early v3 attempts.

**Rule (recorded in memory):** `cam_high`/`cam_low` are *dataset channel names*, not physical heights or camera types — always verify by comparing a real training frame to a live camera capture, never guess from appearance/mounting.

### 10.3 USB-bandwidth gotcha — 4 D405s on one host controller

All 4 D405s enumerate on **USB Bus 004** (single host controller). With 4 active streams, iso scheduling starves at least one camera intermittently:

```
WARNING - Error reading frame in background thread for RealSenseCamera(<serial>): read failed (status=False)
TimeoutError: Timed out waiting for frame from camera RealSenseCamera(<serial>) after <N>ms.
```

Patches applied **in the client venv** (`examples/trossen_ai/.venv/lib/python3.11/site-packages/lerobot/cameras/realsense/camera_realsense.py` — **these are lost on a fresh `uv sync` and must be re-applied**):

- warmup pre-read sleep `1 s → 3 s`
- warmup-loop read timeout `200 ms → 3000 ms`, wrapped in `try/except RuntimeError: pass` so transient first-read failures don't kill `connect`
- `async_read` default timeout `200 ms → 2000 ms` so stale frames during running stall the loop (briefly drop control rate) instead of crashing it

Camera config (in `main.py`): all 4 D405s at `480×270 RGB8 @ 5 fps`, `warmup_s=10`. Dataset is 480×640 (4:3); we send 480×270 (16:9) — AlohaInputs resizes server-side to 224×224 so functionally OK but a 4:3↔16:9 aspect stretch remains (D405 RGB8 maxes out at 480×270; 640×480 is YUYV-only and `lerobot.cameras.realsense` hard-codes `rs.format.rgb8`).

**Real (un-fixed) bottleneck:** move at least one D405 to a USB port on a different host controller (not Bus 004). Then iso bandwidth is split across 2 host controllers and contention vanishes. Until that's done, fps is the only knob.

### 10.4 The "alternating success / fail" pattern and the auto-reset fix

Without intervention: a successful run is reliably followed by a failed run. Cause: the kernel UVC driver doesn't immediately release USB iso-bandwidth reservations when a stream ends, so the next launch starts with a fragmented schedule and one camera is starved. `hardware_reset()` forces each device to renegotiate from scratch.

Wired `_reset_realsense_cameras()` into the start of `TrossenOpenPIBridge.__init__()` — every `main.py` launch now hardware-resets first (~8 s startup hit), so every attempt begins with a clean iso schedule. No manual reset between runs.

### 10.5 Gripper firmware-limit crash and the action clamp

First end-to-end policy run ran ~10 s of motion then crashed with:
```
Joint 6 position limit exceeded: expected [-0.004, 0.044], motor reported 0.044022. Setting to idle.
```
22 μm of floating-point overshoot of the gripper's hard 0.044 m limit → firmware idles the joint → TCP broken pipe → `disconnect()` cleanup ALSO fails (it tries to drive the same out-of-range gripper). Result: arm stuck torque-on with gripper physically at 0.044042, and the only recovery is **physically** nudging the gripper carriage back inside `[0, 0.044]` (power-cycle alone doesn't help — controller re-reads the same encoder value and idles again).

**Fix:** added module-level `ACTION_LO`/`ACTION_HI` from the URDF actuator `ctrlrange` (1 mrad margin on arm joints, 1 mm margin on grippers → `[0.001, 0.043]`); `execute_action()` clamps every action through `np.clip` and logs a warning when any dim is clipped.

### 10.6 Graceful Ctrl+C → home

Wrapped `bridge.autonomous_mode()` in `try/except KeyboardInterrupt / finally: bridge.cleanup()` so Ctrl+C (and any unhandled exception) runs the library's `disconnect()` routine, which smoothly drives both arms to staged then sleep/zero (~4 s/arm). Caveat: this is a **controlled motion**, not an instant freeze — for real emergencies (collision imminent) use the **physical e-stop**, not Ctrl+C. No documented hardware e-stop button exists on the WidowX AI per Trossen docs; the bench's power-kill is the de-facto physical stop.

### 10.7 Replay-overlay verdict: deployment vs. policy

Ran `eval_tools/replay_overlay.py --config pi05_bimanual_rotation --episode 0 --frames 0,50,150` against the running server **before** the camera fix:

| frame | first-action MAE | chunk-mean MAE |
|---|---|---|
| 0 | 0.050 | 0.106 |
| 50 | 0.029 | 0.073 |
| 150 | 0.046 | 0.046 |

Worst dims (avg): **L_j3 (0.16) · R_j3 (0.15) · L_j2 (0.13)** — both **wrist-pitch** joints + an elbow. Wrist pitch is the most camera-dependent dim and was hit hardest, exactly matching the (then-active) D455-in-D405-slot bug. Improving-with-time MAE pattern (0.106 → 0.046) also fits "initial perception is off, policy stabilizes once moving." Conclusion: **mostly deployment** (primarily the camera mis-assignment from §10.2), with some narrow-finetune contribution from 100-episode LoRA.

⚠️ The overlay can NOT detect physical-to-channel-name swaps (it reads from the dataset, never from physical cameras). For that, use the dataset-vs-live frame matching method in §10.2.

### 10.8 Deployment pipeline — operator runbook (current, all fixes applied)

This supersedes §5.B for v3 deployment.

```bash
# 1. Server (once per machine session; ~30 s to load)
cd $HOME/pi0.5-bimanual/openpi
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_bimanual_rotation \
    --policy.dir=$HOME/pi0.5-bimanual/openpi_checkpoints/pi05_bimanual_rotation/pi05_bimanual_rotation_v1/29999

# 2. (offline, optional) replay-overlay sanity check — disambiguates deployment vs policy
HF_TOKEN=<read-token> uv run python ../eval_tools/replay_overlay.py \
    --config pi05_bimanual_rotation --episode 0 --frames 0,50,150

# 3. Place the box on the table in the trained layout. Confirm both arms powered, arm IPs
#    192.168.1.5 (left) / 192.168.1.4 (right) reachable, the four D405s plugged in
#    (D455s irrelevant for the policy).

# 4. (optional) Test mode — no policy-driven motion, but the disconnect at exit DOES move
#    both arms to rest pose (~4 s/arm). Useful for one full pipeline check on real frames.
cd examples/trossen_ai
uv run main.py --mode test --task_prompt "Bimanual Rotate" --max_steps 60

# 5. Autonomous — keep --max_steps small the first time (300 ≈ 10 s of policy motion).
uv run main.py --mode autonomous --task_prompt "Bimanual Rotate" --max_steps 300
```

What `main.py` does at every launch (all baked in via `trossen_ai_client_patch.diff`):
- Hardware-resets all RealSense devices (~8 s) → clean USB iso schedule
- Connects left arm → right arm → all 4 cameras (driver `v1.10.0` matches firmware `v1.10.0`)
- Runs the 30 Hz control loop, requesting a 50-step action chunk every 50 steps
- **Clamps every action** to safe joint ranges before sending
- On Ctrl+C / exception / normal finish → `cleanup()` → `disconnect()` → both arms smoothly home to staged then sleep/zero

**Stopping the run:**
- Ctrl+C → smooth return-to-rest (~4 s/arm). Orderly stop only.
- Bench physical e-stop / power-kill → immediate freeze. Use for real emergencies.

**Known recovery patterns:**
- Cameras report `Device or resource busy`: another process (often yunshuang's `visualize_live.py`) has them open. `sudo kill <PID>`, then retry.
- One D405 still flakes after auto-reset: drop fps lower in `main.py` (already at 5), or — the real fix — move one D405 to a different USB host controller.
- Gripper-limit firmware idle (left arm `0.044+` encoder reading): physically nudge the gripper carriage back into `[0, 0.044]`, then re-home.

### 10.9 Outcomes
- Pipeline confirmed working end-to-end (one no-box run + one ~10 s policy run on the box that crashed at the gripper limit — fixed since).
- With the camera-type fix (§10.2) + all the auxiliary patches, the next autonomous attempt is the **first** that's actually receiving the correct camera distribution; whether it succeeds at the task is now genuinely a test of policy quality on this rig (which the replay overlay suggested could improve a lot).
- **§5 / §8 are deprecated for v3 deployment — follow §10 instead.**

### 10.10 Artifacts (new / updated)
| Artifact | Location | Notes |
|---|---|---|
| Client edits patch | `trossen_ai_client_patch.diff` (root) | camera mapping (4× D405), driver pin, action clamp, Ctrl+C-home, auto-reset; apply in `openpi/` |
| Replay-overlay script | `eval_tools/replay_overlay.py` | now multi-config (v1 / v2 / v3 presets); needs HF token + matching server |
| Dataset reference frames | `artifacts/dataset_{top,bottom,left_wrist,right_wrist}.png` | gold for verifying camera mapping |
| Live D405 captures | `artifacts/d405_<serial>.png` | matched 1:1 against dataset frames |
| Lerobot venv patches | `openpi/examples/trossen_ai/.venv/.../camera_realsense.py` | **not in any patch file — lost on `uv sync`** |
| Replay overlay log | `/tmp/replay_overlay_v3.log` | the run that diagnosed deployment vs policy |

## 11. Eval-desktop session — v2 (flip, 4 cameras) on `coldbrew`, 2026-05-30

Brought up the v2 (flip-4-camera) checkpoint **in parallel with v3** — same client, same camera mapping, same patches, only the policy server config differs. v3 stays one `pkill` + relaunch away, no client changes ever needed.

### 11.1 Why v2 needs zero client changes
v2 (`pi05_bimanual_flip_4_camera`) and v3 (`pi05_bimanual_rotation`) **share the exact same dataset schema**: 4 cameras (`top` / `bottom` / `left_wrist` / `right_wrist`), 14-dim bimanual joints @ 30 Hz, single task per dataset. So everything in §10 (camera mapping, action clamp, USB-bandwidth patches, auto-reset, Ctrl+C-home) applies unchanged. The only per-policy differences are **server-side**:

| Thing that changes | v2 (flip) | v3 (rotation) |
|---|---|---|
| `--policy.config` | `pi05_bimanual_flip_4_camera` | `pi05_bimanual_rotation` |
| `--policy.dir` | `…/pi05_bimanual_flip_4_camera/pi05_bimanual_flip_4_camera_v1/29999` | `…/pi05_bimanual_rotation/pi05_bimanual_rotation_v1/29999` |
| `--task_prompt` (client) | `"Bimanual Flip"` | `"Bimanual Rotate"` |

### 11.2 Setup
- Pulled v2 checkpoint from HF (`BruceZhang0912/pi05-bimanual-flip-4-camera`, ~5.8 GB) into `$HOME/pi0.5-bimanual/openpi_checkpoints/pi05_bimanual_flip_4_camera/pi05_bimanual_flip_4_camera_v1/29999/{params, assets/trossen, _CHECKPOINT_METADATA}`.
- Stopped the v3 server (`pkill -f serve_policy.py`), started v2 server with `--policy.config=pi05_bimanual_flip_4_camera` on the same `:8000`. Both norm stats loaded; ready.

### 11.3 Runbook — v2 flip
```bash
# (one-time per machine session) start the v2 server
cd $HOME/pi0.5-bimanual/openpi
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_bimanual_flip_4_camera \
    --policy.dir=$HOME/pi0.5-bimanual/openpi_checkpoints/pi05_bimanual_flip_4_camera/pi05_bimanual_flip_4_camera_v1/29999

# place flip object(s) in trained layout, then:
cd examples/trossen_ai
uv run main.py --mode autonomous --task_prompt "Bimanual Flip" --max_steps 300
```

### 11.4 Switching between v2 (flip) and v3 (rotation)
Only the server swaps; `main.py` is untouched.

```bash
# v2 (flip)  -> v3 (rotation)
pkill -f serve_policy.py
cd $HOME/pi0.5-bimanual/openpi
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_bimanual_rotation \
    --policy.dir=$HOME/pi0.5-bimanual/openpi_checkpoints/pi05_bimanual_rotation/pi05_bimanual_rotation_v1/29999
# then client: uv run main.py --mode autonomous --task_prompt "Bimanual Rotate" --max_steps 300

# v3 (rotation) -> v2 (flip)
pkill -f serve_policy.py
cd $HOME/pi0.5-bimanual/openpi
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_bimanual_flip_4_camera \
    --policy.dir=$HOME/pi0.5-bimanual/openpi_checkpoints/pi05_bimanual_flip_4_camera/pi05_bimanual_flip_4_camera_v1/29999
# then client: uv run main.py --mode autonomous --task_prompt "Bimanual Flip" --max_steps 300
```

Server load takes ~30 s + a ~1–2 min JIT on the first inference call after each swap. If you'd rather run both servers concurrently (instant switch, no JIT penalty), launch them on different ports with `--port 8000` / `--port 8001` and `XLA_PYTHON_CLIENT_MEM_FRACTION=0.4` per server to fit both on the RTX 4090 (24 GB); then point the client at the right one with `--policy_port`. Not done currently — single-server-at-a-time is the simpler path.

### 11.5 Artifacts (v2-specific only — everything else is shared with §10)
| Artifact | Location |
|---|---|
| v2 checkpoint (gitignored) | `$HOME/pi0.5-bimanual/openpi_checkpoints/pi05_bimanual_flip_4_camera/pi05_bimanual_flip_4_camera_v1/29999/` |
| v2 HF repo (private) | https://huggingface.co/BruceZhang0912/pi05-bimanual-flip-4-camera |
| v2 dataset on HF (private) | https://huggingface.co/datasets/BruceZhang0912/Bimanual-Flip-4-Camera |

## 12. Real-arm eval results — both policies fail to generalize, 2026-05-31

Headline: **0/15 success on every box, on both v2 (flip) and v3 (rotation).** The full deployment pipeline from §10 and §11 was in place — correct 4× D405 camera mapping verified against training frames (§10.2), action clamp (§10.5), auto camera reset, Ctrl+C-home, driver/firmware match. No crashes, no premature termination; the arms moved smoothly through trained-looking trajectories every trial. They simply did not complete the task.

### 12.1 Protocol
- **v2 (`pi05_bimanual_flip_4_camera`)**: each box variant flipped 15 times → **0/15** success on each.
- **v3 (`pi05_bimanual_rotation`)**: each box variant rotated 15 times → **0/15** success on each.
- Client: `uv run main.py --mode autonomous --task_prompt "<Bimanual Flip|Rotate>" --max_steps 300/400` (~10–13 s of policy motion per trial). Server config / prompt swapped per §11.4. Same arms, same camera mount positions, scene set to match the training layout.

### 12.2 What this means now that deployment is mostly fixed

The §10.7 replay overlay had identified deployment (specifically the D455-vs-D405 camera-type bug in `cam_high`/`cam_low`) as the dominant cause of the §8/§10 v1 failures. After fixing that and the rest of the §10 patch stack, **0/15 across both policies on the real arms** says the remaining failure is no longer dominated by deployment — it sits with policy generalization on this rig.

Suspected dominant factors, in priority order:

1. **Narrow training distribution.** Each policy is a LoRA finetune on ~100 episodes of one task, one platform, one controlled scene. Trossen's own openpi notes warn that small datasets generalize poorly to changes in object color/shape/lighting/exact camera pose. Even after the camera fix, residual differences (lighting, exact mount angles, lens calibration drift between training cameras and our serials) are still distribution shift.
2. **Camera resolution / fps still mismatched.** Dataset frames stored at 480×640 (4:3) @ 30 fps; deployment runs at 480×270 (16:9) @ 5 fps because (a) D405 RGB8 maxes out at 480×270 and (b) all four D405s share USB Bus 004 → bandwidth contention forces 5 fps. AlohaInputs resizes to 224×224 either way, but the 4:3↔16:9 aspect stretch and 6× lower frame rate are real shifts.
3. **No success / termination logic.** The client runs a fixed `--max_steps` with no task-success detection — it keeps acting regardless of state. Doesn't cause failure, but means "success" requires the entire trained behavior to fire in the right window.

Things that are NOT likely to be the cause (verified):
- Channel-name swap (cam_left_wrist ↔ cam_right_wrist): replay-overlay error pattern in §10.7 was not symmetric in a way that swap would produce, and the swap-experiment didn't move MAE in the expected direction.
- Joint-order bug: replay MAE was distributed across joints, not concentrated on one.
- Driver/firmware: matched (v1.10.0 both sides), no errors during the runs.
- Camera channel-to-camera-type assignment: verified by dataset-vs-live frame matching in §10.2.

### 12.3 Recommended next investigations

In rough order of cost:

1. **Re-run replay overlay with the corrected camera mapping** (now that §10.2 is fixed). The pre-fix MAE numbers in §10.7 reflected the wrong cameras feeding the policy. A fresh run should show how much MAE the policy currently has on real-frame inputs; if MAE is now small (<0.03 chunk-mean), the policy IS reproducing demonstrations on dataset frames and the gap to hardware is environmental/lens, not the model. If MAE is still ≥ ~0.05, the policy's near-training-MAE floor is the bottleneck and more data is the answer.
2. **Run with the exact same physical object used during data collection** if not already, before concluding generalization fails. Color/material shifts matter for these narrow finetunes.
3. **Increase camera fps** — splitting wrist D405s onto a USB bus that isn't Bus 004 lets us run all four at 30 fps. Probably moderate.
4. **More training data**. Either more episodes per task, or a wider scene/object distribution, or both. Probably the biggest lever long-term.
5. **Consider unfreezing more of the pi0.5 backbone** — current configs LoRA-tune PaliGemma + the 300M action expert; broadening that may help, at training-cost.

### 12.4 What's confirmed working (so this section doesn't get misread)
- End-to-end client/server pipeline: cameras stream, policy queried at 30 Hz (limited by 5 fps wrist), 50-step action chunks decoded and executed; both arms move smoothly.
- Safety: action clamp prevents the gripper-limit firmware crash from §10.5 from recurring; Ctrl+C and unhandled exceptions both route arms back to the rest pose; auto camera reset eliminates the alternating success/fail launch pattern from §10.4.
- Switching policies: server-only restart per §11.4 works as documented; client unchanged.

The infrastructure is solid; the **policy quality on this rig** is the open problem.
