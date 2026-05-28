# Dev log — pi0.5 fine-tunes on bimanual flip datasets

Author: brucezha@usc.edu (also collected all training data)
Machine used for training: yunshuang's box (GPU 3, RTX 6000 Ada 48 GB)

## 1. Goal

Fine-tune the pi0.5 base policy on Trossen `trossen_ai_stationary` bimanual flip
data so it can be evaluated on the real Trossen arms.

Two datasets covered so far (both collected by Bruce, 14-dim bimanual joints,
30 fps, task `"Bimanual Flip"`):

| run | dataset | episodes | frames | cameras | finetuned on |
|---|---|---|---|---|---|
| v1 (2026-05-26) | local `bimanual_flip_5_objects` | 100 | 44 883 | 2 (cam_high, cam_low) | pi05_base + LoRA |
| v2 (2026-05-28) | HF `BruceZhang0912/Bimanual-Flip-4-Camera` | 101 | 30 199 | 4 (top, bottom, left_wrist, right_wrist) | pi05_base + LoRA |

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

### 4.5 Push the 4-camera checkpoint to Hugging Face (TBD)

Same procedure as v1 — upload `29999/{params,assets,_CHECKPOINT_METADATA}` (~5.8 GB, skip `train_state/` ≈ 2.8 GB) to a new private repo, suggested name `BruceZhang0912/pi05-bimanual-flip-4-camera`. Once pushed, also amend the GitHub repo's `pi05_config_patch.diff` to include both the v1 and v2 TrainConfig blocks.

## 5. What to run, step by step

The steps below were written for **v1** (`pi05_bimanual_flip_5_objects`, 2-camera). Substitute the v2 names where indicated to deploy `pi05_bimanual_flip_4_camera` instead — the recipe is identical, only the config name, exp-name, HF model repo (TBD when v2 finishes and gets uploaded), and dataset id change.

### 5.A On THIS machine (`/data/bruce/pi0.5/openpi` on yunshuang's box)

**v1**: nothing further to run — checkpoint trained, HF upload done, dev_log + config patch pushed to GitHub.

**v2**: training **in progress** at the time of writing (PID 3548308 on GPU 3). Once it finishes, follow the same pattern: upload `29999/{params,assets,_CHECKPOINT_METADATA}` to a new HF model repo (e.g. `BruceZhang0912/pi05-bimanual-flip-4-camera`), and amend the GitHub repo's patch to include the v2 block alongside the v1 block.

Note: the `pi05_bimanual_flip_5_objects` block in `src/openpi/training/config.py` is an **uncommitted local edit** on the `trossen-ai` branch. That's fine — we don't push it back to the TrossenRobotics fork. The diff is preserved as:

- GitHub repo `brucezhangcy/pi0.5-bimanual` → `pi05_config_patch.diff`
- Local: regenerate any time with `git diff src/openpi/training/config.py`

If you ever want to retrain on this box, the existing setup still works:

```bash
cd /data/bruce/pi0.5/openpi
CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
    uv run scripts/train.py pi05_bimanual_flip_5_objects \
        --exp-name=pi05_bimanual_flip_5_objects_v2 \
        --overwrite
```

### 5.B On the eval desktop (the box wired to the real Trossen arms)

Two things live in two different places:

- **GitHub** (`brucezhangcy/pi0.5-bimanual`, private): this dev_log + the openpi config patch. Small text files. Will also hold future task variants (rotation, etc.).
- **Hugging Face** (`BruceZhang0912/pi05-bimanual-flip-5-objects`, private): the 5.8 GB trained checkpoint.

**Layout on the eval desktop.** Everything is consolidated under the cloned instructions repo so there's a single self-contained project folder (this is how the eval box is actually set up — `coldbrew`, RTX 4090):

```
$HOME/pi0.5-bimanual/                       # the GitHub instructions repo (this dir)
├── dev_log.md  pi05_config_patch.diff  README.md
├── openpi/                                 # TrossenRobotics/openpi clone (gitignored)
└── openpi_checkpoints/                     # HF checkpoint download (gitignored)
    └── pi05_bimanual_flip_5_objects/pi05_bimanual_flip_5_objects_v1/29999/
```

`openpi/` and `openpi_checkpoints/` are listed in `.gitignore` so they never get committed back into the instructions repo. The dataset is **not** required on the eval box (eval runs from the checkpoint + base Trossen norm stats); it lives only on the training box.

Six steps, in order:

**Step 1 — Clone the instructions repo from GitHub** (gives you this dev_log + the config patch).

```bash
git clone https://github.com/brucezhangcy/pi0.5-bimanual.git
cd pi0.5-bimanual
ls   # dev_log.md  pi05_config_patch.diff  README.md
```

**Step 2 — Clone openpi** (into the instructions repo so everything stays under one folder).

```bash
# from inside $HOME/pi0.5-bimanual (where Step 1 left you)
git clone --recurse-submodules https://github.com/TrossenRobotics/openpi.git
cd openpi
git checkout trossen-ai          # same branch we trained on
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

**Step 3 — Apply the config patch** (without this, `serve_policy.py` doesn't know how to wrap the weights).

```bash
# still in $HOME/pi0.5-bimanual/openpi (the patch is one level up)
git apply ../pi05_config_patch.diff
grep -n "pi05_bimanual_flip_5_objects" src/openpi/training/config.py    # confirm the block is in
```

**Step 4 — Pull the checkpoint from HF** (~5.8 GB).

```bash
# huggingface-cli ships in openpi's uv env; run it from $HOME/pi0.5-bimanual/openpi
HF_TOKEN=<your HF read token> \
    uv run huggingface-cli download BruceZhang0912/pi05-bimanual-flip-5-objects \
        --local-dir $HOME/pi0.5-bimanual/openpi_checkpoints/pi05_bimanual_flip_5_objects/pi05_bimanual_flip_5_objects_v1
```

(The `--local-dir-use-symlinks` flag was removed in current `huggingface_hub` — omit it; the CLI already copies real files into `--local-dir`.)

After this the directory should look like:
```
$HOME/pi0.5-bimanual/openpi_checkpoints/pi05_bimanual_flip_5_objects/pi05_bimanual_flip_5_objects_v1/29999/
    ├── params/
    ├── assets/trossen/
    └── _CHECKPOINT_METADATA
```

**Step 5 — Start the policy server.**

```bash
cd $HOME/pi0.5-bimanual/openpi
uv run scripts/serve_policy.py \
    policy:checkpoint \
    --policy.config=pi05_bimanual_flip_5_objects \
    --policy.dir=$HOME/pi0.5-bimanual/openpi_checkpoints/pi05_bimanual_flip_5_objects/pi05_bimanual_flip_5_objects_v1/29999
```

Defaults to a websocket server on port 8000. First inference call JIT-compiles for 1–2 min — fire a dummy request before letting the robot move.

**Step 6 — Run the Trossen client against the live arms.**

Follow the **Real-Robot Evaluation** section of https://docs.trossenrobotics.com/trossen_arm/v1.8/tutorials/openpi.html. The client must:

- Send `observation.images.cam_high` and `observation.images.cam_low` images each control tick.
- **Not** send wrist-camera images — model was trained without them; `image_mask` for those slots is `False` and real wrist images would be ignored.
- Send the 14-dim joint state in order `left_joint_0..6, right_joint_0..6`.
- Send `prompt="Bimanual Flip"` (same string as `meta/tasks.jsonl`).
- Run at ~30 Hz (the dataset fps); execute the returned action chunk and only re-query the policy once per chunk.

### 5.C Sanity checks before touching the robot

1. **Dry run — ✅ DONE on `coldbrew` 2026-05-26.** Sent a dummy observation (zero 14-dim state + random `cam_high`/`cam_low` images @224 CHW + `prompt="Bimanual Flip"`) through the websocket client. Output action chunk is `(50, 14)` with joint magnitudes in a sane radian range (global abs max ≈ 0.65, no NaNs/blowups). Confirms repack → norm-stats → pi0.5 → action-decode works end-to-end. Values aren't behaviorally meaningful (random images) — that's what #2 is for. Script: `/tmp/dryrun_infer.py`.
2. **Replay overlay** — feed the first frame of a real training episode and see whether the predicted action chunk roughly matches the recorded action. Wildly off → camera mapping or joint ordering is wrong. **Needs the dataset**, which is not on the eval box; copy a few episodes from the training box (`/data/bruce/pi0.5/bimanual_flip_5_objects`) under `$HOME/pi0.5-bimanual/datasets/` to run this.
3. **Slow first physical run** — clamp action rate / scale to ~50 % until you see the policy execute one full flip without surprises. Then ramp up.

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

## 8. Eval-desktop session — `coldbrew` (RTX 4090), 2026-05-27

Brought the trained checkpoint up on the eval desktop wired to the real Trossen arms and ran the first real-robot eval. **Outcome: the policy did not succeed at the flip task (see 8.6).**

### 8.1 Setup (done)
- Cloned `TrossenRobotics/openpi` (branch `trossen-ai`), `uv sync` + `uv pip install -e .`, applied `pi05_config_patch.diff`. Pulled the 5.8 GB checkpoint from HF into `29999/`.
- **Consolidated layout**: everything lives under `$HOME/pi0.5-bimanual/` — the instructions repo, plus `openpi/` and `openpi_checkpoints/` (both gitignored). Moving `openpi/` breaks its uv venv (absolute paths) → re-ran `uv sync` after the move. §4.B commands updated to these paths.
- Policy server: `serve_policy.py policy:checkpoint --policy.config=pi05_bimanual_flip_5_objects` on `:8000`. Dry-run (§4.C #1) passed: `(50,14)`, sane magnitudes.

### 8.2 Camera mapping
- This rig has 2× RealSense **D455** scene cams + 4× **D405** (wrist, unused). `cam_high=338122302972`, `cam_low=333422304645`.
- **Caveat:** both D455s are at the **same physical height** — the high/low assignment was confirmed only by *viewpoint appearance*, NOT by matching against a real training frame. `cam_high`/`cam_low` are dataset *channel names*, not heights; a swap feeds the policy OOD images. This is unverified ground-truth and a prime suspect for 8.6. See `eval_tools/replay_overlay.py`.

### 8.3 Driver/firmware fix (gotcha)
- `main.py`'s arm `connect()` failed: **driver v1.9.0 vs controller firmware v1.10.0** ("major and minor versions must match"). The client env resolved `trossen-arm==1.9.0` (lerobot_trossen only pins `>=1.9.0`), and `uv run` auto-syncs so a manual `uv pip install` got reverted. **Fix: pinned `trossen-arm==1.10.0` in `examples/trossen_ai/pyproject.toml`.**

### 8.4 Client edits (preserved as `trossen_ai_client_patch.diff`)
The `openpi/` tree is gitignored, so client edits are saved as a patch in this repo:
- Set the two camera serials, dropped the wrist-cam entries.
- Pinned `trossen-arm==1.10.0`.
- Wrapped the run loop in `try/finally` so **Ctrl+C also returns the arms to rest** (`cleanup()`→`disconnect()` drives both arms to staged then sleep/zero, ~4 s/arm).

### 8.5 Test mode (`--mode test`) — passed
Full real pipeline verified: both arms connected (driver==firmware==1.10.0), both cameras connected, policy queried → `(50,14)`, logged would-be actions are smooth and within joint ranges. **Note:** test mode does NOT execute the per-step actions, but `disconnect()` on exit DOES move both arms to the rest pose (~4 s/arm) — i.e. test mode is not entirely motion-free at shutdown.

### 8.6 Real eval (`--mode autonomous`) — NO SUCCESS
Ran the policy on the live arms. **It did not complete the bimanual flip.** (Detailed failure behavior not yet logged here — TODO: fill in what the arms actually did: reached wrong spot / failed to grasp / erratic / froze, etc.)

Leading hypotheses, roughly in priority order:
1. **Camera mapping unverified against training data** (8.2) — top suspect. Run `eval_tools/replay_overlay.py` on a real episode: if predicted≈recorded actions, mapping+pipeline are right; if all dims are off with cameras looking fine, cam_high/cam_low are likely swapped.
2. **Scene/distribution mismatch** — pi0.5 here is a narrow LoRA finetune (100 eps, 1 task, controlled scene). Object choice, positions, lighting, and camera *poses* must closely match data collection. "Same setup" was asserted but not validated against training frames.
3. **No success/termination logic** — the client runs a fixed `--max_steps` (default 1000 ≈ 33 s @ 30 Hz) with no task-success detection; it keeps acting regardless. Not a failure cause, but shapes what "no success" looks like.

**Recommended next step:** copy a few episodes from the training box and run the **Tier-1 replay overlay** before the next hardware attempt — it disambiguates camera-mapping vs. policy-quality without risking the arms.

### 8.7 Safety tooling added (`eval_tools/`)
Safety ladder + scripts (see `eval_tools/README.md`): `capture_action_chunk.py`, `sim_playback.py` (Tier-2 MuJoCo limit/velocity check + mp4, runs in the `trossen_sim` conda env), `replay_overlay.py` (Tier-1, needs dataset), `make_demo_trajectory.py`. Physical e-stop for this rig is **not** a documented dedicated button — confirm the bench's power-kill/e-stop; comm-loss and collision auto-idle are the documented software safeties.

### 8.8 New artifacts
| Artifact | Location |
|---|---|
| Eval-desktop openpi clone (gitignored) | `$HOME/pi0.5-bimanual/openpi/` (branch `trossen-ai` + patches) |
| Checkpoint (gitignored) | `$HOME/pi0.5-bimanual/openpi_checkpoints/pi05_bimanual_flip_5_objects/pi05_bimanual_flip_5_objects_v1/29999/` |
| Client edits patch | `trossen_ai_client_patch.diff` (apply in `openpi/`) |
| Safety/eval tooling | `eval_tools/` |
| Test-mode log | `/tmp/client_test.log` |
