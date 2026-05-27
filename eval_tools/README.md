# eval_tools — safety ladder for the bimanual-flip pi0.5 policy

Run these in order, lowest risk first, before letting the real arms move. They all
talk to the policy server from dev_log §4.B step 5 (websocket on `:8000`).

| Tier | Tool | Env | Needs | Risk |
|---|---|---|---|---|
| 0 | dummy dry-run (`/tmp/dryrun_infer.py`) | openpi | nothing | none |
| 1 | `replay_overlay.py` | openpi | dataset on box | none (offline) |
| 2 | `sim_playback.py` | `trossen_sim` conda | mujoco only | none |
| 3 | `examples/trossen_ai/main.py --mode test` | trossen_ai | arms + cameras | very low (no motion) |
| 4 | `examples/trossen_ai/main.py --mode autonomous` | trossen_ai | hardware + operator | real |

**What sim can and cannot do:** pi0.5 was trained on real RealSense images, so a MuJoCo
render is out-of-distribution — closing the loop in sim gives meaningless *behavior*.
Sim (`sim_playback.py`) validates *mechanics* only: joint limits, jumps, gross pose.
The real behavioral/correctness signal is Tier 1 (`replay_overlay.py`) on real frames.

## capture_action_chunk.py  (openpi env)
Saves one action chunk from the server to `.npy` (dummy obs by default) to feed Tier 2.
```bash
cd /home/bruce/pi0.5-bimanual/openpi
uv run python ../eval_tools/capture_action_chunk.py --out ../artifacts/sample_action_chunk.npy
```

## sim_playback.py  (trossen_sim conda env, mujoco 3.x)
Plays a `[T,14]` chunk through `stationary_ai/scene_joint.xml`; prints a limit/velocity
report and renders an mp4. Exit code 2 if any check fails.
```bash
MUJOCO_GL=egl /home/bruce/miniconda3/envs/trossen_sim/bin/python \
    /home/bruce/pi0.5-bimanual/eval_tools/sim_playback.py \
    --chunk /home/bruce/pi0.5-bimanual/artifacts/sample_action_chunk.npy \
    --out   /home/bruce/pi0.5-bimanual/artifacts/sim_playback.mp4
# --camera cam_high  shows roughly the policy's top-down view
```
Action dim order = dataset order: `[L_j0..L_j5, L_grip, R_j0..R_j5, R_grip]`. Arm joints
are radians; gripper dims (6,13) are the carriage slide in **meters [0,0.044]** — if your
dataset stores the gripper differently, those dims will (correctly) trip the range check.
The limit ranges come from the sim model's actuator `ctrlrange` (URDF-derived); the real
arm firmware limits are the ultimate ground truth.

## replay_overlay.py  (openpi env)  — strongest pre-hardware check, needs the dataset
Feeds REAL recorded frames to the server and compares predicted vs recorded actions
(per-joint MAE). Dataset is **not on this box**; copy a few episodes first — see the
header of `replay_overlay.py` for the rsync + symlink commands. Then:
```bash
cd /home/bruce/pi0.5-bimanual/openpi
uv run python ../eval_tools/replay_overlay.py --episode 0 --frame 0
```
Small, uniform MAE ⇒ pipeline + joint mapping correct. One dim huge ⇒ joint-order bug.
All dims off (with cameras looking right) ⇒ cam_high/cam_low swapped.

## Tier 3/4 — real arm  (trossen_ai env)
See dev_log §4.B step 6 / §4.C. Before the first run, edit `examples/trossen_ai/main.py`:
RealSense **serial numbers** (must match this robot and the trained cam_high/cam_low
assignment), **arm IPs**, and always pass `--task_prompt "Bimanual Flip"`.
```bash
cd /home/bruce/pi0.5-bimanual/openpi/examples/trossen_ai
uv run main.py --mode test       --task_prompt "Bimanual Flip"   # logs actions, no motion
uv run main.py --mode autonomous --task_prompt "Bimanual Flip"   # moves — operator + e-stop
```
