"""Tier-1 replay overlay (deployment vs. policy quality disambiguator).

Feed REAL recorded frames from one episode of the training dataset to the running
policy server and compare predicted vs recorded actions per joint. Strongest offline
correctness check we have. Runs in the openpi env, with the matching server already
up on :8000.

Pick a preset that matches your active server config. Add a new preset if you train
a new variant with a different dataset schema.

Example (v3 rotation):
    cd /home/bruce/pi0.5-bimanual/openpi
    HF_TOKEN=... uv run python ../eval_tools/replay_overlay.py \\
        --config pi05_bimanual_rotation \\
        --episode 0 --frames 0,50,150 --horizon 50

Diagnosis guide for the printed MAE table:
 - chunk-mean MAE < ~0.03 uniformly: pipeline + mapping correct, blame deployment
   quirks (cam resolution/fps/aspect) for real-arm misbehavior.
 - ONE dim huge (e.g. 5x others): joint-order bug.
 - All dims off + improving over time: initial perception is off (camera mapping
   or scene mismatch), policy stabilizes once moving.
 - All dims off uniformly: policy isn't generalizing to this rig.

CAVEAT: the overlay reads from the dataset and never touches the physical cameras,
so it CANNOT detect a physical-to-channel-name swap (e.g. serial X is on the left
arm but main.py routes it to cam_right_wrist). For that, compare a dataset frame
to a live camera capture directly (see dev_log v3 §10.2).
"""
import argparse
import numpy as np
import cv2
from openpi_client import websocket_client_policy

DIM_NAMES = [
    "L_j0", "L_j1", "L_j2", "L_j3", "L_j4", "L_j5", "L_grip",
    "R_j0", "R_j1", "R_j2", "R_j3", "R_j4", "R_j5", "R_grip",
]

# (repo_id, channel_map: server_channel -> dataset_key) per config
PRESETS = {
    "pi05_bimanual_flip_5_objects": (
        "bruce/bimanual_flip_5_objects",
        {
            "cam_high": "observation.images.cam_high",
            "cam_low":  "observation.images.cam_low",
        },
    ),
    "pi05_bimanual_flip_4_camera": (
        "BruceZhang0912/Bimanual-Flip-4-Camera",
        {
            "cam_high":        "observation.images.top",
            "cam_low":         "observation.images.bottom",
            "cam_left_wrist":  "observation.images.left_wrist",
            "cam_right_wrist": "observation.images.right_wrist",
        },
    ),
    "pi05_bimanual_rotation": (
        "BruceZhang0912/Bimanual-Recordings",
        {
            "cam_high":        "observation.images.top",
            "cam_low":         "observation.images.bottom",
            "cam_left_wrist":  "observation.images.left_wrist",
            "cam_right_wrist": "observation.images.right_wrist",
        },
    ),
}


def to_client_image(img):
    """Match examples/trossen_ai/main.py's observation format:
    LeRobot returns CHW float[0,1] tensors -> HWC uint8 RGB resized to 224 -> CHW uint8.
    """
    import torch
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] < arr.shape[-1]:
        arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255).round().astype(np.uint8)
    arr = cv2.resize(arr, (224, 224))
    return np.transpose(arr, (2, 0, 1))  # HWC -> CHW uint8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=sorted(PRESETS.keys()))
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--frames", type=str, default="0,50,150",
                    help="comma-separated frame indices within the episode to test")
    ap.add_argument("--horizon", type=int, default=50)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    repo_id, cam_map = PRESETS[args.config]
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    print(f"loading {repo_id}; cam_map={list(cam_map.keys())}")
    meta = LeRobotDatasetMetadata(repo_id)
    H = args.horizon
    ds = LeRobotDataset(
        repo_id,
        episodes=[args.episode],
        delta_timestamps={"action": [k / meta.fps for k in range(H)]},
    )
    print(f"  episode {args.episode}: {len(ds)} frames, task(s)={list(meta.tasks.values())[:2]}")

    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)

    all_results = []
    for fidx in [int(s) for s in args.frames.split(",")]:
        if fidx + H > len(ds):
            print(f"skip frame {fidx}: only {len(ds)} frames in episode")
            continue
        sample = ds[fidx]
        state = np.asarray(sample["observation.state"], dtype=np.float32).reshape(-1)[:14]
        recorded = np.asarray(sample["action"], dtype=np.float64).reshape(H, 14)
        if "task" in sample:
            prompt = sample["task"]
        elif "task_index" in sample:
            prompt = list(meta.tasks.values())[int(sample["task_index"])]
        else:
            prompt = list(meta.tasks.values())[0]

        obs = {
            "state": state,
            "images": {ck: to_client_image(sample[dk]) for ck, dk in cam_map.items()},
            "prompt": prompt,
        }
        print(f"\n--- frame {fidx} (prompt={prompt!r}) ---")
        pred = np.asarray(client.infer(obs)["actions"], dtype=np.float64)[:H]
        n = min(len(pred), len(recorded))
        err = np.abs(pred[:n] - recorded[:n])
        rng = recorded[:n].max(0) - recorded[:n].min(0) + 1e-9
        print(f"{'dim':6s} {'MAE':>8s} {'maxAE':>8s} {'recRng':>8s} {'MAE/rng':>9s}")
        for d in range(14):
            print(f"{DIM_NAMES[d]:6s} {err[:,d].mean():8.4f} {err[:,d].max():8.4f} "
                  f"{rng[d]:8.4f} {err[:,d].mean()/rng[d]*100:8.1f}%")
        print(f"first-action MAE = {err[0].mean():.4f}   chunk-mean MAE = {err.mean():.4f}")
        all_results.append((fidx, err.mean(), err[0].mean(), err.mean(axis=0)))

    if all_results:
        print("\n=== summary ===")
        for fidx, chunk_mae, first_mae, _ in all_results:
            print(f"  frame {fidx}: chunk-mean MAE = {chunk_mae:.4f}   first-action = {first_mae:.4f}")
        avg_per_dim = np.mean([r[3] for r in all_results], axis=0)
        worst = np.argsort(avg_per_dim)[-3:][::-1]
        print(f"\n  worst 3 dims (avg MAE):")
        for d in worst:
            print(f"    {DIM_NAMES[d]:6s} = {avg_per_dim[d]:.4f}")


if __name__ == "__main__":
    main()
