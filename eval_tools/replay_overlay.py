"""Tier-1 replay overlay: the strongest OFFLINE correctness/safety check.

Feed REAL recorded frames from a training episode to the running policy server and
compare the predicted action chunk against the recorded ground-truth actions. If the
policy tracks the demonstrations, (a) it learned the task and (b) its commands stay
near human-demonstrated trajectories -- which is what makes them safe to run.

Requires the dataset on this box (it is NOT here yet). Copy a few episodes from the
training box, e.g.:
    rsync -a <trainbox>:/data/bruce/pi0.5/bimanual_flip_5_objects \
        $HOME/pi0.5-bimanual/datasets/
    # make LeRobot find it by repo_id:
    mkdir -p ~/.cache/huggingface/lerobot/bruce
    ln -sfn $HOME/pi0.5-bimanual/datasets/bimanual_flip_5_objects \
        ~/.cache/huggingface/lerobot/bruce/bimanual_flip_5_objects

Then run in the OPENPI env (has openpi_client + lerobot v0.1.0), with the policy
server already up on :8000:
    cd /home/bruce/pi0.5-bimanual/openpi
    uv run python ../eval_tools/replay_overlay.py --episode 0 --frame 0

What "good" looks like: per-joint MAE small relative to that joint's motion range
over the episode (rule of thumb < ~10-15% of range, gripper looser). A single joint
with huge MAE => likely a joint-ordering bug. All joints uniformly off + cameras
look right => check the cam_high/cam_low assignment.
"""
import argparse
import numpy as np
from openpi_client import websocket_client_policy

REPO_ID = "bruce/bimanual_flip_5_objects"
DIM_NAMES = [
    "L_j0", "L_j1", "L_j2", "L_j3", "L_j4", "L_j5", "L_grip",
    "R_j0", "R_j1", "R_j2", "R_j3", "R_j4", "R_j5", "R_grip",
]


def to_client_image(img):
    """Match examples/trossen_ai/main.py: HxWx3 uint8 (RGB) -> resize 224 -> CHW uint8.

    Handles either HWC-uint8 or CHW-float[0,1] inputs from LeRobot defensively.
    """
    import cv2

    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] < arr.shape[-1]:
        arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255).round().astype(np.uint8)
    arr = cv2.resize(arr, (224, 224))
    return np.transpose(arr, (2, 0, 1))  # HWC -> CHW uint8 (RGB, as the dataset stores it)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--frame", type=int, default=0, help="frame index within the episode")
    ap.add_argument("--horizon", type=int, default=50, help="action chunk length to compare")
    ap.add_argument("--repo-id", default=REPO_ID)
    args = ap.parse_args()

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(args.repo_id)
    H = args.horizon
    # Pull the recorded action chunk [t .. t+H) via delta_timestamps.
    ds = LeRobotDataset(
        args.repo_id,
        delta_timestamps={"action": [k / meta.fps for k in range(H)]},
    )
    ep_from = ds.episode_data_index["from"][args.episode].item()
    ep_to = ds.episode_data_index["to"][args.episode].item()
    idx = ep_from + args.frame
    if idx + H > ep_to:
        raise SystemExit(f"frame {args.frame}+horizon {H} exceeds episode length {ep_to - ep_from}")
    sample = ds[idx]

    state = np.asarray(sample["observation.state"], dtype=np.float32).reshape(-1)[:14]
    recorded = np.asarray(sample["action"], dtype=np.float64).reshape(H, 14)
    prompt = meta.tasks[sample["task_index"].item()] if "task_index" in sample else "Bimanual Flip"

    obs = {
        "state": state,
        "images": {
            "cam_high": to_client_image(sample["observation.images.cam_high"]),
            "cam_low": to_client_image(sample["observation.images.cam_low"]),
        },
        "prompt": prompt,
    }

    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    pred = np.asarray(client.infer(obs)["actions"], dtype=np.float64)[:H]

    n = min(len(pred), len(recorded))
    pred, recorded = pred[:n], recorded[:n]
    err = np.abs(pred - recorded)  # [n,14]
    rng = recorded.max(0) - recorded.min(0) + 1e-9

    print(f"\nepisode {args.episode} frame {args.frame}  prompt={prompt!r}  horizon={n}")
    print(f"{'dim':6s} {'MAE':>8s} {'maxAE':>8s} {'recRange':>9s} {'MAE/range':>10s}")
    for d in range(14):
        print(f"{DIM_NAMES[d]:6s} {err[:,d].mean():8.4f} {err[:,d].max():8.4f} "
              f"{rng[d]:9.4f} {err[:,d].mean()/rng[d]*100:9.1f}%")
    overall = err.mean()
    print(f"\noverall MAE = {overall:.4f} rad/m   (first-action MAE = {err[0].mean():.4f})")
    print("interpretation: small & uniform => pipeline + mapping correct; one dim huge "
          "=> joint-order bug; all dims off => camera mapping / OOD frames")


if __name__ == "__main__":
    main()
