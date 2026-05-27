"""Query the running policy server once and save the action chunk to a .npy file.

Run from the openpi env (it has openpi_client):
    cd /home/bruce/pi0.5-bimanual/openpi
    uv run python ../eval_tools/capture_action_chunk.py --out ../artifacts/sample_action_chunk.npy

By default this sends a DUMMY observation (zero state + random images), so the
resulting actions are NOT behaviorally meaningful -- they only exercise the
pipeline and give the sim playback / limit checker a [T,14] chunk to chew on.
Pass --state to seed a non-zero starting state if you want.
"""
import argparse
import numpy as np
from openpi_client import websocket_client_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--prompt", default="Bimanual Flip")
    ap.add_argument("--out", required=True, help="output .npy path for the [T,14] action chunk")
    args = ap.parse_args()

    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)

    def img():
        return np.random.randint(0, 255, size=(3, 224, 224), dtype=np.uint8)

    obs = {
        "state": np.zeros(14, dtype=np.float32),
        "images": {"cam_high": img(), "cam_low": img()},
        "prompt": args.prompt,
    }
    resp = client.infer(obs)
    actions = np.asarray(resp["actions"], dtype=np.float64)
    assert actions.ndim == 2 and actions.shape[1] == 14, f"expected [T,14], got {actions.shape}"
    np.save(args.out, actions)
    print(f"saved {actions.shape} action chunk -> {args.out}")
    print("per-dim min:", np.round(actions.min(0), 3))
    print("per-dim max:", np.round(actions.max(0), 3))


if __name__ == "__main__":
    main()
