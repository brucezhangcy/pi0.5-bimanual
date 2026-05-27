"""Generate a smooth, in-limits bimanual demo trajectory [T,14] for sim_playback.py.

NOT policy output -- a hand-authored sweep so you can watch the sim arms move through
the workspace and confirm joint mapping / limits look right. Uses smootherstep easing
between waypoints (zero velocity at each waypoint) so it stays well under the velocity
caps and passes the Tier-2 safety check.

    /home/bruce/miniconda3/envs/trossen_sim/bin/python \
        /home/bruce/pi0.5-bimanual/eval_tools/make_demo_trajectory.py \
        --out /home/bruce/pi0.5-bimanual/artifacts/demo_sweep.npy
"""
import argparse
import numpy as np

# (time_s, value) waypoints per LEFT-arm dim; right arm mirrors a few signs below.
# Stays inside ranges: j0 +-3.05, j1 0..3.14, j2 0..2.36, j3/j4 +-1.57, j5 +-3.14, grip 0..0.044
WAYPOINTS_S = [0.0, 2.0, 4.0, 6.0]
LEFT = {
    0: [0.0,  0.30, -0.30, 0.0],   # base yaw sway
    1: [0.0,  1.00,  0.55, 0.0],   # shoulder lift
    2: [0.0,  0.90,  0.50, 0.0],   # elbow
    3: [0.0,  0.00,  0.00, 0.0],
    4: [0.0,  0.50, -0.50, 0.0],   # wrist pitch
    5: [0.0,  1.00, -1.00, 0.0],   # wrist roll
    6: [0.0,  0.044, 0.0,  0.022], # gripper: open, close, half
}
MIRROR_SIGN = {0: -1, 4: -1, 5: -1}  # right arm flips these


def smootherstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (x * 6 - 15) + 10)


def interp_dim(times, t_wp, v_wp):
    out = np.empty_like(times)
    for i in range(len(t_wp) - 1):
        t0, t1 = t_wp[i], t_wp[i + 1]
        seg = (times >= t0) & (times <= t1) if i == len(t_wp) - 2 else (times >= t0) & (times < t1)
        u = smootherstep((times[seg] - t0) / (t1 - t0))
        out[seg] = v_wp[i] + (v_wp[i + 1] - v_wp[i]) * u
    out[times < t_wp[0]] = v_wp[0]
    out[times > t_wp[-1]] = v_wp[-1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    T = WAYPOINTS_S[-1]
    n = int(round(T * args.fps)) + 1
    times = np.linspace(0.0, T, n)
    traj = np.zeros((n, 14))
    for d in range(7):
        v = np.array(LEFT[d], dtype=float)
        traj[:, d] = interp_dim(times, WAYPOINTS_S, v)            # left arm
        traj[:, 7 + d] = interp_dim(times, WAYPOINTS_S, MIRROR_SIGN.get(d, 1) * v)  # right arm

    np.save(args.out, traj)
    print(f"saved demo trajectory {traj.shape} ({T:.0f}s @ {args.fps:.0f}fps) -> {args.out}")


if __name__ == "__main__":
    main()
