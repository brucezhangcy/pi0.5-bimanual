"""Tier-2 mechanical safety check: play a [T,14] action chunk through the Trossen
stationary_ai MuJoCo model and (a) flag joint-limit / velocity violations,
(b) render an mp4 so a human can eyeball the motion before any real arm moves.

IMPORTANT: this validates *mechanics* (limits, jumps, gross pose), NOT policy
behavior. The chunk's images were real-OOD if it came from the dummy dry-run, so
the motion itself is only meaningful when the chunk was produced from real frames
(see replay_overlay.py). Use this to catch "would this command exceed a joint
limit or jump too fast" -- the things that hurt the hardware.

Action dim order (matches the LeRobot dataset and the sim actuators):
    [L_j0 L_j1 L_j2 L_j3 L_j4 L_j5 L_grip  R_j0 R_j1 R_j2 R_j3 R_j4 R_j5 R_grip]
Arm joints are radians; gripper dims (6, 13) are the carriage slide in METERS
[0, 0.044]. If your dataset stores the gripper in other units, dims 6/13 will
trip the range check -- that's the harness doing its job, not a bug.

Run in the mujoco env (conda `trossen_sim`, mujoco 3.x):
    MUJOCO_GL=egl /home/bruce/miniconda3/envs/trossen_sim/bin/python \
        /home/bruce/pi0.5-bimanual/eval_tools/sim_playback.py \
        --chunk /home/bruce/pi0.5-bimanual/artifacts/sample_action_chunk.npy \
        --out   /home/bruce/pi0.5-bimanual/artifacts/sim_playback.mp4
"""
import argparse
import os
import numpy as np

SCENE = "/home/bruce/Aloha_real/trossen_arm_mujoco/trossen_arm_mujoco/assets/stationary_ai/scene_joint.xml"
DIM_NAMES = [
    "L_j0", "L_j1", "L_j2", "L_j3", "L_j4", "L_j5", "L_grip",
    "R_j0", "R_j1", "R_j2", "R_j3", "R_j4", "R_j5", "R_grip",
]


def safety_report(actions, ctrlrange, fps, max_arm_vel, max_grip_vel):
    """Numeric, mujoco-independent. Returns (ok, lines)."""
    T = len(actions)
    lines = []
    ok = True

    # 1) absolute limits (from the sim actuator ctrlrange == joint range)
    lo, hi = ctrlrange[:, 0], ctrlrange[:, 1]
    below = actions < lo - 1e-6
    above = actions > hi + 1e-6
    for d in range(14):
        n = int(below[:, d].sum() + above[:, d].sum())
        if n:
            ok = False
            amin, amax = actions[:, d].min(), actions[:, d].max()
            lines.append(
                f"  [LIMIT] {DIM_NAMES[d]:6s} {n:3d}/{T} frames out of "
                f"[{lo[d]:.3f},{hi[d]:.3f}]  (chunk range [{amin:.3f},{amax:.3f}])"
            )
    if ok:
        lines.append("  [LIMIT] all 14 dims within joint ranges for all frames  OK")

    # 2) per-step velocity (jump) check
    dt = 1.0 / fps
    dq = np.abs(np.diff(actions, axis=0))  # [T-1,14]
    is_grip = np.zeros(14, dtype=bool)
    is_grip[[6, 13]] = True
    vel_cap = np.where(is_grip, max_grip_vel, max_arm_vel) * dt  # max delta per frame
    viol = dq > vel_cap + 1e-9
    vel_ok = True
    for d in range(14):
        n = int(viol[:, d].sum())
        if n:
            vel_ok = False
            ok = False
            worst = dq[:, d].max() * fps  # back to per-second
            lines.append(
                f"  [VEL]   {DIM_NAMES[d]:6s} {n:3d}/{T-1} steps exceed "
                f"{'grip' if is_grip[d] else 'arm'} cap; worst {worst:.2f} "
                f"{'m' if is_grip[d] else 'rad'}/s"
            )
    if vel_ok:
        lines.append(
            f"  [VEL]   all steps within {max_arm_vel:.2f} rad/s (arm) / "
            f"{max_grip_vel:.3f} m/s (grip) @ {fps:.0f}fps  OK"
        )
    return ok, lines


def render(actions, out_path, fps, substeps, width, height, camera):
    import mujoco
    import imageio.v2 as imageio

    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    if model.nu != 14:
        raise RuntimeError(f"expected 14 actuators, model has {model.nu}")

    renderer = mujoco.Renderer(model, height=height, width=width)
    if camera is None:
        # Well-framed free camera looking at the table (matches scene visual hint).
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.0, 0.0, 0.35]
        cam.distance = 1.6
        cam.azimuth = 90
        cam.elevation = -25
        cam_arg = {"camera": cam}
    else:
        cam_arg = {"camera": camera}  # e.g. "cam_high" -> roughly the policy's top view

    frames = []
    mujoco.mj_resetData(model, data)
    for t in range(len(actions)):
        data.ctrl[:14] = actions[t]
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        renderer.update_scene(data, **cam_arg)
        frames.append(renderer.render())
    imageio.mimsave(out_path, frames, fps=int(fps))
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", required=True, help="[T,14] .npy action chunk")
    ap.add_argument("--out", default=None, help="mp4 output path (omit to skip rendering)")
    ap.add_argument("--fps", type=float, default=30.0, help="dataset control rate")
    ap.add_argument("--max-arm-vel", type=float, default=3.14, help="rad/s heuristic cap")
    ap.add_argument("--max-grip-vel", type=float, default=0.10, help="m/s heuristic cap")
    ap.add_argument("--substeps", type=int, default=16, help="sim steps per control frame")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--camera", default=None, help="named mujoco camera (default: free cam)")
    args = ap.parse_args()

    actions = np.load(args.chunk).astype(np.float64)
    assert actions.ndim == 2 and actions.shape[1] == 14, f"expected [T,14], got {actions.shape}"
    print(f"loaded chunk {actions.shape} from {args.chunk}")

    # Pull the authoritative limits straight from the sim model.
    import mujoco
    model = mujoco.MjModel.from_xml_path(SCENE)
    ctrlrange = np.array(model.actuator_ctrlrange)  # [14,2]

    print("\n=== Tier-2 mechanical safety report ===")
    ok, lines = safety_report(actions, ctrlrange, args.fps, args.max_arm_vel, args.max_grip_vel)
    print("\n".join(lines))
    print("=== VERDICT:", "PASS (mechanically safe to visualize/replay)" if ok
          else "FAIL (see flags above; do NOT send to hardware as-is)", "===\n")

    if args.out:
        print(f"rendering {len(actions)} frames -> {args.out} ...")
        render(actions, args.out, args.fps, args.substeps, args.width, args.height, args.camera)
        print(f"wrote {args.out}")

    raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    main()
