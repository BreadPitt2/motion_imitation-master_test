from __future__ import annotations

import argparse
import time
from pathlib import Path

try:
    import pybullet as p
    import pybullet_data as pd
except ModuleNotFoundError as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: pybullet. Install it (in your conda env) with:\n"
        "  pip install pybullet"
    ) from e


def _default_urdf_path() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    return str(repo_root / "bittle" / "urdf" / "bittle.urdf")


def _get_screen_size() -> tuple[int, int]:
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        width = int(root.winfo_screenwidth())
        height = int(root.winfo_screenheight())
        root.destroy()
        if width > 0 and height > 0:
            return width, height
    except Exception:
        pass
    return 1280, 720


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=str, default=_default_urdf_path())
    parser.add_argument(
        "--simulate",
        action="store_true",
        default=False,
        help="Enable physics stepping + gravity (default: off / static viewer).",
    )
    parser.add_argument("--base_z", type=float, default=0.8)
    parser.add_argument("--steps", type=int, default=2400, help="How long to simulate (only with --simulate).")
    parser.add_argument("--dt", type=float, default=1.0 / 240.0, help="Wall-clock sleep per step (only with --simulate).")
    parser.add_argument("--gravity", type=float, default=-9.8, help="Gravity z (only with --simulate).")
    parser.add_argument(
        "--label_scale",
        type=float,
        default=0.8,
        help="Text size for debug labels.",
    )
    parser.add_argument(
        "--show_fixed",
        action="store_true",
        default=False,
        help="Also label FIXED joints (default: label only moving joints).",
    )
    parser.add_argument(
        "--only_named",
        action="store_true",
        default=False,
        help="Label only the 8 main Bittle joints (shoulders + knees).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=960,
        help="GUI window width (default: 960).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=540,
        help="GUI window height (default: 540).",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        default=False,
        help="Start with a fullscreen-sized window (may be too big on some setups).",
    )
    args = parser.parse_args()

    options = ""
    width = int(args.width)
    height = int(args.height)
    if args.fullscreen:
        width, height = _get_screen_size()
    options = f"--width={width} --height={height}"

    print(f"Launching PyBullet GUI with options: {options}")
    p.connect(p.GUI, options=options)
    p.setAdditionalSearchPath(pd.getDataPath())
    if args.simulate:
        p.setGravity(0, 0, float(args.gravity))
    else:
        p.setGravity(0, 0, 0)
        p.setRealTimeSimulation(0)

    plane = p.loadURDF("plane.urdf")
    robot = p.loadURDF(
        args.urdf,
        basePosition=[0, 0, float(args.base_z)],
        useFixedBase=(not args.simulate),
    )

    num = p.getNumJoints(robot)
    idx_to_joint = {}
    idx_to_link = {}
    for i in range(num):
        info = p.getJointInfo(robot, i)
        idx_to_joint[i] = info[1].decode("utf-8")
        idx_to_link[i] = info[12].decode("utf-8")

    wanted_joint_names = {
        "left-front-shoulder-joint",
        "left-back-shoulder-joint",
        "right-front-shoulder-joint",
        "right-back-shoulder-joint",
        "left-front-knee-joint",
        "left-back-knee-joint",
        "right-front-knee-joint",
        "right-back-knee-joint",
    }

    # Create labels (and keep ids so we can replace/update them each step).
    label_ids: dict[int, int] = {}
    for i in range(num):
        pos = p.getLinkState(robot, i, computeForwardKinematics=True)[4]
        info = p.getJointInfo(robot, i)
        joint_name = idx_to_joint[i]
        joint_type = int(info[2])
        if (not args.show_fixed) and joint_type == p.JOINT_FIXED:
            continue
        if args.only_named and (joint_name not in wanted_joint_names):
            continue

        txt = f"{i}: {idx_to_link[i]}\n  joint={joint_name}"
        label_ids[i] = p.addUserDebugText(
            txt,
            pos,
            textColorRGB=[0.2, 0.9, 0.2],
            textSize=float(args.label_scale),
        )

    # Label the base as -1.
    base_label_id = p.addUserDebugText(
        "-1: BASE",
        p.getBasePositionAndOrientation(robot)[0],
        textColorRGB=[0.2, 0.6, 1.0],
        textSize=float(args.label_scale),
    )

    print("Close the PyBullet window to stop.")
    if args.simulate:
        print("Watching contacts between robot and plane. Contacting link labels turn RED.")
    else:
        print("Static viewer (no physics): robot will not move/fly away.")

    last_print = 0.0
    start = time.time()
    step = 0
    while p.isConnected():
        if args.simulate and (step >= int(args.steps)):
            break

        if args.simulate:
            p.stepSimulation()

        # Which robot links are touching the ground plane?
        if args.simulate:
            contacts = p.getContactPoints(bodyA=robot, bodyB=plane)
            touched = {c[3] for c in contacts}  # linkIndexA
        else:
            touched = set()

        # Update base label position.
        base_pos = p.getBasePositionAndOrientation(robot)[0]
        p.addUserDebugText(
            "-1: BASE",
            base_pos,
            textColorRGB=[0.2, 0.6, 1.0],
            textSize=float(args.label_scale),
            replaceItemUniqueId=base_label_id,
        )

        # Update each link label position and color.
        for i in list(label_ids.keys()):
            pos = p.getLinkState(robot, i, computeForwardKinematics=True)[4]
            is_touching = i in touched
            color = [1.0, 0.2, 0.2] if is_touching else [0.2, 0.9, 0.2]
            suffix = "  CONTACT" if is_touching else ""
            txt = f"{i}: {idx_to_link[i]}\n  joint={idx_to_joint[i]}{suffix}"
            p.addUserDebugText(
                txt,
                pos,
                textColorRGB=color,
                textSize=float(args.label_scale),
                replaceItemUniqueId=label_ids[i],
            )

        now = time.time()
        if args.simulate and (now - last_print > 1.0):
            touched_sorted = sorted(touched)
            print(f"[t={now - start:5.1f}s] touched link indices: {touched_sorted}")
            if touched_sorted:
                for idx in touched_sorted:
                    name = "BASE" if idx == -1 else idx_to_link.get(idx, "?")
                    jname = "" if idx == -1 else idx_to_joint.get(idx, "")
                    print(f"  {idx:2d} -> link={name} joint={jname}")
            last_print = now

        if args.simulate:
            time.sleep(float(args.dt))
            step += 1
        else:
            time.sleep(0.1)


if __name__ == "__main__":
    main()
