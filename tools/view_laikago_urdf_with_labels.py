from __future__ import annotations

import argparse
import math
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
    # Training uses laikago_toes_limits.urdf; it also contains toe links.
    return str(repo_root / "laikago" / "laikago_toes_limits.urdf")


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


def _compute_robot_aabb(robot: int, num_joints: int) -> tuple[list[float], list[float]]:
    aabb_min = [float("inf"), float("inf"), float("inf")]
    aabb_max = [float("-inf"), float("-inf"), float("-inf")]

    def expand(aabb):
        lo, hi = aabb
        for k in range(3):
            aabb_min[k] = min(aabb_min[k], lo[k])
            aabb_max[k] = max(aabb_max[k], hi[k])

    expand(p.getAABB(robot, -1))
    for i in range(num_joints):
        expand(p.getAABB(robot, i))

    return aabb_min, aabb_max


def _raise_robot_above_ground(robot: int, num_joints: int, margin: float) -> None:
    """Shifts base so the robot AABB min-z is at least `margin` above z=0."""
    aabb_min, _ = _compute_robot_aabb(robot, num_joints)
    min_z = float(aabb_min[2])
    if min_z >= margin:
        return
    base_pos, base_orn = p.getBasePositionAndOrientation(robot)
    dz = margin - min_z
    p.resetBasePositionAndOrientation(
        robot, [base_pos[0], base_pos[1], base_pos[2] + dz], base_orn
    )


def _fit_camera_to_robot(robot: int, num_joints: int) -> None:
    aabb_min, aabb_max = _compute_robot_aabb(robot, num_joints)
    center = [(aabb_min[k] + aabb_max[k]) * 0.5 for k in range(3)]
    extent = [aabb_max[k] - aabb_min[k] for k in range(3)]
    dist = max(extent) * 1.8 if max(extent) > 0 else 1.0
    p.resetDebugVisualizerCamera(
        cameraDistance=float(dist),
        cameraYaw=35,
        cameraPitch=-25,
        cameraTargetPosition=center,
    )


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
    parser.add_argument(
        "--identity_orientation",
        action="store_true",
        default=False,
        help="Use identity base orientation (default: use Laikago's standard init orientation).",
    )
    parser.add_argument(
        "--ground_margin",
        type=float,
        default=0.02,
        help="When not simulating, auto-raise robot so it is this far above z=0.",
    )
    parser.add_argument(
        "--no_auto_ground",
        action="store_true",
        default=False,
        help="Disable the auto-raise above ground behavior.",
    )
    parser.add_argument("--steps", type=int, default=2400, help="How long to simulate (only with --simulate).")
    parser.add_argument("--dt", type=float, default=1.0 / 240.0, help="Wall-clock sleep per step (only with --simulate).")
    parser.add_argument("--gravity", type=float, default=-9.8, help="Gravity z (only with --simulate).")
    parser.add_argument(
        "--label_scale",
        type=float,
        default=0.7,
        help="Text size for debug labels.",
    )
    parser.add_argument(
        "--show_fixed",
        action="store_true",
        default=False,
        help="Also label FIXED joints (default: label only moving joints).",
    )
    parser.add_argument(
        "--only_toes",
        action="store_true",
        default=False,
        help="Label only toe links/joints (name contains 'toe' or 'jtoe').",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="",
        help="Only label joints/links whose name contains this substring.",
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
    if args.identity_orientation:
        base_orn = [0, 0, 0, 1]
    else:
        # Match the orientation used by the Laikago robot class in this repo:
        # heading towards -x direction and z axis is up.
        base_orn = p.getQuaternionFromEuler([math.pi / 2.0, 0.0, math.pi / 2.0])
    robot = p.loadURDF(
        args.urdf,
        basePosition=[0, 0, float(args.base_z)],
        baseOrientation=base_orn,
        useFixedBase=(not args.simulate),
    )

    num = p.getNumJoints(robot)
    if (not args.simulate) and (not args.no_auto_ground):
        _raise_robot_above_ground(robot, num, margin=float(args.ground_margin))
    _fit_camera_to_robot(robot, num)

    idx_to_joint = {}
    idx_to_link = {}
    joint_types = {}
    for i in range(num):
        info = p.getJointInfo(robot, i)
        idx_to_joint[i] = info[1].decode("utf-8")
        idx_to_link[i] = info[12].decode("utf-8")
        joint_types[i] = int(info[2])

    name_filter = args.filter.strip().lower()

    label_ids: dict[int, int] = {}
    for i in range(num):
        joint_name = idx_to_joint[i]
        link_name = idx_to_link[i]
        joint_type = joint_types[i]

        if (not args.show_fixed) and joint_type == p.JOINT_FIXED:
            continue

        if args.only_toes:
            key = (joint_name + " " + link_name).lower()
            if ("toe" not in key) and ("jtoe" not in key):
                continue

        if name_filter:
            key = (joint_name + " " + link_name).lower()
            if name_filter not in key:
                continue

        pos = p.getLinkState(robot, i, computeForwardKinematics=True)[4]
        txt = f"{i}: {link_name}\n  joint={joint_name}"
        label_ids[i] = p.addUserDebugText(
            txt,
            pos,
            textColorRGB=[0.2, 0.9, 0.2],
            textSize=float(args.label_scale),
        )

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
            contacts = p.getContactPoints(bodyA=robot, bodyB=plane)
            touched = {c[3] for c in contacts}  # linkIndexA
        else:
            touched = set()

        base_pos = p.getBasePositionAndOrientation(robot)[0]
        p.addUserDebugText(
            "-1: BASE",
            base_pos,
            textColorRGB=[0.2, 0.6, 1.0],
            textSize=float(args.label_scale),
            replaceItemUniqueId=base_label_id,
        )

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
            last_print = now

        if args.simulate:
            time.sleep(float(args.dt))
            step += 1
        else:
            time.sleep(0.1)


if __name__ == "__main__":
    main()
