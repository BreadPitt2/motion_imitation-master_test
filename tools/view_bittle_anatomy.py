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
    parser = argparse.ArgumentParser(
        description="Static (no-physics) Bittle URDF viewer with joint/link labels."
    )
    parser.add_argument("--urdf", type=str, default=_default_urdf_path())
    parser.add_argument("--base_x", type=float, default=0.0)
    parser.add_argument("--base_y", type=float, default=0.0)
    parser.add_argument("--base_z", type=float, default=0.0)
    parser.add_argument(
        "--label_scale",
        type=float,
        default=1.2,
        help="Text size for debug labels.",
    )
    parser.add_argument(
        "--windowed",
        action="store_true",
        default=False,
        help="Do not try to start fullscreen-sized window.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="GUI window width (0 = auto).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=0,
        help="GUI window height (0 = auto).",
    )
    parser.add_argument(
        "--show_fixed",
        action="store_true",
        default=False,
        help="Also label FIXED joints (default: label only moving joints).",
    )
    parser.add_argument(
        "--print_map",
        action="store_true",
        default=False,
        help="Print index -> joint/link names to the terminal.",
    )
    args = parser.parse_args()

    options = ""
    if not args.windowed:
        width = int(args.width)
        height = int(args.height)
        if width <= 0 or height <= 0:
            width, height = _get_screen_size()
        options = f"--width={width} --height={height}"

    p.connect(p.GUI, options=options)
    p.setAdditionalSearchPath(pd.getDataPath())
    p.resetSimulation()

    # No physics: no gravity, no stepping.
    p.setGravity(0, 0, 0)
    p.setRealTimeSimulation(0)

    # Optional: draw a ground plane for visual reference only.
    p.loadURDF("plane.urdf")

    base_pos = [float(args.base_x), float(args.base_y), float(args.base_z)]
    base_orn = [0, 0, 0, 1]
    robot = p.loadURDF(args.urdf, basePosition=base_pos, baseOrientation=base_orn, useFixedBase=True)

    num = p.getNumJoints(robot)

    if args.print_map:
        print("URDF:", args.urdf)
        print("num_joints:", num)
        print("index -> joint_name | child_link_name | joint_type")

    # Base label (-1 in PyBullet).
    p.addUserDebugText(
        "-1: BASE",
        base_pos,
        textColorRGB=[0.2, 0.6, 1.0],
        textSize=float(args.label_scale),
    )

    joint_type_name = {
        p.JOINT_REVOLUTE: "REVOLUTE",
        p.JOINT_PRISMATIC: "PRISMATIC",
        p.JOINT_SPHERICAL: "SPHERICAL",
        p.JOINT_PLANAR: "PLANAR",
        p.JOINT_FIXED: "FIXED",
    }

    for i in range(num):
        info = p.getJointInfo(robot, i)
        joint_type = int(info[2])
        if (not args.show_fixed) and joint_type == p.JOINT_FIXED:
            continue

        joint_name = info[1].decode("utf-8")
        link_name = info[12].decode("utf-8")
        type_str = joint_type_name.get(joint_type, str(joint_type))

        # Link index == joint index for the child link in PyBullet.
        pos = p.getLinkState(robot, i, computeForwardKinematics=True)[4]
        txt = f"{i}: {link_name}\n  joint={joint_name}\n  type={type_str}"
        p.addUserDebugText(
            txt,
            pos,
            textColorRGB=[0.2, 0.9, 0.2],
            textSize=float(args.label_scale),
        )

        if args.print_map:
            print(f"{i:2d} -> {joint_name} | {link_name} | {type_str}")

    print("Static viewer loaded. Close the PyBullet window when done.")
    while p.isConnected():
        time.sleep(0.1)


if __name__ == "__main__":
    main()
