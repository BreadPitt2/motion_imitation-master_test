from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=str, default=_default_urdf_path())
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parent / "bittle_contact_report.txt"),
        help="Output .txt path (default: tools/bittle_contact_report.txt)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=480,
        help="Number of simulation steps to settle (default: 480 ~2s at 240Hz)",
    )
    parser.add_argument(
        "--base_z",
        type=float,
        default=0.25,
        help="Initial base height in meters (default: 0.25)",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        default=False,
        help="Run with PyBullet GUI (default: off)",
    )
    args = parser.parse_args()

    connection_mode = p.GUI if args.gui else p.DIRECT
    p.connect(connection_mode)
    p.setAdditionalSearchPath(pd.getDataPath())
    p.setGravity(0, 0, -9.8)

    plane = p.loadURDF("plane.urdf")
    robot = p.loadURDF(
        args.urdf, basePosition=[0, 0, float(args.base_z)], useFixedBase=False
    )

    num = p.getNumJoints(robot)
    idx_to_joint = {}
    idx_to_link = {}
    for i in range(num):
        info = p.getJointInfo(robot, i)
        idx_to_joint[i] = info[1].decode("utf-8")
        idx_to_link[i] = info[12].decode("utf-8")

    for _ in range(int(args.steps)):
        p.stepSimulation()

    contacts = p.getContactPoints(bodyA=robot, bodyB=plane)
    touched = sorted({c[3] for c in contacts})  # c[3] = linkIndexA

    lines: list[str] = []
    lines.append(f"URDF: {args.urdf}")
    lines.append(f"steps: {int(args.steps)}")
    lines.append(f"base_z: {float(args.base_z)}")
    lines.append("")
    lines.append(f"Touched link indices: {touched}")
    for idx in touched:
        if idx == -1:
            lines.append("  -1 -> BASE")
        else:
            lines.append(
                f"  {idx:2d} -> link={idx_to_link[idx]} joint={idx_to_joint[idx]}"
            )

    p.disconnect()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
