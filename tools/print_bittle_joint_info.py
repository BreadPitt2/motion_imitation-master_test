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


JOINT_TYPE = {
    p.JOINT_REVOLUTE: "REVOLUTE",
    p.JOINT_PRISMATIC: "PRISMATIC",
    p.JOINT_SPHERICAL: "SPHERICAL",
    p.JOINT_PLANAR: "PLANAR",
    p.JOINT_FIXED: "FIXED",
}


def _default_urdf_path() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    return str(repo_root / "bittle" / "urdf" / "bittle.urdf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=str, default=_default_urdf_path())
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parent / "bittle_joint_info.txt"),
        help="Output .txt path (default: tools/bittle_joint_info.txt)",
    )
    args = parser.parse_args()

    lines: list[str] = []

    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pd.getDataPath())

    robot = p.loadURDF(args.urdf, useFixedBase=True)
    num = p.getNumJoints(robot)

    lines.append(f"URDF: {args.urdf}")
    lines.append(f"num_joints: {num}")
    lines.append("")

    child_link_name = {}
    joint_name = {}
    for i in range(num):
        info = p.getJointInfo(robot, i)
        joint_name[i] = info[1].decode("utf-8")
        child_link_name[i] = info[12].decode("utf-8")

    for i in range(num):
        info = p.getJointInfo(robot, i)
        jname = info[1].decode("utf-8")
        lname = info[12].decode("utf-8")
        jtype = JOINT_TYPE.get(info[2], str(info[2]))
        q_index, u_index = info[3], info[4]
        parent = info[16]
        parent_lname = "BASE" if parent == -1 else child_link_name[parent]
        axis = info[13]
        ll, ul = info[8], info[9]
        lines.append(
            f"[{i:2d}] {jtype:8s} q={q_index:2d} u={u_index:2d} parent={parent:2d}({parent_lname}) "
            f"child={lname:28s} joint={jname:30s} axis={tuple(axis)} lim=({ll:.4f},{ul:.4f})"
        )

    lines.append("")
    lines.append("--- Named joints (index in PyBullet) ---")
    wanted = [
        "left-front-shoulder-joint",
        "left-back-shoulder-joint",
        "right-front-shoulder-joint",
        "right-back-shoulder-joint",
        "left-front-knee-joint",
        "left-back-knee-joint",
        "right-front-knee-joint",
        "right-back-knee-joint",
    ]
    name_to_idx = {joint_name[i]: i for i in range(num)}
    for n in wanted:
        lines.append(f"{n:28s} -> {name_to_idx.get(n)}")

    lines.append("")
    lines.append("--- Candidate arrays in retarget order [LF, LB, RF, RB] ---")
    shoulders = [
        name_to_idx.get("left-front-shoulder-joint"),
        name_to_idx.get("left-back-shoulder-joint"),
        name_to_idx.get("right-front-shoulder-joint"),
        name_to_idx.get("right-back-shoulder-joint"),
    ]
    knees = [
        name_to_idx.get("left-front-knee-joint"),
        name_to_idx.get("left-back-knee-joint"),
        name_to_idx.get("right-front-knee-joint"),
        name_to_idx.get("right-back-knee-joint"),
    ]
    lines.append(f"SIM_HIP_JOINT_IDS candidates (shoulder link indices): {shoulders}")
    lines.append(f"SIM_TOE_JOINT_IDS candidates (knee/end-eff link indices): {knees}")

    lines.append("")
    lines.append("--- World positions (sanity check) ---")
    for label, idxs in [("shoulder", shoulders), ("knee", knees)]:
        for idx in idxs:
            if idx is None:
                continue
            ls = p.getLinkState(robot, idx, computeForwardKinematics=True)
            pos = ls[4]
            lines.append(
                f"{label:8s} linkIndex={idx:2d} linkName={child_link_name[idx]:28s} "
                f"world_pos={tuple(round(x, 4) for x in pos)}"
            )

    p.disconnect()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
