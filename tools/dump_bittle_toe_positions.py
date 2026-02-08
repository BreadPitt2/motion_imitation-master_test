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
    return str(repo_root / "bittle" / "urdf" / "bittle_toes.urdf")


def _link_index_by_name(body: int, link_name: str) -> int | None:
    num = p.getNumJoints(body)
    for i in range(num):
        info = p.getJointInfo(body, i)
        child_link_name = info[12].decode("utf-8")
        if child_link_name == link_name:
            return i
    return None


def _world_pos(body: int, link_index: int) -> tuple[float, float, float]:
    pos = p.getLinkState(body, link_index, computeForwardKinematics=True)[4]
    return float(pos[0]), float(pos[1]), float(pos[2])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=str, default=_default_urdf_path())
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parent / "bittle_toe_positions.txt"),
    )
    parser.add_argument("--base_z", type=float, default=0.8)
    args = parser.parse_args()

    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pd.getDataPath())

    p.loadURDF("plane.urdf")
    robot = p.loadURDF(args.urdf, basePosition=[0, 0, float(args.base_z)], useFixedBase=True)

    pairs = [
        ("toeLF", "left-front-knee-link"),
        ("toeLB", "left-back-knee-link"),
        ("toeRF", "right-front-knee-link"),
        ("toeRB", "right-back-knee-link"),
    ]

    lines: list[str] = []
    lines.append(f"URDF: {args.urdf}")
    lines.append(f"base_z: {float(args.base_z)}")
    lines.append("")

    for toe_name, knee_name in pairs:
        toe_idx = _link_index_by_name(robot, toe_name)
        knee_idx = _link_index_by_name(robot, knee_name)
        lines.append(f"{toe_name} index: {toe_idx}")
        lines.append(f"{knee_name} index: {knee_idx}")
        if toe_idx is None or knee_idx is None:
            lines.append("  ERROR: missing link index (check URDF link names).")
            lines.append("")
            continue

        toe_pos = _world_pos(robot, toe_idx)
        knee_pos = _world_pos(robot, knee_idx)
        delta = (toe_pos[0] - knee_pos[0], toe_pos[1] - knee_pos[1], toe_pos[2] - knee_pos[2])
        lines.append(f"  toe_pos:  {tuple(round(v, 6) for v in toe_pos)}")
        lines.append(f"  knee_pos: {tuple(round(v, 6) for v in knee_pos)}")
        lines.append(f"  delta:    {tuple(round(v, 6) for v in delta)}")
        lines.append("")

    p.disconnect()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

