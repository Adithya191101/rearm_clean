"""Regenerate config/rebot_grasps_soup_can.yaml.

    python3 -m rebot_b601dm_perception.write_grasps [output_path]

The grasp YAML is generated rather than hand-maintained so that its jaw gaps are
the output of ``jaws.gap_for_command`` instead of numbers someone typed. The
focused authoring test compares it byte-for-byte.

ROS-free: writes a file, imports nothing but this package.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .grasps import author_grasp_set, dump_grasp_yaml

DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / "config" / "rebot_grasps_soup_can.yaml"


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    out = Path(argv[0]) if argv else DEFAULT_OUTPUT
    grasp_set = author_grasp_set()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_grasp_yaml(grasp_set), encoding="utf-8")
    print(f"wrote {len(grasp_set)} grasps to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
