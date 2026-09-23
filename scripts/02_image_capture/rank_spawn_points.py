#!/usr/bin/env python3
"""Rank CARLA map spawn points by approximate clearance from static obstacles."""

from __future__ import annotations

import argparse
import math

import carla


LABEL_NAMES = (
    "Buildings",
    "Poles",
    "Walls",
    "Fences",
    "TrafficSigns",
    "TrafficLight",
    "GuardRail",
    "Static",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="按周围已知静态物体的近似水平净空对出生点排序。"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    spawn_points = world.get_map().get_spawn_points()

    obstacles: list[tuple[str, float, float, float]] = []
    for label_name in LABEL_NAMES:
        label = getattr(carla.CityObjectLabel, label_name)
        for box in world.get_level_bbs(label):
            # 用水平外接圆近似旋转包围盒，评分偏保守但便于快速筛选。
            horizontal_radius = math.hypot(box.extent.x, box.extent.y)
            obstacles.append(
                (
                    label_name,
                    box.location.x,
                    box.location.y,
                    horizontal_radius,
                )
            )

    ranked = []
    for index, transform in enumerate(spawn_points):
        x = transform.location.x
        y = transform.location.y
        clearance, nearest_label = min(
            (
                math.hypot(x - ox, y - oy) - obstacle_radius,
                label_name,
            )
            for label_name, ox, oy, obstacle_radius in obstacles
        )
        ranked.append(
            (
                clearance,
                index,
                nearest_label,
                x,
                y,
                transform.rotation.yaw,
            )
        )

    print(f"地图: {world.get_map().name}，出生点: {len(spawn_points)}")
    print("净空是静态包围盒近似值，必须再用实际相机图像确认。")
    print("clearance  index  nearest       x        y      yaw")
    for clearance, index, label, x, y, yaw in sorted(ranked, reverse=True)[
        : args.top
    ]:
        print(
            f"{clearance:9.2f}  {index:5d}  {label:12s} "
            f"{x:8.2f} {y:8.2f} {yaw:8.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
