#!/usr/bin/env python3
"""Capture synchronized RGB and optional semantic data around a CARLA vehicle."""

from __future__ import annotations

import argparse
import json
import math
import queue
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import carla


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "raw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="围绕静止车辆同步采集 RGB 图像和相机元数据。"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--session",
        required=True,
        help="输出目录名称，只允许字母、数字、点、下划线和连字符。",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "vehicle_blueprint",
        "spawn_point_index",
        "radius_m",
        "image_width",
        "image_height",
        "fov_degrees",
        "fixed_delta_seconds",
        "vehicle_settle_frames",
        "camera_warmup_frames",
        "sensor_timeout_seconds",
        "weather",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"配置缺少字段: {', '.join(missing)}")
    uses_single_height = {
        "views",
        "camera_height_above_target_m",
    }.issubset(config)
    uses_multi_height = {
        "views_per_height",
        "camera_height_levels_m",
    }.issubset(config)
    if uses_single_height == uses_multi_height:
        raise ValueError(
            "必须且只能配置一种轨迹：views + camera_height_above_target_m，"
            "或 views_per_height + camera_height_levels_m"
        )
    if uses_single_height and int(config["views"]) < 2:
        raise ValueError("views 必须至少为 2")
    if uses_multi_height:
        levels = config["camera_height_levels_m"]
        if not isinstance(levels, list) or not levels:
            raise ValueError("camera_height_levels_m 必须是非空数组")
        if int(config["views_per_height"]) < 2:
            raise ValueError("views_per_height 必须至少为 2")
        offsets = config.get("angle_offset_per_height_degrees", [0.0] * len(levels))
        if not isinstance(offsets, list) or len(offsets) != len(levels):
            raise ValueError(
                "angle_offset_per_height_degrees 的数量必须与高度层数一致"
            )
        look_offsets = config.get("look_at_z_offsets_m")
        if look_offsets is not None and (
            not isinstance(look_offsets, list) or len(look_offsets) != len(levels)
        ):
            raise ValueError("look_at_z_offsets_m 的数量必须与高度层数一致")
    if float(config["radius_m"]) <= 0:
        raise ValueError("radius_m 必须大于 0")
    if int(config["image_width"]) <= 0 or int(config["image_height"]) <= 0:
        raise ValueError("图像宽高必须大于 0")
    return config


def build_view_plan(config: dict[str, Any]) -> list[dict[str, float | int]]:
    """Expand either trajectory format into one ordered list of camera views."""
    if "camera_height_levels_m" not in config:
        views = int(config["views"])
        height = float(config["camera_height_above_target_m"])
        look_offset = float(config.get("look_at_z_offset_m", 0.0))
        return [
            {
                "height_level_index": 0,
                "camera_height_above_target_m": height,
                "angle_index_within_height": index,
                "relative_orbit_angle_degrees": index * 360.0 / views,
                "look_at_z_offset_m": look_offset,
            }
            for index in range(views)
        ]

    levels = [float(value) for value in config["camera_height_levels_m"]]
    views_per_height = int(config["views_per_height"])
    offsets = [
        float(value)
        for value in config.get(
            "angle_offset_per_height_degrees", [0.0] * len(levels)
        )
    ]
    look_offsets = [
        float(value)
        for value in config.get("look_at_z_offsets_m", [0.0] * len(levels))
    ]
    plan: list[dict[str, float | int]] = []
    for level_index, (height, offset) in enumerate(zip(levels, offsets)):
        for angle_index in range(views_per_height):
            plan.append(
                {
                    "height_level_index": level_index,
                    "camera_height_above_target_m": height,
                    "angle_index_within_height": angle_index,
                    "relative_orbit_angle_degrees": (
                        offset + angle_index * 360.0 / views_per_height
                    )
                    % 360.0,
                    "look_at_z_offset_m": look_offsets[level_index],
                }
            )
    return plan


def location_dict(location: carla.Location) -> dict[str, float]:
    return {"x": location.x, "y": location.y, "z": location.z}


def rotation_dict(rotation: carla.Rotation) -> dict[str, float]:
    return {
        "pitch": rotation.pitch,
        "yaw": rotation.yaw,
        "roll": rotation.roll,
    }


def transform_dict(transform: carla.Transform) -> dict[str, Any]:
    return {
        "location_m": location_dict(transform.location),
        "rotation_degrees": rotation_dict(transform.rotation),
        "camera_to_world_matrix": transform.get_matrix(),
        "world_to_camera_matrix": transform.get_inverse_matrix(),
    }


def make_look_at_transform(
    target: carla.Location,
    radius_m: float,
    camera_height_above_target_m: float,
    orbit_angle_degrees: float,
    look_at_z_offset_m: float = 0.0,
) -> carla.Transform:
    angle = math.radians(orbit_angle_degrees)
    camera_location = carla.Location(
        x=target.x + radius_m * math.cos(angle),
        y=target.y + radius_m * math.sin(angle),
        z=target.z + camera_height_above_target_m,
    )
    dx = target.x - camera_location.x
    dy = target.y - camera_location.y
    dz = target.z + look_at_z_offset_m - camera_location.z
    horizontal = math.hypot(dx, dy)
    rotation = carla.Rotation(
        pitch=math.degrees(math.atan2(dz, horizontal)),
        yaw=math.degrees(math.atan2(dy, dx)),
        roll=0.0,
    )
    return carla.Transform(camera_location, rotation)


def wait_for_image(
    image_queue: queue.Queue[carla.Image], expected_frame: int, timeout: float
) -> carla.Image:
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(f"等待相机帧 {expected_frame} 超时")
        try:
            image = image_queue.get(timeout=remaining)
        except queue.Empty as error:
            raise RuntimeError(f"等待相机帧 {expected_frame} 超时") from error
        if image.frame < expected_frame:
            continue
        if image.frame > expected_frame:
            raise RuntimeError(
                f"相机跳过目标帧 {expected_frame}，收到帧 {image.frame}"
            )
        return image


def camera_intrinsics(width: int, height: int, fov_degrees: float) -> dict[str, Any]:
    focal = width / (2.0 * math.tan(math.radians(fov_degrees) / 2.0))
    cx = width / 2.0
    cy = height / 2.0
    return {
        "model": "PINHOLE",
        "width": width,
        "height": height,
        "horizontal_fov_degrees": fov_degrees,
        "fx": focal,
        "fy": focal,
        "cx": cx,
        "cy": cy,
        "K": [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
        "note": "Derived from CARLA horizontal FOV with square pixels.",
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.session):
        print("session 名称包含不允许的字符。", file=sys.stderr)
        return 2

    try:
        config = load_config(args.config.resolve())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"读取配置失败: {error}", file=sys.stderr)
        return 2

    session_dir = args.output_root.resolve() / args.session
    images_dir = session_dir / "images"
    capture_semantic = bool(config.get("capture_semantic_vehicle_mask", False))
    semantic_dir = session_dir / "semantic_raw"
    if session_dir.exists():
        print(f"输出目录已存在，为避免覆盖已停止: {session_dir}", file=sys.stderr)
        return 2
    images_dir.mkdir(parents=True)
    if capture_semantic:
        semantic_dir.mkdir()

    client = carla.Client(args.host, args.port)
    client.set_timeout(float(config["sensor_timeout_seconds"]))

    world = None
    original_settings = None
    original_weather = None
    vehicle = None
    camera = None
    semantic_camera = None
    frame_records: list[dict[str, Any]] = []
    session_record: dict[str, Any] = {
        "session": args.session,
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_file": str(args.config.resolve()),
        "config": config,
        "coordinate_convention": (
            "CARLA/Unreal world coordinates: x forward, y right, z up; "
            "transform matrices are saved exactly as returned by CARLA."
        ),
    }
    write_json(session_dir / "session.json", session_record)

    try:
        client_version = client.get_client_version()
        server_version = client.get_server_version()
        if client_version != server_version:
            raise RuntimeError(
                f"客户端/服务端版本不一致: {client_version} / {server_version}"
            )

        world = client.get_world()
        requested_map = config.get("map_name")
        if requested_map is not None:
            requested_map_name = str(requested_map).split("/")[-1]
            current_map_name = world.get_map().name.split("/")[-1]
            if current_map_name != requested_map_name:
                print(
                    f"切换地图: {current_map_name} -> {requested_map_name}",
                    flush=True,
                )
                world = client.load_world(requested_map_name)
        original_settings = world.get_settings()
        original_weather = world.get_weather()

        capture_settings = world.get_settings()
        capture_settings.synchronous_mode = True
        capture_settings.fixed_delta_seconds = float(config["fixed_delta_seconds"])
        world.apply_settings(capture_settings)

        weather_name = str(config["weather"])
        if not hasattr(carla.WeatherParameters, weather_name):
            raise RuntimeError(f"未知 CARLA 天气预设: {weather_name}")
        world.set_weather(getattr(carla.WeatherParameters, weather_name))
        world.tick()

        carla_map = world.get_map()
        spawn_points = carla_map.get_spawn_points()
        spawn_index = int(config["spawn_point_index"])
        if not 0 <= spawn_index < len(spawn_points):
            raise RuntimeError(
                f"spawn_point_index={spawn_index} 超出范围 0..{len(spawn_points)-1}"
            )

        vehicle_matches = world.get_blueprint_library().filter(
            str(config["vehicle_blueprint"])
        )
        if not vehicle_matches:
            raise RuntimeError(f"找不到车辆蓝图: {config['vehicle_blueprint']}")
        vehicle_bp = vehicle_matches[0]
        if vehicle_bp.has_attribute("role_name"):
            vehicle_bp.set_attribute("role_name", "gs_capture_target")
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[spawn_index])
        if vehicle is None:
            raise RuntimeError(f"车辆出生点 {spawn_index} 被占用")

        for _ in range(int(config["vehicle_settle_frames"])):
            world.tick()
        vehicle.set_simulate_physics(False)
        world.tick()
        vehicle_transform = vehicle.get_transform()

        target = carla.Location(
            x=vehicle_transform.location.x,
            y=vehicle_transform.location.y,
            z=vehicle_transform.location.z + vehicle.bounding_box.location.z,
        )
        view_plan = build_view_plan(config)

        width = int(config["image_width"])
        height = int(config["image_height"])
        fov = float(config["fov_degrees"])
        intrinsics = camera_intrinsics(width, height, fov)
        write_json(session_dir / "camera_intrinsics.json", intrinsics)

        camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(width))
        camera_bp.set_attribute("image_size_y", str(height))
        camera_bp.set_attribute("fov", str(fov))
        camera_bp.set_attribute("sensor_tick", "0.0")

        first_transform = make_look_at_transform(
            target,
            float(config["radius_m"]),
            float(view_plan[0]["camera_height_above_target_m"]),
            vehicle_transform.rotation.yaw
            + float(view_plan[0]["relative_orbit_angle_degrees"]),
            float(view_plan[0]["look_at_z_offset_m"]),
        )
        camera = world.spawn_actor(camera_bp, first_transform)
        image_queue: queue.Queue[carla.Image] = queue.Queue()
        camera.listen(image_queue.put)

        semantic_queue: queue.Queue[carla.Image] | None = None
        if capture_semantic:
            semantic_bp = world.get_blueprint_library().find(
                "sensor.camera.semantic_segmentation"
            )
            semantic_bp.set_attribute("image_size_x", str(width))
            semantic_bp.set_attribute("image_size_y", str(height))
            semantic_bp.set_attribute("fov", str(fov))
            semantic_bp.set_attribute("sensor_tick", "0.0")
            semantic_camera = world.spawn_actor(semantic_bp, first_transform)
            semantic_queue = queue.Queue()
            semantic_camera.listen(semantic_queue.put)
        world.tick()

        total_views = len(view_plan)
        warmup_frames = int(config["camera_warmup_frames"])
        timeout = float(config["sensor_timeout_seconds"])
        spectator = world.get_spectator()

        for index, planned_view in enumerate(view_plan):
            relative_angle = float(planned_view["relative_orbit_angle_degrees"])
            camera_height = float(planned_view["camera_height_above_target_m"])
            orbit_angle = vehicle_transform.rotation.yaw + relative_angle
            desired_transform = make_look_at_transform(
                target,
                float(config["radius_m"]),
                camera_height,
                orbit_angle,
                float(planned_view["look_at_z_offset_m"]),
            )
            camera.set_transform(desired_transform)
            if semantic_camera is not None:
                semantic_camera.set_transform(desired_transform)
            spectator.set_transform(desired_transform)

            image = None
            semantic_image = None
            for _ in range(warmup_frames):
                expected_frame = world.tick()
                image = wait_for_image(image_queue, expected_frame, timeout)
                if semantic_queue is not None:
                    semantic_image = wait_for_image(
                        semantic_queue, expected_frame, timeout
                    )
            if image is None:
                raise RuntimeError("没有收到 RGB 图像")
            if capture_semantic and semantic_image is None:
                raise RuntimeError("没有收到语义分割图像")

            filename = f"{index:06d}.png"
            image_path = images_dir / filename
            image.save_to_disk(str(image_path))
            if not image_path.is_file() or image_path.stat().st_size == 0:
                raise RuntimeError(f"图像保存失败: {image_path}")
            if semantic_image is not None:
                semantic_path = semantic_dir / filename
                semantic_image.save_to_disk(
                    str(semantic_path), carla.ColorConverter.Raw
                )
                if not semantic_path.is_file() or semantic_path.stat().st_size == 0:
                    raise RuntimeError(f"语义图保存失败: {semantic_path}")

            record = {
                "index": index,
                "filename": filename,
                "height_level_index": int(planned_view["height_level_index"]),
                "camera_height_above_target_m": camera_height,
                "angle_index_within_height": int(
                    planned_view["angle_index_within_height"]
                ),
                "relative_orbit_angle_degrees": relative_angle,
                "world_orbit_angle_degrees": orbit_angle,
                "look_at_z_offset_m": float(planned_view["look_at_z_offset_m"]),
                "frame": image.frame,
                "semantic_frame": (
                    semantic_image.frame if semantic_image is not None else None
                ),
                "timestamp_seconds": image.timestamp,
                "camera_transform": transform_dict(image.transform),
            }
            frame_records.append(record)
            print(
                f"[{index + 1}/{total_views}] {filename} frame={image.frame} "
                f"level={int(planned_view['height_level_index']) + 1} "
                f"height={camera_height:.1f}m angle={relative_angle:.1f}°"
            )

        frames_path = session_dir / "frames.jsonl"
        frames_path.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n"
                for record in frame_records
            ),
            encoding="utf-8",
        )

        session_record.update(
            {
                "status": "complete",
                "client_version": client_version,
                "server_version": server_version,
                "map": carla_map.name,
                "vehicle_id": vehicle.id,
                "vehicle_type": vehicle.type_id,
                "vehicle_transform": transform_dict(vehicle_transform),
                "target_location_m": location_dict(target),
                "captured_images": len(frame_records),
                "captured_semantic_images": (
                    len(frame_records) if capture_semantic else 0
                ),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        write_json(session_dir / "session.json", session_record)
        print(f"采集完成: {session_dir}")
        return 0

    except (RuntimeError, OSError, ValueError) as error:
        session_record.update(
            {
                "status": "failed",
                "error": str(error),
                "captured_images": len(frame_records),
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        write_json(session_dir / "session.json", session_record)
        print(f"采集失败: {error}", file=sys.stderr)
        print(f"失败记录: {session_dir / 'session.json'}", file=sys.stderr)
        return 1

    finally:
        if semantic_camera is not None:
            try:
                semantic_camera.stop()
                semantic_camera.destroy()
            except RuntimeError as error:
                print(f"清理语义相机失败: {error}", file=sys.stderr)
        if camera is not None:
            try:
                camera.stop()
                camera.destroy()
            except RuntimeError as error:
                print(f"清理相机失败: {error}", file=sys.stderr)
        if vehicle is not None:
            try:
                vehicle.destroy()
            except RuntimeError as error:
                print(f"清理车辆失败: {error}", file=sys.stderr)
        if world is not None and original_weather is not None:
            try:
                world.set_weather(original_weather)
            except RuntimeError as error:
                print(f"恢复天气失败: {error}", file=sys.stderr)
        if world is not None and original_settings is not None:
            try:
                world.apply_settings(original_settings)
            except RuntimeError as error:
                print(f"恢复 world settings 失败: {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
