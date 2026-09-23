#!/usr/bin/env python3
"""Connect to CARLA 0.9.16, spawn a stationary vehicle, and show it."""

from __future__ import annotations

import argparse
import math
import select
import sys
import time

import carla


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在正在运行的 CARLA 中生成一辆静止车辆。"
    )
    parser.add_argument("--host", default="127.0.0.1", help="CARLA 服务端地址")
    parser.add_argument("--port", type=int, default=2000, help="CARLA RPC 端口")
    parser.add_argument(
        "--vehicle",
        default="vehicle.tesla.model3",
        help="CARLA 车辆蓝图 ID",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="连接超时秒数")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    vehicle = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(args.timeout)

        client_version = client.get_client_version()
        server_version = client.get_server_version()
        print(f"客户端版本: {client_version}")
        print(f"服务端版本: {server_version}")

        if client_version != server_version:
            raise RuntimeError(
                "客户端与服务端版本不一致，请使用与 CARLA 服务端版本匹配的 Python API。"
            )

        world = client.get_world()
        carla_map = world.get_map()
        print(f"当前地图: {carla_map.name}")

        blueprint_library = world.get_blueprint_library()
        matches = blueprint_library.filter(args.vehicle)
        if not matches:
            raise RuntimeError(f"找不到车辆蓝图: {args.vehicle}")

        blueprint = matches[0]
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "gs_capture_target")

        spawn_points = carla_map.get_spawn_points()
        spawn_index = 126
        if spawn_index >= len(spawn_points):
            raise RuntimeError(f"出生点 {spawn_index} 不存在")
        vehicle = world.try_spawn_actor(blueprint, spawn_points[spawn_index])
        if vehicle is None:
            raise RuntimeError(f"出生点 {spawn_index} 被占用")

        # try_spawn_actor 返回时，服务端可能尚未把新 actor 的真实出生位姿
        # 同步给客户端；此时 get_transform() 会短暂返回 (0, 0, 0)。
        # 等待一个仿真帧后再读取，才能用真实位置计算 spectator 视角。
        world.wait_for_tick(args.timeout)
        transform = vehicle.get_transform()
        forward = transform.get_forward_vector()
        spectator_transform = carla.Transform(
            transform.location - forward * 7.0 + carla.Location(z=3.0),
            carla.Rotation(pitch=-15.0, yaw=transform.rotation.yaw),
        )
        spectator = world.get_spectator()

        # spectator.set_transform 也可能延迟到后续帧才生效。重复设置并读取，
        # 直到服务端报告的位置与目标足够接近，再向用户报告成功。
        deadline = time.monotonic() + args.timeout
        while True:
            spectator.set_transform(spectator_transform)
            world.wait_for_tick(min(1.0, args.timeout))
            applied_transform = spectator.get_transform()
            delta = applied_transform.location - spectator_transform.location
            distance = math.sqrt(delta.x**2 + delta.y**2 + delta.z**2)
            if distance < 0.1:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("观察者视角未能移动到车辆附近")

        print(f"车辆类型: {vehicle.type_id}")
        print(f"车辆 ID: {vehicle.id}")
        print(f"出生点编号: {spawn_index}")
        print(f"车辆位置: {transform.location}")
        print(f"观察者位置: {applied_transform.location}")
        print("车辆已生成。现在查看 CARLA 窗口。")
        print("按 Enter 删除车辆并退出（也可以按 Ctrl+C）...")

        # 在等待用户检查画面期间保持目标 transform。
        while True:
            spectator.set_transform(spectator_transform)
            ready, _, _ = select.select([sys.stdin], [], [], 0.25)
            if ready:
                sys.stdin.readline()
                break
        return 0

    except (KeyboardInterrupt, EOFError):
        print("\n收到退出信号。")
        return 0
    except RuntimeError as error:
        print(f"运行失败: {error}", file=sys.stderr)
        print(
            "请确认 CARLA 已启动、2000 端口正在监听，并且客户端版本为 0.9.16。",
            file=sys.stderr,
        )
        return 1
    finally:
        if vehicle is not None:
            try:
                destroyed = vehicle.destroy()
                if destroyed:
                    print("车辆已删除。")
                else:
                    print("车辆删除请求未成功，请检查 CARLA 中的 actor。", file=sys.stderr)
            except RuntimeError as error:
                print(f"清理车辆时出错: {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
