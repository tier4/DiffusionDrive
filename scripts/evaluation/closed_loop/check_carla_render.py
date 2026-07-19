"""Phase-0 gate: verify CARLA 0.9.15 renders on this GPU inside Docker.
Run INSIDE the agent container with the CARLA server container up.
Connects, loads Town01, spawns an RGB camera, captures one frame,
and fails if the frame is missing or all-black."""
import argparse
import queue
import sys

import numpy as np

import carla


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="carla")
    parser.add_argument("--port", type=int, default=2000)
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0)
    world = client.load_world("Town01")
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", "800")
    bp.set_attribute("image_size_y", "600")
    spawn = world.get_map().get_spawn_points()[0]
    cam = world.spawn_actor(bp, carla.Transform(
        spawn.location + carla.Location(z=2.0), spawn.rotation))
    q = queue.Queue()
    cam.listen(q.put)
    try:
        for _ in range(20):
            world.tick()
        image = q.get(timeout=30.0)
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
            image.height, image.width, 4)
        mean = float(arr[:, :, :3].mean())
        print(f"Captured {image.width}x{image.height} frame, mean intensity {mean:.1f}")
        if mean < 1.0:
            print("FAIL: frame is black -- rendering not working on this GPU")
            sys.exit(1)
        print("PHASE-0 PASS: CARLA renders on this GPU")
    finally:
        cam.destroy()


if __name__ == "__main__":
    main()
