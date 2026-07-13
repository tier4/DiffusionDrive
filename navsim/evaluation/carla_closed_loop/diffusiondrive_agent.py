"""CARLA leaderboard agent for the B2D-trained DiffusionDrive model.

Sensors replicate the dataset collection rig (Bench2Drive tools/data_collect.py:430-692):
3 front cameras (fov 70), LIDAR_TOP at (-0.39, 0, 1.84), IMU/GPS at (-1.4, 0, 0).
Runs inference every tick (20 Hz), like the reference VAD agent.
"""
import datetime
import json
import math
import os
import pathlib

import numpy as np

import carla
from leaderboard.autoagents import autonomous_agent
from scipy.optimize import fsolve

from navsim.common.bench2drive_dataloader import map_carla_command_to_discrete
from navsim.evaluation.carla_closed_loop.input_adapter import build_agent_input
from navsim.evaluation.carla_closed_loop.model_runner import ModelRunner
from navsim.evaluation.carla_closed_loop.route_planner import (
    EARTH_RADIUS_EQUA,
    RoutePlanner,
)
from navsim.evaluation.carla_closed_loop.waypoint_controller import WaypointController

SAVE_PATH = os.environ.get("SAVE_PATH", None)
COMMAND_TEXT = {0: "LEFT", 1: "STRAIGHT", 2: "RIGHT", 3: "UNKNOWN"}
SAVE_EVERY_N_TICKS = 10


def get_entry_point():
    return "DiffusionDriveCloseLoopAgent"


class DiffusionDriveCloseLoopAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        parts = path_to_conf_file.split("+")
        ckpt_candidates = [p for p in parts if p.endswith((".ckpt", ".pth"))]
        if not ckpt_candidates:
            raise ValueError(
                f"No checkpoint (.ckpt/.pth) in agent config string: {path_to_conf_file}"
            )
        self.ckpt_path = ckpt_candidates[0]
        extras = [p for p in parts if p != self.ckpt_path]
        self.save_name = extras[0] if extras else datetime.datetime.now().strftime(
            "%m_%d_%H_%M_%S"
        )

        self.model_runner = ModelRunner(checkpoint_path=self.ckpt_path)
        self.controller = WaypointController()
        self.step = -1
        self._initialized = False

        self.save_path = None
        if SAVE_PATH is not None:
            route_stem = pathlib.Path(os.environ["ROUTES"]).stem
            self.save_path = pathlib.Path(SAVE_PATH) / f"{route_stem}_{self.save_name}"
            (self.save_path / "rgb_front").mkdir(parents=True, exist_ok=False)
            (self.save_path / "meta").mkdir(parents=True, exist_ok=False)

    def _init(self):
        # lat/lon reference solved from first route waypoint -- verbatim method
        # from Bench2DriveZoo/team_code/vad_b2d_agent.py:173-189.
        locx, locy = (
            self._global_plan_world_coord[0][0].location.x,
            self._global_plan_world_coord[0][0].location.y,
        )
        lon, lat = self._global_plan[0][0]["lon"], self._global_plan[0][0]["lat"]

        def equations(vars):
            x, y = vars
            eq1 = (
                lon * math.cos(x * math.pi / 180)
                - (locx * x * 180) / (math.pi * EARTH_RADIUS_EQUA)
                - math.cos(x * math.pi / 180) * y
            )
            eq2 = (
                math.log(math.tan((lat + 90) * math.pi / 360))
                * EARTH_RADIUS_EQUA
                * math.cos(x * math.pi / 180)
                + locy
                - math.cos(x * math.pi / 180)
                * EARTH_RADIUS_EQUA
                * math.log(math.tan((90 + x) * math.pi / 360))
            )
            return [eq1, eq2]

        self.lat_ref, self.lon_ref = fsolve(equations, [0, 0])
        self._route_planner = RoutePlanner(
            4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref
        )
        self._route_planner.set_route(self._global_plan, True)
        self._initialized = True

    def sensors(self):
        # Exact dataset-collection rig: data_collect.py:433-453, 605-616, 626-692.
        return [
            {"type": "sensor.camera.rgb", "x": 0.80, "y": 0.0, "z": 1.60,
             "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
             "width": 1600, "height": 900, "fov": 70, "id": "CAM_FRONT"},
            {"type": "sensor.camera.rgb", "x": 0.27, "y": -0.55, "z": 1.60,
             "roll": 0.0, "pitch": 0.0, "yaw": -55.0,
             "width": 1600, "height": 900, "fov": 70, "id": "CAM_FRONT_LEFT"},
            {"type": "sensor.camera.rgb", "x": 0.27, "y": 0.55, "z": 1.60,
             "roll": 0.0, "pitch": 0.0, "yaw": 55.0,
             "width": 1600, "height": 900, "fov": 70, "id": "CAM_FRONT_RIGHT"},
            {"type": "sensor.lidar.ray_cast", "x": -0.39, "y": 0.0, "z": 1.84,
             "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "range": 85,
             "rotation_frequency": 10, "channels": 64, "points_per_second": 600000,
             "id": "LIDAR_TOP"},
            {"type": "sensor.other.imu", "x": -1.4, "y": 0.0, "z": 0.0,
             "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "sensor_tick": 0.05, "id": "IMU"},
            {"type": "sensor.other.gnss", "x": -1.4, "y": 0.0, "z": 0.0,
             "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "sensor_tick": 0.01, "id": "GPS"},
            {"type": "sensor.speedometer", "reading_frequency": 20, "id": "SPEED"},
        ]

    def run_step(self, input_data, timestamp):
        self.step += 1
        if not self._initialized:
            self._init()

        gps = input_data["GPS"][1][:2]           # [lat, lon]
        imu = input_data["IMU"][1]                # [ax,ay,az,gx,gy,gz,compass]
        speed = float(input_data["SPEED"][1]["speed"])
        compass = float(imu[-1])
        if math.isnan(compass):  # known CARLA quirk on first frames
            compass = 0.0

        pos = self._route_planner.gps_to_location(np.array([gps[0], gps[1]]))
        target_xy, road_option = self._route_planner.run_step(pos)

        agent_input = build_agent_input(
            camera_images={k: input_data[k][1] for k in
                           ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")},
            lidar_points=input_data["LIDAR_TOP"][1],
            speed=speed,
            compass=compass,
            imu_accel_xy=np.asarray(imu[:2], dtype=np.float32),
            road_option_value=int(road_option.value),
            map_xy=pos,
        )
        trajectory = self.model_runner.infer(agent_input)  # [8,3] ego frame

        result = self.controller.control(
            trajectory=trajectory, speed=speed,
            target_map_xy=np.asarray(target_xy, dtype=np.float64),
            ego_map_xy=pos, compass=compass,
        )

        control = carla.VehicleControl()
        control.steer = float(np.clip(result["steer"], -1.0, 1.0))
        control.throttle = float(np.clip(result["throttle"], 0.0, 0.75))
        control.brake = float(np.clip(result["brake"], 0.0, 1.0))

        if self.save_path is not None and self.step % SAVE_EVERY_N_TICKS == 0:
            self._save_frame(input_data, control, trajectory, int(road_option.value), speed)
        return control

    def _save_frame(self, input_data, control, trajectory, road_option_value, speed):
        from PIL import Image
        frame = self.step // SAVE_EVERY_N_TICKS
        rgb = input_data["CAM_FRONT"][1][:, :, 2::-1]
        Image.fromarray(rgb).save(self.save_path / "rgb_front" / f"{frame:05d}.jpg", quality=85)
        command_id = int(map_carla_command_to_discrete(road_option_value))
        meta = {
            "step": self.step,
            "control": {"steer": control.steer, "throttle": control.throttle,
                        "brake": control.brake},
            "speed": speed,
            "command": COMMAND_TEXT.get(command_id, "UNKNOWN"),
            "command_id": command_id,
            # generate_bev_video_ddrive.py expects x-forward / y-LEFT: negate model y.
            "predicted_trajectory": [[float(p[0]), float(-p[1])] for p in trajectory],
            "trajectory_raw_ego_lh": [[float(v) for v in p] for p in trajectory],
        }
        with open(self.save_path / "meta" / f"{frame:05d}.json", "w") as f:
            json.dump(meta, f, indent=1)

    def destroy(self):
        if hasattr(self, "model_runner"):
            del self.model_runner
        import torch
        torch.cuda.empty_cache()
