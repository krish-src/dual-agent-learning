#!/usr/bin/env python3
import carla
import gymnasium as gym
import numpy as np
import math
import random
import weakref
import pygame

from gymnasium import spaces
from stable_baselines3 import PPO


# ==========================
# CONFIG
# ==========================
CARLA_HOST = "localhost"
CARLA_PORT = 2000

DT = 0.05
TARGET_SPEED = 8.0
MAX_LANE_OFFSET = 3.0

CAM_W, CAM_H = 640, 480
CAM_FOV = 90

# Episodes per curriculum stage
CURRICULUM = [
    ("straight", 100),
    ("gentle", 150),
    ("sharp", 200),
    ("intersection", 200),
    ("traffic", 300),
]


# ==========================
# CAMERA
# ==========================
class CameraManager:
    def __init__(self, w, h):
        self.surface = pygame.Surface((w, h))
        self.w, self.h = w, h

    def callback(self, image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((self.h, self.w, 4))[:, :, :3]
        arr = arr[:, :, ::-1]
        pygame.surfarray.blit_array(self.surface, arr.swapaxes(0, 1))


# ==========================
# ENV
# ==========================
class CarlaCurriculumEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self.client = carla.Client(CARLA_HOST, CARLA_PORT)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.map = self.world.get_map()

        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DT
        self.world.apply_settings(settings)

        self.bp_lib = self.world.get_blueprint_library()
        self.actors = []

        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )

        self.observation_space = spaces.Box(
            low=-50,
            high=50,
            shape=(4,),
            dtype=np.float32,
        )

        pygame.init()
        self.display = pygame.display.set_mode((CAM_W, CAM_H))
        pygame.display.set_caption("CARLA RL Curriculum Training")
        self.font = pygame.font.SysFont("monospace", 14)
        self.clock = pygame.time.Clock()

        self.cam_mgr = CameraManager(CAM_W, CAM_H)

        self.episode_count = 0
        self.prev_speed = 0.0

    # --------------------------
    # Curriculum logic
    # --------------------------
    def _current_stage(self):
        ep = self.episode_count
        count = 0
        for stage, length in CURRICULUM:
            count += length
            if ep < count:
                return stage
        return CURRICULUM[-1][0]

    def _curvature(self, wp):
        nxt = wp.next(5.0)
        if not nxt:
            return 0.0
        return abs(wp.transform.rotation.yaw - nxt[0].transform.rotation.yaw)

    def _filtered_spawn_points(self, stage):
        candidates = []
        for sp in self.map.get_spawn_points():
            wp = self.map.get_waypoint(sp.location, project_to_road=True)
            if not wp:
                continue

            curv = self._curvature(wp)

            if stage == "straight" and curv < 2:
                candidates.append(sp)
            elif stage == "gentle" and 2 <= curv < 8:
                candidates.append(sp)
            elif stage == "sharp" and curv >= 8:
                candidates.append(sp)
            elif stage == "intersection" and wp.is_junction:
                candidates.append(sp)
            elif stage == "traffic":
                candidates.append(sp)

        return candidates if candidates else self.map.get_spawn_points()

    # --------------------------
    def _spawn_vehicle(self, stage):
        bp = self.bp_lib.find("vehicle.tesla.model3")
        spawn_points = self._filtered_spawn_points(stage)
        spawn = random.choice(spawn_points)
        self.vehicle = self.world.spawn_actor(bp, spawn)
        self.actors.append(self.vehicle)

        cam_bp = self.bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(CAM_W))
        cam_bp.set_attribute("image_size_y", str(CAM_H))
        cam_bp.set_attribute("fov", str(CAM_FOV))

        cam_tf = carla.Transform(
            carla.Location(x=-6, z=3),
            carla.Rotation(pitch=-15),
        )

        cam = self.world.spawn_actor(cam_bp, cam_tf, attach_to=self.vehicle)
        self.actors.append(cam)

        weak = weakref.ref(self.cam_mgr)
        cam.listen(lambda img: weak().callback(img))

        # Traffic only in final stage
        if stage == "traffic":
            tm = self.client.get_trafficmanager()
            tm.set_synchronous_mode(True)
            for _ in range(15):
                v = self.world.spawn_actor(
                    random.choice(self.bp_lib.filter("vehicle.*")),
                    random.choice(self.map.get_spawn_points()),
                )
                v.set_autopilot(True, tm.get_port())
                self.actors.append(v)

    # --------------------------
    def _get_obs(self):
        vel = self.vehicle.get_velocity()
        speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

        tf = self.vehicle.get_transform()
        wp = self.map.get_waypoint(tf.location, project_to_road=True)

        lane_offset = tf.location.distance(wp.transform.location)
        yaw_error = (tf.rotation.yaw - wp.transform.rotation.yaw + 180) % 360 - 180

        accel = (speed - self.prev_speed) / DT
        self.prev_speed = speed

        return np.array([speed, lane_offset, yaw_error, accel], dtype=np.float32)

    # --------------------------
    def reset(self, seed=None, options=None):
        for a in self.actors:
            try: a.destroy()
            except: pass
        self.actors.clear()

        stage = self._current_stage()
        print(f"[CURRICULUM] Episode {self.episode_count} → Stage: {stage}")

        self._spawn_vehicle(stage)
        self.prev_speed = 0.0
        self.world.tick()

        self.episode_count += 1
        return self._get_obs(), {}

    # --------------------------
    def step(self, action):
        steer, throttle, brake = action

        self.vehicle.apply_control(
            carla.VehicleControl(
                steer=float(steer),
                throttle=float(throttle),
                brake=float(brake),
            )
        )

        self.world.tick()
        obs = self._get_obs()
        speed, lane_offset, yaw_error, accel = obs

        reward = (
            +1.0
            - 0.6 * lane_offset
            - 0.05 * abs(yaw_error)
            - 0.05 * abs(speed - TARGET_SPEED)
            - 0.02 * abs(steer)
        )

        done = lane_offset > MAX_LANE_OFFSET

        self.display.blit(self.cam_mgr.surface, (0, 0))
        txt = self.font.render(
            f"Stage: {self._current_stage()} | Reward: {reward:.2f}",
            True,
            (255, 255, 255),
        )
        self.display.blit(txt, (10, 10))
        pygame.display.flip()
        self.clock.tick(20)

        return obs, reward, done, False, {}

    # --------------------------
    def close(self):
        for a in self.actors:
            try: a.destroy()
            except: pass
        self.world.apply_settings(self.original_settings)
        pygame.quit()


# ==========================
# TRAIN
# ==========================
if __name__ == "__main__":
    env = CarlaCurriculumEnv()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=128,
        learning_rate=3e-4,
        gamma=0.99,
    )

    model.learn(total_timesteps=600_000)
    model.save("ppo_carla_curriculum")

    env.close()
