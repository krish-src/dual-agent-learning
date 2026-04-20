"""
CARLA Reinforcement Learning Environment
=========================================

Wraps CARLA as a Gymnasium environment for training autonomous driving agents.

SENSOR RANKING (by usefulness for RL driving):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Rank │ Sensor             │ Why                                          │ Used here?
 ─────┼────────────────────┼──────────────────────────────────────────────┼──────────
  1   │ RGB Camera (front) │ Sees road, cars, signs, lanes - everything  │ YES (primary)
  2   │ Semantic Seg Camera│ Labels road/car/person directly - pixel      │ YES (observation)
      │                    │ perfect vehicle detection for the agent      │
  3   │ Vehicle Speed      │ Must know speed to control throttle/brake   │ YES (state)
  4   │ Collision Detector │ Tells you when you crashed → reward signal  │ YES (reward)
  5   │ Waypoint Direction │ Which way to go at intersections            │ YES (state)
  6   │ Lane Invasion Det. │ Tells you when crossing lane marks → reward │ YES (reward)
  7   │ Distance from center│ How far from lane center → reward          │ YES (computed)
  8   │ Radar              │ Object velocity ahead - detects approaching │ YES (state)
      │                    │ vehicles with distance and speed             │
  9   │ Depth Camera       │ Distance info, but speed+camera give enough │ NO (redundant)
  10  │ LiDAR              │ 3D distances, great but expensive to process│ NO (overkill
      │                    │ and RGB covers most cases for urban driving │  for starter)
  11  │ IMU                │ Acceleration/gyro - derivable from speed    │ NO (redundant)
  12  │ GNSS               │ GPS coords - too coarse, waypoints better   │ NO (redundant)

OUTPUTS (what the agent controls):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  action[0] = steer  [-1.0 to +1.0]  (steering wheel: left to right)
  action[1] = accel  [-1.0 to +1.0]  (combined pedal: >0 = gas, <0 = brake)

  Using a single accel axis instead of separate throttle/brake because:
  - Prevents the agent from pressing gas+brake simultaneously
  - Random exploration actually produces movement (critical for early SAC learning)
  - Reduces action dimensionality → easier exploration
  - Gear is automatic, handbrake/lights/signals are not relevant for RL
"""

import math
import random
import weakref

import carla
import gymnasium
import numpy as np


# ==============================================================================
# -- Constants -----------------------------------------------------------------
# ==============================================================================

OBS_WIDTH = 84              # NN input resolution (camera captures at this size directly)
OBS_HEIGHT = 84
TARGET_SPEED_KMH = 30.0     # Fallback cruising speed if CARLA returns no limit
MAX_EPISODE_STEPS = 4000    # ~200 seconds at 20fps sync
LANE_WIDTH = 3.5            # Approximate lane width in meters
MIN_EPISODE_REWARD = -500.0 # Kill episode if total reward drops below this


# ==============================================================================
# -- CarlaRLEnv ----------------------------------------------------------------
# ==============================================================================

class CarlaRLEnv(gymnasium.Env):
    """
    Gymnasium environment for training an RL agent to drive in CARLA.

    Observation: Dict with:
      - 'camera': (84, 84, 3) RGB image from front camera
      - 'seg_camera': (84, 84) uint8 single-channel semantic class ID map
          Classes: road=7, vehicle=10, pedestrian=4, traffic_light=6, traffic_sign=12
      - 'state':  (8,) float32 vector:
          [0] speed / speed_limit (normalized, 0-2ish)
          [1] distance from lane center / (lane_width/2) (normalized, -1 to 1)
          [2] angle to next waypoint in degrees / 90 (normalized, -1 to 1)
          [3] 1.0 if at junction, 0.0 otherwise
          [4] nearest vehicle distance from radar (normalized, 0-1; 1=no detection)
          [5] nearest vehicle relative speed from radar (normalized, -1 to 1)
          [6] traffic light state (0.0=none/green, 0.5=yellow, 1.0=red)
          [7] speed limit normalized (speed_limit / 120.0)

    Action: Box(2,) float32:
      [0] steer:  -1.0 (full left) to +1.0 (full right)
      [1] accel:  -1.0 (full brake) to +1.0 (full throttle)
          Positive = gas pedal, negative = brake pedal.
          Single axis prevents simultaneous throttle+brake.
    """

    metadata = {"render_modes": [], "render_fps": 20}

    def __init__(self, host='127.0.0.1', port=2000, render_mode=None,
                 fixed_delta=0.05, num_npc_vehicles=15, num_npc_walkers=5,
                 no_rendering=True):
        super().__init__()

        self.render_mode = render_mode
        self._num_npc_vehicles = num_npc_vehicles
        self._num_npc_walkers = num_npc_walkers
        self._no_rendering = no_rendering

        # -- Action space: [steer, accel] — single pedal axis
        #    accel > 0 = throttle, accel < 0 = brake
        self.action_space = gymnasium.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )

        # -- Observation space: camera image + semantic seg + state vector
        self.observation_space = gymnasium.spaces.Dict({
            'camera': gymnasium.spaces.Box(
                low=0, high=255,
                shape=(OBS_HEIGHT, OBS_WIDTH, 3),
                dtype=np.uint8
            ),
            'seg_camera': gymnasium.spaces.Box(
                low=0, high=12,
                shape=(OBS_HEIGHT, OBS_WIDTH),
                dtype=np.uint8
            ),
            'state': gymnasium.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(8,),
                dtype=np.float32
            ),
        })

        # -- Connect to CARLA
        self.client = carla.Client(host, port)
        self.client.set_timeout(60.0)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.bp_library = self.world.get_blueprint_library()
        self.spawn_points = self.map.get_spawn_points()

        # -- Enable synchronous mode (+ no_rendering for headless speed)
        self._original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = fixed_delta
        settings.no_rendering_mode = self._no_rendering
        self.world.apply_settings(settings)

        self.traffic_manager = self.client.get_trafficmanager()
        self.traffic_manager.set_synchronous_mode(True)

        # -- Actor references (cleaned up each reset)
        self.vehicle = None
        self.camera_sensor = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.seg_camera_sensor = None
        self.radar_sensor = None
        self._actor_list = []

        # -- NPC traffic (spawned each episode for realistic environment)
        self._npc_vehicle_ids = []
        self._npc_walker_ids = []
        self._npc_controller_ids = []

        # -- Sensor data buffers
        self._camera_image = np.zeros((OBS_HEIGHT, OBS_WIDTH, 3), dtype=np.uint8)
        self._seg_image = np.zeros((OBS_HEIGHT, OBS_WIDTH), dtype=np.uint8)
        self._collision_occurred = False
        self._collision_intensity = 0.0
        self._lane_invasion_occurred = False

        # -- Radar data buffer
        self._radar_nearest_dist = 1.0   # normalized: 1.0 = no detection / max range
        self._radar_nearest_speed = 0.0  # normalized relative speed
        self._RADAR_MAX_RANGE = 100.0    # meters

        # -- State tracking
        self._step_count = 0
        self._total_reward = 0.0
        self._episode_count = 0
        self._prev_steer = 0.0
        self._prev_accel = 0.0
        self._standstill_steps = 0
        self._offroad_steps = 0
        self._waypoints = []
        self._current_wp_index = 0
        self._ran_red_light = False
        self._ran_yellow_light = False

        # -- Reward history
        self._reward_history = []

    # ==========================================================================
    # -- Gymnasium interface ---------------------------------------------------
    # ==========================================================================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Destroy previous actors (ego + NPC traffic)
        self._cleanup()

        # Reset state
        self._collision_occurred = False
        self._collision_intensity = 0.0
        self._lane_invasion_occurred = False
        self._step_count = 0
        self._total_reward = 0.0
        self._prev_steer = 0.0
        self._prev_accel = 0.0
        self._standstill_steps = 0
        self._offroad_steps = 0
        self._radar_nearest_dist = 1.0
        self._radar_nearest_speed = 0.0
        self._ran_red_light = False
        self._ran_yellow_light = False

        # Spawn ego vehicle
        vehicle_bp = self.bp_library.filter('vehicle.tesla.model3')[0]
        vehicle_bp.set_attribute('role_name', 'hero')
        spawn_point = random.choice(self.spawn_points)
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self._actor_list.append(self.vehicle)

        # Set up sensors
        self._setup_camera()
        self._setup_seg_camera()
        self._setup_radar()
        self._setup_collision_sensor()
        self._setup_lane_invasion_sensor()

        # Spawn NPC traffic (other cars + pedestrians for realistic driving)
        self._spawn_npc_traffic()

        # Randomize weather each episode for training generalization
        self._randomize_weather()

        # Build waypoint route from current position
        self._build_waypoint_route()

        # Let the world settle
        for _ in range(3):
            self.world.tick()

        self._episode_count += 1

        obs = self._get_observation()
        info = {}
        return obs, info

    def step(self, action):
        self._step_count += 1

        # -- Parse action: [steer, accel]
        #    accel > 0 → throttle, accel < 0 → brake (mutually exclusive)
        steer = float(np.clip(action[0], -1.0, 1.0))
        accel = float(np.clip(action[1], -1.0, 1.0))

        if accel >= 0:
            throttle = accel
            brake = 0.0
        else:
            throttle = 0.0
            brake = -accel

        # -- Apply control to vehicle
        control = carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
        )
        self.vehicle.apply_control(control)

        # -- Tick simulation
        self.world.tick()

        # -- Get observation
        obs = self._get_observation()

        # -- Compute reward
        reward, terminated = self._compute_reward(steer, accel, throttle, brake)

        # -- Truncated if max steps reached
        truncated = self._step_count >= MAX_EPISODE_STEPS

        self._total_reward += reward
        self._prev_steer = steer
        self._prev_accel = accel

        info = {
            'speed_kmh': self._get_speed_kmh(),
            'episode_reward': self._total_reward,
            'step': self._step_count,
        }

        if terminated or truncated:
            self._reward_history.append(self._total_reward)

        return obs, reward, terminated, truncated, info

    def close(self):
        self._cleanup()
        # Restore original settings
        try:
            self.world.apply_settings(self._original_settings)
        except Exception:
            pass

    # ==========================================================================
    # -- Sensor Setup ----------------------------------------------------------
    # ==========================================================================

    def _setup_camera(self):
        cam_bp = self.bp_library.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(OBS_WIDTH))
        cam_bp.set_attribute('image_size_y', str(OBS_HEIGHT))
        cam_bp.set_attribute('fov', '100')
        cam_transform = carla.Transform(
            carla.Location(x=1.5, z=2.4),
            carla.Rotation(pitch=-10)
        )
        self.camera_sensor = self.world.spawn_actor(
            cam_bp, cam_transform, attach_to=self.vehicle
        )
        self._actor_list.append(self.camera_sensor)

        weak_self = weakref.ref(self)
        self.camera_sensor.listen(lambda img: CarlaRLEnv._on_camera(weak_self, img))

    def _setup_seg_camera(self):
        """Semantic segmentation camera — same viewpoint as main RGB camera.

        Produces a single-channel class ID image at observation resolution.
        Classes: road=7, vehicle=10, pedestrian=4, traffic_light=6, traffic_sign=12
        """
        seg_bp = self.bp_library.find('sensor.camera.semantic_segmentation')
        seg_bp.set_attribute('image_size_x', str(OBS_WIDTH))
        seg_bp.set_attribute('image_size_y', str(OBS_HEIGHT))
        seg_bp.set_attribute('fov', '100')
        seg_transform = carla.Transform(
            carla.Location(x=1.5, z=2.4),
            carla.Rotation(pitch=-10)
        )
        self.seg_camera_sensor = self.world.spawn_actor(
            seg_bp, seg_transform, attach_to=self.vehicle
        )
        self._actor_list.append(self.seg_camera_sensor)

        weak_self = weakref.ref(self)
        self.seg_camera_sensor.listen(
            lambda img: CarlaRLEnv._on_seg_camera(weak_self, img))

    def _setup_radar(self):
        """Radar on front of vehicle — detects objects ahead with distance and velocity."""
        radar_bp = self.bp_library.find('sensor.other.radar')
        radar_bp.set_attribute('horizontal_fov', '30')
        radar_bp.set_attribute('vertical_fov', '20')
        radar_bp.set_attribute('range', str(self._RADAR_MAX_RANGE))
        radar_bp.set_attribute('points_per_second', '200')
        radar_transform = carla.Transform(
            carla.Location(x=2.5, z=1.0),
            carla.Rotation(pitch=5)
        )
        self.radar_sensor = self.world.spawn_actor(
            radar_bp, radar_transform, attach_to=self.vehicle
        )
        self._actor_list.append(self.radar_sensor)

        weak_self = weakref.ref(self)
        self.radar_sensor.listen(
            lambda data: CarlaRLEnv._on_radar(weak_self, data))

    def _setup_collision_sensor(self):
        col_bp = self.bp_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(
            col_bp, carla.Transform(), attach_to=self.vehicle
        )
        self._actor_list.append(self.collision_sensor)

        weak_self = weakref.ref(self)
        self.collision_sensor.listen(
            lambda event: CarlaRLEnv._on_collision(weak_self, event)
        )

    def _setup_lane_invasion_sensor(self):
        lane_bp = self.bp_library.find('sensor.other.lane_invasion')
        self.lane_invasion_sensor = self.world.spawn_actor(
            lane_bp, carla.Transform(), attach_to=self.vehicle
        )
        self._actor_list.append(self.lane_invasion_sensor)

        weak_self = weakref.ref(self)
        self.lane_invasion_sensor.listen(
            lambda event: CarlaRLEnv._on_lane_invasion(weak_self, event)
        )

    # ==========================================================================
    # -- Sensor Callbacks ------------------------------------------------------
    # ==========================================================================

    @staticmethod
    def _on_camera(weak_self, image):
        self = weak_self()
        if not self:
            return
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((OBS_HEIGHT, OBS_WIDTH, 4))[:, :, :3]  # BGRA→BGR
        self._camera_image = array[:, :, ::-1].copy()  # BGR→RGB

    @staticmethod
    def _on_seg_camera(weak_self, image):
        """Extract single-channel class ID map from raw semantic segmentation.

        In CARLA's raw semantic seg output (BGRA), the red channel (index 2)
        contains the class tag ID directly. We do NOT apply CityScapes
        conversion here — that's only for display colorization.
        """
        self = weak_self()
        if not self:
            return
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((OBS_HEIGHT, OBS_WIDTH, 4))
        # Red channel (index 2 in BGRA layout) = class tag ID
        self._seg_image = array[:, :, 2].copy()

    @staticmethod
    def _on_radar(weak_self, radar_data):
        """Process radar detections to find nearest vehicle distance and speed."""
        self = weak_self()
        if not self:
            return

        nearest_dist = self._RADAR_MAX_RANGE
        nearest_speed = 0.0

        for detection in radar_data:
            dist = detection.depth  # meters
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_speed = detection.velocity  # m/s, negative = approaching

        # Normalize: distance 0-1 (1 = max range / no detection), speed to -1..1
        self._radar_nearest_dist = min(nearest_dist / self._RADAR_MAX_RANGE, 1.0)
        self._radar_nearest_speed = np.clip(nearest_speed / 30.0, -1.0, 1.0)

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self._collision_occurred = True
        self._collision_intensity = intensity

    @staticmethod
    def _on_lane_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        self._lane_invasion_occurred = True

    # ==========================================================================
    # -- Observation -----------------------------------------------------------
    # ==========================================================================

    def _get_observation(self):
        # Camera: already captured at OBS_WIDTH x OBS_HEIGHT
        camera_obs = self._camera_image.copy()

        # Semantic segmentation: already at observation resolution
        seg_obs = self._seg_image.copy()

        # Dynamic speed limit from CARLA (falls back to TARGET_SPEED_KMH)
        speed_limit = self._get_speed_limit()

        # State vector (8 values)
        speed_norm = self._get_speed_kmh() / speed_limit
        dist_center = self._get_distance_from_center()
        dist_norm = np.clip(dist_center / (LANE_WIDTH / 2), -1.0, 1.0)
        angle_norm = np.clip(self._get_angle_to_waypoint() / 90.0, -1.0, 1.0)
        at_junction = 1.0 if self._is_at_junction() else 0.0
        tl_state = self._get_traffic_light_state()
        speed_limit_norm = speed_limit / 120.0

        state = np.array([
            speed_norm,
            dist_norm,
            angle_norm,
            at_junction,
            self._radar_nearest_dist,
            self._radar_nearest_speed,
            tl_state,
            speed_limit_norm,
        ], dtype=np.float32)

        return {'camera': camera_obs, 'seg_camera': seg_obs, 'state': state}

    def _get_speed_kmh(self):
        vel = self.vehicle.get_velocity()
        return 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

    def _get_distance_from_center(self):
        """Signed distance from lane center. Positive = right of center."""
        vehicle_loc = self.vehicle.get_location()
        wp = self.map.get_waypoint(vehicle_loc, project_to_road=True,
                                   lane_type=carla.LaneType.Driving)
        if wp is None:
            return LANE_WIDTH  # Far off-road

        # Vector from waypoint center to vehicle
        wp_loc = wp.transform.location
        dx = vehicle_loc.x - wp_loc.x
        dy = vehicle_loc.y - wp_loc.y

        # Right vector of the waypoint
        yaw = math.radians(wp.transform.rotation.yaw)
        right_x = math.cos(yaw + math.pi / 2)
        right_y = math.sin(yaw + math.pi / 2)

        return dx * right_x + dy * right_y

    def _get_angle_to_waypoint(self):
        """Angle between vehicle heading and direction to next waypoint (degrees)."""
        if not self._waypoints or self._current_wp_index >= len(self._waypoints):
            return 0.0

        wp = self._waypoints[self._current_wp_index]
        vehicle_t = self.vehicle.get_transform()
        vehicle_loc = vehicle_t.location

        # Direction to waypoint
        dx = wp.transform.location.x - vehicle_loc.x
        dy = wp.transform.location.y - vehicle_loc.y
        target_yaw = math.degrees(math.atan2(dy, dx))

        vehicle_yaw = vehicle_t.rotation.yaw

        # Angle difference, normalized to [-180, 180]
        angle = target_yaw - vehicle_yaw
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360

        # Advance waypoint if close enough
        dist = math.sqrt(dx**2 + dy**2)
        if dist < 5.0 and self._current_wp_index < len(self._waypoints) - 1:
            self._current_wp_index += 1

        return angle

    def _is_at_junction(self):
        vehicle_loc = self.vehicle.get_location()
        wp = self.map.get_waypoint(vehicle_loc, project_to_road=True)
        return wp is not None and wp.is_junction

    def _is_on_road(self):
        vehicle_loc = self.vehicle.get_location()
        wp = self.map.get_waypoint(vehicle_loc, project_to_road=True,
                                   lane_type=carla.LaneType.Driving)
        if wp is None:
            return False
        dist = vehicle_loc.distance(wp.transform.location)
        return dist < (LANE_WIDTH / 2 + 1.0)  # Some tolerance

    def _is_wrong_way(self):
        """Check if vehicle is driving against the lane direction (opposite lane).

        Compares vehicle yaw with the nearest waypoint's yaw. If the angle
        between them exceeds 120°, the car is going the wrong way.
        """
        vehicle_t = self.vehicle.get_transform()
        vehicle_loc = vehicle_t.location
        wp = self.map.get_waypoint(vehicle_loc, project_to_road=True,
                                   lane_type=carla.LaneType.Driving)
        if wp is None:
            return False

        # Angle difference between vehicle heading and lane direction
        diff = vehicle_t.rotation.yaw - wp.transform.rotation.yaw
        # Normalize to [-180, 180]
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360

        return abs(diff) > 120.0

    def _get_speed_limit(self):
        """Get the current posted speed limit from CARLA, with fallback."""
        try:
            limit = self.vehicle.get_speed_limit()
            if limit is not None and limit > 0:
                return float(limit)
        except Exception:
            pass
        return TARGET_SPEED_KMH

    def _get_traffic_light_state(self):
        """Get traffic light state as a normalized float.

        Returns: 0.0 = no light / green, 0.5 = yellow, 1.0 = red
        """
        try:
            if self.vehicle.is_at_traffic_light():
                tl = self.vehicle.get_traffic_light()
                if tl is not None:
                    state = tl.get_state()
                    if state == carla.TrafficLightState.Red:
                        return 1.0
                    elif state == carla.TrafficLightState.Yellow:
                        return 0.5
        except Exception:
            pass
        return 0.0

    # ==========================================================================
    # -- Weather ---------------------------------------------------------------
    # ==========================================================================

    _WEATHER_PRESETS = [
        carla.WeatherParameters.ClearNoon,
        carla.WeatherParameters.CloudyNoon,
        carla.WeatherParameters.WetNoon,
        carla.WeatherParameters.WetCloudyNoon,
        carla.WeatherParameters.SoftRainNoon,
        carla.WeatherParameters.HardRainNoon,
    ]

    def _randomize_weather(self):
        """Set a random weather preset each episode for training variety."""
        weather = random.choice(self._WEATHER_PRESETS)
        self.world.set_weather(weather)

    # ==========================================================================
    # -- NPC Traffic -----------------------------------------------------------
    # ==========================================================================

    def _spawn_npc_traffic(self):
        """Spawn NPC vehicles and walkers to simulate real traffic.

        Wrapped in try/except so a CARLA timeout doesn't crash training —
        an episode with fewer NPCs is better than a crash.
        """
        self._npc_vehicle_ids = []
        self._npc_walker_ids = []
        self._npc_controller_ids = []

        try:
            self._spawn_npc_vehicles()
            self._spawn_npc_walkers()
        except RuntimeError as e:
            print(f'  [WARNING] NPC spawn timeout: {e}')
            print(f'  Continuing with {len(self._npc_vehicle_ids)} vehicles, '
                  f'{len(self._npc_walker_ids)} walkers')

    def _spawn_npc_vehicles(self):
        if self._num_npc_vehicles <= 0:
            return

        vehicle_bps = self.bp_library.filter('vehicle.*')
        vehicle_bps = [x for x in vehicle_bps
                       if int(x.get_attribute('number_of_wheels')) >= 4]

        available_spawns = list(self.spawn_points)
        random.shuffle(available_spawns)

        # Don't spawn on top of the ego vehicle
        ego_loc = self.vehicle.get_location()
        available_spawns = [
            sp for sp in available_spawns
            if sp.location.distance(ego_loc) > 10.0
        ]

        num = min(self._num_npc_vehicles, len(available_spawns))
        batch = []
        for i in range(num):
            bp = random.choice(vehicle_bps)
            if bp.has_attribute('color'):
                bp.set_attribute('color',
                    random.choice(bp.get_attribute('color').recommended_values))
            if bp.has_attribute('driver_id'):
                bp.set_attribute('driver_id',
                    random.choice(bp.get_attribute('driver_id').recommended_values))
            bp.set_attribute('role_name', 'autopilot')
            batch.append(
                carla.command.SpawnActor(bp, available_spawns[i])
                .then(carla.command.SetAutopilot(
                    carla.command.FutureActor, True,
                    self.traffic_manager.get_port()))
            )

        results = self.client.apply_batch_sync(batch, True)
        for r in results:
            if not r.error:
                self._npc_vehicle_ids.append(r.actor_id)

    def _spawn_npc_walkers(self):
        if self._num_npc_walkers <= 0:
            return

        walker_bps = self.bp_library.filter('walker.pedestrian.*')
        controller_bp = self.bp_library.find('controller.ai.walker')

        # Get random sidewalk locations
        spawn_locs = []
        for _ in range(self._num_npc_walkers * 2):  # Try extra in case some fail
            loc = self.world.get_random_location_from_navigation()
            if loc is not None:
                spawn_locs.append(carla.Transform(location=loc))
            if len(spawn_locs) >= self._num_npc_walkers:
                break

        # Spawn walker actors
        batch = []
        for sp in spawn_locs:
            bp = random.choice(walker_bps)
            if bp.has_attribute('is_invincible'):
                bp.set_attribute('is_invincible', 'false')
            batch.append(carla.command.SpawnActor(bp, sp))

        results = self.client.apply_batch_sync(batch, True)
        for r in results:
            if not r.error:
                self._npc_walker_ids.append(r.actor_id)

        # Spawn AI controllers for each walker
        batch = []
        for wid in self._npc_walker_ids:
            batch.append(carla.command.SpawnActor(
                controller_bp, carla.Transform(), wid))
        results = self.client.apply_batch_sync(batch, True)
        for r in results:
            if not r.error:
                self._npc_controller_ids.append(r.actor_id)

        # Start walking
        self.world.tick()
        all_controllers = self.world.get_actors(self._npc_controller_ids)
        for ctrl in all_controllers:
            ctrl.start()
            dest = self.world.get_random_location_from_navigation()
            if dest:
                ctrl.go_to_location(dest)
            ctrl.set_max_speed(1.0 + random.random() * 1.5)

    def _destroy_npc_traffic(self):
        """Destroy all NPC vehicles and walkers."""
        # Stop walker controllers first
        if self._npc_controller_ids:
            controllers = self.world.get_actors(self._npc_controller_ids)
            for ctrl in controllers:
                try:
                    ctrl.stop()
                except Exception:
                    pass
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self._npc_controller_ids])
            self._npc_controller_ids.clear()

        # Destroy walkers
        if self._npc_walker_ids:
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self._npc_walker_ids])
            self._npc_walker_ids.clear()

        # Destroy vehicles
        if self._npc_vehicle_ids:
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self._npc_vehicle_ids])
            self._npc_vehicle_ids.clear()

    def _build_waypoint_route(self):
        """Build a route of waypoints ahead of the vehicle."""
        vehicle_loc = self.vehicle.get_location()
        wp = self.map.get_waypoint(vehicle_loc, project_to_road=True)
        if wp is None:
            self._waypoints = []
            return

        self._waypoints = []
        self._current_wp_index = 0

        current = wp
        for _ in range(200):  # ~200 waypoints ahead
            next_wps = current.next(3.0)  # 3m spacing
            if not next_wps:
                break
            # At junctions, pick a random path
            current = random.choice(next_wps)
            self._waypoints.append(current)

    # ==========================================================================
    # -- Reward Function -------------------------------------------------------
    # ==========================================================================

    def _compute_reward(self, steer, accel, throttle, brake):
        """
        REWARD BREAKDOWN:
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Component            │ Range          │ Purpose
        ─────────────────────┼────────────────┼──────────────────────────────────
        Speed reward         │ +0.0 to +1.0   │ Drive at posted speed limit
        Centering reward     │ +0.0 to +0.4   │ Stay in lane center
        Heading reward       │ -0.4 to +0.4   │ Face the right direction
        ─────────────────────┼────────────────┼──────────────────────────────────
        Collision penalty    │ -300           │ Crashed → episode ends
        Lane invasion        │ -5.0           │ Crossed lane markings
        Off-road penalty     │ -3.0/step      │ Driving on sidewalk/grass
        Off-road terminate   │ -100           │ Off-road too long → ends
        Steer jerk penalty   │ -0.5 * |Δs|    │ Smooth steering
        Accel jerk penalty   │ -0.3 * |Δa|    │ Smooth acceleration
        Wrong-way penalty    │ -5.0/step      │ Driving against traffic
        Red light penalty    │ -50.0          │ Running a red light
        Yellow light penalty │ -10.0          │ Running a yellow light
        Speeding penalty     │ scaled/step    │ Over posted speed limit
        Tailgating penalty   │ -1.0 to -3.0   │ Too close to car ahead
        Standstill penalty   │ -2.0/step      │ Not moving (exempt at red light)
        Standstill terminate │ -100           │ Still 5s (exempt at red light)
        Reward floor         │ -500 total     │ Cut hopeless episodes short
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        Total per-step when driving well: ~+1.4 to +1.8
        A 2000-step episode with good driving: ~3000 total reward
        An episode that crashes at step 50: ~50*1.5 - 300 = -225 reward
        → Agent quickly learns: staying alive & driving > crashing
        """
        reward = 0.0
        terminated = False

        speed_kmh = self._get_speed_kmh()
        speed_limit = self._get_speed_limit()

        # ── 1. SPEED REWARD: main positive signal ──
        # Linear ramp up to posted speed limit, slight penalty above 1.5x limit
        if speed_kmh <= speed_limit:
            speed_reward = speed_kmh / speed_limit  # 0.0 → 1.0
        elif speed_kmh <= speed_limit * 1.5:
            speed_reward = 1.0 - (speed_kmh - speed_limit) / speed_limit
        else:
            speed_reward = -0.5  # Way too fast
        reward += speed_reward

        # ── 2. CENTERING REWARD: stay in lane center ──
        dist_center = abs(self._get_distance_from_center())
        half_lane = LANE_WIDTH / 2
        center_reward = max(0.0, 1.0 - dist_center / half_lane) * 0.4
        reward += center_reward

        # ── 3. HEADING REWARD: face the right direction ──
        angle = self._get_angle_to_waypoint()
        heading_reward = math.cos(math.radians(angle)) * 0.4
        reward += heading_reward

        # ── 4. COLLISION PENALTY ──
        if self._collision_occurred:
            reward -= 300.0
            terminated = True

        # ── 5. LANE INVASION PENALTY ──
        if self._lane_invasion_occurred:
            reward -= 5.0
            self._lane_invasion_occurred = False

        # ── 6. OFF-ROAD PENALTY ──
        if not self._is_on_road():
            reward -= 3.0
            self._offroad_steps += 1
            if self._offroad_steps > 40:  # ~2 seconds off-road
                reward -= 100.0
                terminated = True
        else:
            self._offroad_steps = 0

        # ── 7. STEER SMOOTHNESS ──
        steer_change = abs(steer - self._prev_steer)
        reward -= steer_change * 0.5

        # ── 8. ACCEL SMOOTHNESS ──
        accel_change = abs(accel - self._prev_accel)
        reward -= accel_change * 0.3

        # ── 9. WRONG-WAY PENALTY ──
        if self._is_wrong_way():
            reward -= 5.0

        # ── 10. TRAFFIC LIGHT COMPLIANCE ──
        tl_state = self._get_traffic_light_state()
        self._ran_red_light = False
        self._ran_yellow_light = False
        if tl_state == 1.0 and speed_kmh > 5.0:
            # Running a red light at speed
            reward -= 50.0
            self._ran_red_light = True
        elif tl_state == 0.5 and speed_kmh > 5.0:
            # Running a yellow light at speed
            reward -= 10.0
            self._ran_yellow_light = True

        # ── 11. SPEEDING PENALTY ──
        # Scaled penalty when exceeding 110% of posted speed limit
        if speed_kmh > speed_limit * 1.1:
            excess_kmh = speed_kmh - speed_limit
            reward -= 0.5 * (excess_kmh / speed_limit)

        # ── 12. TAILGATING PENALTY ──
        # Use radar data to penalize following too closely
        radar_dist_m = self._radar_nearest_dist * self._RADAR_MAX_RANGE
        if self._radar_nearest_dist < 1.0 and speed_kmh > 10.0:
            speed_ms = speed_kmh / 3.6
            time_gap = radar_dist_m / speed_ms if speed_ms > 0.1 else 999.0
            if time_gap < 1.0:
                reward -= 3.0  # Dangerous tailgating
            elif time_gap < 2.0:
                reward -= 1.0  # Too close

        # ── 13. STANDSTILL PENALTY ──
        # Exempt when stopped at a red light — that's correct behavior
        at_red_light = (tl_state == 1.0)
        if speed_kmh < 1.0:
            if not at_red_light:
                self._standstill_steps += 1
                reward -= 2.0
                if self._standstill_steps > 100:  # ~5 seconds still → kill
                    reward -= 100.0
                    terminated = True
            else:
                # Stopped at red light — don't penalize, don't accumulate
                self._standstill_steps = 0
        else:
            self._standstill_steps = 0

        # ── 14. TOTAL REWARD FLOOR ──
        if not terminated and (self._total_reward + reward) < MIN_EPISODE_REWARD:
            terminated = True

        return reward, terminated

    # ==========================================================================
    # -- Cleanup ---------------------------------------------------------------
    # ==========================================================================

    def _cleanup(self):
        # Destroy NPC traffic first
        self._destroy_npc_traffic()

        # Destroy ego vehicle and its sensors
        for actor in reversed(self._actor_list):
            try:
                if actor is not None and actor.is_alive:
                    if hasattr(actor, 'stop'):
                        actor.stop()
                    actor.destroy()
            except Exception:
                pass
        self._actor_list.clear()
        self.vehicle = None
        self.camera_sensor = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.seg_camera_sensor = None
        self.radar_sensor = None
