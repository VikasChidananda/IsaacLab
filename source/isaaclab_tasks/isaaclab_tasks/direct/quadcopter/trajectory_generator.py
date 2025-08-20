import numpy as np
from scipy.interpolate import CubicSpline
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrajectoryGenerator:
    @staticmethod
    def circle(t, radius=2.0, freq=0.5, z_height=1.0):
        """ Circular trajectory """
        x = radius * np.cos(2 * np.pi * freq * t)
        y = radius * np.sin(2 * np.pi * freq * t)
        z = z_height * np.ones_like(t)
        return np.stack([x, y, z], axis=1)  # [N, 3]

    @staticmethod
    def lemniscate(t, scale=3.0, z_height=1.0):
        """ Figure-8 trajectory (Lemniscate of Bernoulli) """
        x = scale * np.cos(t) / (1 + np.sin(t)**2)
        y = scale * np.sin(t) * np.cos(t) / (1 + np.sin(t)**2)
        z = z_height * np.ones_like(t)
        return np.stack([x, y, z], axis=1)
    
    @staticmethod
    def waypoints(t, points):
        """ Interpolate between user-defined waypoints """
        points = np.array(points)  # [M, 3]
        t_waypoints = np.linspace(0, 1, len(points))
        spline = CubicSpline(t_waypoints, points, axis=0)
        return spline(t % 1.0)
    

class TrajectoryManager:
    def __init__(self, cfg, traj_type=0, verbose=True):
        self.cfg = cfg
        self.episode_length_s = self.cfg.episode_length_s
        self.dt = self.cfg.sim.dt

        self.traj_type = traj_type  # 0: circle, 1: lemniscate, 2: waypoints
        self._desired_pos_w = None
        self._generate_trajectory()
        self.verbose = verbose

        print(f"cfg episode_length_s: {self.cfg.episode_length_s}")

    @property
    def desired_pos_w(self):
        return self._desired_pos_w

    def _generate_trajectory(self):
        # generate trajectory based on the selected type

        t = torch.linspace(0, self.cfg.episode_length_s, steps=int(self.cfg.episode_length_s / self.dt))
        if True:
            # print cfgs
            print(f"self.cfg.episode_length_s: {self.cfg.episode_length_s}")
            print(f"Generating trajectory of type {self.traj_type} with {len(t)} points.")

        if self.traj_type == 0:
            trajectory_points = TrajectoryGenerator.circle(t, radius=2.0, freq=0.5, z_height=1.0)
        elif self.traj_type == 1:
            trajectory_points = TrajectoryGenerator.lemniscate(t, scale=3.0, z_height=1.0)
        elif self.traj_type == 2:
            # Example waypoints for a simple trajectory
            waypoints = [[0, 0, 1], [2, 2, 1], [0, 2, 1], [-2, 0, 1], [0, -2, 1]]
            trajectory_points = TrajectoryGenerator.waypoints(t, waypoints)
        
        return torch.tensor(trajectory_points, dtype=torch.float32, device=device)

    def get_next_position(self, step):
        """ Get the next position in the trajectory """
        if self._desired_pos_w is None:
            self._desired_pos_w = self._generate_trajectory()
        
        # Ensure step is within bounds
        print(f"Step requested: {step}, Total steps: {len(self._desired_pos_w)}")
        step = min(step, len(self._desired_pos_w) - 1)
        return self._desired_pos_w[step]