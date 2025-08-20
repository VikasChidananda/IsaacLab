# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to simulate a quadcopter.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/quadcopter.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to simulate a quadcopter.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort:skip

def rotation_matrix_from_tilt(angle):
    # Tilt around X axis
    return torch.tensor([
        [1, 0, 0],
        [0, torch.cos(angle), -torch.sin(angle)],
        [0, torch.sin(angle), torch.cos(angle)]
    ])

def get_tilted_thrust(thrust_magnitude, tilt_angle, rotor_direction):
    # rotor_direction: unit vector in default thrust direction (e.g., [0, 0, 1])
    # tilt_angle: angle in radians
    # Rotate around X or Y axis
    R = rotation_matrix_from_tilt(tilt_angle)
    return thrust_magnitude * (R @ rotor_direction)

theta_dist = torch.distributions.Uniform(-torch.pi/6, torch.pi/6)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[0.5, 0.5, 1.0], target=[0.0, 0.0, 0.5])

    # Spawn things into stage
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Robots
    robot_cfg = CRAZYFLIE_CFG.replace(prim_path="/World/Crazyflie")
    robot_cfg.spawn.func("/World/Crazyflie", robot_cfg.spawn, translation=robot_cfg.init_state.pos)


    # thursts
    thrust_magnitude = torch.tensor([0.1])
    rotor_direction = torch.tensor([0.,0.,1.])
    # create handles for the robots
    robot = Articulation(robot_cfg)

    # Play the simulator
    sim.reset()

    # Fetch relevant parameters to make the quadcopter hover in place
    prop_body_ids = robot.find_bodies("m.*_prop")[0]
    robot_mass = robot.root_physx_view.get_masses().sum()
    gravity = torch.tensor(sim.cfg.gravity, device=sim.device).norm()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    # Simulate physics
    theta = theta_dist.sample()
    while simulation_app.is_running():
        # reset
        if count % 200 == 0:
            print(f"tilt angle : {theta}")
            # reset counters
            theta = theta_dist.sample()
            sim_time = 0.0
            count = 0
            # reset dof state
            joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])
            robot.write_root_velocity_to_sim(robot.data.default_root_state[:, 7:])
            robot.reset()
            # reset command
            print(">>>>>>>> Reset!")
        # apply action to the robot (make the robot float in place)
        
        # print(theta)
        forces = torch.zeros(robot.num_instances, 4, 3, device=sim.device)
        tilt_angles = torch.ones(robot.num_instances, 4, device=sim.device) * theta
        
        thrust = get_tilted_thrust(thrust_magnitude, theta, rotor_direction).to(sim.device)

        torques = torch.zeros_like(forces)
        torques[..., 1] += torch.tensor(0.5, device=sim.device)

        # forces[..., :] = thrust
        forces[..., 2] += robot_mass * gravity / 4.0
        robot.set_external_force_and_torque(forces, torques, body_ids=prop_body_ids)
        robot.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        robot.update(sim_dt)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
