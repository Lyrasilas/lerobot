import logging
import time
from functools import cached_property
from typing import Any
import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_racecar import RacecarConfig   

logger = logging.getLogger(__name__)


class Racecar(Robot):
    """
    The Racecar robot platform, often used for autonomous driving research and education.
    """

    config_class = RacecarConfig
    name = "racecar"

    def __init__(self, config: RacecarConfig):
        super().__init__(config)
        self.config = config
        self.motors = {
            "steering": Motor(1, "generic", MotorNormMode.RANGE_M1_1),
            "throttle": Motor(2, "generic", MotorNormMode.RANGE_0_1),
            "brake": Motor(3, "generic", MotorNormMode.RANGE_0_1),
        }
        self.cameras = make_cameras_from_configs(config.cameras)
        
    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.motors}
    
    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }
        
    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}
    
    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft
    
    @property
    def is_connected(self) -> bool:
        # As the racecar is a simulated robot, we consider it always connected
        return True
    
    @property
    def is_calibrated(self) -> bool:
        return all(motor in self.calibration for motor in self.motors)
    
    def calibrate(self) -> None:
        if self.is_calibrated:
            logger.info("Racecar is already calibrated.")
            return

        logger.info("Calibrating Racecar...")
        for motor_name in self.motors:
            if motor_name not in self.calibration:
                self.calibration[motor_name] = MotorCalibration(
                    id=self.motors[motor_name].id,
                    drive_mode=0,
                    homing_offset=0.0,
                    range_min=-1.0 if self.motors[motor_name].norm_mode == MotorNormMode.RANGE_M1_1 else 0.0,
                    range_max=1.0 if self.motors[motor_name].norm_mode == MotorNormMode.RANGE_M1_1 else 1.0,
                )
                logger.info(f"Calibrated {motor_name} with default values.")
        # self._save_calibration()
        logger.info("Racecar calibration completed.")
        
    def configure(self):
        pass  # No specific configuration needed for the racecar
    
    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Simulate reading motor positions
        start = time.perf_counter()
        obs_dict = {f"{motor}.pos": self.motors[motor].goal_position for motor in self.motors}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Simulate capturing images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict
    
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Command racecar to move to a target configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises
        ------
        DeviceNotConnectedError
            If the racecar is not connected.

        Returns
        -------
        dict
            The action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Extract goal positions from action dict
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # Cap goal position if max_relative_target is set (simulated logic)
        if hasattr(self.config, "max_relative_target") and self.config.max_relative_target is not None:
            present_pos = {motor: motor_obj.goal_position for motor, motor_obj in self.motors.items()}
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # Set goal position for each motor
        for motor_name, val in goal_pos.items():
            self.motors[motor_name].set_goal_position(val)
            logger.debug(f"Set {motor_name} to {val}")

        # Simulate sending commands to the motors
        time.sleep(0.1)
        logger.info("Actions sent to Racecar motors.")

        # Return the actual action sent
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}
    
    def disconnect(self):
        pass  # Optionally add any simulated disconnect logic here
    
    def connect(self, calibrate: bool = True) -> None:
        """Simulated connect method for Racecar."""
        pass