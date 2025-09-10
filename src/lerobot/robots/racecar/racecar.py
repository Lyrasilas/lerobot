import logging
import time
from functools import cached_property
from typing import Any

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
                self.calibration[motor_name] = MotorCalibration(offset=0.0, scale=1.0)
                logger.info(f"Calibrated {motor_name} with default values.")
        self._save_calibration()
        logger.info("Racecar calibration completed.")
        
    def configure(self):
        pass  # No specific configuration needed for the racecar
    
    def get_observation(self):
        pass
    
    def send_action(self, action: dict[str, Any]) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        for motor_name, motor in self.motors.items():
            if motor_name in action:
                goal_pos = ensure_safe_goal_position(
                    motor_name,
                    action[motor_name],
                    self.calibration.get(motor_name),
                    motor.norm_mode,
                )
                motor.set_goal_position(goal_pos)
                logger.debug(f"Set {motor_name} to {goal_pos}")
            else:
                logger.warning(f"Action for {motor_name} not provided.")
        
        # Simulate sending commands to the motors
        time.sleep(0.1)  # Simulate some delay
        logger.info("Actions sent to Racecar motors.")
        
    def disconnect(self):
        self.is_connected = False