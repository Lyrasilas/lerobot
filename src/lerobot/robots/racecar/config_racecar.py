from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig

@RobotConfig.register_subclass("racecar")
@dataclass
class RacecarConfig(RobotConfig):
    # Port to connect to the racecar (if applicable)
    port: str | None = None # Racecar is often simulated, so port can be None
    
    disable_torque_on_disconnect: bool = True
    
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your racecar.
    max_relative_target: int | None = None
    
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    
    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False
    
    track_style: str = "default"