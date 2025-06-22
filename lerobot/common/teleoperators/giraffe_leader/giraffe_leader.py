#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
import math
import collections
from pathlib import Path
from typing import Optional
import json

import serial
import numpy as np

from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.motors import MotorCalibration

from ..teleoperator import Teleoperator
from .config_giraffe_leader import GiraffeLeaderConfig
from .leader_calibration import CalibrationDataGenerator

logger = logging.getLogger(__name__)

# Add this at the top-level (after logger = ...)
file_handler = logging.FileHandler('giraffe_leader.log')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    logger.addHandler(file_handler)

def signed_delta(raw: int, ref: int) -> int:
    return ((raw - ref + 2048) % 4096) - 2048

class GiraffeLeader(Teleoperator):
    """
    GiraffeLeader for reading AS5600 sensor data from the Giraffe hardware
    """

    config_class = GiraffeLeaderConfig
    name = "giraffe_leader"

    def __init__(self, config: GiraffeLeaderConfig):
        super().__init__(config)
        self.config = config
        self.serial_port: Optional[serial.Serial] = None
        self.angle_windows = [collections.deque(maxlen=self.config.window_size) for _ in range(6)]
        self.dummy_angles = [0.0] * 6
        self.zero_pose: Optional[list[int]] = None
        self.joint_ranges: Optional[list[tuple[int, int]]] = None
        self.slopes: Optional[list[float]] = None
        self.calibration: dict[str, MotorCalibration] = {}

    @property
    def action_features(self) -> dict[str, type]:
        return {
            "shoulder_pan.pos": float,
            "shoulder_lift.pos": float,
            "elbow_flex.pos": float,
            "wrist_flex.pos": float,
            "wrist_roll.pos": float,
            "gripper.pos": float,
        }

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.serial_port is not None and self.serial_port.is_open

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        try:
            self.serial_port = serial.Serial(
                self.config.port,
                self.config.baud_rate,
                timeout=1
            )
            
            if not self.is_calibrated and calibrate:
                self.calibrate()

            self.configure()  # Call configure after connection
            logger.info(f"{self} connected.")
        except Exception as e:
            raise DeviceNotConnectedError(f"Failed to connect {self}: {e}")

    def configure(self) -> None:
        """Configure the device after connection."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
            
        logger.info(f"Configuring {self}...")
        # No specific configuration needed for AS5600 sensors
        # Just verify we can read data
        try:
            if self.serial_port is None:
                raise DeviceNotConnectedError(f"{self} is not connected")
            self.serial_port.readline()  # Clear any buffered data
            logger.info(f"{self} configured successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to configure {self}: {e}")

    @property
    def is_calibrated(self) -> bool:
        return self.zero_pose is not None and self.joint_ranges is not None and self.slopes is not None

    def calibrate(self) -> None:
        print(f"\nRunning calibration of {self}")
        if not self.calibration_fpath.exists():
            print("Calibration file not found. Running calibration process.")
            generator = CalibrationDataGenerator(
                serial_port=self.config.port,
                baud_rate=self.config.baud_rate,
                samples=10,
                sample_delay=0.05,
                device=self
            )
            generator.generate(self.calibration_fpath)

        self._load_calibration()
        print(f"Calibration loaded from {self.calibration_fpath}")

    def _load_calibration(self, fpath: Path | None = None) -> None:
        """
        Load calibration data from the specified file.
        """
        fpath = self.calibration_fpath if fpath is None else fpath
        with open(fpath) as f:
            data = json.load(f)
            
            # Load middle position values
            middle_pos = data["middle_position"]
            if not isinstance(middle_pos, list) or len(middle_pos) != 6:
                raise ValueError("Middle position must contain 6 values.")
            self.zero_pose = middle_pos
            
            # Load joint ranges and create MotorCalibration objects
            self.joint_ranges = []
            self.calibration = {}
            for i, joint in enumerate(["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]):
                joint_data = data[joint]
                min_val = joint_data["range_min"]
                max_val = joint_data["range_max"]
                self.joint_ranges.append((min_val, max_val))
                
                # Create MotorCalibration object for each joint
                self.calibration[joint] = MotorCalibration(
                    id=i + 1,  # Use 1-based indexing for motor IDs
                    drive_mode=0,
                    homing_offset=middle_pos[i],
                    range_min=min_val,
                    range_max=max_val
                )
            
            # Compute slopes based on joint ranges
            self.slopes = []
            for min_val, max_val in self.joint_ranges:
                range_size = max_val - min_val
                if range_size == 0:
                    self.slopes.append(0.0)
                else:
                    # Map the full range to [-90, 90] degrees
                    self.slopes.append(180.0 / range_size)
            
            print(f"Middle Position: {self.zero_pose}")
            print(f"Joint Ranges: {self.joint_ranges}")
            print(f"Slopes: {self.slopes}")

    def _save_calibration(self, fpath: Path | None = None) -> None:
        """
        Save calibration data to the specified file.
        """
        if self.zero_pose is None or self.joint_ranges is None:
            raise RuntimeError("Cannot save calibration: calibration data is not initialized")
            
        fpath = self.calibration_fpath if fpath is None else fpath
        data = {
            "middle_position": self.zero_pose,
            "shoulder_pan": {"range_min": self.joint_ranges[0][0], "range_max": self.joint_ranges[0][1]},
            "shoulder_lift": {"range_min": self.joint_ranges[1][0], "range_max": self.joint_ranges[1][1]},
            "elbow_flex": {"range_min": self.joint_ranges[2][0], "range_max": self.joint_ranges[2][1]},
            "wrist_flex": {"range_min": self.joint_ranges[3][0], "range_max": self.joint_ranges[3][1]},
            "wrist_roll": {"range_min": self.joint_ranges[4][0], "range_max": self.joint_ranges[4][1]},
            "gripper": {"range_min": self.joint_ranges[5][0], "range_max": self.joint_ranges[5][1]},
        }
        with open(fpath, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Calibration saved to {fpath}")

    def convert_raw_to_degrees(self, raw_values: list[int]) -> list[float]:
        if not self.is_calibrated:
            raise RuntimeError("Device must be calibrated before reading values")
            
        degrees = []
        for i, raw in enumerate(raw_values):
            # Calculate position relative to middle position
            d = signed_delta(raw, self.zero_pose[i])  # type: ignore
            # Convert to degrees using the slope
            angle = d * self.slopes[i]  # type: ignore
            degrees.append(angle)
        return degrees

    def apply_median_filter(self, angle_values: list[float]) -> list[float]:
        filtered_angles = []
        for i, angle in enumerate(angle_values):
            self.angle_windows[i].append(angle)
            angles_window = np.array(self.angle_windows[i])
            angles_unwrapped = np.unwrap(np.deg2rad(angles_window))
            median_unwrapped = np.median(angles_unwrapped)
            median_deg = math.degrees(median_unwrapped)
            median_deg = (median_deg + 180) % 360 - 180
            filtered_angles.append(median_deg)
        return filtered_angles

    def map_value(self, x: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
        return max(out_min, min(out_max, (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min))

    def clip_angle(self, angle: float, min_angle: float, max_angle: float) -> float:
        return max(min(angle, max_angle), min_angle)

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        if not self.is_calibrated:
            raise RuntimeError("Device must be calibrated before reading values")

        start = time.perf_counter()
        try:
            if self.serial_port is None:
                raise DeviceNotConnectedError(f"{self} is not connected")
                
            # Read until we get valid data
            max_attempts = 5
            for _ in range(max_attempts):
                try:
                    data = self.serial_port.readline().decode('utf-8').strip()
                    if not data:
                        continue
                        
                    # Split by comma and filter out empty strings
                    values = [v.strip() for v in data.split(',') if v.strip()]
                    
                    # Verify we have exactly 6 values
                    if len(values) != 6:
                        logger.warning(f"Expected 6 values, got {len(values)}: {values}")
                        continue
                        
                    # Convert to signed integers (may be negative for reversed joints)
                    try:
                        raw_values = [int(v) for v in values]
                    except ValueError as e:
                        logger.warning(f"Invalid number format: {e}")
                        continue
                        
                    # Process the valid data
                    if self.joint_ranges is None or self.zero_pose is None:
                        raise RuntimeError("Device must be calibrated before reading values")
                    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
                    
                    # Calculate normalized values for each joint
                    normalized_values = []
                    for i, joint in enumerate(joint_names):
                        raw = raw_values[i]
                        sign = 1 if raw >= 0 else -1
                        abs_raw = abs(raw)
                        min_val, max_val = self.joint_ranges[i]
                        middle = self.zero_pose[i]
                        shifted = ((abs_raw - middle + 2048) % 4096) - 2048
                        range_size = max_val - min_val
                        if range_size == 0:
                            normalized = 0.0
                        elif joint == "gripper":
                            normalized = (shifted - min_val) / range_size * 100
                            normalized = max(0.0, min(100.0, normalized))
                            # If sign is negative, invert gripper (100-normalized)
                            if sign == -1:
                                normalized = 100.0 - normalized
                        elif joint == "wrist_roll":
                            # For continuous rotation, map 0-4095 to -100 to 100
                            normalized = ((shifted - min_val) / range_size) * 200 - 100
                            normalized = max(-100.0, min(100.0, normalized))                                
                        else:
                            normalized = ((shifted - (min_val + max_val) / 2) / (range_size / 2)) * 100
                            normalized = max(-100.0, min(100.0, normalized))
                            # Reverse sign for all non-gripper joints
                            normalized = sign * normalized
                        normalized_values.append(normalized)
                    
                    # Apply median filter to the normalized values
                    filtered_values = self.apply_median_filter(normalized_values)
                    
                    # Create action dictionary with filtered values
                    action = {}
                    for i, joint in enumerate(joint_names):
                        action[f"{joint}.pos"] = filtered_values[i]
                    
                    dt_ms = (time.perf_counter() - start) * 1e3
                    logger.debug(f"{self} read action: {dt_ms:.1f}ms")
                    return action
                        
                except Exception as e:
                    logger.warning(f"Error reading data: {e}")
                    continue
                        
            raise RuntimeError(f"Failed to read valid data after {max_attempts} attempts")
                
        except Exception as e:
            logger.error(f"Error reading from {self}: {e}")
            raise RuntimeError(f"Failed to read action: {e}")

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # No feedback to send for this device
        pass

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        if self.serial_port:
            self.serial_port.close()
            self.serial_port = None
        logger.info(f"{self} disconnected.")

    def __str__(self) -> str:
        """Return a string representation of the device."""
        return f"{self.id} {self.__class__.__name__}"

    def _normalize(self, value: int, joint_name: str) -> float:
        """Normalize joint value: gripper to [0, 100], others to [-100, 100] (0=center)."""
        if self.joint_ranges is None:
            raise RuntimeError("Joint ranges not initialized. Please run calibration first.")
        if joint_name not in self.joint_ranges:
            raise ValueError(f"Unknown joint: {joint_name}")
        joint_range = self.joint_ranges[joint_name]
        range_min = joint_range["range_min"]
        range_max = joint_range["range_max"]
        middle = joint_range["middle_position"] if "middle_position" in joint_range else (range_min + range_max) // 2
        range_size = range_max - range_min
        if range_size == 0:
            return 0.0
        if joint_name == "gripper":
            # Normalize gripper to [0, 100]
            normalized = (value - range_min) / (range_size) * 100
            return max(0.0, min(100.0, normalized))
        else:
            # Normalize others to [-100, 100] (0=center)
            normalized = ((value - middle) / (range_size / 2)) * 100
            return max(-100.0, min(100.0, normalized))

    def _denormalize(self, value: float, joint_name: str) -> int:
        """Denormalize joint value: gripper from [0, 100], others from [-100, 100]."""
        if self.joint_ranges is None:
            raise RuntimeError("Joint ranges not initialized. Please run calibration first.")
        if joint_name not in self.joint_ranges:
            raise ValueError(f"Unknown joint: {joint_name}")
        joint_range = self.joint_ranges[joint_name]
        range_min = joint_range["range_min"]
        range_max = joint_range["range_max"]
        middle = joint_range["middle_position"] if "middle_position" in joint_range else (range_min + range_max) // 2
        range_size = range_max - range_min
        if range_size == 0:
            return middle
        if joint_name == "gripper":
            # Denormalize gripper from [0, 100]
            raw = range_min + (value / 100) * range_size
            return max(range_min, min(range_max, int(raw)))
        else:
            # Denormalize others from [-100, 100]
            raw = middle + (value / 100) * (range_size / 2)
            return max(range_min, min(range_max, int(raw)))
