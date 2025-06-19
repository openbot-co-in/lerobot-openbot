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
import serial
import time
import threading
import json
from pathlib import Path
from typing import List, Optional
from tabulate import tabulate
import numpy as np

logger = logging.getLogger(__name__)

class JointData:
    """Class to store joint data during calibration."""
    def __init__(self, name: str):
        self.name = name
        self.current_val = 0
        self.min_val = 4095  # Start with max value to find true min
        self.max_val = 0     # Start with min value to find true max
        self.samples = 0

    def update(self, value: int):
        """Update joint data with new value."""
        self.current_val = value
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)
        self.samples += 1

    def get_range_size(self) -> int:
        """Get the size of the joint's range."""
        return self.max_val - self.min_val

    @staticmethod
    def name_to_index(name: str) -> int:
        mapping = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        return mapping.index(name)

class CalibrationDataGenerator:
    """
    Collects AS5600 readings for calibration.
    Records middle position (homing) and full range of motion for each joint.
    """

    def __init__(
        self,
        serial_port: str,
        baud_rate: int = 115200,
        samples: int = 10,
        sample_delay: float = 0.05,
    ):
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.samples = samples
        self.sample_delay = sample_delay
        self.ser: Optional[serial.Serial] = None
        self.values: List[int] = []
        self.values_lock = threading.Lock()
        self.run_lock = threading.Lock()
        self.running = False
        self.display_running = False
        self.joints = [
            JointData("shoulder_pan"),
            JointData("shoulder_lift"),
            JointData("elbow_flex"),
            JointData("wrist_flex"),
            JointData("wrist_roll"),
            JointData("gripper")
        ]
        self.middle_position: Optional[list[int]] = None  # Store after step 1
        self.shifted_min = [4095] * 6  # For step 2 display
        self.shifted_max = [0] * 6

    def _read_one_raw(self) -> None:
        """Continuously read values from serial port."""
        if self.ser is None:
            raise RuntimeError("Serial port is not initialized")
            
        while self.running:
            try:
                line = self.ser.readline().decode('utf-8').strip()
                if not line:
                    continue
                    
                # Split by comma and filter out empty strings
                values = [v.strip() for v in line.split(',') if v.strip()]
                
                # Verify we have exactly 6 values
                if len(values) != 6:
                    continue
                    
                # Convert to integers
                try:
                    raw_values = [int(v) for v in values]
                except ValueError:
                    continue
                    
                # Update joint data
                with self.values_lock:
                    self.values = raw_values
                    for joint, value in zip(self.joints, raw_values):
                        joint.update(value)
                        
            except Exception as e:
                logger.warning(f"Error reading from serial port: {e}")
                time.sleep(0.1)  # Add delay on error to prevent tight loop

    def _shift_for_display(self, value: int, middle: int) -> int:
        """Shift encoder value so that middle is at 2048, wrap to [0, 4095]."""
        return (value - middle + 2048) % 4096

    def _display_joint_values(self, step: int) -> None:
        """Display current joint values in a table format."""
        while self.display_running:
            try:
                table_data = []
                if step == 2 and self.middle_position is not None:
                    # Display shifted values for step 2
                    for i, joint in enumerate(self.joints):
                        shifted_current = self._shift_for_display(joint.current_val, self.middle_position[i])
                        self.shifted_min[i] = min(self.shifted_min[i], shifted_current)
                        self.shifted_max[i] = max(self.shifted_max[i], shifted_current)
                        shifted_range = self.shifted_max[i] - self.shifted_min[i]
                        row = [
                            joint.name,
                            f"{shifted_current:4d}",
                            f"{self.shifted_min[i]:4d}",
                            f"{self.shifted_max[i]:4d}",
                            f"{shifted_range:4d}"
                        ]
                        table_data.append(row)
                else:
                    # Step 1 or fallback: show raw values
                    for joint in self.joints:
                        row = [
                            joint.name,
                            f"{joint.current_val:4d}",
                            f"{joint.min_val:4d}",
                            f"{joint.max_val:4d}",
                            f"{joint.get_range_size():4d}"
                        ]
                        table_data.append(row)

                print("\033[2J\033[H")  # Clear screen and move cursor to top
                if step == 1:
                    print("\n=== STEP 1: MIDDLE POSITION ===")
                    print("1. Move all joints to their middle positions")
                    print("2. Press ENTER to start sampling")
                elif step == 2:
                    print("\n=== STEP 2: FULL RANGE OF MOTION ===")
                    print("1. Move all joints through their full range of motion")
                    print("2. Press ENTER when done")
                print("\nCurrent Values:")
                print(tabulate(
                    table_data,
                    headers=["Joint", "Current", "Min", "Max", "Range"],
                    tablefmt="grid"
                ))
                time.sleep(0.05)
            except Exception as e:
                logger.warning(f"Error displaying values: {e}")
                time.sleep(0.1)

    def _monitor_values(self, step: int) -> None:
        """Monitor and display joint values during calibration."""
        self.display_running = True
        display_thread = threading.Thread(target=self._display_joint_values, args=(step,))
        display_thread.daemon = True
        display_thread.start()

    def _shift_and_unwrap(self, values: list[int], middle: int) -> list[float]:
        """Shift encoder readings so that middle is zero, then unwrap as continuous ticks."""
        # Shift readings so that middle is logical zero
        shifted = [((v - middle + 2048) % 4096) - 2048 for v in values]
        # Convert to radians
        radians = np.array(shifted) * 2 * np.pi / 4096
        unwrapped = np.unwrap(radians)
        # Convert back to ticks
        ticks = unwrapped * 4096 / (2 * np.pi)
        return ticks.tolist()

    def generate(self, output_file: str | Path) -> None:
        """
        Generate calibration data by collecting AS5600 readings in two poses:
        1. Middle position (homing)
        2. Full range of motion for each joint (unwrapped)
        """
        logger.info("\n=== CALIBRATION DATA GENERATION ===\n")
        
        try:
            # Delete existing calibration file if it exists
            output_path = Path(output_file).resolve()
            if output_path.exists():
                try:
                    logger.info(f"Deleting existing calibration file: {output_path}")
                    output_path.unlink(missing_ok=True)
                    logger.info("Successfully deleted existing calibration file")
                except Exception as e:
                    logger.error(f"Failed to delete existing calibration file: {e}")
                    raise
            
            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
            with self.run_lock:
                self.running = True
            
            # Start reading thread
            read_thread = threading.Thread(target=self._read_one_raw)
            read_thread.daemon = True
            read_thread.start()
            
            # Step 1: Middle position
            logger.info("Starting Step 1: Middle Position")
            self._monitor_values(1)
            input("\nPress ENTER to start sampling middle position...")
            # Sample middle position
            middle_values = []
            for _ in range(self.samples):
                with self.values_lock:
                    if len(self.values) == 6:
                        middle_values.append(self.values.copy())
                time.sleep(self.sample_delay)
            if not middle_values:
                raise RuntimeError("No valid middle position values collected")
            # Average the middle position values
            middle_position = [sum(vals) // len(vals) for vals in zip(*middle_values)]
            self.middle_position = middle_position  # Store for step 2
            logger.info(f"Middle position values: {middle_position}")
            # Step 2: Record full range of motion (unwrapped)
            logger.info("\nStarting Step 2: Full Range of Motion")
            for joint in self.joints:
                joint.min_val = 4095
                joint.max_val = 0
                joint.samples = 0
            self.shifted_min = [4095] * 6
            self.shifted_max = [0] * 6
            self.display_running = False
            time.sleep(0.1)
            self._monitor_values(2)
            # Clear and start buffer collection BEFORE prompting user
            step2_buffer = [[] for _ in range(6)]
            collecting = True
            def collect_step2():
                while collecting:
                    with self.values_lock:
                        if len(self.values) == 6:
                            for i in range(6):
                                step2_buffer[i].append(self.values[i])
                    time.sleep(self.sample_delay)
            collector_thread = threading.Thread(target=collect_step2)
            collector_thread.daemon = True
            collector_thread.start()
            print("\nMove ALL joints through their FULL range of motion NOW.\nWhen done, press ENTER to finish step 2...")
            input()  # Wait for user to finish moving joints
            collecting = False
            collector_thread.join()
            # Shift and unwrap all values for each joint
            shifted_unwrapped_ranges = []
            for i in range(6):
                shifted_unwrapped = self._shift_and_unwrap(step2_buffer[i], middle_position[i])
                shifted_unwrapped_ranges.append(shifted_unwrapped)
            joint_ranges = []
            for i in range(6):
                min_val = int(np.min(shifted_unwrapped_ranges[i]))
                max_val = int(np.max(shifted_unwrapped_ranges[i]))
                joint_ranges.append((min_val, max_val))
            calibration_data = {
                "middle_position": middle_position,
                "offsets": middle_position,
            }
            for i, joint in enumerate(self.joints):
                calibration_data[joint.name] = {
                    "range_min": joint_ranges[i][0],
                    "range_max": joint_ranges[i][1],
                    "offset": middle_position[i]
                }
            with open(output_file, "w") as f:
                json.dump(calibration_data, f, indent=4)
            logger.info(f"\nCalibration data saved to {output_file}")
        except Exception as e:
            logger.error(f"Error during calibration: {e}")
            raise
        finally:
            with self.run_lock:
                self.running = False
                self.display_running = False
            if self.ser:
                self.ser.close()
                self.ser = None


if __name__ == "__main__":
    generator = CalibrationDataGenerator(
        serial_port='/dev/ttyUSB0',
        baud_rate=115200,
        samples=10,
        sample_delay=0.05
    )
    generator.generate("calibration.json")
    
