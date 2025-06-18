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
import os
import serial.tools.list_ports
from tabulate import tabulate

from .config_giraffe_leader import GiraffeLeaderConfig
from .giraffe_leader import GiraffeLeader

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def list_available_ports():
    """List all available serial ports."""
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No serial ports found!")
        return []
        
    print("\nAvailable ports:")
    port_info = []
    for port in ports:
        info = {
            "Port": port.device,
            "Description": port.description,
            "Hardware ID": port.hwid
        }
        port_info.append(info)
    
    print(tabulate(port_info, headers="keys", tablefmt="grid"))
    return [port.device for port in ports]

def display_joint_positions(action):
    """Display joint positions in a table format."""
    joint_data = []
    for joint, value in action.items():
        joint_data.append([joint, f"{value:.2f}"])
    print(tabulate(joint_data, headers=["Joint", "Position"], tablefmt="grid"))

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # List available ports
    available_ports = list_available_ports()
    if not available_ports:
        return

    # Get port from user
    port = input("\nEnter the port to connect to (e.g., /dev/ttyUSB0): ").strip()
    if not port:
        logger.error("No port specified")
        return

    try:
        # Create config
        config = GiraffeLeaderConfig(
            port=port,
            id="test"  # Use a test ID for calibration file
        )

        # Create and connect to device
        logger.info("Connecting to device...")
        device = GiraffeLeader(config)
        device.connect(calibrate=True)

        # Test reading actions for 5 seconds
        logger.info("\nReading actions for 5 seconds...")
        print("Press Ctrl+C to stop early")
        
        start_time = time.time()
        while time.time() - start_time < 5:
            try:
                action = device.get_action()
                clear_screen()
                print("\nCurrent Joint Positions:")
                display_joint_positions(action)
                time.sleep(0.05)  # Update at 20Hz for smoother display
            except KeyboardInterrupt:
                print("\nStopped by user")
                break
            except Exception as e:
                logger.error(f"Error reading action: {e}")
                break

    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        if 'device' in locals():
            device.disconnect()
            logger.info("Device disconnected")

if __name__ == "__main__":
    main() 