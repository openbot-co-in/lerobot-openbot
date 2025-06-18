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

import time
import logging
from pathlib import Path

from lerobot.common.teleoperators.giraffe_leader.config_giraffe_leader import GiraffeLeaderConfig
from lerobot.common.teleoperators.giraffe_leader.giraffe_leader import GiraffeLeader
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_available_ports():
    """Find available USB serial ports."""
    # Only look for USB devices
    ports = [str(path) for path in Path("/dev").glob("ttyUSB*")]
    if not ports:
        # Also check for ACM devices (common on some systems)
        ports = [str(path) for path in Path("/dev").glob("ttyACM*")]
    return ports

def main():
    # Find available ports
    ports = find_available_ports()
    logger.info(f"Available USB ports: {ports}")
    
    if not ports:
        logger.error("No USB serial ports found! Please connect your device.")
        return

    # Create configuration
    config = GiraffeLeaderConfig(
        port=ports[0],  # Use the first available USB port
        baud_rate=115200,
        window_size=10,
        calibration_file="Calibration_Data.txt",
        id="giraffe_leader_1"  # Add an ID to the device
    )

    # Create and connect to the device
    try:
        logger.info("Initializing GiraffeLeader...")
        leader = GiraffeLeader(config)
        
        logger.info("Connecting to device...")
        leader.connect(calibrate=True)  # Will calibrate if needed
        
        logger.info("Starting to read sensor data...")
        logger.info("Press Ctrl+C to stop")
        
        # Read sensor data for 10 seconds
        start_time = time.time()
        while time.time() - start_time < 10:
            try:
                action = leader.get_action()
                if action:
                    logger.info("Current joint positions:")
                    for joint, pos in action.items():
                        logger.info(f"  {joint}: {pos:.2f}")
                time.sleep(0.1)  # Read at 10Hz
            except Exception as e:
                logger.error(f"Error reading action: {e}")
                break
                
    except KeyboardInterrupt:
        logger.info("Stopping...")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        if 'leader' in locals():
            logger.info("Disconnecting...")
            leader.disconnect()
            logger.info("Disconnected")

if __name__ == "__main__":
    main() 