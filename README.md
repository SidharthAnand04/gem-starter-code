# Autonomous Lane Following System 

## Motivation
This project implements an autonomous lane following system with integrated  for autonomous vehicles. The system is designed to:
- Follow lane markings safely and smoothly
- Provide real-time visual feedback of lane detection and vehicle control

## Demo Video
[![Autonomous Lane Following Demo](https://img.youtube.com/vi/EK7IHKS61hU/0.jpg)](https://youtu.be/EK7IHKS61hU)
*Click the image above to watch the demonstration video*

## Models Used
The system utilizes two pre-trained deep learning models:

- **YOLOPv2 for lane detection**: https://github.com/CAIC-AD/YOLOPv2
  
  Instructions for use:
  1. Download the model from https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt
  2. Move the .pt model file into the weights folder

- **YOLOv5s-seg for stop sign detection**: https://github.com/ultralytics/yolov5
  
  Instructions for use:
  - Model is automatically downloaded when first running the system
  - Internet connection is required for initial download


## Technical Approach

### System Architecture
The system consists of two main components:
1. Lane Detection Node (`lane_detection.py`)
2. Vehicle Controller Node (`controller.py`)

#### Lane Detection
- Uses a deep learning-based approach with YOLOPv2 for lane detection
- Implements a custom image processing pipeline:
  - HSV color space filtering for yellow lane marking detection
  - Grayscale conversion and thresholding for enhanced lane visibility
  - Frame buffering for stable detection
- Generates waypoints for vehicle navigation
- Stop sign detection can be introduced using YOLOv5

#### Vehicle Control
- Implements PID controllers for both speed and steering
- Features:
  - Adaptive steering control based on lateral error
  - Speed control with acceleration limits
  - Smooth gear shifting and brake control

### Key Features

#### Lane Following
- Real-time lane boundary detection
- Dynamic waypoint generation
- Continuous path planning
- Lateral error correction

## Implementation Details

### Dependencies
- ROS (Robot Operating System)
- PyTorch
- OpenCV
- NumPy
- PACMOD vehicle interface

### Key Parameters
- Default speed: 1.5 m/s
- Maximum acceleration: 2.5 m/s²
- Steering PID: Kp=0.01, Ki=0.0, Kd=0.005
- Speed PID: Kp=0.5, Ki=0.0, Kd=0.1
- Stop sign brake distance: 5 meters
- Stop duration: 3 seconds

### ROS Topics
#### Subscribed Topics
- `/oak/rgb/image_raw`: Camera feed
- `/pacmod/as_tx/enable`: Vehicle enable status
- `/pacmod/parsed_tx/vehicle_speed_rpt`: Vehicle speed

#### Published Topics
- `/lane_detection/waypoints`: Navigation waypoints
- `/lane_detection/annotate`: Annotated video feed
- `/pacmod/as_rx/steer_cmd`: Steering commands
- `/pacmod/as_rx/accel_cmd`: Acceleration commands

## Usage
1. Launch ROS master node

2. Start the lane detection node:
   ```bash
   python lane_detection_node.py
   ```
3. Start the controller node:
   ```bash
   python lane_follow_controller.py
   ```

## Performance Considerations
- Frame processing runs at 10Hz
- Stop sign detection includes aspect ratio verification
- PID controllers include anti-windup mechanisms
- Smooth acceleration and deceleration profiles
- Robust to varying lighting conditions

## Future Improvements
- Integration of additional traffic sign detection
- Dynamic speed adjustment based on road conditions
- Enhanced lane detection in adverse weather
- Implementation of path prediction algorithms
- Integration with global path planning
