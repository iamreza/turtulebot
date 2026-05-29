<!--
Course: Autonomous Robotics
Supervisor: Prof. Dr.-Ing. Reinhard Gerndt
Semester: Sommer Semester
Group: 7

Team:
- Reza Babaee, 70498082
- Hamid Safisamghabadi, 70497663
- Emad Mohammadi, 70494663
- Azarjan Gharibian
-->

# Magic Card

ROS 2 and Flask application for detecting playing-card orientation changes with a TurtleBot and camera.

## Versions

- `v1/`: first stable version.
- `v2/`: second stable version.
- `v3/`: latest stable version.

The main program logic is the same across versions: each version is started from its own `run.py` file.

`v2` rescans cards after odom closed-loop 180° turns and cross-matches the
check scan against the references, comparing each card in normal vs 180°
orientation. `v3` adds floor placement: the cards lie flat on the floor and the
fixed forward camera views them at an angle, so each detected card is
perspective-rectified (warped to a head-on canonical rectangle) before
matching, and a high-res central-pip cue refines the orientation verdict for
low-detail or near-symmetric cards.

## Project Structure

```text
magic_card/
├── README.md
├── v1/
│   ├── run.py
│   └── app/
│       ├── README.md
│       ├── __init__.py
│       ├── __main__.py
│       ├── config.py
│       ├── main.py
│       ├── robot_node.py
│       ├── templates.py
│       ├── utils.py
│       └── web.py
├── v2/
│   ├── run.py
│   └── app/
│       ├── __init__.py
│       ├── __main__.py
│       ├── camera_node.py
│       ├── config.py
│       ├── detector.py
│       ├── logger.py
│       ├── main.py
│       ├── state.py
│       ├── templates.py
│       ├── utils.py
│       ├── web.py
│       └── workflow.py
└── v3/
    ├── run.py
    └── app/
        ├── __init__.py
        ├── __main__.py
        ├── camera_node.py
        ├── config.py
        ├── detector.py
        ├── logger.py
        ├── main.py
        ├── state.py
        ├── templates.py
        ├── utils.py
        ├── web.py
        └── workflow.py
```

## Main Components

- `run.py`: version entrypoint; starts the selected Magic Card application.
- `app/main.py`: initializes ROS 2, application state, the robot/camera node, and the Flask web server.
- `app/config.py`: stores ROS topics, server port, detector thresholds, timing values, and workflow constants.
- `app/templates.py`: contains the embedded Flask HTML interface.
- `app/web.py`: defines Flask routes for the web dashboard, status API, camera feeds, and user actions.
- `app/utils.py`: contains shared helper functions for image preparation, comparison, geometry, and angle handling.
- `v1/app/robot_node.py`: V1 ROS node and workflow implementation.
- `v2/app/camera_node.py`: V2 ROS node for camera, odometry, motor power, battery, and motion control.
- `v2/app/detector.py`: V2 card segmentation, contour filtering, crop extraction, and debug overlay generation.
- `v2/app/workflow.py`: V2 scan, turn, rescan, cross-match, and 180-degree rotation detection workflow.
- `v3/app/detector.py`: V3 detector for floor-placed cards; minAreaRect geometry, frame-border rejection, and per-card perspective rectification to a head-on canonical crop.
- `v3/app/workflow.py`: V3 workflow with the center-fine central-pip orientation override for low-detail / near-symmetric cards.
- `app/state.py`: shared runtime state for camera frames, detected slots, workflow status, and UI data.
- `app/logger.py`: runtime logging and saved image output helpers.

## Run

Use three terminals.

### Terminal 1: TurtleBot bringup

```bash
ros2 launch turtlebot3_bringup robot.launch.py
```

### Terminal 2: Camera node

```bash
ros2 run camera_ros camera_node --ros-args -p width:=1280 -p height:=960 -p format:=RGB888 -p jpeg_quality:=90
```

### Terminal 3: Magic Card app

For version 1:

```bash
cd v1
python3 run.py
```

For version 2:

```bash
cd v2
python3 run.py
```

For version 3:

```bash
cd v3
python3 run.py
```

The web interface starts on port `8086`.
