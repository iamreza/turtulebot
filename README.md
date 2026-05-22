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

The main program logic is the same across versions: each version is started from its own `run.py` file.

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

The web interface starts on port `8086`.
