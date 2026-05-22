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

# Magic Card Package Layout

- `config.py`: configuration constants, ROS topics, motion tuning, detector tuning, and log paths.
- `templates.py`: embedded Flask HTML page.
- `utils.py`: logging helpers, time helpers, angle helpers, image crop preparation, and comparison utilities.
- `robot_node.py`: ROS node, camera processing, card detection, scan workflows, motion control, reports, and robot state.
- `web.py`: Flask application, status/video/log routes, and action dispatch.
- `main.py`: application bootstrap for logging, ROS init/spin, Flask startup, and shutdown cleanup.
- `__main__.py`: allows running the package with `python3 -m app`.

`run.py` is the project entrypoint and simply calls `app.main.main()`.
