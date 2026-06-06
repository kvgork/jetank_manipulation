# jetank_manipulation

Open-loop preset grasp action server for the JeTank arm via MoveIt 2. Exposes a
`GraspObject` action that runs a fixed approach → grasp → retreat → park
sequence.

## ROS 2 API

This package provides a single runtime node plus a custom action interface.

### Nodes

| Node name | Executable | Role |
|---|---|---|
| `grasp_server` | `grasp_server` (`ros2 run jetank_manipulation grasp_server`) | Open-loop preset grasp action server; drives the JeTank arm through a fixed grasp sequence via MoveIt2's `move_group` and commands the gripper via the gripper action controller. |

### Actions

| Action name | Type | Role |
|---|---|---|
| `grasp_object` | `jetank_manipulation/action/GraspObject` | **Server.** Runs the full grasp sequence (approach → pre-grasp → open → grasp → close → retreat → park) and publishes `stage` feedback. |
| `/move_action` | `moveit_msgs/action/MoveGroup` | **Client.** Plans + executes each named arm target against the running `move_group`. |
| `/gripper_controller/gripper_cmd` | `control_msgs/action/GripperCommand` | **Client.** Opens/closes the gripper via the `GripperActionController`. |

The `GraspObject.action` definition (owned by this package):

```
# Goal
string object_hint   # unused in Phase 1; reserved for Phase 2 vision-targeted grasp
---
# Result
bool success
string message
---
# Feedback
string stage
```

`stage` feedback values emitted by the server: `moving_to_approach`, `moving_to_pre_grasp`, `opening_gripper`, `moving_to_grasp`, `closing_gripper`, `retreating`, `parking`, `done`, `aborting_retreat`.

> This package publishes/subscribes to no topics and offers no services of its own — all robot interaction is through the three actions above.

### Key parameters

Defaults below are the node-declared defaults; the launched values come from `config/grasp_poses.yaml` (noted where they differ).

| Parameter | Default (node) | Launch value (`grasp_poses.yaml`) | Meaning |
|---|---|---|---|
| `motion.velocity_scaling` | `0.3` | `0.3` | Fraction of max joint velocity (0–1). |
| `motion.acceleration_scaling` | `0.3` | `0.3` | Fraction of max joint acceleration (0–1). |
| `motion.allowed_planning_time_s` | `5.0` | `5.0` | OMPL planning time budget per stage (s). |
| `motion.num_planning_attempts` | `3` | `3` | Planning retries per stage before abort. |
| `gripper.open_width` | `0.04` | `0.04` | Gripper open command position (m). |
| `gripper.close_width` | `0.0` | `0.0` | Gripper close command position (m). |
| `gripper.max_effort` | `5.0` | (not set, uses default) | Max effort for GripperCommand (N). |
| `gripper.dwell_after_open_s` | `0.5` | `0.5` | Sleep after opening gripper (s). |
| `gripper.dwell_after_close_s` | `0.8` | `0.8` | Sleep after closing gripper (s). |
| `arm_targets.approach` | `ready` | `grasp_pre` | SRDF named target for the approach pose. |
| `arm_targets.pre_grasp` | `grasp_pre` | `grasp_pre` | SRDF named target for the pre-grasp pose. |
| `arm_targets.grasp` | `grasp_reach` | `grasp_reach` | SRDF named target for the grasp pose. |
| `arm_targets.retreat` | `ready` | `grasp_pre` | SRDF named target for the retreat pose. |
| `arm_targets.park` | `home` | `home` | SRDF named target for the final park pose. |
| `use_sim_time` | (standard) | `true` | Use the Gazebo clock; launch arg defaults to `true`. |

Named arm targets (`home`, `ready`, `grasp_pre`, `grasp_reach`) and their joint values are hard-coded in `grasp_server.py` (`_SRDF_STATES`) as a manual mirror of `jetank_moveit_config/config/jetank.srdf`. The planning group is `arm` and the planner is `RRTConnect`.

### Launch

- `ros2 launch jetank_manipulation grasp.launch.py` — starts `grasp_server` with `config/grasp_poses.yaml` and `use_sim_time:=true` (override with `use_sim_time:=false`). Requires `move_group` to already be running (e.g. `jetank_moveit_config/launch/moveit_sim.launch.py` or `sim_demo.launch.py`).

### Run / invoke

```bash
ros2 run jetank_manipulation grasp_server --ros-args -p use_sim_time:=true
ros2 action send_goal /grasp_object jetank_manipulation/action/GraspObject '{}'
```
