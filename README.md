# jetank_manipulation

Mobile-manipulation grasp pipeline for the JeTank arm (4-DOF: S1/S2/S3/S5 +
gripper) via MoveIt 2. The package decomposes the "pick up a sock" task into
three modular nodes wired together by ROS interfaces:

- **`grasp_server`** — open-loop preset (or pose-targeted) grasp action server.
- **`base_approach_node`** — diff-drive servo that drives the base to a standoff
  ahead of a sensed point.
- **`mobile_grasp_coordinator`** — thin FSM that orchestrates segmentation, base
  approach and grasp into one `Trigger`-driven sock pickup.

The arm camera is fixed to the `S1` link with no tilt, so a floor sock leaves the
camera FOV at close range. The pipeline therefore **remembers the grasp pose in a
world-fixed frame (`odom`) while the sock is still visible**, drives the base up,
and grasps **open-loop** from the remembered pose.

## ROS 2 API

### Nodes

| Node name | Executable | Role |
|---|---|---|
| `grasp_server` | `grasp_server` | Preset / pose-targeted grasp action server; drives the arm via MoveIt2's `move_group` and the gripper action controller. |
| `base_approach_node` | `base_approach_node` | `ApproachTarget` action server; proportional rotate-then-drive diff-drive servo to a standoff ahead of a point. |
| `mobile_grasp_coordinator` | `mobile_grasp_coordinator` | FSM coordinator; `Trigger` service that runs SEGMENT → REACH_CHECK → [APPROACH] → GRASP. Owns no perception/motion/driving logic — pure ROS-interface orchestration. |

`grasp_pose_node` (sock cloud → grasp pose, Phase 6) is also packaged; its
`top_down_quaternion` helper is reused by the coordinator.

---

## `mobile_grasp_coordinator`

Triggered by the `~/execute_sock_grasp` service (`std_srvs/srv/Trigger`). On
trigger it runs a state machine, returning a graceful `success=false` Trigger
response (never raising) on any missing server / timeout / rejection / TF failure:

1. **SEGMENT** — call `/segment_socks` (`jetank_detection/SegmentSocks`) for the
   sock centroid in `target_frame` (default `base_link`). Build the top-down
   grasp `PoseStamped` and transform it into `world_frame` (default `odom`),
   **remembering it while the sock is still in view**. No sock → fail.
2. **REACH_CHECK** — is the centroid within `arm_reach` (default 0.22 m) of the
   arm mount `arm_base_xy` (default `[0.06, 0.0]`)? If yes, skip APPROACH.
3. **APPROACH** *(conditional)* — call `/approach_target` (`ApproachTarget`) with
   the centroid and `approach_standoff` (default 0.18 m) to drive the base up.
4. **GRASP** — call `/grasp_object` (`GraspObject`). In `grasp_mode:=preset`
   (default) it sends an **empty** pose so `grasp_server` runs its tuned preset
   joint sequence — a free-form Cartesian floor pose is infeasible on this 4-DOF
   arm (wrist self-collides with the arm-mounted camera, OMPL finds no valid IK).
   In `grasp_mode:=pose` it recovers the remembered `odom` pose into `target_frame`
   at the latest TF and sends it as `target_pose` (for elevated/reachable targets).
5. **DONE / FAILED** — report the sock centroid in the Trigger response.

Latches the grasp pose on `~/grasp_pose` (`PoseStamped`, transient-local) for RViz.

### Key parameters

| Parameter | Default | Meaning |
|---|---|---|
| `grasp_mode` | `preset` | `preset` (tuned joint reach, no pose) or `pose` (send remembered pose). |
| `target_frame` | `base_link` | Frame for segmentation + grasp. |
| `world_frame` | `odom` | World-fixed frame the grasp pose is remembered in. |
| `arm_reach` | `0.22` | Reachable radius (m) about the arm mount. |
| `arm_base_xy` | `[0.06, 0.0]` | Arm mount XY in `target_frame`. |
| `approach_standoff` | `0.18` | Standoff (m) handed to `ApproachTarget`. |
| `min_score` / `max_range` | `0.3` / `3.0` | Segmentation goal thresholds. |
| `segment_timeout_s` / `approach_timeout_s` / `grasp_timeout_s` | `10` / `30` / `60` | Per-step timeouts (s). |
| `segment_action` / `approach_action` / `grasp_action` | `/segment_socks` / `/approach_target` / `/grasp_object` | Action names. |

```bash
ros2 run jetank_manipulation mobile_grasp_coordinator --ros-args -p use_sim_time:=true
ros2 service call /mobile_grasp_coordinator/execute_sock_grasp std_srvs/srv/Trigger '{}'
```

---

## `base_approach_node`

Hosts the `/approach_target` action (`jetank_manipulation/action/ApproachTarget`):
drive the tracked base until a target point is `standoff` metres ahead, with a
proportional **rotate-to-face then drive** servo.

Per tick it snapshots the goal `target` once into `stable_frame` (default `odom`,
a world-fixed point so the distance actually shrinks as the base drives), then
re-TFs `stable_frame → base_frame` each tick to get `(dx, dy)`. The pure control
law `approach_control(...)` (no ROS types, unit-tested) rotates in place while
`|heading| > heading_tol`, then drives forward until `dist <= standoff + arrive_tol`
— the forward term `k_lin*(dist-standoff)` asymptotes, so the tolerance band is
what trips arrival. Commands are published as `geometry_msgs/TwistStamped` on
`cmd_vel_topic` (default `/diff_drive_controller/cmd_vel`); a zero Twist is sent on
arrival, timeout, cancel, or exit.

### Key parameters

| Parameter | Default | Meaning |
|---|---|---|
| `base_frame` | `base_link` | Frame the target is reduced to for control. |
| `cmd_vel_topic` | `/diff_drive_controller/cmd_vel` | TwistStamped command topic. |
| `stable_frame` | `odom` | World-fixed frame the target is snapshotted into. |
| `k_lin` / `k_ang` | `0.6` / `1.2` | Linear / angular proportional gains. |
| `max_lin` / `max_ang` | `0.15` / `0.8` | Linear (m/s) / angular (rad/s) saturation. |
| `heading_tol` | `0.15` | Heading error (rad) below which forward drive starts. |
| `arrive_tol` | `0.03` | Distance band (m) past `standoff` that trips arrival. |
| `control_rate` | `10.0` | Control loop rate (Hz). |

```bash
ros2 run jetank_manipulation base_approach_node --ros-args -p use_sim_time:=true
ros2 action send_goal /approach_target jetank_manipulation/action/ApproachTarget \
  '{target: {header: {frame_id: base_link}, point: {x: 0.5, y: 0.0, z: 0.0}}, standoff: 0.18, timeout: 20.0}'
```

---

## `grasp_server`

Open-loop grasp action server. Each named arm target is planned as a joint goal
(per-DOF `JointConstraint`s, `±0.01` tol) against `move_group`; planning group
`arm`, planner `RRTConnect`.

### Preset grasp sequence (speed-optimised)

The default sequence is **`grasp_pre → open → grasp_reach → close → home`**. The
legacy `ready` **approach** and **retreat-via-ready** moves are **SKIPPED** (their
target params default to empty) — both were redundant *backward* swings
(`ready` is `S2=-45°`) before/after the forward grasp. The arm now goes
home/current → `grasp_pre` directly and retreats straight to `home` (= park).

`stage` feedback values: `moving_to_pre_grasp`, `opening_gripper`,
`moving_to_grasp`, `closing_gripper`, `parking`, `done`, `aborting_retreat`
(plus `moving_to_approach` / `retreating` only if those targets are re-enabled).

If a `GraspObject` goal carries a non-empty `target_pose.header.frame_id`, the
**pose-targeted** path runs instead (`grasp_pre → pre-grasp at +approach_height →
open → reach at target → close → retreat → home`), driving the EE to the
Cartesian pose (position-only; `pose_grasp.pose_use_orientation` is false because
the 4-DOF arm + KDL IK cannot satisfy a full 6-DOF pose).

### Named arm targets (`_SRDF_STATES`)

Hard-coded in `grasp_server.py` as a manual mirror of
`jetank_moveit_config/config/jetank.srdf` (S1/S2/S3/S5 rad):

| Target | S2 | S3 | Notes |
|---|---|---|---|
| `home` | 0.0 | 0.0 | Park pose. |
| `ready` | -0.785 | 1.047 | Legacy backward-tilted approach (no longer in the default sequence). |
| `grasp_pre` | 1.222 (70°) | -0.262 | Raised forward approach. |
| `grasp_reach` | **1.8326 (105°)** | -0.262 | Floor reach. **≥107° jams the gripper into the floor (controller TIMED_OUT/CONTROL_FAILED)**; lower angles leave the gripper hovering above the sock. Re-tune if the floor height changes. |

### Key parameters (node defaults, tuned for speed)

| Parameter | Default | Meaning |
|---|---|---|
| `motion.velocity_scaling` | `0.6` | Fraction of max joint velocity (0–1). |
| `motion.acceleration_scaling` | `0.5` | Fraction of max joint acceleration (0–1). |
| `motion.allowed_planning_time_s` | `1.5` | OMPL planning budget per stage (s). |
| `motion.num_planning_attempts` | `1` | Planning retries per stage. |
| `gripper.open_width` / `close_width` | `0.04` / `0.0` | Gripper command positions (m). |
| `gripper.max_effort` | `5.0` | GripperCommand max effort (N). |
| `gripper.dwell_after_open_s` | `0.2` | Sleep after opening gripper (s). |
| `gripper.dwell_after_close_s` | `0.8` | Sleep after closing gripper (s). |
| `arm_targets.approach` | `""` (skip) | Approach target (empty → skipped). |
| `arm_targets.pre_grasp` | `grasp_pre` | Pre-grasp target. |
| `arm_targets.grasp` | `grasp_reach` | Grasp target. |
| `arm_targets.retreat` | `""` (skip) | Retreat target (empty → skipped). |
| `arm_targets.park` | `home` | Final park target. |
| `pose_grasp.approach_height_m` | `0.06` | +Z standoff (m) for pose-targeted pre-grasp/retreat. |
| `pose_grasp.pose_use_orientation` | `false` | Constrain EE orientation (keep off on the 4-DOF arm). |

> `config/grasp_poses.yaml` currently still carries the **pre-tuning** values
> (`approach`/`retreat: grasp_pre`, `velocity/acceleration_scaling: 0.3`,
> `allowed_planning_time_s: 5.0`, `num_planning_attempts: 3`,
> `dwell_after_open_s: 0.5`). The node-declared defaults above are the
> speed-optimised values; pass the new values explicitly (or update the YAML) to
> get the tuned behaviour when launching with that config.

### Launch / invoke

```bash
ros2 launch jetank_manipulation grasp.launch.py   # grasp_server + config, use_sim_time:=true
ros2 action send_goal /grasp_object jetank_manipulation/action/GraspObject '{}'  # preset
```

Requires `move_group` to be running (e.g. `jetank_moveit_config moveit_sim.launch.py`).

---

## Action interfaces (owned by this package)

### `GraspObject.action`

```
# Goal
string object_hint                   # reserved for vision-targeted grasp
geometry_msgs/PoseStamped target_pose  # non-empty frame_id => pose-targeted; empty => preset
float32 approach_height              # +Z pre-grasp/retreat standoff (m); <=0 => param default
---
# Result
bool success
string message
---
# Feedback
string stage
```

Also a **client** of `/move_action` (`moveit_msgs/MoveGroup`) and
`/gripper_controller/gripper_cmd` (`control_msgs/GripperCommand`).

### `ApproachTarget.action`

```
# Goal
geometry_msgs/PointStamped target   # point to approach (any frame; reduced to base frame)
float32 standoff                    # stop when target is this far ahead (m)
float32 timeout                     # s, abort if exceeded
---
# Result
bool success
float32 final_distance
string message
---
# Feedback
float32 distance
float32 heading_error
```

---

## Tests

`test/` loads the source modules by file path and exercises pure logic, stubbing
heavy ROS deps only when absent (so the suite runs in a bare env or against the
real packages):

- `test_import.py` — `grasp_server` module constants, `_SRDF_STATES` (exactly
  `home`/`ready`/`grasp_pre`/`grasp_reach`, 4 DOF each), config named targets
  resolve, and `_named_target_request` / pose-request field correctness.
- `test_base_control.py` — the pure `approach_control` servo law (rotate-then-drive,
  clamping, arrival band).
- `test_grasp_pose.py` — `grasp_pose_node` geometry (`top_down_quaternion`, cloud→pose).

```bash
# standalone (bare env, deps stubbed)
pixi run -- bash -c 'cd src/jetank_manipulation && python -m pytest test/ -q'

# under colcon
colcon test --packages-select jetank_manipulation && colcon test-result --verbose
```
