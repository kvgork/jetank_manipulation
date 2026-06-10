# Plan — Optimize grasp speed (slow arm motion)

## Symptom (observed 2026-06-10)
The preset grasp is very slow and moves oddly: the arm first swings **backward**
(to `ready`, S2=−45°), then **forward** to a pre-grasp (`grasp_pre`, S2=+70°),
**waits a long time**, only then opens the gripper and goes down to `grasp_reach`.
Full grasp ≈ 60-90 s.

## Sequence today (`grasp_server._execute_cb`, preset path)
`ready → grasp_pre → open → grasp_reach → close → ready (retreat) → home`

Measured stage times (this session's logs):
| Stage | Target (S2) | ~time |
|---|---|---|
| moving_to_approach | `ready` (−45°) | ~9.6 s ← backward swing |
| moving_to_pre_grasp | `grasp_pre` (+70°) | ~16.5 s ← the "long wait" |
| opening_gripper | — | ~1 s (+0.5 s dwell) |
| moving_to_grasp | `grasp_reach` (105°) | ~7-9 s |
| closing_gripper | — | ~1 s (+0.8 s dwell) |
| retreating | `ready` (−45°) | ~15-19 s |
| parking | `home` (0°) | ~6-9 s |

## Root causes
1. **Redundant `ready` approach.** The arm starts at `home` (0°). Going to
   `ready` (−45°, *backward*) before `grasp_pre` (+70°, forward) adds a full
   backward→forward swing (~9.6 s) for no reason — `home`/current → `grasp_pre`
   is a direct forward move.
2. **Low velocity/acceleration scaling.** `motion.velocity_scaling=0.3`,
   `acceleration_scaling=0.3` → arm crawls. Doubles every execution time.
3. **Over-budgeted planning.** `allowed_planning_time_s=5.0` +
   `num_planning_attempts=3` for **named JOINT targets**. Joint-space goals plan
   almost instantly with RRTConnect; 5 s × 3 attempts is the bulk of the "long
   wait" (planner burns budget / retries before returning).
4. **Retreat via `ready`.** Retreat goes to `ready` (−45° backward) then `home`
   — another backward swing. Retreat straight to `home`.
5. **Dwells.** `dwell_after_open=0.5`, `dwell_after_close=0.8` — minor, keep the
   close dwell (grip settle), trim the open dwell.

## Optimizations (ordered by impact, all param/sequence — no new logic)
1. **Drop the `ready` approach + retreat-via-ready.** Set the sequence to
   `grasp_pre → open → grasp_reach → close → home` (skip step 1 `ready`; retreat
   `ready`→ drop, go straight `home`). Saves ~9.6 s (approach) + ~15 s (retreat
   uses home directly instead of ready+home). Implement: `arm_targets.approach`
   and `arm_targets.retreat` — make them skippable (empty = skip) OR set
   approach=`grasp_pre`-adjacent / retreat=`home`. Simplest: guard the move when
   the target equals the previous/again, and set `retreat`==`park`==`home`.
2. **Raise speed:** `motion.velocity_scaling 0.3 → 0.6`, `acceleration_scaling
   0.3 → 0.5`. Roughly halves execution time on every move. (Keep ≤0.6 first;
   verify no overshoot/CONTROL_FAILED on the floor reach.)
3. **Cut planning budget for joint targets:** `allowed_planning_time_s 5.0 →
   1.5`, `num_planning_attempts 3 → 1`. Joint goals solve immediately; this
   removes most of the per-move "wait." (Pose-targeted path can keep a larger
   budget — gate by path if needed.)
4. **Trim dwell:** `dwell_after_open_s 0.5 → 0.2`. Keep `dwell_after_close_s 0.8`.

Expected: ~60-90 s → ~20-30 s per grasp.

## Files / params
- `src/jetank_manipulation/jetank_manipulation/grasp_server.py`
  - `_execute_cb` preset sequence: skip `ready` approach; retreat → `home` only.
  - param defaults (lines ~366-374): velocity_scaling, acceleration_scaling,
    allowed_planning_time_s, num_planning_attempts, dwell_after_open_s.
- Optionally expose via `grasp_poses.yaml` so it's tunable without rebuild.
- **Rebuild required:** `colcon build --symlink-install --packages-select
  jetank_manipulation` (Python here is an installed COPY, not symlinked).

## Test
- Trigger `/grasp_object` (preset) and time `Stage:` logs start→done.
- Verify still reaches `grasp_reach` cleanly (no CONTROL_FAILED) at higher speed.
- Confirm motion path: no backward swing; `grasp_pre → grasp_reach` is forward+down.
- Full mission still completes (NAVIGATE→SEARCH→PICK→DEPOSIT→DONE).

## Risks
- Higher vel/acc near the floor could re-introduce CONTROL_FAILED at
  `grasp_reach` (jam) — bump speed in steps, re-verify the 105° reach.
- Skipping `ready` assumes the start pose (home/post-search) can plan directly to
  `grasp_pre` collision-free — true for this arm; verify in RViz.
- Related: grasp_reach S2=105° ([[jetank-grasp-reach-floor-angle]]).
