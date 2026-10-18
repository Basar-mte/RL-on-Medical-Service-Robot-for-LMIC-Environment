#!/bin/bash
# Publication media.
#   sim/make_publication_media.sh [all|runs|videos|gazebo]
#
#   runs    two video runs (walled / fixed_equal with 8 and with 45 walkers:
#           the live RViz mapping stream + 4K RViz stills) and three 4K RViz
#           still runs
#   videos  for each video run: the cinematic Gazebo footage re-rendered from
#           the run's own logs (replay_video.py - robot, camera and every
#           walker posed inside the simulator), then the composite with the
#           live RViz inset (compose_video.py) -> paper_figures/video/
#   gazebo  4K anti-aliased Gazebo renders of every tier; walkers are frozen
#           at their SIMULATED (social-force) poses from a robot-free crowd log
#           (crowd_log.py), so nobody stands inside anybody else
# Every run here is a demonstration run, reported - if at all - as one outcome.

HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$HERE/.."
FR="$ROOT/results_figures/figure_runs"; HI="$ROOT/results_figures/hires"; VID="$ROOT/paper_figures/video"
source /opt/ros/jazzy/setup.bash
source ~/hn_ws_paper/install/setup.bash
cd "$HERE"
ONLY="${1:-all}"
run() { echo "=== $(date +%T) $1 ==="; shift; python3 -u figure_run.py "$@" 2>&1 | tail -2; }

if [ "$ONLY" = "all" ] || [ "$ONLY" = "runs" ]; then
run "video run: walled / fixed_equal / busy (8 walkers)" --preset fixed_equal --tier walled --peds busy \
    --out-dir "$FR/walled_fixed_equal_video" --rviz-stream \
    --still paper_orbit.rviz:simtime:55:rviz_orbit.png \
    --still paper_slam.rviz:done:rviz_slam_final.png \
    --still paper_topdown.rviz:done:rviz_topdown.png
run "video run: walled / fixed_equal / crowd45 (45 walkers)" --preset fixed_equal --tier walled --peds crowd45 \
    --out-dir "$FR/walled_fixed_equal_crowd45_video" --rviz-stream \
    --still paper_orbit.rviz:simtime:60:rviz_orbit.png \
    --still paper_slam.rviz:done:rviz_slam_final.png \
    --still paper_topdown.rviz:done:rviz_topdown.png
run "still run: open / vanilla / busy" --preset vanilla --tier open --peds busy \
    --out-dir "$FR/open_vanilla" --still paper_topdown.rviz:done:rviz_topdown.png
run "still run: open / fixed_equal / busy" --preset fixed_equal --tier open --peds busy \
    --out-dir "$FR/open_fixed_equal" --still paper_topdown.rviz:done:rviz_topdown.png
run "still run: furnished / fixed_equal / busy" --preset fixed_equal --tier furnished --peds busy \
    --out-dir "$FR/furnished_fixed_equal" --still paper_topdown.rviz:done:rviz_topdown.png \
    --still paper_orbit.rviz:simtime:50:rviz_orbit.png
fi

if [ "$ONLY" = "all" ] || [ "$ONLY" = "videos" ]; then
mkdir -p "$VID"
for v in "walled_fixed_equal_video busy hospitalnav_demo.mp4 8" \
         "walled_fixed_equal_crowd45_video crowd45 hospitalnav_crowd45.mp4 45" \
         "walled_fixed_equal_crowd45_video2 crowd45 hospitalnav_crowd45_run2.mp4 45"; do
  set -- $v
  [ -d "$FR/$1" ] || continue
  echo "=== $(date +%T) replay footage: $1 ==="
  python3 -u replay_video.py --run-dir "$FR/$1" --tier walled --peds $2 --out "$FR/$1/gazebo.mp4" 2>&1 | tail -1
  echo "=== $(date +%T) composite: $3 ==="
  python3 -u compose_video.py --run-dir "$FR/$1" --out "$VID/$3" --policy fixed_equal \
      --scene "Walled hospital ward · $4 walking pedestrians" 2>&1 | tail -1
done
fi

[ "$ONLY" = "runs" ] || [ "$ONLY" = "videos" ] && { echo "=== $(date +%T) MEDIA DONE ==="; exit 0; }
export HN_SCOPE=probe
crowd() { local f="$ROOT/results_figures/crowd_${1}_${2}.csv"; [ -f "$f" ] || python3 crowd_log.py --level $2 --tier $1 --seconds 200 --out "$f" >/dev/null; echo "$f"; }
EYE=corridor,nursedesk,ward,prayer,emergency,chase,nursedesk_east,robot_eye
for t in furnished walled open; do
  P=busy; [ "$t" = "open" ] && P=light
  S=$EYE; [ "$t" != "furnished" ] && S=corridor,chase,ward
  [ "$t" = "walled" ] && S=$S,door_ward,door_er
  LOGT=walled; [ "$t" = "open" ] && LOGT=open
  [ "$P" = "light" ] && PL="" || PL="--peds-log $(crowd $LOGT $P) --peds-at 120"
  echo "=== $(date +%T) gazebo 4K: $t eye-level ==="
  python3 -u capture_figures.py --tier $t --peds $P $PL --robot 33.0,14.9,3.1416 --ceiling --shots $S \
      --scale 2 --out "$HI" --settle 40 2>&1 | grep -E "saved|no frame|could not|figure"
  if [ "$t" = "furnished" ]; then      # close-up of the robot without passers-by in front of the lens
    python3 -u capture_figures.py --tier $t --peds none --robot 33.0,14.9,3.1416 --ceiling --shots robot_closeup \
        --scale 2 --out "$HI" --settle 30 2>&1 | grep -E "saved|no frame|could not"
  fi
  echo "=== $(date +%T) gazebo 4K: $t overview ==="
  python3 -u capture_figures.py --tier $t --peds $P $PL --robot 33.0,14.9,3.1416 --shots floorplan,oblique \
      --scale 2 --out "$HI" --settle 40 2>&1 | grep -E "saved|no frame|could not|figure"
done
echo "=== $(date +%T) gazebo 4K: 45 walkers (walled) ==="
python3 -u capture_figures.py --tier walled --peds crowd45 --peds-log "$(crowd walled crowd45)" --peds-at 120 \
    --robot 33.0,14.9,3.1416 --shots floorplan,oblique --scale 2 --out "$HI" --settle 40 --suffix _crowd45 \
    2>&1 | grep -E "saved|no frame|could not|figure"
python3 -u capture_figures.py --tier walled --peds crowd45 --peds-log "$(crowd walled crowd45)" --peds-at 120 \
    --robot 33.0,14.9,3.1416 --ceiling --shots corridor,chase --scale 2 --out "$HI" --settle 40 --suffix _crowd45 \
    2>&1 | grep -E "saved|no frame|could not|figure"
echo "=== $(date +%T) gazebo 4K: reference (the trained geometry, as the lidar sees it) ==="
python3 -u capture_figures.py --tier reference --peds none --robot 33.0,14.9,3.1416 --shots floorplan,oblique \
    --scale 2 --out "$HI" --settle 25 2>&1 | grep -E "saved|no frame|could not|figure"
echo "=== $(date +%T) ALL MEDIA DONE ==="
