#!/bin/bash
# Reset the machine to "nothing running" before a sim run.
#
# A stale node makes the next run fail in confusing ways: a second
# /slam_toolbox loses its CONFIGURE transition, and a leftover bridge keeps
# publishing into the new run.
#
# SELF-PROTECTION. `pkill -f <pattern>` matches any process whose command line
# contains the pattern - including the shell that invoked this script, when the
# caller's own command line happens to mention one of these names. That killed
# the caller mid-script repeatedly (a sync would silently not happen, and the
# next build would quietly use stale sources). So this never calls pkill:
# it collects candidate PIDs, drops itself and every one of its ancestors, and
# only then kills.

self=$$
# SCOPE. Several simulators may legitimately run side by side (the campaign in
# one gz transport partition, a probe or figure capture in another). A process
# started with HN_SCOPE=<name> in its environment belongs to that scope, and
# this script - called as `clean_sim.sh <name>` - only kills processes of the
# same scope or with no scope at all (stale leftovers from older tooling).
scope="${1:-}"
in_scope() {
  local env_scope
  env_scope=$(tr '\0' '\n' < "/proc/$1/environ" 2>/dev/null | sed -n 's/^HN_SCOPE=//p' | head -1)
  [ -z "$env_scope" ] || [ -z "$scope" ] || [ "$env_scope" = "$scope" ]
}
# every ancestor of this script, so we can never kill the caller
ancestors=" "
p=$self
while [ -n "$p" ] && [ "$p" -gt 1 ] 2>/dev/null; do
  ancestors="$ancestors$p "
  p=$(awk '{print $4}' "/proc/$p/stat" 2>/dev/null)
done

PATTERNS=(
  "lib/hospitalnav_deploy/"
  "ros2 launch hospitalnav_deploy"
  "lib/slam_toolbox/"
  "lib/nav2_"
  "lib/ros_gz_bridge/"
  "lib/ros_gz_sim/"
  "lib/robot_state_publisher/"
  "lib/tf2_ros/"
)
EXACT=(gz-sim-server rviz2 ruby)

kill_pid() {
  case "$ancestors" in *" $1 "*) return;; esac
  [ "$1" = "$self" ] && return
  in_scope "$1" || return
  kill -9 "$1" 2>/dev/null
}

for pat in "${PATTERNS[@]}"; do
  for pid in $(pgrep -f -- "$pat" 2>/dev/null); do kill_pid "$pid"; done
done
for name in "${EXACT[@]}"; do
  for pid in $(pgrep -x -- "$name" 2>/dev/null); do kill_pid "$pid"; done
done

for _ in $(seq 20); do
  left=0
  for pid in $(pgrep -x gz-sim-server 2>/dev/null) $(pgrep -f -- "lib/slam_toolbox/" 2>/dev/null); do
    in_scope "$pid" && left=$((left + 1))
  done
  [ "$left" -eq 0 ] && break
  sleep 0.5
done

# SIGKILLed DDS participants leave shared-memory segments behind; the next run
# then fails to lock fastrtps_port7000 and discovery silently degrades - nodes
# start but never see each other's topics, which looks like "SLAM mapped almost
# nothing" rather than like a transport fault.
if ! pgrep -x gz-sim-server >/dev/null 2>&1; then
  rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_* 2>/dev/null
fi
exit 0
