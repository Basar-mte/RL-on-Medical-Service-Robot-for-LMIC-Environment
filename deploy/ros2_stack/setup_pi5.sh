#!/usr/bin/env bash
# One-time setup for the Raspberry Pi 5 (Ubuntu Server 22.04, arm64) OR your WSL
# Ubuntu 22.04 dev machine. Installs ROS 2 Humble + SLAM/Nav2 + the deps this
# package needs. Run:  bash setup_pi5.sh
set -e

echo "[1/5] locale + apt sources"
sudo apt update
sudo apt install -y locales curl gnupg lsb-release
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
     -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | \
     sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

echo "[2/5] ROS 2 Humble + SLAM + Nav2"
sudo apt update
sudo apt install -y ros-humble-ros-base python3-colcon-common-extensions \
     ros-humble-slam-toolbox ros-humble-navigation2 ros-humble-nav2-bringup \
     ros-humble-tf-transformations ros-humble-rviz2
# Optional off-the-shelf LiDAR driver (skip if using our self-contained lidar_node)
sudo apt install -y ros-humble-ldlidar-stl-ros2 || true

echo "[3/5] python deps"
python3 -m pip install --upgrade pip
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install stable-baselines3 numpy pillow pyserial

echo "[4/5] environment"
grep -q 'source /opt/ros/humble/setup.bash' ~/.bashrc || \
     echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
grep -q 'ROS_DOMAIN_ID' ~/.bashrc || echo 'export ROS_DOMAIN_ID=17' >> ~/.bashrc
grep -q 'ROS_LOCALHOST_ONLY' ~/.bashrc || echo 'export ROS_LOCALHOST_ONLY=0' >> ~/.bashrc

echo "[5/5] done. Open a NEW terminal, then:"
echo "   cd ~/hn_ws && colcon build && source install/setup.bash"
