# Hiwonder MentorPi meshes (not included)

`urdf/mentorpi_real.urdf` uses the vendor CAD meshes of the Hiwonder MentorPi Ackermann chassis. They are Hiwonder's files and are not redistributed in this repository.

To use the real robot model, copy these files from the MentorPi ROS 2 package supplied with the robot into this folder:

```
base_link.STL  cam_Link.STL  imu_link.stl  lidar_Link.STL
wheel_lb_Link.STL  wheel_lf_Link.STL  wheel_rb_Link.STL  wheel_rf_Link.STL
```

Without them, use the simplified robot model `urdf/hospitalnav_robot.urdf`, which needs no extra files.
