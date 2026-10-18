// ReplayRig: poses models from recorded tracks, every physics step, inside the
// simulator - the video renderer's robot and camera.
//
//   <plugin filename="hospitalnav_replay_rig" name="hospitalnav::ReplayRig">
//     <track><model>mentorpi</model><csv>/path/robot.csv</csv></track>
//     <track><model>vidcam</model><csv>/path/camera.csv</csv></track>
//   </plugin>
//
// Each CSV is "t,x,y,z,qw,qx,qy,qz" (sim time, world pose), sorted by t. At
// every PreUpdate each model gets the pose interpolated at the current sim
// time (linear position, slerp orientation; clamped outside the track) as a
// world-pose command, so a rendering sensor that updates in the same step
// renders exactly those poses. Posing them from outside instead - a gz
// service call per frame and a lock-stepped world - lost requests and dropped
// frames (measured: 55 of 112, and a render frozen for hours on a lost step).
#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gz/common/Console.hh>
#include <gz/math/Pose3.hh>
#include <gz/plugin/Register.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/components/AngularVelocityCmd.hh>
#include <gz/sim/components/LinearVelocityCmd.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/Static.hh>

namespace hospitalnav
{
using namespace gz;

struct RigTrack
{
  std::string model;
  sim::Entity entity{sim::kNullEntity};
  std::vector<double> t;
  std::vector<math::Pose3d> p;
};

class ReplayRig : public sim::System, public sim::ISystemConfigure, public sim::ISystemPreUpdate
{
  public: void Configure(const sim::Entity &, const std::shared_ptr<const sdf::Element> &_sdf,
                         sim::EntityComponentManager &, sim::EventManager &) override
  {
    auto sdf = std::const_pointer_cast<sdf::Element>(_sdf);
    for (auto e = sdf->FindElement("track"); e; e = e->GetNextElement("track"))
    {
      RigTrack tr;
      tr.model = e->Get<std::string>("model");
      const std::string path = e->Get<std::string>("csv");
      std::ifstream f(path);
      std::string line;
      std::getline(f, line);                          // header
      while (std::getline(f, line))
      {
        std::stringstream ls(line);
        std::string c;
        std::vector<double> v;
        while (std::getline(ls, c, ','))
          v.push_back(std::stod(c));
        if (v.size() < 8)
          continue;
        tr.t.push_back(v[0]);
        tr.p.emplace_back(math::Vector3d(v[1], v[2], v[3]), math::Quaterniond(v[4], v[5], v[6], v[7]));
      }
      gzmsg << "ReplayRig: " << tr.model << " <- " << tr.t.size() << " samples from " << path << std::endl;
      if (tr.t.size() >= 2)
        this->tracks.push_back(tr);
    }
  }

  public: void PreUpdate(const sim::UpdateInfo &_info, sim::EntityComponentManager &_ecm) override
  {
    const double now = std::chrono::duration<double>(_info.simTime).count();
    for (auto &tr : this->tracks)
    {
      if (tr.entity == sim::kNullEntity)
        tr.entity = _ecm.EntityByComponents(sim::components::Model(), sim::components::Name(tr.model));
      if (tr.entity == sim::kNullEntity)
        continue;
      auto it = std::lower_bound(tr.t.begin(), tr.t.end(), now);
      size_t i = static_cast<size_t>(std::distance(tr.t.begin(), it));
      i = std::clamp<size_t>(i, 1, tr.t.size() - 1);
      const double t0 = tr.t[i - 1], t1 = tr.t[i];
      const double f = std::clamp(t1 > t0 ? (now - t0) / (t1 - t0) : 0.0, 0.0, 1.0);
      const auto &a = tr.p[i - 1];
      const auto &b = tr.p[i];
      const math::Pose3d pose(a.Pos() + (b.Pos() - a.Pos()) * f,
                              math::Quaterniond::Slerp(f, a.Rot(), b.Rot(), true));
      sim::Model(tr.entity).SetWorldPoseCmd(_ecm, pose);
      // a posed dynamic model must also carry no velocity: a pose command
      // resets where it is, not how fast it is going, and anything that
      // accelerates it (measured: gravity on the replay robot's wheels)
      // accumulates step after step until the solver blows up and the model
      // stops rendering - the robot vanished ~29 s into every replay
      const auto *st = _ecm.Component<sim::components::Static>(tr.entity);
      if (!st || !st->Data())
      {
        _ecm.SetComponentData<sim::components::LinearVelocityCmd>(tr.entity, math::Vector3d::Zero);
        _ecm.SetComponentData<sim::components::AngularVelocityCmd>(tr.entity, math::Vector3d::Zero);
      }
    }
  }

  private: std::vector<RigTrack> tracks;
};
}  // namespace hospitalnav

GZ_ADD_PLUGIN(hospitalnav::ReplayRig, gz::sim::System,
              hospitalnav::ReplayRig::ISystemConfigure,
              hospitalnav::ReplayRig::ISystemPreUpdate)
GZ_ADD_PLUGIN_ALIAS(hospitalnav::ReplayRig, "hospitalnav::ReplayRig")
