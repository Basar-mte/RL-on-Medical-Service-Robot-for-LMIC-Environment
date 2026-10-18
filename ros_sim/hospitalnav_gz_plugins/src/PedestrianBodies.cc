// PedestrianBodies: walking pedestrians for the HospitalNav worlds - the
// actor everyone sees, the physical body the robot bumps into, and the ground
// truth the run is scored against, all from ONE state per walker.
//
// WALKERS
//   Every <actor> with a scripted trajectory is a walker on a closed loop
//   (ped_routes.py writes the loops as dense, equal-time keyframes). This
//   system does not let gz play the script by sim time; it owns each walker's
//   progress along it, tau (script seconds), and every step sets the actor's
//   trajectory pose and walk-cycle time itself (Actor::SetTrajectoryPose /
//   SetAnimationTime, which override the SDF script for rendering AND for the
//   gpu_lidar). A walker that is not blocked advances tau at exactly sim rate,
//   so it walks its loop on the same schedule as the scripted actor would.
//
// WALKING (<yield>true</yield>, the default): social force
//   Each walker heads for a point LEAD_S ahead on its own loop (the loop's
//   clock only runs while the walker keeps up, so one held up by the crowd
//   or the robot resumes its route where it left it) and is pushed by the
//   classic social forces of Helbing & Molnar (1995): relaxation toward its
//   own pace, exponential repulsion from every other person (anisotropic -
//   people react little to what is behind them), a stronger, wider one from
//   the robot, and repulsion from the walls of the tier's occupancy map (a
//   distance field). A small sideways share of a head-on repulsion breaks the
//   standoff a purely radial force leaves. People do not plan around the
//   robot; they give it room the way they give each other room - and slow,
//   stop, or step around when it is in the way. Positions never enter a
//   wall: a step into one slides along it.
//   A walker never puts its body inside the robot's footprint (the CAD
//   rectangle, rotated with the robot, plus its body radius): squeezed against
//   the robot or bumped by it, it gives way; resting contact is allowed.
//   With <yield>false</yield> walkers follow the original non-yielding schedule.
//
//   Why: measured 2026-10-17 with 45 NON-yielding walkers, the 2 kg robot was
//   knocked onto its side (tilt 90-180 deg) in 2 of 2 runs - an artifact of a
//   person modelled as an unstoppable 70 kg piston; and walkers on a fixed
//   schedule walk through each other (6 touching pairs at any instant).
//
// BODIES
//   Each walker has a gravity-free 70 kg collision cylinder "<actor>_body".
//   Each step it is commanded the velocity that tracks the walker - feed-
//   forward plus TAU-second correction of the position error - starting from
//   the velocity it ACTUALLY had over the last step (pose difference, i.e.
//   after contacts) and changing by at most A_MAX * dt: a free body follows to
//   a few cm, a blocked one pushes with a bounded force (~ 70 kg * A_MAX)
//   instead of the unbounded force of a re-asserted velocity (measured before
//   that limit: a crate caught between two walkers was squeezed out at about
//   45 m/s). Angular velocity is commanded to zero so a collision cannot tip
//   or spin it. A body TELEPORT_M behind its walker is put back onto it. The
//   bodies' collision bitmasks (ped_routes.py) let them pass through EACH
//   OTHER - the social force already keeps people apart, and a body shoved
//   off its actor would be an obstacle where the lidar sees nobody.
//
// OUTPUTS
//   <topic>: gz.msgs.Pose_V of every walker (name, x, y, yaw), stamped with
//   sim time, at PUBLISH_HZ - hospitalnav_deploy's ped_gt turns it into ROS
//   /pedestrian_poses, so ground truth is what was simulated, not a schedule.
//   <log>: CSV of t and every walker's x, y, yaw and walk-cycle time at
//   LOG_HZ, which <playback> replays exactly (the video renderer).
//
// WALK CYCLE
//   The Fuel walk clips are in-place cycles (gz warns "Animation has no x
//   displacement" and ignores <interpolate_x>), so a scripted actor's cycle
//   simply runs on sim time. Here it runs at the walker's current pace.
#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <gz/common/Console.hh>
#include <gz/math/Pose3.hh>
#include <gz/math/Vector2.hh>
#include <gz/msgs/pose_v.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/Actor.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/Actor.hh>
#include <gz/sim/components/AngularVelocityCmd.hh>
#include <gz/sim/components/LinearVelocityCmd.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/Pose.hh>
#include <gz/transport/Node.hh>
#include <sdf/Actor.hh>

namespace hospitalnav
{
using namespace gz;

// body drive
constexpr double TAU = 0.15;        // s, position-error time constant
constexpr double V_MAX = 2.5;       // m/s, fastest catch-up
constexpr double A_MAX = 12.0;      // m/s^2 -> push bounded at ~840 N for 70 kg
constexpr double TELEPORT_M = 1.5;  // m behind its walker before a body is put back
// walkers (social force, Helbing & Molnar 1995; per unit mass)
constexpr double BODY_R = 0.22;     // m, matches ped_routes.BODY_RADIUS
constexpr double LEAD_S = 1.0;      // s of route ahead the walker heads for
constexpr double LAG_OK = 0.4;      // m behind its route clock: the clock runs at full rate...
constexpr double LAG_MAX = 1.2;     // ...and stops at this lag, until the walker catches up
constexpr double TAU_R = 0.5;       // s, relaxation toward the desired velocity
constexpr double A_PED = 2.1;       // m/s^2, person-person repulsion strength
constexpr double B_PED = 0.30;      // m, its range
constexpr double A_ROBOT = 4.0;     // people keep a wider berth from a robot
constexpr double B_ROBOT = 0.35;
constexpr double A_WALL = 6.0;
constexpr double B_WALL = 0.12;
constexpr double LAMBDA = 0.35;     // anisotropy: weight of what is behind
constexpr double TANGENT = 0.35;    // sideways share of a head-on repulsion
constexpr double F_MAX = 12.0;      // m/s^2 cap on the summed force
constexpr double V_WALK_MAX = 1.8;  // m/s
constexpr double YAW_RATE = 8.0;    // rad/s, fastest heading change
constexpr double PUBLISH_HZ = 25.0;
constexpr double LOG_HZ = 20.0;

struct Walker
{
  std::string name;
  sim::Entity actor{sim::kNullEntity};
  sim::Entity body{sim::kNullEntity};
  std::vector<double> t;
  std::vector<math::Pose3d> p;
  double period{0.0};
  double speed{1.0};         // the walker's own walking pace (m/s)
  // state
  double tau{0.0};
  double rate{1.0};          // how fast the route clock runs (0..1)
  math::Vector2d carrot{0, 0};
  bool haveCarrot{false};
  math::Vector2d pos{0, 0};
  math::Vector2d vel{0, 0};  // m/s over the last step
  double yaw{0.0};
  bool havePose{false};
  double animT{0.0};         // walk-cycle time
  // body
  math::Vector3d prev;
  bool havePrev{false};
};

class PedestrianBodies : public sim::System,
                         public sim::ISystemConfigure,
                         public sim::ISystemPreUpdate
{
  public: void Configure(const sim::Entity &, const std::shared_ptr<const sdf::Element> &_sdf,
                         sim::EntityComponentManager &, sim::EventManager &) override
  {
    auto get = [&](const char *k, auto def) {
      return _sdf->HasElement(k) ? _sdf->Get<decltype(def)>(k) : def; };
    this->yield = get("yield", true);
    this->robotName = get("robot", std::string("mentorpi"));
    this->robotR = get("robot_radius", 0.18);
    this->robotHL = get("robot_half_length", 0.11);
    this->robotHW = get("robot_half_width", 0.09);
    this->topic = get("topic", std::string("/hospitalnav/pedestrians"));
    this->logPath = get("log", std::string());
    this->playPath = get("playback", std::string());
    const std::string map = get("map", std::string());
    if (!map.empty())
      this->LoadMap(map, get("map_origin", math::Vector2d(0, 0)), get("map_resolution", 0.05));
    // <phase walker="name">tau0</phase>: where on its loop a walker starts
    // (ped_routes.phases(level, seed); absent = the loop's start)
    auto el = std::const_pointer_cast<sdf::Element>(_sdf);
    for (auto e = el->FindElement("phase"); e; e = e->GetNextElement("phase"))
      if (e->HasAttribute("walker"))
        this->phase[e->GetAttribute("walker")->GetAsString()] = e->Get<double>();
    this->pub = this->node.Advertise<msgs::Pose_V>(this->topic);
    if (!this->playPath.empty())
      this->LoadPlayback(this->playPath);
  }

  public: void PreUpdate(const sim::UpdateInfo &_info, sim::EntityComponentManager &_ecm) override
  {
    if (!this->initialised)
      this->Init(_ecm);
    if (_info.paused)
      return;
    const double now = std::chrono::duration<double>(_info.simTime).count();
    const double dt = std::max(1e-4, std::chrono::duration<double>(_info.dt).count());
    this->simNow = now;

    if (!this->play.empty())
    {
      this->Playback(_ecm, now);
      return;
    }

    bool haveRobot = false;
    math::Vector2d robot;
    if (this->robot == sim::kNullEntity)
      this->robot = _ecm.EntityByComponents(sim::components::Model(),
                                            sim::components::Name(this->robotName));
    if (this->robot != sim::kNullEntity)
      if (const auto *rp = _ecm.Component<sim::components::Pose>(this->robot))
      {
        robot.Set(rp->Data().Pos().X(), rp->Data().Pos().Y());
        this->robotYaw = rp->Data().Rot().Yaw();
        haveRobot = true;
      }

    for (size_t i = 0; i < this->walkers.size(); ++i)
    {
      Walker &w = this->walkers[i];
      this->StepWalker(w, i, dt, haveRobot, robot);
      this->ApplyActor(_ecm, w);
      this->DriveBody(_ecm, w, dt);
    }
    if (now - this->lastPub >= 1.0 / PUBLISH_HZ - 1e-9)
    {
      this->lastPub = now;
      this->Publish(now);
    }
    if (this->log && now - this->lastLog >= 1.0 / LOG_HZ - 1e-9)
    {
      this->lastLog = now;
      *this->log << std::fixed;
      this->log->precision(3);
      *this->log << now;
      for (const auto &w : this->walkers)
        *this->log << ',' << w.pos.X() << ',' << w.pos.Y() << ',' << w.yaw << ',' << w.animT;
      *this->log << '\n';
      this->log->flush();
    }
  }

  // ---- walkers ------------------------------------------------------------
  private: void StepWalker(Walker &w, size_t idx, double dt, bool haveRobot, const math::Vector2d &robot)
  {
    if (!this->yield)
    {
      // the original non-yielding schedule
      w.tau = w.period > 0.0 ? std::fmod(w.tau + dt, w.period) : w.tau + dt;
      const math::Vector2d pos = this->At2(w, w.tau);
      w.vel = w.havePose ? (pos - w.pos) / dt : math::Vector2d(0, 0);
      w.pos = pos;
      w.yaw = this->Yaw(w, w.tau);
      w.havePose = true;
      w.rate = 1.0;
      w.animT += dt;
      return;
    }
    if (!w.havePose)
    {
      w.pos = this->At2(w, w.tau);
      w.yaw = this->Yaw(w, w.tau);
      w.havePose = true;
    }
    // ---- the route: a carrot LEAD_S ahead on the loop; the loop's clock
    // tau only runs while the walker keeps up with it, so a walker held up by
    // the crowd or the robot resumes its route where it left it
    const math::Vector2d base = this->At2(w, w.tau);
    const double lag = (w.pos - base).Length();
    w.rate = std::clamp((LAG_MAX - lag) / (LAG_MAX - LAG_OK), 0.0, 1.0);
    w.tau += w.rate * dt;
    if (w.period > 0.0)
      w.tau = std::fmod(w.tau, w.period);
    // the carrot must be in plain sight: the farthest route point up to LEAD_S
    // ahead with a clear straight line to it (heading for a point behind a
    // wall corner would pin a walker against the wall - measured: 10 s stuck
    // in a doorway)
    math::Vector2d carrot = w.carrot;
    bool seen = false;
    for (double lead = LEAD_S; lead > -1e-9 && !seen; lead -= LEAD_S / 4)
    {
      const math::Vector2d c = this->At2(w, w.tau + lead);
      if (this->SegFree(w.pos, c, BODY_R * 0.9))
      {
        carrot = c;
        seen = true;
      }
    }
    if (!seen && !w.haveCarrot)
      carrot = base;
    w.carrot = carrot;
    w.haveCarrot = true;
    math::Vector2d toC = carrot - w.pos;
    const double dC = toC.Length();
    const math::Vector2d e = dC > 1e-3 ? toC / dC : math::Vector2d(std::cos(w.yaw), std::sin(w.yaw));
    // desired speed: the walker's own pace, easing off as it closes on the
    // carrot (which stands still while the loop pauses at a corner)
    const double v0 = w.speed * std::clamp(dC / (w.speed * LEAD_S), 0.0, 1.0);

    // ---- social force (Helbing & Molnar 1995), per unit mass
    math::Vector2d F = (e * v0 - w.vel) / TAU_R;
    auto repel = [&](const math::Vector2d &other, double reach, double A, double B)
    {
      math::Vector2d d = w.pos - other;
      const double dist = d.Length();
      if (dist > 4.0 || dist < 1e-6)
        return;
      const math::Vector2d n = d / dist;
      // anisotropy: people react far less to what is behind them
      const double cosphi = -n.Dot(e);
      const double aniso = LAMBDA + (1.0 - LAMBDA) * (1.0 + cosphi) / 2.0;
      const double mag = A * std::exp((reach - dist) / B) * aniso;
      F += n * mag;
      // a small sideways push breaks the head-on standoff a purely radial
      // force leaves (both would stop dead): pass on whichever side the
      // other already is NOT, or to the right if it is dead ahead
      if (cosphi > 0.5)
      {
        const math::Vector2d left(-e.Y(), e.X());
        const double side = (other - w.pos).Dot(left) > 0.02 ? -1.0 : 1.0;
        F += left * (side * TANGENT * mag);
      }
    };
    for (size_t k = 0; k < this->walkers.size(); ++k)
      if (k != idx && this->walkers[k].havePose)
        repel(this->walkers[k].pos, 2 * BODY_R, A_PED, B_PED);
    if (haveRobot)
      repel(robot, BODY_R + this->robotR, A_ROBOT, B_ROBOT);
    if (!this->dist.empty())
    {
      math::Vector2d g;
      const double dw = this->WallDist(w.pos, g);
      if (dw < 1.0)
      {
        // walls push sideways, never back against where the walker is going:
        // a jamb corner's repulsion would otherwise cancel a slow walker's drive
        // into a doorway (measured: a 0.52 m/s patient held 26 s in a door)
        math::Vector2d fw = g * (A_WALL * std::exp((BODY_R - dw) / B_WALL));
        const double back = fw.Dot(e);
        if (back < 0.0)
          fw -= e * back;
        F += fw;
      }
    }
    if (!this->debugName.empty() && w.name == this->debugName && std::fmod(this->simNow, 1.0) < dt * 0.99)
      gzmsg << "DBG " << w.name << " t=" << this->simNow << " pos=" << w.pos << " base=" << base
            << " carrot=" << carrot << " seen=" << seen << " v0=" << v0 << " vel=" << w.vel
            << " F=" << F << " rate=" << w.rate << " wall=" << (this->dist.empty() ? -1.0 : [&]{ math::Vector2d g; return this->WallDist(w.pos, g); }())
            << std::endl;
    if (F.Length() > F_MAX)
      F = F.Normalized() * F_MAX;
    w.vel += F * dt;
    const double vmax = std::min(1.3 * w.speed + 0.1, V_WALK_MAX);
    if (w.vel.Length() > vmax)
      w.vel = w.vel.Normalized() * vmax;
    // never step into a wall: slide along it instead
    math::Vector2d np = w.pos + w.vel * dt;
    if (!this->Free(np, BODY_R * 0.8))
    {
      const math::Vector2d nx(np.X(), w.pos.Y()), ny(w.pos.X(), np.Y());
      if (this->Free(nx, BODY_R * 0.8))
      { np = nx; w.vel.Y(0); }
      else if (this->Free(ny, BODY_R * 0.8))
      { np = ny; w.vel.X(0); }
      else
      { np = w.pos; w.vel.Set(0, 0); }
    }
    // A walker never puts its body INSIDE the robot's footprint (rectangle
    // + body radius, rotated with the robot): squeezed against it, or bumped
    // by it, the walker gives way instead. Without this the walker's 70 kg
    // body, driven toward a position overlapping the 2 kg robot, pushed it
    // with up to ~840 N and toppled it - measured 2026-10-17: 2 of 480 runs,
    // a walker squeezing past a stopped robot rolled it onto its back.
    // Resting contact (a walker against the robot's side) is still allowed.
    if (haveRobot)
    {
      const double c = std::cos(this->robotYaw), sn = std::sin(this->robotYaw);
      const math::Vector2d rel = np - robot;
      const math::Vector2d loc(c * rel.X() + sn * rel.Y(), -sn * rel.X() + c * rel.Y());
      const math::Vector2d near(std::clamp(loc.X(), -this->robotHL, this->robotHL),
                                std::clamp(loc.Y(), -this->robotHW, this->robotHW));
      math::Vector2d out = loc - near;
      double d = out.Length();
      if (d < BODY_R)
      {
        if (d < 1e-6)
        {
          // centre inside the rectangle: leave by the nearest side
          const double gx = this->robotHL - std::abs(loc.X()), gy = this->robotHW - std::abs(loc.Y());
          out = gx < gy ? math::Vector2d(loc.X() >= 0 ? 1 : -1, 0) : math::Vector2d(0, loc.Y() >= 0 ? 1 : -1);
          d = 0.0;
        }
        else
          out /= d;
        const math::Vector2d fixed = near + out * BODY_R;
        const math::Vector2d nw(robot.X() + c * fixed.X() - sn * fixed.Y(), robot.Y() + sn * fixed.X() + c * fixed.Y());
        if (this->Free(nw, BODY_R * 0.8))
        {
          np = nw;
          const math::Vector2d nWorld(c * out.X() - sn * out.Y(), sn * out.X() + c * out.Y());
          const double into = w.vel.Dot(nWorld);
          if (into < 0.0)
            w.vel -= nWorld * into;                  // no velocity into the robot
        }
      }
    }
    w.pos = np;
    // heading: along the motion when walking, the script's at a standstill
    const double sp = w.vel.Length();
    const double yawT = sp > 0.15 ? std::atan2(w.vel.Y(), w.vel.X()) : this->Yaw(w, w.tau);
    w.yaw += std::clamp(std::remainder(yawT - w.yaw, 2 * M_PI), -YAW_RATE * dt, YAW_RATE * dt);
    // The walk clips are in-place cycles (no root motion - gz itself ignores
    // <interpolate_x> for them), so a scripted actor's cycle runs on sim time.
    // Here it runs at the walker's pace: full speed animates exactly as the
    // scripted actor did; a slowing walker's stride slows to a standstill.
    w.animT += dt * std::clamp(sp / std::max(w.speed, 0.3), 0.0, 1.0);
  }

  // distance (m) to the nearest map obstacle and the unit gradient away from it
  private: double WallDist(const math::Vector2d &_p, math::Vector2d &_g) const
  {
    auto at = [&](double x, double y)
    {
      int c = std::clamp(static_cast<int>(std::floor((x - this->origin.X()) / this->res)), 0, this->gw - 1);
      int r = std::clamp(static_cast<int>(std::floor((y - this->origin.Y()) / this->res)), 0, this->gh - 1);
      return this->dist[static_cast<size_t>(r) * this->gw + c];
    };
    const double h = this->res;
    const double d = at(_p.X(), _p.Y());
    _g.Set(at(_p.X() + h, _p.Y()) - at(_p.X() - h, _p.Y()), at(_p.X(), _p.Y() + h) - at(_p.X(), _p.Y() - h));
    if (_g.Length() > 1e-9)
      _g.Normalize();
    return d;
  }

  private: void ApplyActor(sim::EntityComponentManager &_ecm, const Walker &w)
  {
    sim::Actor a(w.actor);
    a.SetTrajectoryPose(_ecm, math::Pose3d(w.pos.X(), w.pos.Y(), 0.0, 0, 0, w.yaw));
    a.SetAnimationTime(_ecm, std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                               std::chrono::duration<double>(w.animT)));
  }

  private: void DriveBody(sim::EntityComponentManager &_ecm, Walker &w, double dt)
  {
    if (w.body == sim::kNullEntity)
      return;
    const auto *pc = _ecm.Component<sim::components::Pose>(w.body);
    if (!pc)
      return;
    const math::Pose3d cur = pc->Data();
    const math::Vector3d p0(w.pos.X(), w.pos.Y(), 0.0);
    const math::Vector3d vff(w.vel.X(), w.vel.Y(), 0.0);
    const math::Vector3d err = p0 - cur.Pos();
    math::Vector3d vcmd;
    if (err.Length() > TELEPORT_M)
    {
      sim::Model(w.body).SetWorldPoseCmd(_ecm, math::Pose3d(p0, cur.Rot()));
      w.prev = p0;
      vcmd = vff;
    }
    else
    {
      const math::Vector3d vact = w.havePrev ? (cur.Pos() - w.prev) / dt : math::Vector3d::Zero;
      math::Vector3d vdes = vff + err / TAU;
      if (vdes.Length() > V_MAX)
        vdes = vdes.Normalized() * V_MAX;
      math::Vector3d dv = vdes - vact;
      if (dv.Length() > A_MAX * dt)
        dv = dv.Normalized() * (A_MAX * dt);
      vcmd = vact + dv;
      w.prev = cur.Pos();
    }
    w.havePrev = true;
    // the command is in the body frame: rotate the world velocity into it
    _ecm.SetComponentData<sim::components::LinearVelocityCmd>(w.body, cur.Rot().RotateVectorReverse(vcmd));
    _ecm.SetComponentData<sim::components::AngularVelocityCmd>(w.body, math::Vector3d::Zero);
  }

  private: void Publish(double now)
  {
    msgs::Pose_V msg;
    msg.mutable_header()->mutable_stamp()->set_sec(static_cast<int64_t>(std::floor(now)));
    msg.mutable_header()->mutable_stamp()->set_nsec(static_cast<int32_t>((now - std::floor(now)) * 1e9));
    for (const auto &w : this->walkers)
    {
      auto *p = msg.add_pose();
      p->set_name(w.name);
      p->mutable_position()->set_x(w.pos.X());
      p->mutable_position()->set_y(w.pos.Y());
      p->mutable_orientation()->set_z(std::sin(w.yaw / 2));
      p->mutable_orientation()->set_w(std::cos(w.yaw / 2));
    }
    this->pub.Publish(msg);
  }

  // ---- playback (video replay of a logged run) -----------------------------
  private: void LoadPlayback(const std::string &_path)
  {
    std::ifstream f(_path);
    std::string line;
    if (!std::getline(f, line))
    {
      gzerr << "PedestrianBodies: cannot read playback log " << _path << std::endl;
      return;
    }
    std::stringstream hs(line);
    std::string col;
    std::getline(hs, col, ',');                       // "t"
    while (std::getline(hs, col, ','))
      if (col.size() > 2 && col.substr(col.size() - 2) == "_x")
        this->playNames.push_back(col.substr(0, col.size() - 2));
    while (std::getline(f, line))
    {
      std::stringstream ls(line);
      std::vector<double> row;
      while (std::getline(ls, col, ','))
        row.push_back(std::stod(col));
      if (row.size() == 1 + 4 * this->playNames.size())
        this->play.push_back(row);
    }
    gzmsg << "PedestrianBodies: playback of " << this->playNames.size() << " walkers, "
          << this->play.size() << " samples from " << _path << std::endl;
  }

  private: void Playback(sim::EntityComponentManager &_ecm, double now)
  {
    auto it = std::lower_bound(this->play.begin(), this->play.end(), now,
                               [](const std::vector<double> &r, double t) { return r[0] < t; });
    size_t i = static_cast<size_t>(std::distance(this->play.begin(), it));
    i = std::clamp<size_t>(i, 1, this->play.size() - 1);
    const auto &a = this->play[i - 1];
    const auto &b = this->play[i];
    const double f = std::clamp(b[0] > a[0] ? (now - a[0]) / (b[0] - a[0]) : 0.0, 0.0, 1.0);
    for (size_t k = 0; k < this->playNames.size(); ++k)
    {
      auto wi = this->byName.find(this->playNames[k]);
      if (wi == this->byName.end())
        continue;
      Walker &w = this->walkers[wi->second];
      const size_t c = 1 + 4 * k;
      w.pos.Set(a[c] + f * (b[c] - a[c]), a[c + 1] + f * (b[c + 1] - a[c + 1]));
      w.yaw = a[c + 2] + f * std::remainder(b[c + 2] - a[c + 2], 2 * M_PI);
      // the walk cycle wraps: interpolate only within a cycle
      w.animT = b[c + 3] >= a[c + 3] ? a[c + 3] + f * (b[c + 3] - a[c + 3]) : (f < 0.5 ? a[c + 3] : b[c + 3]);
      this->ApplyActor(_ecm, w);
    }
  }

  // ---- setup ----------------------------------------------------------------
  private: void Init(sim::EntityComponentManager &_ecm)
  {
    _ecm.Each<sim::components::Actor, sim::components::Name>(
      [&](const sim::Entity &_e, const sim::components::Actor *_a, const sim::components::Name *_n)
      {
        const sdf::Actor &sa = _a->Data();
        if (sa.TrajectoryCount() == 0)
          return true;
        const sdf::Trajectory *tj = sa.TrajectoryByIndex(0);
        if (tj->WaypointCount() < 2)
          return true;
        Walker w;
        w.name = _n->Data();
        w.actor = _e;
        for (uint64_t i = 0; i < tj->WaypointCount(); ++i)
        {
          w.t.push_back(tj->WaypointByIndex(i)->Time());
          w.p.push_back(tj->WaypointByIndex(i)->Pose());
        }
        w.period = w.t.back();
        if (auto ph = this->phase.find(w.name); ph != this->phase.end() && w.period > 0.0)
          w.tau = std::fmod(ph->second, w.period);
        // the walker's pace: its route's walking speed (legs are constant-speed)
        double vmax = 0.0;
        for (size_t i = 1; i < w.t.size(); ++i)
          if (w.t[i] > w.t[i - 1])
            vmax = std::max(vmax, (w.p[i].Pos() - w.p[i - 1].Pos()).Length() / (w.t[i] - w.t[i - 1]));
        w.speed = vmax > 0.05 ? vmax : 1.0;
        w.body = _ecm.EntityByComponents(sim::components::Model(),
                                         sim::components::Name(w.name + "_body"));
        sim::Actor(_e).SetAnimationName(_ecm, sa.AnimationCount() ? sa.AnimationByIndex(0)->Name() : "walk");
        this->byName[w.name] = this->walkers.size();
        this->walkers.push_back(w);
        return true;
      });
    int bodies = 0;
    for (const auto &w : this->walkers)
      bodies += w.body != sim::kNullEntity;
    gzmsg << "PedestrianBodies: " << this->walkers.size() << " walkers, " << bodies
          << " with a physical body, " << this->phase.size() << " seeded start phases; yield="
          << (this->yield ? "on" : "off") << ", map=" << (this->grid.empty() ? "none" : "loaded")
          << ", poses on " << this->topic << std::endl;
    if (!this->logPath.empty())
    {
      this->log = std::make_unique<std::ofstream>(this->logPath);
      *this->log << "t";
      for (const auto &w : this->walkers)
        *this->log << ',' << w.name << "_x," << w.name << "_y," << w.name << "_yaw," << w.name << "_anim";
      *this->log << '\n';
    }
    this->initialised = true;
  }

  // ---- script interpolation -------------------------------------------------
  private: size_t Seg(const Walker &w, double u, double &f) const
  {
    if (w.period > 0.0)
      u = std::fmod(std::fmod(u, w.period) + w.period, w.period);
    auto it = std::upper_bound(w.t.begin(), w.t.end(), u);
    size_t i = static_cast<size_t>(std::distance(w.t.begin(), it));
    i = std::clamp<size_t>(i, 1, w.t.size() - 1);
    const double t0 = w.t[i - 1], t1 = w.t[i];
    f = t1 > t0 ? std::clamp((u - t0) / (t1 - t0), 0.0, 1.0) : 0.0;
    return i;
  }

  private: math::Vector2d At2(const Walker &w, double u) const
  {
    double f;
    const size_t i = this->Seg(w, u, f);
    const auto &a = w.p[i - 1].Pos();
    const auto &b = w.p[i].Pos();
    return math::Vector2d(a.X() + f * (b.X() - a.X()), a.Y() + f * (b.Y() - a.Y()));
  }

  private: double Yaw(const Walker &w, double u) const
  {
    double f;
    const size_t i = this->Seg(w, u, f);
    const double a = w.p[i - 1].Rot().Yaw(), b = w.p[i].Rot().Yaw();
    return a + f * std::remainder(b - a, 2 * M_PI);
  }

  // ---- occupancy map (where a sidestep may go) --------------------------------
  private: void LoadMap(const std::string &_pgm, const math::Vector2d &_origin, double _res)
  {
    std::ifstream f(_pgm, std::ios::binary);
    std::string magic;
    f >> magic;
    auto skip = [&]() { f >> std::ws; while (f.peek() == '#') { std::string c; std::getline(f, c); f >> std::ws; } };
    int w = 0, h = 0, maxv = 0;
    skip(); f >> w; skip(); f >> h; skip(); f >> maxv;
    f.get();
    if (magic != "P5" || w <= 0 || h <= 0)
    {
      gzerr << "PedestrianBodies: cannot read map " << _pgm << std::endl;
      return;
    }
    std::vector<unsigned char> raw(static_cast<size_t>(w) * h);
    f.read(reinterpret_cast<char *>(raw.data()), raw.size());
    this->grid.assign(raw.size(), 0);
    for (int r = 0; r < h; ++r)              // PGM row 0 is the TOP (max y)
      for (int c = 0; c < w; ++c)
        this->grid[static_cast<size_t>(h - 1 - r) * w + c] = raw[static_cast<size_t>(r) * w + c] < 128;
    this->gw = w;
    this->gh = h;
    this->origin = _origin;
    this->res = _res;
    // distance to the nearest obstacle (two-pass chamfer 3-4, in metres) for
    // the walls' social force
    const float big = 1e9f;
    std::vector<float> d(this->grid.size(), big);
    for (size_t i = 0; i < d.size(); ++i)
      if (this->grid[i])
        d[i] = 0.f;
    auto idx = [&](int r, int c) { return static_cast<size_t>(r) * w + c; };
    for (int r = 0; r < h; ++r)
      for (int c = 0; c < w; ++c)
      {
        float &v = d[idx(r, c)];
        if (c > 0) v = std::min(v, d[idx(r, c - 1)] + 3.f);
        if (r > 0) v = std::min(v, d[idx(r - 1, c)] + 3.f);
        if (r > 0 && c > 0) v = std::min(v, d[idx(r - 1, c - 1)] + 4.f);
        if (r > 0 && c < w - 1) v = std::min(v, d[idx(r - 1, c + 1)] + 4.f);
      }
    for (int r = h - 1; r >= 0; --r)
      for (int c = w - 1; c >= 0; --c)
      {
        float &v = d[idx(r, c)];
        if (c < w - 1) v = std::min(v, d[idx(r, c + 1)] + 3.f);
        if (r < h - 1) v = std::min(v, d[idx(r + 1, c)] + 3.f);
        if (r < h - 1 && c < w - 1) v = std::min(v, d[idx(r + 1, c + 1)] + 4.f);
        if (r < h - 1 && c > 0) v = std::min(v, d[idx(r + 1, c - 1)] + 4.f);
      }
    this->dist.resize(d.size());
    for (size_t i = 0; i < d.size(); ++i)
      this->dist[i] = d[i] / 3.f * static_cast<float>(_res);
  }

  private: bool SegFree(const math::Vector2d &_a, const math::Vector2d &_b, double _r) const
  {
    if (this->grid.empty())
      return true;
    const double L = (_b - _a).Length();
    const int n = std::max(1, static_cast<int>(std::ceil(L / 0.1)));
    for (int k = 1; k <= n; ++k)
      if (!this->Free(_a + (_b - _a) * (static_cast<double>(k) / n), _r))
        return false;
    return true;
  }

  private: bool Free(const math::Vector2d &_p, double _r) const
  {
    if (this->grid.empty())
      return true;                            // open tier / no map: anywhere
    for (int k = -1; k < 8; ++k)
    {
      const double a = k * M_PI / 4;
      const math::Vector2d q = k < 0 ? _p : _p + math::Vector2d(std::cos(a), std::sin(a)) * _r;
      const int c = static_cast<int>(std::floor((q.X() - this->origin.X()) / this->res));
      const int r = static_cast<int>(std::floor((q.Y() - this->origin.Y()) / this->res));
      if (c < 0 || r < 0 || c >= this->gw || r >= this->gh || this->grid[static_cast<size_t>(r) * this->gw + c])
        return false;
    }
    return true;
  }

  private: bool initialised{false};
  private: std::vector<Walker> walkers;
  private: std::map<std::string, size_t> byName;
  private: std::map<std::string, double> phase;
  private: bool yield{true};
  private: std::string robotName;
  private: sim::Entity robot{sim::kNullEntity};
  private: double robotR{0.18};
  private: double robotYaw{0.0};
  private: double robotHL{0.11};   // m, half-length of the robot's footprint (chassis CAD -0.104..0.110)
  private: double robotHW{0.09};   // m, half-width (wheels to 0.087, steering margin)
  private: std::string topic;
  private: transport::Node node;
  private: transport::Node::Publisher pub;
  private: double lastPub{-1e9};
  private: std::string logPath;
  private: std::unique_ptr<std::ofstream> log;
  private: double lastLog{-1e9};
  private: std::string playPath;
  private: std::vector<std::string> playNames;
  private: std::vector<std::vector<double>> play;
  private: double simNow{0.0};
  private: std::string debugName{getenv("HN_PED_DEBUG") ? getenv("HN_PED_DEBUG") : ""};
  private: std::vector<char> grid;
  private: std::vector<float> dist;
  private: int gw{0}, gh{0};
  private: math::Vector2d origin{0, 0};
  private: double res{0.05};
};
}  // namespace hospitalnav

GZ_ADD_PLUGIN(hospitalnav::PedestrianBodies, gz::sim::System,
              hospitalnav::PedestrianBodies::ISystemConfigure,
              hospitalnav::PedestrianBodies::ISystemPreUpdate)
GZ_ADD_PLUGIN_ALIAS(hospitalnav::PedestrianBodies, "hospitalnav::PedestrianBodies")
