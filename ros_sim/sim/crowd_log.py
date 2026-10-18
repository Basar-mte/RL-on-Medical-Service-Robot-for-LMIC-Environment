"""Records a robot-free walk of a pedestrian level (PedestrianBodies' social-
force walkers, the same plugin the runs use) to a peds.csv-format log - the
crowd that stills (capture_figures.py --peds-log) and crowd statistics use.

    python3 crowd_log.py --level crowd45 --tier walled --seconds 200 --out ../results_figures/crowd_walled_crowd45.csv
"""
import argparse
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, "..", "hospitalnav_deploy"))
sys.path.insert(0, os.path.join(PKG, "hospitalnav_deploy"))
import ped_routes as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="crowd45")
    ap.add_argument("--tier", default="walled")
    ap.add_argument("--seconds", type=float, default=200.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    sdf = open(os.path.join(PKG, "worlds", f"hospital_{a.tier}.world")).read()
    mp = os.path.join(PKG, "worlds", "hospital_map_%s.yaml" % ("open" if a.tier == "open" else "walled"))
    sdf = P.inject_actors(sdf, a.level, map_yaml=mp, log=out)
    tmp = tempfile.mkdtemp(prefix="crowdlog_")
    world = os.path.join(tmp, "crowd.world")
    open(world, "w").write(sdf)
    env = dict(os.environ, GZ_PARTITION=f"crowdlog_{os.getpid()}", HN_SCOPE="probe",
               GZ_SIM_RESOURCE_PATH=os.pathsep.join([PKG, os.path.dirname(PKG)]))
    steps = int(round(a.seconds / 0.01))
    subprocess.run(["gz", "sim", "-s", "-r", "-v1", "--iterations", str(steps), world], env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=3600)
    print(f"wrote {out}: {a.level} on {a.tier}, {a.seconds:.0f} s")


if __name__ == "__main__":
    main()
