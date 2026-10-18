"""Thermal guard for long training campaigns (Windows, no admin rights needed).

Real CPU core temperatures need admin rights, so the guard watches what IS
readable and reliable:
  * '% Processor Performance' — CPU clock relative to nominal. Under full load a
    healthy Core Ultra 9 runs ~140% (turbo); thermal/power throttling pulls it
    below 100%. This is the primary overheating signal.
  * Thermal-zone 'Throttle Reasons' (non-zero = platform is throttling) and the
    ACPI thermal-zone temperature (a board sensor, not the CPU core).
  * GPU temperature via nvidia-smi.

Levels (checked every SAMPLE_SEC by the campaign):
  WARNING  : clock < 100% under load, any throttle flag, zone > 85 C, GPU > 83 C
             -> log + dashboard + desktop pop-up (rate-limited)
  CRITICAL : clock < 75% for 3 samples in a row, or throttle flag 3 in a row,
             or zone > 95 C, or GPU > 90 C
             -> campaign sheds STEP runs (newest checkpoints first; they resume later)
  RECOVERY : RECOVER_SAMPLES consecutive clean samples -> restore STEP runs
Every sample is appended to <runs>/thermal.csv; status in <runs>/thermal_status.json.
"""
from __future__ import annotations
import json, os, subprocess, time
from pathlib import Path

SAMPLE_SEC = 60
STEP, MAX_SHED = 4, 12
RECOVER_SAMPLES = 30
POPUP_EVERY_S = 2 * 3600


def _ps(cmd: str) -> str:
    try:
        return subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True,
                              text=True, timeout=30).stdout
    except Exception:
        return ""


def sample() -> dict:
    out = _ps("$c = Get-Counter '\\Processor Information(_Total)\\% Processor Performance',"
              "'\\Processor(_Total)\\% Processor Time','\\Thermal Zone Information(*)\\Throttle Reasons',"
              "'\\Thermal Zone Information(*)\\Temperature' -ErrorAction SilentlyContinue; "
              "$c.CounterSamples | ForEach-Object { $_.Path + '=' + $_.CookedValue }")
    s = dict(perf=None, util=None, throttle=0.0, zone_c=None, gpu_c=None)
    for line in out.splitlines():
        if "=" not in line:
            continue
        path, val = line.rsplit("=", 1)
        try:
            v = float(val)
        except ValueError:
            continue
        p = path.lower()
        if "% processor performance" in p:
            # A 0 reading is a counter glitch (seen ~1-3x/day), not a 0 % clock:
            # treat as missing rather than raising a false WARNING/CRITICAL.
            s["perf"] = v if v > 0 else None
        elif "% processor time" in p:
            s["util"] = v
        elif "throttle reasons" in p:
            s["throttle"] = max(s["throttle"], v)
        elif "thermal zone information" in p and p.endswith("\\temperature"):
            s["zone_c"] = max(s["zone_c"] or -99, v - 273.15)
    try:
        g = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        s["gpu_c"] = max(float(x) for x in g) if g else None
    except Exception:
        pass
    return s


class ThermalGuard:
    def __init__(self, runs_dir: Path, log):
        self.dir, self.log = Path(runs_dir), log
        self.shed = 0
        self.crit_streak = self.clean_streak = 0
        self.last_t = 0.0
        self.last_popup = 0.0
        self.last_warn_log = 0.0
        self.csv = self.dir / "thermal.csv"
        if not self.csv.exists():
            self.csv.write_text("time,perf_pct,cpu_util_pct,throttle,zone_c,gpu_c,shed_runs,level\n")

    def _popup(self, text: str):
        if time.time() - self.last_popup < POPUP_EVERY_S:
            return
        self.last_popup = time.time()
        try:
            subprocess.Popen(["msg", os.environ.get("USERNAME", "*"), "/TIME:3600", text],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    def tick(self) -> int:
        """Sample if due; return how many runs to shed below the normal cap."""
        if time.time() - self.last_t < SAMPLE_SEC:
            return self.shed
        self.last_t = time.time()
        s = sample()
        perf, util = s["perf"], s["util"]
        loaded = util is not None and util > 70
        warn = ((perf is not None and loaded and perf < 100) or s["throttle"] > 0
                or (s["zone_c"] or 0) > 85 or (s["gpu_c"] or 0) > 83)
        crit_now = ((perf is not None and loaded and perf < 75) or s["throttle"] > 0)
        hard = (s["zone_c"] or 0) > 95 or (s["gpu_c"] or 0) > 90
        self.crit_streak = self.crit_streak + 1 if crit_now else 0
        self.clean_streak = 0 if warn else self.clean_streak + 1
        level = "ok"
        if hard or self.crit_streak >= 3:
            level = "CRITICAL"
            if self.shed < MAX_SHED:
                self.shed = min(MAX_SHED, self.shed + STEP)
                self.log(f"THERMAL CRITICAL ({self._fmt(s)}) -> shedding to {self.shed} fewer runs")
                self._popup(f"Training PC THERMAL CRITICAL: {self._fmt(s)}. "
                            f"Automatically reduced load by {self.shed} runs. Check cooling.")
            self.crit_streak = 0
        elif warn:
            level = "WARNING"
            if time.time() - self.last_warn_log > 1800:
                self.last_warn_log = time.time()
                self.log(f"THERMAL WARNING ({self._fmt(s)})")
                self._popup(f"Training PC thermal warning: {self._fmt(s)}. Training continues; check cooling.")
        elif self.shed and self.clean_streak >= RECOVER_SAMPLES:
            self.shed = max(0, self.shed - STEP)
            self.clean_streak = 0
            self.log(f"thermal recovered ({self._fmt(s)}) -> restoring capacity (shed now {self.shed})")
        with open(self.csv, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{perf},{util},{s['throttle']},"
                    f"{s['zone_c']},{s['gpu_c']},{self.shed},{level}\n")
        json.dump({**s, "shed": self.shed, "level": level, "time": time.strftime("%Y-%m-%d %H:%M:%S")},
                  open(self.dir / "thermal_status.json", "w"))
        return self.shed

    @staticmethod
    def _fmt(s):
        f = lambda v, u: "n/a" if v is None else f"{v:.0f}{u}"  # noqa: E731
        return (f"CPU clock {f(s['perf'], '%')} of nominal, util {f(s['util'], '%')}, throttle flag "
                f"{s['throttle']:.0f}, board {f(s['zone_c'], 'C')}, GPU {f(s['gpu_c'], 'C')}")


if __name__ == "__main__":
    print(json.dumps(sample()))
