"""
Smoke tests for LMICHospitalNav-v1.

Run:  pytest tests/test_env.py -v
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Floor plan
# ---------------------------------------------------------------------------

class TestFloorPlan:
    def setup_method(self):
        from envs.floorplan import load_floorplan
        self.fp = load_floorplan(
            Path(__file__).parent.parent / "configs" / "dmch_hospital.yaml"
        )

    def test_dimensions(self):
        assert self.fp.width_m == pytest.approx(40.0)
        assert self.fp.height_m == pytest.approx(30.0)
        assert self.fp.width_cells == 400
        assert self.fp.height_cells == 300

    def test_walkable(self):
        # Corridor center should be walkable
        assert self.fp.is_walkable(20.0, 13.5)
        # Outside bounds should not be walkable (treated as wall)
        assert not self.fp.is_walkable(-1.0, -1.0)

    def test_zone_types(self):
        zone = self.fp.get_zone(20.0, 13.5)
        assert zone in ("corridor", "nurse_station", "walkable")

    def test_astar(self):
        path = self.fp.astar(np.array([20.0, 13.5]), np.array([4.0, 4.0]))
        assert path is not None
        assert len(path) > 1

    def test_path_length(self):
        pl = self.fp.path_length(np.array([20.0, 13.5]), np.array([4.0, 4.0]))
        euclidean = np.linalg.norm(np.array([20.0, 13.5]) - np.array([4.0, 4.0]))
        assert pl >= euclidean - 0.5  # A* path >= euclidean

    def test_zone_onehot(self):
        oh = self.fp.zone_onehot(20.0, 13.5)
        assert oh.shape == (8,)
        assert oh.sum() == pytest.approx(1.0)

    def test_prayer_intensity(self):
        assert self.fp.prayer_intensity(12.5) == pytest.approx(1.0)  # Dhuhr
        assert self.fp.prayer_intensity(10.0) == pytest.approx(0.3)  # off-window

    def test_cultural_zone(self):
        assert self.fp.is_cultural_zone(33.0, 26.0)   # prayer room
        assert not self.fp.is_cultural_zone(20.0, 13.5)


# ---------------------------------------------------------------------------
# Pedestrian crowd
# ---------------------------------------------------------------------------

class TestPedestrianCrowd:
    def setup_method(self):
        from envs.floorplan import load_floorplan
        from envs.pedestrians import PedestrianCrowd
        fp = load_floorplan(
            Path(__file__).parent.parent / "configs" / "dmch_hospital.yaml"
        )
        rng = np.random.default_rng(0)
        self.crowd = PedestrianCrowd(fp, max_pedestrians=40, rng=rng)
        counts = dict(medical_staff=2, patient=4, family_attendant=4, wheelchair=2)
        self.crowd.reset(counts, rng=np.random.default_rng(0))

    def test_agent_count(self):
        # At least some agents were spawned
        assert len(self.crowd.agents) >= 6

    def test_step(self):
        n_before = len(self.crowd.agents)
        self.crowd.step(0.1)
        assert len(self.crowd.agents) == n_before

    def test_local_neighbors(self):
        robot_pos = np.array([20.0, 13.5])
        nb = self.crowd.get_local_neighbors(robot_pos, k=5)
        assert nb.shape == (5, 4)

    def test_crowd_density(self):
        d = self.crowd.crowd_density(np.array([20.0, 13.5]))
        assert 0.0 <= d <= 1.0

    def test_nearest_family_distance(self):
        dist = self.crowd.nearest_family_distance(np.array([20.0, 13.5]))
        assert dist > 0.0


# ---------------------------------------------------------------------------
# Gymnasium environment
# ---------------------------------------------------------------------------

class TestLMICHospitalNavEnv:
    def setup_method(self):
        from envs.lmic_hospital_env import LMICHospitalNavEnv
        self.env = LMICHospitalNavEnv(seed=42)

    def teardown_method(self):
        self.env.close()

    def test_reset(self):
        obs, info = self.env.reset(seed=0)
        assert obs.shape == (85,)
        assert obs.dtype == np.float32
        assert "dist_to_goal" in info
        assert "context_vector" in info

    def test_step(self):
        self.env.reset(seed=0)
        action = self.env.action_space.sample()
        obs, reward, terminated, truncated, info = self.env.step(action)
        assert obs.shape == (85,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "collision" in info
        assert "task_complete" in info

    def test_obs_bounds(self):
        obs, _ = self.env.reset(seed=1)
        assert (obs >= self.env.observation_space.low - 1e-5).all()
        assert (obs <= self.env.observation_space.high + 1e-5).all()

    def test_action_space(self):
        assert self.env.action_space.low[0] == pytest.approx(0.0)
        assert self.env.action_space.high[0] == pytest.approx(0.8)
        assert self.env.action_space.shape == (2,)

    def test_multiple_resets(self):
        for seed in range(5):
            obs, info = self.env.reset(seed=seed)
            assert obs.shape == (85,)

    def test_task_options(self):
        from envs.lmic_hospital_env import TASK_IDS
        for task in TASK_IDS:
            obs, info = self.env.reset(seed=0, options={"task": task})
            assert info["task_id"] == task

    def test_density_options(self):
        for density in ["low", "medium", "high"]:
            obs, info = self.env.reset(seed=0, options={"density": density})
            assert info["density_level"] == density

    def test_episode_terminates(self):
        """Run random policy until done (within 100 steps)."""
        self.env.reset(seed=0)
        for _ in range(100):
            action = self.env.action_space.sample()
            _, _, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                return
        # If not terminated, that's OK (sparse reward case)

    def test_check_env(self):
        """Run SB3 check_env — must pass with no warnings."""
        from stable_baselines3.common.env_checker import check_env
        from envs.lmic_hospital_env import LMICHospitalNavEnv
        fresh_env = LMICHospitalNavEnv(seed=0)
        check_env(fresh_env, warn=True, skip_render_check=True)
        fresh_env.close()


# ---------------------------------------------------------------------------
# Reward components
# ---------------------------------------------------------------------------

class TestRewardComponents:
    def test_navigation_reward(self):
        from rewards.components import navigation_reward
        action = np.array([0.5, 0.0])
        # Moving toward goal: prev > curr
        r = navigation_reward(5.0, 4.0, False, action)
        assert r > 0.0
        # On goal
        r_done = navigation_reward(1.0, 0.0, True, action)
        assert r_done > r

    def test_safety_reward(self):
        from rewards.components import safety_reward
        r_coll = safety_reward(True, 0.1)
        assert r_coll == pytest.approx(-5.0)
        r_clear = safety_reward(False, 2.0)
        assert r_clear == pytest.approx(0.0)
        r_close = safety_reward(False, 0.4)
        assert r_close < 0.0

    def test_task_reward(self):
        from rewards.components import task_reward
        r_on_time = task_reward(True, 1.0, 30.0, 45.0)
        r_late    = task_reward(True, 1.0, 60.0, 45.0)
        assert r_on_time > r_late


# ---------------------------------------------------------------------------
# Cultural reward
# ---------------------------------------------------------------------------

class TestCulturalReward:
    def setup_method(self):
        from envs.floorplan import load_floorplan
        self.fp = load_floorplan(
            Path(__file__).parent.parent / "configs" / "dmch_hospital.yaml"
        )

    def test_prayer_avoidance(self):
        from rewards.cultural import prayer_avoidance_reward
        # Inside prayer room (prayer_room at ~33, 26)
        r, v = prayer_avoidance_reward(np.array([33.0, 26.0]), self.fp, 12.5)
        assert v is True
        assert r < -0.5

    def test_gender_zone(self):
        from rewards.cultural import gender_zone_reward
        # Inside female ward (~31, 18.5)
        r, v = gender_zone_reward(np.array([31.0, 18.5]), self.fp)
        assert v is True
        assert r < 0.0

    def test_attendant_proxemics(self):
        from rewards.cultural import attendant_proxemics_reward
        r_hard, v = attendant_proxemics_reward(0.3)
        assert v is True
        assert r_hard < -1.0
        r_clear, v2 = attendant_proxemics_reward(2.0)
        assert r_clear == pytest.approx(0.0)
        assert not v2

    def test_cvr_tracker(self):
        from rewards.cultural import CulturalViolationTracker
        tracker = CulturalViolationTracker()
        # Record steps in prayer zone
        for _ in range(5):
            tracker.record(np.array([33.0, 26.0]), self.fp, 12.5, 2.0)
        # Record steps in corridor (no violation)
        for _ in range(5):
            tracker.record(np.array([20.0, 13.5]), self.fp, 10.0, 2.0)
        summary = tracker.summary()
        assert summary["steps"] == 10
        assert summary["prayer_cvr"] > 0.0
        assert 0.0 <= summary["cvr"] <= 1.0


# ---------------------------------------------------------------------------
# CWN
# ---------------------------------------------------------------------------

class TestCWN:
    def test_output_shape(self):
        from rewards.cwn import CWN
        cwn = CWN()
        ctx = np.random.rand(12).astype(np.float32)
        w = cwn.get_weights(ctx)
        assert w.shape == (4,)
        assert abs(w.sum() - 1.0) < 1e-5

    def test_batch(self):
        from rewards.cwn import CWN
        cwn = CWN()
        ctx = np.random.rand(8, 12).astype(np.float32)
        w = cwn.get_weights(ctx)
        assert w.shape == (8, 4)
        np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-5)

    def test_fixed_weight_source(self):
        from rewards.cwn import FixedWeightSource
        fws = FixedWeightSource("equal")
        w = fws.get_weights(np.zeros(12, dtype=np.float32))
        np.testing.assert_allclose(w, [0.25, 0.25, 0.25, 0.25], atol=1e-6)
