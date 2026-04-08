"""
env/cyber_env.py

CyberDefenseEnv — the core environment logic.

This class is framework-agnostic. It is wrapped by:
  - app.py         (FastAPI/OpenEnv HTTP server)
  - inference.py   (LLM agent runner)

State: 8 partially-observable features (Gaussian noise applied)
Actions: 6 discrete defense actions
"""
import numpy as np
from env.attack_simulator import AttackSimulator
from env.reward import RewardCalculator

# ── Action registry ───────────────────────────────────────────────────────────
ACTIONS = {
    0: "do_nothing",
    1: "monitor_traffic",
    2: "block_ip",
    3: "scan_system",
    4: "throttle_bandwidth",
    5: "isolate_subsystem",
}
ACTION_NAMES = list(ACTIONS.values())

ACTION_COSTS = {
    "do_nothing":         0.00,
    "monitor_traffic":    0.05,
    "block_ip":           0.10,
    "scan_system":        0.20,
    "throttle_bandwidth": 0.10,
    "isolate_subsystem":  0.30,
}

# ── Observation feature descriptions (for LLM prompts) ───────────────────────
OBS_FEATURES = [
    "network_traffic_anomaly",
    "failed_login_rate",
    "suspicious_process_score",
    "cpu_usage",
    "memory_usage",
    "alert_level",
    "bandwidth_usage",
    "time_step_norm",
]

# True signal templates per stage [stage 0..4]
_STAGE_SIGNALS = {
    "network_traffic_anomaly":  [0.00, 0.30, 0.60, 0.40, 0.90],
    "failed_login_rate":        [0.00, 0.50, 0.80, 0.30, 0.20],
    "suspicious_process_score": [0.00, 0.10, 0.50, 0.90, 0.70],
    "cpu_usage":                [0.30, 0.40, 0.60, 0.80, 0.70],
    "memory_usage":             [0.20, 0.30, 0.40, 0.80, 0.70],
    "alert_level":              [0.00, 0.40, 0.60, 0.50, 0.80],
    "bandwidth_usage":          [0.20, 0.40, 0.50, 0.40, 0.90],
}


class CyberDefenseEnv:
    OBS_DIM = 8
    N_ACTIONS = 6

    def __init__(self, max_steps: int = 50, noise_level: float = 0.15):
        self.max_steps = max_steps
        self.noise_level = noise_level
        self._atk = AttackSimulator()
        self._rew = RewardCalculator()
        self._step_count = 0
        self._health = 1.0
        self._budget = 1.0
        self._monitoring = False
        self._isolated = False
        self._atk_state: dict = {}
        self._done = False

    # ── Gym-style API ─────────────────────────────────────────────────────────

    def reset(self, seed: int = None) -> tuple:
        """Returns (obs_array, info_dict)."""
        np.random.seed(seed)
        self._step_count = 0
        self._health = 1.0
        self._budget = 1.0
        self._monitoring = False
        self._isolated = False
        self._done = False
        self._atk_state = self._atk.reset(seed=seed)
        obs = self._observe()
        return obs, self._info()

    def step(self, action) -> tuple:
        """
        action: int (0-5) or str (action name)
        Returns (obs, reward, terminated, truncated, info)
        """
        assert not self._done, "Episode finished — call reset() first"

        if isinstance(action, int):
            action_name = ACTIONS[action]
        else:
            action_name = str(action).strip().lower()
            if action_name not in ACTION_NAMES:
                action_name = "do_nothing"

        self._step_count += 1
        prev_stage = self._atk_state["stage"]

        # Apply action
        cost = ACTION_COSTS[action_name]
        self._budget = max(0.0, self._budget - cost)
        attack_active = prev_stage > 0
        false_positive = not attack_active and action_name not in ("do_nothing", "monitor_traffic")

        if action_name == "monitor_traffic":
            self._monitoring = True
        if action_name == "isolate_subsystem":
            self._isolated = True
            self._health -= 0.05  # isolation hurts uptime slightly

        # Advance attack
        self._atk_state = self._atk.step(action_name, self._isolated)
        new_stage = self._atk_state["stage"]
        self._health = max(0.0, self._health - self._atk_state["damage_this_step"])

        # Reward
        reward = self._rew.compute(
            action=action_name,
            prev_stage=prev_stage,
            new_stage=new_stage,
            attack_stopped=self._atk_state["stopped"],
            system_health=self._health,
            resource_cost=cost,
            false_positive=false_positive,
            monitoring_active=self._monitoring,
        )

        terminated = (
            self._health <= 0.0
            or self._atk_state["stopped"]
            or self._atk_state["exfil_complete"]
        )
        truncated = self._step_count >= self.max_steps
        self._done = terminated or truncated

        obs = self._observe()
        info = self._info()
        info["action_taken"] = action_name
        info["reward_breakdown"] = self._rew.last_breakdown
        info["false_positive"] = false_positive

        return obs, reward, terminated, truncated, info

    # ── Observation helpers ───────────────────────────────────────────────────

    def _observe(self) -> np.ndarray:
        stage = self._atk_state.get("stage", 0)
        intensity = self._atk_state.get("intensity", 0.0)

        raw = []
        for feat in OBS_FEATURES[:-1]:  # skip time_step_norm
            base = _STAGE_SIGNALS[feat][stage]
            raw.append(min(1.0, base * intensity if intensity > 0 else base * 0.1))

        noise_scale = self.noise_level * (0.5 if self._monitoring else 1.0)
        obs = np.array(raw, dtype=np.float32)
        obs += np.random.normal(0, noise_scale, size=len(raw)).astype(np.float32)
        obs = np.clip(obs, 0.0, 1.0)
        obs = np.append(obs, np.float32(self._step_count / self.max_steps))
        return obs

    def obs_to_dict(self) -> dict:
        """Human-readable observation dict (for LLM prompts)."""
        obs = self._observe()
        return {feat: round(float(obs[i]), 3) for i, feat in enumerate(OBS_FEATURES)}

    def obs_to_text(self, obs_dict: dict = None) -> str:
        """Natural language observation for LLM agent.
        
        Accepts an optional pre-computed obs_dict to avoid calling _observe() twice
        and generating two different noisy observations in the same step.
        """
        d = obs_dict if obs_dict is not None else self.obs_to_dict()
        lines = [
            f"- Network traffic anomaly:    {d['network_traffic_anomaly']:.2f}  (0=normal, 1=severe)",
            f"- Failed login rate:          {d['failed_login_rate']:.2f}",
            f"- Suspicious process score:   {d['suspicious_process_score']:.2f}",
            f"- CPU usage:                  {d['cpu_usage']:.2f}",
            f"- Memory usage:               {d['memory_usage']:.2f}",
            f"- Alert level:                {d['alert_level']:.2f}",
            f"- Bandwidth usage:            {d['bandwidth_usage']:.2f}",
            f"- Episode progress:           {d['time_step_norm']:.2f}  (0=start, 1=end)",
            f"- System health:              {self._health:.2f}",
            f"- Resource budget remaining:  {self._budget:.2f}",
        ]
        return "\n".join(lines)

    def _info(self) -> dict:
        return {
            "step": self._step_count,
            "attack_stage": self._atk_state.get("stage", 0),
            "attack_stage_name": self._atk_state.get("stage_name", "dormant"),
            "system_health": round(self._health, 4),
            "resource_budget": round(self._budget, 4),
            "attack_stopped": self._atk_state.get("stopped", False),
            "exfil_complete": self._atk_state.get("exfil_complete", False),
            "monitoring_active": self._monitoring,
            "isolated": self._isolated,
        }

    @property
    def done(self) -> bool:
        return self._done
