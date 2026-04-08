"""
env/attack_simulator.py

Multi-stage cyber attack simulator following the kill chain:
  0 = Dormant   (attack not yet started)
  1 = Recon     (passive scanning, info gathering)
  2 = Exploit   (active breach attempt)
  3 = Persist   (malware installed, backdoors set)
  4 = Exfil     (data actively being stolen)
"""
import numpy as np


STAGE_NAMES = {0: "dormant", 1: "recon", 2: "exploit", 3: "persist", 4: "exfil"}

# Per-stage probability of advancing each step (if not blocked)
ADVANCE_PROB = {0: 0.0, 1: 0.30, 2: 0.40, 3: 0.35, 4: 0.0}

# Damage dealt to system health per step at each stage
STAGE_DAMAGE = {0: 0.00, 1: 0.00, 2: 0.02, 3: 0.04, 4: 0.09}

# How much each defender action slows advance probability (multiplier)
ACTION_SLOWDOWN = {
    "do_nothing":         1.00,
    "monitor_traffic":    0.92,
    "block_ip":           0.45,
    "scan_system":        0.55,
    "throttle_bandwidth": 0.38,
    "isolate_subsystem":  0.04,
}

# Probability of STOPPING the attack outright per action per stage
STOP_PROB = {
    1: {"block_ip": 0.55, "scan_system": 0.20, "isolate_subsystem": 0.82},
    2: {"block_ip": 0.42, "scan_system": 0.32, "isolate_subsystem": 0.78},
    3: {"block_ip": 0.10, "scan_system": 0.58, "isolate_subsystem": 0.88},
    4: {"throttle_bandwidth": 0.48, "isolate_subsystem": 0.72},
}


class AttackSimulator:
    def __init__(self, start_delay: tuple = (3, 8), seed: int = None):
        self._rng = np.random.RandomState(seed)
        self._start_delay_range = start_delay
        self._state = self._init()

    # ── Public ────────────────────────────────────────────────────────────────

    def reset(self, seed: int = None) -> dict:
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self._state = self._init()
        return self._output()

    def step(self, action: str, isolated: bool) -> dict:
        s = self._state

        # Countdown before attack begins
        if s["countdown"] > 0:
            s["countdown"] -= 1
            s["damage_this_step"] = 0.0
            return self._output()

        # Launch attack if still dormant
        if s["stage"] == 0:
            s["stage"] = 1
            s["intensity"] = float(self._rng.uniform(0.3, 0.6))

        stage = s["stage"]

        # Check if this action stops the attack
        stop_chance = STOP_PROB.get(stage, {}).get(action, 0.0)
        if isolated:
            stop_chance = max(stop_chance, 0.88)

        if self._rng.random() < stop_chance:
            s["stopped"] = True
            s["damage_this_step"] = 0.0
            return self._output()

        # Damage this step
        dmg = STAGE_DAMAGE[stage] * s["intensity"]
        s["damage_this_step"] = dmg
        s["total_damage"] += dmg

        # Intensity grows
        s["intensity"] = min(1.0, s["intensity"] + float(self._rng.uniform(0.02, 0.06)))

        # Try to advance stage
        slowdown = ACTION_SLOWDOWN.get(action, 1.0)
        if stage < 4 and self._rng.random() < ADVANCE_PROB[stage] * slowdown:
            s["stage"] += 1

        # Track exfil progress
        if s["stage"] == 4:
            s["exfil_progress"] = min(1.0, s["exfil_progress"] + 0.18 * s["intensity"])
            if s["exfil_progress"] >= 1.0:
                s["exfil_complete"] = True

        return self._output()

    def get_stage_name(self) -> str:
        return STAGE_NAMES.get(self._state["stage"], "unknown")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _init(self) -> dict:
        return {
            "stage": 0,
            "intensity": 0.0,
            "countdown": int(self._rng.randint(*self._start_delay_range)),
            "stopped": False,
            "exfil_complete": False,
            "exfil_progress": 0.0,
            "damage_this_step": 0.0,
            "total_damage": 0.0,
        }

    def _output(self) -> dict:
        s = self._state
        return {
            "stage": s["stage"],
            "stage_name": STAGE_NAMES[s["stage"]],
            "intensity": round(s["intensity"], 4),
            "stopped": s["stopped"],
            "exfil_complete": s["exfil_complete"],
            "damage_this_step": round(s["damage_this_step"], 4),
            "total_damage": round(s["total_damage"], 4),
        }
