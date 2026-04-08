"""
env/reward.py

Multi-objective reward function.

R = R_detection + R_stopping + R_health + R_efficiency
  - P_false_positive - P_resource_waste - P_missed_attack

Reward is bounded approximately in [-5, +15] per step.
"""


class RewardCalculator:
    W = {
        "detection":       1.0,
        "early_stop":      5.0,
        "late_stop":       2.0,
        "health":          0.5,
        "efficiency":      1.5,
        "false_positive": -1.0,
        "resource_waste": -0.5,
        "missed_attack":  -2.0,
    }

    def __init__(self):
        self.last_breakdown: dict = {}

    def compute(
        self,
        action: str,
        prev_stage: int,
        new_stage: int,
        attack_stopped: bool,
        system_health: float,
        resource_cost: float,
        false_positive: bool,
        monitoring_active: bool,
    ) -> float:
        b = {}

        # Detection bonus — agent was monitoring when stage advanced
        b["detection"] = self.W["detection"] if (new_stage > prev_stage and monitoring_active) else 0.0

        # Stop reward — more for stopping early
        if attack_stopped:
            b["stop"] = self.W["early_stop"] * max(1, 3 - prev_stage) if prev_stage <= 2 else self.W["late_stop"]
        else:
            b["stop"] = 0.0

        # Continuous health reward
        b["health"] = self.W["health"] * system_health

        # Efficiency bonus
        b["efficiency"] = self.W["efficiency"] if (attack_stopped and resource_cost <= 0.10) else 0.0

        # Penalties
        b["false_positive"] = self.W["false_positive"] if false_positive else 0.0

        b["resource_waste"] = (self.W["resource_waste"] * resource_cost
                               if (resource_cost > 0.10 and new_stage > prev_stage) else 0.0)

        # Missed attack — scaled by stage severity
        severity = {0: 0.0, 1: 0.0, 2: 0.3, 3: 0.6, 4: 1.0}.get(prev_stage, 0.0)
        b["missed_attack"] = self.W["missed_attack"] * severity if (severity > 0 and action == "do_nothing") else 0.0

        total = sum(b.values())
        self.last_breakdown = {k: round(v, 4) for k, v in b.items()}
        self.last_breakdown["total"] = round(total, 4)
        return float(total)
