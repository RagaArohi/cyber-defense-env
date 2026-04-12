"""
tasks/graders.py
"""
from typing import List


def _clamp(score):
    return round(max(0.01, min(0.99, float(score))), 4)


def grade_detect_recon(episode_log):
    if not episode_log:
        return {"score": 0.05, "reason": "Empty episode log"}
    recon_step = None
    response_step = None
    final_health = episode_log[-1]["info"].get("system_health", 0.0)
    attack_stopped = episode_log[-1]["info"].get("attack_stopped", False)
    proactive = {"monitor_traffic", "block_ip", "scan_system"}
    for i, e in enumerate(episode_log):
        stage = e["info"].get("attack_stage", 0)
        action = e.get("action", "do_nothing")
        if stage >= 1 and recon_step is None:
            recon_step = i
        if recon_step is not None and response_step is None and action in proactive:
            response_step = i
    if recon_step is None:
        return {"score": 0.5, "reason": "No attack occurred"}
    health_score = max(0.0, min(0.98, (final_health - 0.5) / 0.5))
    if response_step is None:
        return {"score": _clamp(0.1 * health_score + 0.02), "reason": "No response"}
    delay = response_step - recon_step
    d_score = 0.92 if delay <= 2 else (0.68 if delay <= 5 else 0.3)
    stop_bonus = 0.08 if attack_stopped else 0.0
    return {"score": _clamp(d_score * 0.6 + health_score * 0.4 + stop_bonus), "reason": f"delay={delay}"}


def grade_stop_exploit(episode_log):
    if not episode_log:
        return {"score": 0.05, "reason": "Empty episode log"}
    final = episode_log[-1]
    final_health = final["info"].get("system_health", 0.0)
    attack_stopped = final["info"].get("attack_stopped", False)
    stop_stage = None
    max_stage = 0
    total_cost = 0.0
    costs = {"do_nothing": 0.0, "monitor_traffic": 0.05, "block_ip": 0.10,
             "scan_system": 0.20, "throttle_bandwidth": 0.10, "isolate_subsystem": 0.30}
    for e in episode_log:
        stage = e["info"].get("attack_stage", 0)
        max_stage = max(max_stage, stage)
        total_cost += costs.get(e.get("action", "do_nothing"), 0.0)
        if e["info"].get("attack_stopped", False) and stop_stage is None:
            stop_stage = stage
    if not attack_stopped:
        base = max(0.02, 0.15 - 0.03 * max_stage) * max(0.02, final_health)
        return {"score": _clamp(base), "reason": f"Not stopped, max_stage={max_stage}"}
    stage_scores = {0: 0.5, 1: 0.88, 2: 0.75, 3: 0.5, 4: 0.2}
    ss = stage_scores.get(stop_stage, 0.2)
    hs = max(0.0, min(0.92, (final_health - 0.4) / 0.6))
    eff = min(0.88, max(0.0, 1.0 - total_cost) + 0.4)
    return {"score": _clamp(ss * 0.5 + hs * 0.3 + eff * 0.2), "reason": f"stage={stop_stage}"}


def grade_prevent_exfil(episode_log):
    if not episode_log:
        return {"score": 0.05, "reason": "Empty episode log"}
    final = episode_log[-1]
    exfil = final["info"].get("exfil_complete", False)
    final_health = final["info"].get("system_health", 0.0)
    attack_stopped = final["info"].get("attack_stopped", False)
    fp = sum(1 for e in episode_log if e.get("false_positive", False))
    fp_rate = fp / max(1, len(episode_log))
    lsc, lst = 0, 0
    for e in episode_log:
        if e["info"].get("attack_stage", 0) >= 4:
            lst += 1
            if e.get("action") in ("throttle_bandwidth", "isolate_subsystem", "scan_system"):
                lsc += 1
    if exfil:
        return {"score": 0.02, "reason": "Exfiltration completed"}
    hs = max(0.0, min(0.92, (final_health - 0.5) / 0.5))
    aq = (lsc / lst) if lst > 0 else 0.5
    sb = 0.08 if attack_stopped else 0.0
    return {"score": _clamp(hs * 0.45 + aq * 0.35 + sb - min(0.22, fp_rate * 0.5) + 0.02), "reason": f"health={final_health:.2f}"}


GRADERS = {
    "detect_recon":  grade_detect_recon,
    "stop_exploit":  grade_stop_exploit,
    "prevent_exfil": grade_prevent_exfil,
}


def grade(task_id, episode_log):
    grader = GRADERS.get(task_id)
    if grader is None:
        return {"score": 0.05, "reason": f"Unknown: {task_id}"}
    result = grader(episode_log)
    result["score"] = _clamp(result["score"])
    return result
# ── Grader classes for direct import by OpenEnv validator ────────────────────

class GradeDetectRecon:
    def grade(self, episode_log):
        result = grade_detect_recon(episode_log or [])
        return _clamp(result["score"])

class GradeStopExploit:
    def grade(self, episode_log):
        result = grade_stop_exploit(episode_log or [])
        return _clamp(result["score"])

class GradePreventExfil:
    def grade(self, episode_log):
        result = grade_prevent_exfil(episode_log or [])
        return _clamp(result["score"])
