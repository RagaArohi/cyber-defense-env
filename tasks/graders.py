"""
tasks/graders.py

Three graded tasks for OpenEnv submission.

Each grader receives the full episode log and returns a score 0.0–1.0.

Task 1 (Easy)   — detect_recon:     Identify and respond to reconnaissance
Task 2 (Medium) — stop_exploit:     Stop the attack before persistence stage
Task 3 (Hard)   — prevent_exfil:    Prevent data exfiltration with minimal resources
"""
from typing import List


def grade_detect_recon(episode_log: List[dict]) -> dict:
    """
    EASY — Detect Reconnaissance
    
    Score 1.0 if agent:
      - Takes a monitoring or blocking action within first 5 steps of recon starting
      - Keeps system health above 0.7
    
    Partial credit for late detection or moderate health loss.
    """
    if not episode_log:
        return {"score": 0.0, "reason": "Empty episode log"}

    recon_step = None
    response_step = None
    final_health = episode_log[-1]["info"].get("system_health", 0.0)
    attack_stopped = episode_log[-1]["info"].get("attack_stopped", False)

    proactive_actions = {"monitor_traffic", "block_ip", "scan_system"}

    for i, entry in enumerate(episode_log):
        stage = entry["info"].get("attack_stage", 0)
        action = entry.get("action", "do_nothing")

        if stage >= 1 and recon_step is None:
            recon_step = i

        if recon_step is not None and response_step is None:
            if action in proactive_actions:
                response_step = i

    # Scoring
    if recon_step is None:
        # No attack occurred — trivially easy, give partial
        return {"score": 0.5, "reason": "No attack occurred during episode"}

    health_score = max(0.0, min(1.0, (final_health - 0.5) / 0.5))

    if response_step is None:
        return {"score": 0.1 * health_score, "reason": "Agent never responded to recon"}

    response_delay = response_step - recon_step
    if response_delay <= 2:
        detection_score = 1.0
    elif response_delay <= 5:
        detection_score = 0.7
    else:
        detection_score = 0.3

    stop_bonus = 0.2 if attack_stopped else 0.0
    score = min(1.0, detection_score * 0.6 + health_score * 0.4 + stop_bonus)

    return {
        "score": round(score, 4),
        "reason": f"Recon at step {recon_step}, response at step {response_step} (delay={response_delay}), health={final_health:.2f}",
        "detection_score": round(detection_score, 4),
        "health_score": round(health_score, 4),
    }


def grade_stop_exploit(episode_log: List[dict]) -> dict:
    """
    MEDIUM — Stop Exploitation
    
    Score 1.0 if agent:
      - Stops the attack before it reaches stage 3 (persistence)
      - Maintains system health above 0.6
      - Uses fewer than 40% of total resource budget
    
    Partial credit for late stops or high resource use.
    """
    if not episode_log:
        return {"score": 0.0, "reason": "Empty episode log"}

    final = episode_log[-1]
    final_health = final["info"].get("system_health", 0.0)
    attack_stopped = final["info"].get("attack_stopped", False)

    # Find at what stage the attack was stopped
    stop_stage = None
    max_stage_reached = 0
    total_cost = 0.0

    action_costs = {
        "do_nothing": 0.0, "monitor_traffic": 0.05, "block_ip": 0.10,
        "scan_system": 0.20, "throttle_bandwidth": 0.10, "isolate_subsystem": 0.30
    }

    for entry in episode_log:
        stage = entry["info"].get("attack_stage", 0)
        max_stage_reached = max(max_stage_reached, stage)
        action = entry.get("action", "do_nothing")
        total_cost += action_costs.get(action, 0.0)

        if entry["info"].get("attack_stopped", False) and stop_stage is None:
            stop_stage = stage

    # Resource efficiency (budget = 1.0 at start)
    resource_efficiency = max(0.0, 1.0 - total_cost)

    if not attack_stopped:
        # Partial credit based on how far attack got
        stage_penalty = max_stage_reached / 4.0
        score = max(0.0, 0.2 - 0.05 * stage_penalty) * (final_health)
        return {
            "score": round(score, 4),
            "reason": f"Attack not stopped. Max stage reached: {max_stage_reached}",
        }

    # Stage-based scoring
    stage_scores = {0: 0.5, 1: 1.0, 2: 0.85, 3: 0.5, 4: 0.2}
    stage_score = stage_scores.get(stop_stage, 0.2)

    health_score = max(0.0, min(1.0, (final_health - 0.4) / 0.6))
    efficiency_score = min(1.0, resource_efficiency + 0.5)  # lenient

    score = min(1.0, stage_score * 0.5 + health_score * 0.3 + efficiency_score * 0.2)

    return {
        "score": round(score, 4),
        "reason": f"Stopped at stage {stop_stage}, health={final_health:.2f}, cost={total_cost:.2f}",
        "stage_score": round(stage_score, 4),
        "health_score": round(health_score, 4),
        "efficiency_score": round(efficiency_score, 4),
    }


def grade_prevent_exfil(episode_log: List[dict]) -> dict:
    """
    HARD — Prevent Data Exfiltration
    
    Score 1.0 if agent:
      - Prevents exfil_complete from ever becoming True
      - Keeps system health above 0.75
      - Uses targeted actions (throttle_bandwidth or isolate when at stage 4)
      - Minimises false positives
    
    This requires the agent to correctly read late-stage signals.
    """
    if not episode_log:
        return {"score": 0.0, "reason": "Empty episode log"}

    final = episode_log[-1]
    exfil_complete = final["info"].get("exfil_complete", False)
    final_health = final["info"].get("system_health", 0.0)
    attack_stopped = final["info"].get("attack_stopped", False)

    false_positives = sum(
        1 for e in episode_log if e.get("false_positive", False)
    )
    total_steps = len(episode_log)
    fp_rate = false_positives / max(1, total_steps)

    # Check for appropriate action at stage 4
    late_stage_correct = 0
    late_stage_total = 0
    for entry in episode_log:
        if entry["info"].get("attack_stage", 0) >= 4:
            late_stage_total += 1
            if entry.get("action") in ("throttle_bandwidth", "isolate_subsystem", "scan_system"):
                late_stage_correct += 1

    if exfil_complete:
        return {
            "score": 0.0,
            "reason": "Exfiltration completed — data stolen",
        }

    # Base score from health
    health_score = max(0.0, min(1.0, (final_health - 0.5) / 0.5))

    # Precision bonus
    fp_penalty = min(0.3, fp_rate * 0.5)

    # Late-stage action quality
    if late_stage_total > 0:
        action_quality = late_stage_correct / late_stage_total
    else:
        action_quality = 0.5  # never reached stage 4, good

    stop_bonus = 0.15 if attack_stopped else 0.0

    score = min(1.0, health_score * 0.45 + action_quality * 0.35 + stop_bonus - fp_penalty)
    score = max(0.0, score)

    return {
        "score": round(score, 4),
        "reason": (
            f"Exfil prevented. Health={final_health:.2f}, "
            f"FP rate={fp_rate:.2f}, late-stage accuracy={action_quality:.2f}"
        ),
        "health_score": round(health_score, 4),
        "action_quality": round(action_quality, 4),
        "false_positive_rate": round(fp_rate, 4),
    }


GRADERS = {
    "detect_recon":  grade_detect_recon,
    "stop_exploit":  grade_stop_exploit,
    "prevent_exfil": grade_prevent_exfil,
}


def grade(task_id: str, episode_log: List[dict]) -> dict:
    """Dispatch to the correct grader."""
    grader = GRADERS.get(task_id)
    if grader is None:
        return {"score": 0.0, "reason": f"Unknown task: {task_id}"}
    return grader(episode_log)
