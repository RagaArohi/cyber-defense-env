"""
inference.py — LLM Agent for CyberDefenseEnv

OpenEnv Hackathon-compliant inference script.

Environment variables (set in HuggingFace Space secrets or .env):
  API_BASE_URL     — OpenAI-compatible API base URL (has default)
  MODEL_NAME       — Model to use (has default)
  HF_TOKEN         — API token / HuggingFace token (NO default — required for LLM)
  LOCAL_IMAGE_NAME — Optional, for from_docker_image()

Output format (OpenEnv spec):
  [START] task=<task_id> env=cyber-defense-env model=<model>
  [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
"""

import os
import sys
import json
import argparse
from openai import OpenAI

# ── Hackathon-required environment variables ──────────────────────────────────
API_BASE_URL     = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1/")
MODEL_NAME       = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN         = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

ENV_NAME = "cyber-defense-env"
SUCCESS_THRESHOLD = 0.5

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a senior cybersecurity analyst operating an automated network defense system.

You receive 8 sensor readings every step and must choose one defense action.

The attacker follows a kill chain you CANNOT see directly:
Stage 0 = Dormant → Stage 1 = Recon → Stage 2 = Exploit → Stage 3 = Persist → Stage 4 = Exfil

AVAILABLE ACTIONS — choose exactly one:
do_nothing         Cost 0.00  No effect. Use only when all signals are truly low.
monitor_traffic    Cost 0.05  Halves sensor noise next step. Use when uncertain.
block_ip           Cost 0.10  Stop connections. Best vs Recon (stage 1) and Exploit (stage 2).
scan_system        Cost 0.20  Deep scan. Best vs Persistence (stage 3).
throttle_bandwidth Cost 0.10  Limit outbound data. Best vs Exfiltration (stage 4).
isolate_subsystem  Cost 0.30  Emergency full stop. Use when health < 0.40 OR all signals maxed.

STAGE INFERENCE RULES:
• High network_traffic_anomaly + high failed_login_rate → Stage 1 or 2 → block_ip
• High suspicious_process_score (> 0.55) → Stage 3 → scan_system
• High bandwidth_usage (> 0.60) + high alert_level → Stage 4 → throttle_bandwidth
• System health < 0.40 → Emergency → isolate_subsystem
• All signals < 0.25 → No threat → do_nothing
• Signals mixed or unclear → monitor_traffic

Respond in EXACTLY this format with no other text:
ACTION: <action_name>
REASONING: <one sentence>"""

TASK_PROMPTS = {
    "detect_recon": (
        "TASK GOAL: Detect reconnaissance early. "
        "At the first sign of elevated network_traffic_anomaly or failed_login_rate, "
        "respond immediately with monitor_traffic or block_ip."
    ),
    "stop_exploit": (
        "TASK GOAL: Stop active exploitation before it reaches persistence (stage 3). "
        "Use block_ip aggressively when login failures and traffic anomalies are high. "
        "If bandwidth_usage exceeds 0.60, switch immediately to throttle_bandwidth."
    ),
    "prevent_exfil": (
        "TASK GOAL: Prevent data exfiltration at ALL costs. "
        "Watch bandwidth_usage every step — if it exceeds 0.55, use throttle_bandwidth NOW. "
        "If it exceeds 0.70 or health drops below 0.40, use isolate_subsystem."
    ),
}

# ── Logging helpers (exact OpenEnv spec) ─────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error=None) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str} -> Avg score '{task}': {score:.4f}", flush=True)


# ── LLM call ──────────────────────────────────────────────────────────────────

def call_llm(observation_text: str, history: list, task_id: str, step: int) -> tuple:
    task_context = TASK_PROMPTS.get(task_id, "")
    user_message = (
        f"TASK: {task_id}\n"
        f"GOAL: {task_context}\n\n"
        f"Step {step} sensor readings:\n{observation_text}\n\n"
        "Choose your action:"
    )
    if HF_TOKEN:
        return _call_openai_client(user_message, history)
    return _rule_based_fallback(observation_text)


def _call_openai_client(user_message: str, history: list) -> tuple:
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=150,
            messages=(
                [{"role": "system", "content": SYSTEM_PROMPT}]
                + history
                + [{"role": "user", "content": user_message}]
            ),
        )
        raw = resp.choices[0].message.content
        return _parse_response(raw)
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", file=sys.stderr, flush=True)
        return _rule_based_fallback(user_message)


def _parse_response(raw: str) -> tuple:
    valid = ["do_nothing", "monitor_traffic", "block_ip", "scan_system",
             "throttle_bandwidth", "isolate_subsystem"]
    action = "do_nothing"
    reasoning = "No reasoning provided"
    for line in raw.strip().splitlines():
        upper = line.upper()
        if upper.startswith("ACTION:"):
            candidate = line.split(":", 1)[1].strip().lower().replace("-", "_")
            if candidate in valid:
                action = candidate
            else:
                for v in valid:
                    if v.startswith(candidate[:5]):
                        action = v
                        break
        elif upper.startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()
    return action, reasoning


def _rule_based_fallback(obs_text: str) -> tuple:
    def extract(keyword: str) -> float:
        kw = keyword.lower()
        for line in obs_text.splitlines():
            if kw in line.lower():
                for token in line.split():
                    try:
                        v = float(token)
                        if 0.0 <= v <= 1.0:
                            return v
                    except ValueError:
                        continue
        return 0.0

    health = extract("health")
    bw     = extract("bandwidth")
    proc   = extract("process")
    login  = extract("login")
    alert  = extract("alert")
    net    = extract("traffic")

    if health < 0.35:
        return "isolate_subsystem", f"Health critical ({health:.2f}) — emergency isolation"
    if bw > 0.60:
        return "throttle_bandwidth", f"Bandwidth {bw:.2f} — exfiltration in progress, throttling"
    if proc > 0.55:
        return "scan_system", f"Suspicious processes ({proc:.2f}) — persistence likely, scanning"
    if login > 0.45 and alert > 0.35:
        return "block_ip", f"Login failures ({login:.2f}) + alerts ({alert:.2f}) — blocking"
    if net > 0.30 or alert > 0.28:
        return "monitor_traffic", f"Elevated signals (net={net:.2f}, alert={alert:.2f}) — monitoring"
    return "do_nothing", f"All signals low (health={health:.2f}) — no action needed"


# ── Standalone runner ─────────────────────────────────────────────────────────

def run_standalone(task_id: str, seed: int = None, verbose: bool = True) -> dict:
    from env.cyber_env import CyberDefenseEnv
    from tasks.graders import grade

    max_steps = {"detect_recon": 30, "stop_exploit": 40, "prevent_exfil": 50}
    env = CyberDefenseEnv(max_steps=max_steps.get(task_id, 50))
    obs_arr, info = env.reset(seed=seed)

    episode_log = []
    rewards = []
    total_reward = 0.0
    history = []

    log_start(task=task_id, env=ENV_NAME, model=MODEL_NAME)

    while not env.done:
        step = info["step"]
        obs_dict = env.obs_to_dict()
        obs_text = env.obs_to_text(obs_dict)

        action, reasoning = call_llm(obs_text, history, task_id, step + 1)

        obs_arr, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        rewards.append(reward)

        log_entry = {
            "step": info["step"],
            "action": action,
            "reasoning": reasoning,
            "reward": round(reward, 4),
            "info": info,
            "false_positive": info.get("false_positive", False),
        }
        episode_log.append(log_entry)

        history.append({"role": "user",      "content": f"Step {step + 1}:\n{obs_text}"})
        history.append({"role": "assistant", "content": f"ACTION: {action}\nREASONING: {reasoning}"})
        if len(history) > 12:
            history = history[-12:]

        log_step(step=info["step"], action=action, reward=reward, done=env.done)

    grade_result = grade(task_id, episode_log)
    score = max(0.01, min(0.99, float(grade_result.get("score", 0.5))))
    success = score >= SUCCESS_THRESHOLD

    log_end(success=success, steps=len(episode_log), score=score, rewards=rewards)

    return {
        "task_id":      task_id,
        "score":        score,
        "success":      success,
        "total_reward": round(total_reward, 4),
        "steps":        len(episode_log),
        "grade_detail": grade_result,
        "log":          episode_log,
    }


# ── HTTP runner ───────────────────────────────────────────────────────────────

def run_http(task_id: str, server_url: str, seed: int = None, verbose: bool = True) -> dict:
    import urllib.request

    def post(path: str, body: dict) -> dict:
        data = json.dumps(body).encode()
        req  = urllib.request.Request(
            f"{server_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    resp       = post("/reset", {"task_id": task_id, "seed": seed})
    session_id = resp["session_id"]
    history, total_reward, steps = [], 0.0, 0
    rewards = []

    log_start(task=task_id, env=ENV_NAME, model=MODEL_NAME)

    while not resp.get("done", False) and not resp.get("episode_complete", False):
        obs_text          = resp["observation_text"]
        action, reasoning = call_llm(obs_text, history, task_id, steps + 1)
        resp              = post("/step", {"session_id": session_id, "action": action, "reasoning": reasoning})
        steps            += 1
        reward            = resp.get("reward", 0.0)
        total_reward     += reward
        rewards.append(reward)
        done              = resp.get("done", False) or resp.get("episode_complete", False)

        history.append({"role": "user",      "content": f"Step {steps}:\n{obs_text}"})
        history.append({"role": "assistant", "content": f"ACTION: {action}\nREASONING: {reasoning}"})
        if len(history) > 12:
            history = history[-12:]

        log_step(step=steps, action=action, reward=reward, done=done)

        if resp.get("episode_complete"):
            break

    grade_result = resp.get("grade", {})
    score   = max(0.01, min(0.99, float(grade_result.get("score", 0.5))))
    success = score >= SUCCESS_THRESHOLD

    log_end(success=success, steps=steps, score=score, rewards=rewards)

    return {"task_id": task_id, "score": score, "steps": steps}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ACD-Env LLM Agent Runner")
    parser.add_argument("--task", default="stop_exploit",
                        choices=["detect_recon", "stop_exploit", "prevent_exfil"])
    parser.add_argument("--all-tasks",  action="store_true", help="Run all three tasks")
    parser.add_argument("--standalone", action="store_true",
                        help="Run directly without HTTP server (default if no --server)")
    parser.add_argument("--server",     default=None,
                        help="Server URL e.g. http://localhost:7860")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--quiet",      action="store_true")
    args = parser.parse_args()

    tasks   = ["detect_recon", "stop_exploit", "prevent_exfil"] if args.all_tasks else [args.task]
    results = []

    for task in tasks:
        if args.server and not args.standalone:
            r = run_http(task, args.server, seed=args.seed, verbose=not args.quiet)
        else:
            r = run_standalone(task, seed=args.seed, verbose=not args.quiet)
        results.append(r)

    if len(results) > 1:
        avg = sum(r["score"] for r in results) / len(results)
        print(f"\nAverage score: {avg:.4f}")


if __name__ == "__main__":
    main()
