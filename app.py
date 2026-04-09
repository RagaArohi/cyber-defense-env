"""
app.py — OpenEnv-compatible FastAPI server for CyberDefenseEnv

Endpoints:
  GET  /                    Health check + env info
  GET  /tasks               List available tasks
  POST /reset               Start new episode
  POST /step                Take one action
  GET  /state               Current observation (for polling agents)
  POST /grade               Grade a completed episode
  GET  /actions             List valid actions with descriptions
  GET  /observation_space   Describe all observation features

Deploy: uvicorn app:app --host 0.0.0.0 --port 7860
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
import uuid
import time

from env.cyber_env import CyberDefenseEnv, ACTION_NAMES, ACTION_COSTS, OBS_FEATURES
from tasks.graders import grade, GRADERS
from config import cfg

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Adaptive Cyber Defense Environment",
    description=(
        "An OpenEnv-compatible reinforcement learning environment where an LLM agent "
        "must detect and stop multi-stage cyber attacks under partial observability."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    cfg.validate()   # prints API key status + tips to console


# ── Session store (in-memory) ─────────────────────────────────────────────────
# Maps session_id → {"env": CyberDefenseEnv, "log": [...], "task_id": str, ...}
_sessions: Dict[str, dict] = {}

TASK_CATALOG = {
    "detect_recon": {
        "id": "detect_recon",
        "name": "Detect Reconnaissance",
        "difficulty": "easy",
        "description": (
            "An attacker has started passive reconnaissance. "
            "Identify the threat early and respond with monitoring or blocking actions "
            "before the attack escalates. Keep system health above 0.70."
        ),
        "max_steps": 30,
        "success_threshold": 0.7,
        "grader": "tasks.graders.grade_detect_recon",
        "has_grader": True,
    },
    "stop_exploit": {
        "id": "stop_exploit",
        "name": "Stop Active Exploitation",
        "difficulty": "medium",
        "description": (
            "An attacker is actively attempting to exploit a vulnerability. "
            "Stop the attack before it reaches the persistence stage (stage 3). "
            "Use targeted actions and manage your resource budget carefully."
        ),
        "max_steps": 40,
        "success_threshold": 0.6,
        "grader": "tasks.graders.grade_stop_exploit",
        "has_grader": True,
    },
    "prevent_exfil": {
        "id": "prevent_exfil",
        "name": "Prevent Data Exfiltration",
        "difficulty": "hard",
        "description": (
            "The attacker has progressed through multiple stages and is attempting "
            "to exfiltrate data. Prevent data theft using the right actions at the right time. "
            "Avoid false positives and maintain system availability."
        ),
        "max_steps": 50,
        "success_threshold": 0.5,
        "grader": "tasks.graders.grade_prevent_exfil",
        "has_grader": True,
    },
}

ACTION_DESCRIPTIONS = {
    "do_nothing":         "Take no action. Free, but allows attack to progress.",
    "monitor_traffic":    "Monitor network traffic. Cost 0.05. Reduces observation noise by 50%.",
    "block_ip":           "Block suspicious IP addresses. Cost 0.10. Effective against recon/exploit.",
    "scan_system":        "Full system scan. Cost 0.20. Effective against persistence.",
    "throttle_bandwidth": "Throttle outbound bandwidth. Cost 0.10. Very effective against exfiltration.",
    "isolate_subsystem":  "Isolate affected subsystem. Cost 0.30. Nearly stops all attack stages but hurts uptime.",
}

# ── Request/Response models ───────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = Field("stop_exploit", description="Task to run")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")

    @field_validator("task_id")
    @classmethod
    def valid_task(cls, v):
        if v not in TASK_CATALOG:
            raise ValueError(f"Unknown task '{v}'. Valid: {list(TASK_CATALOG)}")
        return v


class StepRequest(BaseModel):
    session_id: str = Field(..., description="Session ID from /reset")
    action: str = Field(..., description="Action name string")
    reasoning: Optional[str] = Field(None, description="Agent's reasoning (logged, not required)")

    @field_validator("action")
    @classmethod
    def valid_action(cls, v):
        v = v.strip().lower()
        if v not in ACTION_NAMES:
            # Fuzzy match
            for name in ACTION_NAMES:
                if name.startswith(v[:4]):
                    return name
            return "do_nothing"
        return v


class GradeRequest(BaseModel):
    session_id: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_session(session_id: str) -> dict:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. Call /reset first.")
    return _sessions[session_id]


def _obs_payload(env: CyberDefenseEnv, info: dict, reward: float = None) -> dict:
    obs = env.obs_to_dict()           # single call to _observe() — consistent noise
    obs_text = env.obs_to_text(obs)   # reuse the same obs dict, no second _observe()
    payload = {
        "observation": obs,
        "observation_text": obs_text,
        "system_health": info["system_health"],
        "resource_budget": info["resource_budget"],
        "step": info["step"],
        "done": env.done,
        "terminated": info.get("attack_stopped", False) or info.get("exfil_complete", False) or info["system_health"] <= 0,
        "truncated": info["step"] >= env.max_steps,
    }
    if reward is not None:
        payload["reward"] = round(reward, 4)
        payload["reward_breakdown"] = info.get("reward_breakdown", {})
    return payload


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "Adaptive Cyber Defense Environment",
        "version": "1.0.0",
        "framework": "OpenEnv",
        "status": "ready",
        "active_sessions": len(_sessions),
        "llm_provider": cfg.llm_provider,
        "active_model": cfg.active_model,
        "llm_enabled": cfg.using_llm,
        "api_base_url": cfg.api_base_url,
        "model_name": cfg.model_name,
        "endpoints": ["/tasks", "/reset", "/step", "/state", "/grade", "/actions", "/observation_space"],
    }


@app.get("/tasks")
def list_tasks():
    return {"tasks": list(TASK_CATALOG.values())}


@app.get("/actions")
def list_actions():
    return {
        "actions": [
            {
                "name": name,
                "cost": ACTION_COSTS[name],
                "description": ACTION_DESCRIPTIONS[name],
            }
            for name in ACTION_NAMES
        ]
    }


@app.get("/observation_space")
def observation_space():
    return {
        "type": "continuous",
        "shape": [8],
        "features": [
            {"name": feat, "range": [0.0, 1.0], "description": feat.replace("_", " ").title()}
            for feat in OBS_FEATURES
        ],
        "note": (
            "Observations are NOISY (Gaussian noise applied). "
            "The true attack stage is NOT directly visible. "
            "Agent must infer threat level from correlated signals."
        ),
    }


@app.post("/reset")
def reset(req: ResetRequest):
    task = TASK_CATALOG[req.task_id]
    env = CyberDefenseEnv(max_steps=task["max_steps"])
    obs_arr, info = env.reset(seed=req.seed)

    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "env": env,
        "task_id": req.task_id,
        "log": [],
        "started_at": time.time(),
        "total_reward": 0.0,
        "steps": 0,
    }

    return {
        "session_id": session_id,
        "task": task,
        **_obs_payload(env, info),
        "message": (
            "Episode started. Send POST /step with your action. "
            f"Available actions: {ACTION_NAMES}"
        ),
    }


@app.post("/step")
def step(req: StepRequest):
    sess = _get_session(req.session_id)
    env: CyberDefenseEnv = sess["env"]

    if env.done:
        raise HTTPException(status_code=400, detail="Episode is finished. Call /reset to start a new one.")

    obs_arr, reward, terminated, truncated, info = env.step(req.action)
    sess["total_reward"] += reward
    sess["steps"] += 1

    # Log this step
    log_entry = {
        "step": info["step"],
        "action": req.action,
        "reasoning": req.reasoning,
        "reward": round(reward, 4),
        "info": info,
        "false_positive": info.get("false_positive", False),
    }
    sess["log"].append(log_entry)

    payload = _obs_payload(env, info, reward)
    payload["session_id"] = req.session_id
    payload["total_reward"] = round(sess["total_reward"], 4)

    if env.done:
        # Auto-grade when episode ends
        grade_result = grade(sess["task_id"], sess["log"])
        payload["episode_complete"] = True
        payload["grade"] = grade_result
        payload["message"] = f"Episode complete. Score: {grade_result['score']:.2f}. Call /reset for a new episode."
    else:
        payload["episode_complete"] = False
        payload["actions_available"] = ACTION_NAMES

    return payload


@app.get("/state")
def state(session_id: str):
    sess = _get_session(session_id)
    env: CyberDefenseEnv = sess["env"]
    info = env._info()
    return {
        "session_id": session_id,
        "task_id": sess["task_id"],
        "total_reward": round(sess["total_reward"], 4),
        **_obs_payload(env, info),
    }


@app.post("/grade")
def grade_episode(req: GradeRequest):
    sess = _get_session(req.session_id)

    if not sess["log"]:
        raise HTTPException(status_code=400, detail="No steps recorded yet.")

    result = grade(sess["task_id"], sess["log"])
    result["task_id"] = sess["task_id"]
    result["total_reward"] = round(sess["total_reward"], 4)
    result["steps_taken"] = sess["steps"]
    result["session_id"] = req.session_id
    return result
