"""
tests/test_presubmission.py

Pre-submission validation suite.
Runs all checks required before submitting to OpenEnv hackathon.

Usage:
    python tests/test_presubmission.py

Every test must PASS for a valid submission.
"""
import sys
import os
import json
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PASS = "PASS"
FAIL = "FAIL"
results = []
_registry = []   # ordered list of (name, wrapper) — all test functions registered here


def test(name: str):
    def decorator(fn):
        def wrapper():
            try:
                fn()
                results.append((PASS, name, ""))
                print(f"  {PASS}  {name}")
            except Exception as e:
                msg = str(e)
                results.append((FAIL, name, msg))
                print(f"  {FAIL}  {name}")
                print(f"       → {msg}")
        _registry.append((name, wrapper))  # register every test in order
        return wrapper
    return decorator


# ── FILE STRUCTURE ────────────────────────────────────────────────────────────

@test("Required files exist")
def _():
    required = [
        "app.py", "inference.py", "openenv.yaml",
        "requirements.txt", "Dockerfile", "README.md",
        "env/cyber_env.py", "env/attack_simulator.py", "env/reward.py",
        "tasks/graders.py",
    ]
    missing = [f for f in required if not os.path.exists(f)]
    assert not missing, f"Missing files: {missing}"


@test("openenv.yaml is valid and contains required fields")
def _():
    with open("openenv.yaml") as f:
        content = f.read()
    required_keys = ["name", "tasks", "environment", "server", "inference"]
    for k in required_keys:
        assert k in content, f"Missing key '{k}' in openenv.yaml"
    assert "detect_recon" in content
    assert "stop_exploit" in content
    assert "prevent_exfil" in content


@test("requirements.txt contains all necessary packages")
def _():
    with open("requirements.txt") as f:
        reqs = f.read()
    for pkg in ["fastapi", "uvicorn", "pydantic", "numpy"]:
        assert pkg in reqs, f"Missing package: {pkg}"


@test("Dockerfile exposes port 7860 and uses uvicorn")
def _():
    with open("Dockerfile") as f:
        content = f.read()
    assert "7860" in content, "Dockerfile must EXPOSE 7860"
    assert "uvicorn" in content, "Dockerfile must start uvicorn"
    assert "app:app" in content, "Dockerfile must reference app:app"


# ── IMPORTS ───────────────────────────────────────────────────────────────────

@test("env.cyber_env imports cleanly")
def _():
    from env.cyber_env import CyberDefenseEnv, ACTION_NAMES, OBS_FEATURES
    assert len(ACTION_NAMES) == 6
    assert len(OBS_FEATURES) == 8


@test("env.attack_simulator imports cleanly")
def _():
    from env.attack_simulator import AttackSimulator
    sim = AttackSimulator()
    assert sim is not None


@test("env.reward imports cleanly")
def _():
    from env.reward import RewardCalculator
    r = RewardCalculator()
    assert hasattr(r, "compute")


@test("tasks.graders imports cleanly")
def _():
    from tasks.graders import grade, GRADERS
    assert "detect_recon" in GRADERS
    assert "stop_exploit" in GRADERS
    assert "prevent_exfil" in GRADERS


@test("app.py imports cleanly (FastAPI app)")
def _():
    from app import app
    assert app is not None
    assert app.title == "Adaptive Cyber Defense Environment"


@test("inference.py imports cleanly")
def _():
    import importlib.util
    spec = importlib.util.spec_from_file_location("inference", "inference.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "run_standalone")
    assert hasattr(mod, "call_llm")
    assert hasattr(mod, "SYSTEM_PROMPT")


# ── ENVIRONMENT LOGIC ─────────────────────────────────────────────────────────

@test("CyberDefenseEnv reset() returns correct shapes")
def _():
    import numpy as np
    from env.cyber_env import CyberDefenseEnv
    env = CyberDefenseEnv()
    obs, info = env.reset(seed=42)
    assert obs.shape == (8,), f"Expected (8,), got {obs.shape}"
    assert obs.dtype == np.float32
    assert all(0.0 <= x <= 1.0 for x in obs), "Obs values out of [0,1]"
    assert "system_health" in info
    assert "attack_stage" in info
    assert info["system_health"] == 1.0


@test("CyberDefenseEnv step() with all 6 valid actions")
def _():
    from env.cyber_env import CyberDefenseEnv, ACTION_NAMES
    env = CyberDefenseEnv()
    env.reset(seed=0)
    for action in ACTION_NAMES:
        env2 = CyberDefenseEnv()
        env2.reset(seed=1)
        obs, reward, terminated, truncated, info = env2.step(action)
        assert obs.shape == (8,), f"Bad obs shape for action '{action}'"
        assert isinstance(reward, float), f"Reward must be float for '{action}'"
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "action_taken" in info


@test("CyberDefenseEnv step() with integer actions")
def _():
    from env.cyber_env import CyberDefenseEnv
    env = CyberDefenseEnv()
    env.reset(seed=5)
    for i in range(6):
        env2 = CyberDefenseEnv()
        env2.reset(seed=5)
        obs, reward, terminated, truncated, info = env2.step(i)
        assert obs.shape == (8,)


@test("Episode terminates correctly (truncated at max_steps)")
def _():
    from env.cyber_env import CyberDefenseEnv
    env = CyberDefenseEnv(max_steps=5)
    env.reset(seed=99)
    for _ in range(5):
        obs, r, term, trunc, info = env.step("do_nothing")
    assert trunc or term, "Should have terminated by step 5"


@test("Episode terminates on attack stopped")
def _():
    from env.cyber_env import CyberDefenseEnv
    import numpy as np
    for seed in range(20):
        env = CyberDefenseEnv(max_steps=100)
        env.reset(seed=seed)
        done = False
        steps = 0
        while not done and steps < 100:
            _, _, term, trunc, info = env.step("isolate_subsystem")
            done = term or trunc
            steps += 1
            if info.get("attack_stopped"):
                assert term, "terminated should be True when attack stopped"
                break


@test("Observation values are within [0, 1] for 50 episodes")
def _():
    from env.cyber_env import CyberDefenseEnv
    import numpy as np
    env = CyberDefenseEnv()
    for seed in range(50):
        obs, _ = env.reset(seed=seed)
        assert all(0.0 <= x <= 1.0 for x in obs), f"Obs out of range at seed {seed}: {obs}"


@test("Seeded reset() is reproducible")
def _():
    import numpy as np
    from env.cyber_env import CyberDefenseEnv
    env = CyberDefenseEnv()
    obs_a, _ = env.reset(seed=77)
    obs_b, _ = env.reset(seed=77)
    assert np.allclose(obs_a, obs_b), "Same seed must produce same obs"


@test("obs_to_text() produces non-empty string")
def _():
    from env.cyber_env import CyberDefenseEnv
    env = CyberDefenseEnv()
    env.reset(seed=0)
    text = env.obs_to_text()
    assert isinstance(text, str) and len(text) > 50
    assert "health" in text.lower()
    assert "bandwidth" in text.lower()


# ── ATTACK SIMULATOR ─────────────────────────────────────────────────────────

@test("AttackSimulator progresses through all stages")
def _():
    from env.attack_simulator import AttackSimulator
    sim = AttackSimulator(seed=1)
    sim.reset(seed=1)
    stages_seen = set()
    for _ in range(300):
        state = sim.step("do_nothing", False)
        stages_seen.add(state["stage"])
    assert len(stages_seen) >= 3, f"Only saw stages: {stages_seen}"
    assert max(stages_seen) >= 3, "Never reached late stages"


@test("Isolate action stops attack with high probability")
def _():
    from env.attack_simulator import AttackSimulator
    stops = 0
    for seed in range(30):
        sim = AttackSimulator(seed=seed)
        sim.reset(seed=seed)
        for _ in range(20):
            sim.step("do_nothing", False)
        for _ in range(5):
            result = sim.step("isolate_subsystem", True)
            if result["stopped"]:
                stops += 1
                break
    assert stops >= 20, f"Isolation only stopped {stops}/30 — too low"


@test("AttackSimulator damage is 0 when stopped")
def _():
    from env.attack_simulator import AttackSimulator
    sim = AttackSimulator(seed=0)
    sim.reset(seed=0)
    for _ in range(15):
        sim.step("do_nothing", False)
    # Force stop
    for _ in range(10):
        result = sim.step("isolate_subsystem", True)
        if result["stopped"]:
            assert result["damage_this_step"] == 0.0
            break


# ── REWARD FUNCTION ───────────────────────────────────────────────────────────

@test("Reward signs are correct (positive for good, negative for bad)")
def _():
    from env.reward import RewardCalculator
    r = RewardCalculator()

    cases = [
        # (label, kwargs, expected_sign)
        ("Early stop",     dict(action="block_ip",   prev_stage=1, new_stage=1, attack_stopped=True,  system_health=0.9, resource_cost=0.10, false_positive=False, monitoring_active=True),  "+"),
        ("False positive", dict(action="block_ip",   prev_stage=0, new_stage=0, attack_stopped=False, system_health=1.0, resource_cost=0.10, false_positive=True,  monitoring_active=False), "-"),
        ("Missed attack",  dict(action="do_nothing", prev_stage=3, new_stage=4, attack_stopped=False, system_health=0.5, resource_cost=0.00, false_positive=False, monitoring_active=False), "-"),
        ("Efficient stop", dict(action="monitor_traffic", prev_stage=1, new_stage=1, attack_stopped=True, system_health=0.9, resource_cost=0.05, false_positive=False, monitoring_active=True), "+"),
    ]
    for label, kwargs, sign in cases:
        val = r.compute(**kwargs)
        if sign == "+":
            assert val > 0, f"{label}: expected positive reward, got {val}"
        else:
            assert val < 0, f"{label}: expected negative reward, got {val}"


@test("Reward breakdown dict is populated after compute()")
def _():
    from env.reward import RewardCalculator
    r = RewardCalculator()
    r.compute("block_ip", 1, 1, True, 0.9, 0.1, False, True)
    assert isinstance(r.last_breakdown, dict)
    assert "total" in r.last_breakdown
    assert "stop" in r.last_breakdown


# ── TASK GRADERS ──────────────────────────────────────────────────────────────

def _make_log(actions_stages: list) -> list:
    """Helper to build a minimal episode log."""
    log = []
    for i, (action, stage) in enumerate(actions_stages):
        log.append({
            "step": i + 1,
            "action": action,
            "reasoning": "test",
            "reward": 1.0,
            "false_positive": False,
            "info": {
                "step": i + 1,
                "attack_stage": stage,
                "attack_stage_name": ["dormant","recon","exploit","persist","exfil"][stage],
                "system_health": 0.9,
                "attack_stopped": (i == len(actions_stages) - 1),
                "exfil_complete": False,
            }
        })
    return log


@test("grade_detect_recon returns 0.0–1.0 float")
def _():
    from tasks.graders import grade_detect_recon
    log = _make_log([("do_nothing",0),("do_nothing",1),("monitor_traffic",1),("block_ip",1)])
    result = grade_detect_recon(log)
    assert "score" in result
    assert 0.0 <= result["score"] <= 1.0, f"Score out of range: {result['score']}"


@test("grade_stop_exploit returns 0.0–1.0 float")
def _():
    from tasks.graders import grade_stop_exploit
    log = _make_log([("do_nothing",1),("block_ip",2),("block_ip",2)])
    result = grade_stop_exploit(log)
    assert 0.0 <= result["score"] <= 1.0


@test("grade_prevent_exfil returns 0.0 when exfil_complete")
def _():
    from tasks.graders import grade_prevent_exfil
    log = _make_log([("do_nothing",4),("do_nothing",4)])
    log[-1]["info"]["exfil_complete"] = True
    log[-1]["info"]["attack_stopped"] = False
    result = grade_prevent_exfil(log)
    assert result["score"] == 0.0, f"Expected 0.0 when exfil complete, got {result['score']}"


@test("grade() dispatcher works for all task IDs")
def _():
    from tasks.graders import grade
    log = _make_log([("block_ip",1),("block_ip",1)])
    for task_id in ["detect_recon","stop_exploit","prevent_exfil"]:
        result = grade(task_id, log)
        assert "score" in result
        assert 0.0 <= result["score"] <= 1.0, f"{task_id}: score={result['score']}"


# ── FastAPI ROUTES ────────────────────────────────────────────────────────────

@test("FastAPI app has all required routes")
def _():
    from app import app
    routes = {r.path for r in app.routes}
    required = {"/", "/tasks", "/reset", "/step", "/state", "/grade", "/actions", "/observation_space"}
    missing = required - routes
    assert not missing, f"Missing routes: {missing}"


@test("FastAPI /reset returns session_id and observation")
def _():
    from fastapi.testclient import TestClient
    from app import app
    client = TestClient(app)
    resp = client.post("/reset", json={"task_id": "stop_exploit", "seed": 0})
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert "observation" in data
    assert "task" in data
    assert len(data["observation"]) == 8


@test("FastAPI /step returns reward and observation")
def _():
    from fastapi.testclient import TestClient
    from app import app
    client = TestClient(app)
    resp = client.post("/reset", json={"task_id": "stop_exploit", "seed": 1})
    sid = resp.json()["session_id"]
    resp2 = client.post("/step", json={"session_id": sid, "action": "monitor_traffic", "reasoning": "test"})
    assert resp2.status_code == 200
    data = resp2.json()
    assert "reward" in data
    assert "observation" in data
    assert "system_health" in data


@test("FastAPI /step with invalid session returns 404")
def _():
    from fastapi.testclient import TestClient
    from app import app
    client = TestClient(app)
    resp = client.post("/step", json={"session_id": "invalid-session-xyz", "action": "do_nothing"})
    assert resp.status_code == 404


@test("FastAPI /step with invalid action defaults to do_nothing")
def _():
    from fastapi.testclient import TestClient
    from app import app
    client = TestClient(app)
    resp = client.post("/reset", json={"task_id": "detect_recon", "seed": 2})
    sid = resp.json()["session_id"]
    resp2 = client.post("/step", json={"session_id": sid, "action": "INVALID_ACTION_XYZ"})
    assert resp2.status_code == 200
    assert resp2.json()["observation"] is not None


@test("FastAPI full episode completes and returns grade")
def _():
    from fastapi.testclient import TestClient
    from app import app
    client = TestClient(app)
    resp = client.post("/reset", json={"task_id": "detect_recon", "seed": 3})
    sid = resp.json()["session_id"]

    for _ in range(35):  # max_steps=30 + buffer
        r = client.post("/step", json={"session_id": sid, "action": "block_ip"})
        data = r.json()
        if data.get("episode_complete") or data.get("done"):
            assert "grade" in data
            assert 0.0 <= data["grade"]["score"] <= 1.0
            break
    else:
        raise AssertionError("Episode never completed")


@test("FastAPI /grade endpoint returns score for finished session")
def _():
    from fastapi.testclient import TestClient
    from app import app
    client = TestClient(app)
    resp = client.post("/reset", json={"task_id": "stop_exploit", "seed": 9})
    sid = resp.json()["session_id"]
    for _ in range(50):
        r = client.post("/step", json={"session_id": sid, "action": "isolate_subsystem"})
        if r.json().get("episode_complete"):
            break
    grade_resp = client.post("/grade", json={"session_id": sid})
    assert grade_resp.status_code == 200
    assert "score" in grade_resp.json()


@test("FastAPI /tasks returns 3 tasks")
def _():
    from fastapi.testclient import TestClient
    from app import app
    client = TestClient(app)
    resp = client.get("/tasks")
    assert resp.status_code == 200
    tasks = resp.json()["tasks"]
    assert len(tasks) == 3
    ids = {t["id"] for t in tasks}
    assert ids == {"detect_recon", "stop_exploit", "prevent_exfil"}


@test("FastAPI /actions returns 6 actions")
def _():
    from fastapi.testclient import TestClient
    from app import app
    client = TestClient(app)
    resp = client.get("/actions")
    assert resp.status_code == 200
    actions = resp.json()["actions"]
    assert len(actions) == 6


@test("FastAPI /observation_space describes all 8 features")
def _():
    from fastapi.testclient import TestClient
    from app import app
    client = TestClient(app)
    resp = client.get("/observation_space")
    assert resp.status_code == 200
    features = resp.json()["features"]
    assert len(features) == 8


# ── INFERENCE ─────────────────────────────────────────────────────────────────

@test("inference.py run_standalone completes an episode")
def _():
    import importlib.util
    spec = importlib.util.spec_from_file_location("inference", "inference.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    result = mod.run_standalone("detect_recon", seed=0, verbose=False)
    assert "score" in result
    assert 0.0 <= result["score"] <= 1.0
    assert result["steps"] > 0


@test("inference.py run_standalone works for all 3 tasks")
def _():
    import importlib.util
    spec = importlib.util.spec_from_file_location("inference", "inference.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for task in ["detect_recon", "stop_exploit", "prevent_exfil"]:
        result = mod.run_standalone(task, seed=42, verbose=False)
        assert 0.0 <= result["score"] <= 1.0, f"{task} returned invalid score"


@test("SYSTEM_PROMPT is non-trivial and mentions all actions")
def _():
    import importlib.util
    spec = importlib.util.spec_from_file_location("inference", "inference.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    prompt = mod.SYSTEM_PROMPT
    assert len(prompt) > 200, "SYSTEM_PROMPT too short"
    for action in ["block_ip","scan_system","throttle_bandwidth","isolate_subsystem"]:
        assert action in prompt, f"SYSTEM_PROMPT missing action: {action}"


# ── END-TO-END ────────────────────────────────────────────────────────────────

@test("End-to-end: 10 episodes with rule-based agent, all complete cleanly")
def _():
    from env.cyber_env import CyberDefenseEnv
    from tasks.graders import grade

    for seed in range(10):
        env = CyberDefenseEnv(max_steps=50)
        obs, info = env.reset(seed=seed)
        log = []
        while not env.done:
            # Simple rule-based
            obs_d = env.obs_to_dict()
            bw    = obs_d["bandwidth_usage"]
            proc  = obs_d["suspicious_process_score"]
            login = obs_d["failed_login_rate"]
            alert = obs_d["alert_level"]
            h     = info["system_health"]

            if h < 0.3:           action = "isolate_subsystem"
            elif bw > 0.7:        action = "throttle_bandwidth"
            elif proc > 0.6:      action = "scan_system"
            elif login > 0.5:     action = "block_ip"
            elif alert > 0.3:     action = "monitor_traffic"
            else:                 action = "do_nothing"

            obs, reward, term, trunc, info = env.step(action)
            log.append({"step":info["step"],"action":action,"reasoning":"auto",
                        "reward":reward,"info":info,"false_positive":info.get("false_positive",False)})

        result = grade("stop_exploit", log)
        assert 0.0 <= result["score"] <= 1.0, f"Seed {seed}: bad score {result['score']}"
        assert info["system_health"] >= 0.0


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  ACD-Env Pre-Submission Test Suite")
    print("="*60 + "\n")

    os.chdir(os.path.join(os.path.dirname(__file__), ".."))

    # Run every test in registration order
    for _name, _fn in _registry:
        _fn()

    passed = sum(1 for r in results if r[0] == PASS)
    failed = sum(1 for r in results if r[0] == FAIL)
    total  = len(results)

    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{total} passed")
    if failed:
        print(f"\n  FAILED TESTS:")
        for status, name, msg in results:
            if status == FAIL:
                print(f"    ✗ {name}")
                print(f"      {msg}")
    print("="*60)

    if failed > 0:
        print("\n  ✗ NOT READY — fix failing tests before submitting\n")
        sys.exit(1)
    else:
        print("\n  ✓ ALL TESTS PASSED — ready to submit!\n")
        sys.exit(0)
