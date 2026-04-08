---
title: Cyber Defense Env
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---
# 🛡️ Adaptive Cyber Defense Environment (ACD-Env)

> An OpenEnv-compatible environment where an LLM agent acts as a cybersecurity analyst, detecting and stopping multi-stage cyber attacks under partial observability and resource constraints.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://openenv.ai)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-orange)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 🎯 Problem Statement

Modern Security Operations Centers receive thousands of alerts daily. Human analysts cannot respond to every threat in real-time. This environment simulates that challenge:

- An **attacker** executes a realistic multi-stage cyber attack (Recon → Exploit → Persist → Exfil)
- An **LLM agent** must detect and stop the attack using 6 discrete defense actions
- Observations are **noisy and incomplete** — the agent must infer threat level, not read it directly
- Actions have **resource costs** — the agent must balance security vs. system availability

**Why LLMs?** Rule-based systems are brittle against adaptive attackers. LLMs can reason about ambiguous, partially observable situations the same way a skilled analyst does.

---

## ⚡ Quick Start

### Step 0 — API Key setup (optional but recommended)

The environment runs fully without any API key using the built-in rule-based agent. Adding a key enables LLM-powered reasoning for higher scores.

```bash
# Copy the template and fill in your key
cp .env.example .env
# Then edit .env and set HF_TOKEN=hf_...
```

Get your HuggingFace token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

> **You can skip this entirely.** All 3 tasks pass their success thresholds with the rule-based fallback. Just run any option below and it works immediately.

---

### Option A — Direct Python (no server needed)
```bash
pip install -r requirements.txt

# No API key — uses built-in rule-based agent
python inference.py --all-tasks --standalone

# With API key in .env (auto-loaded)
python inference.py --all-tasks --standalone

# Or pass the token inline
HF_TOKEN=hf_... python inference.py --all-tasks --standalone
```

### Option B — FastAPI Server
```bash
# Terminal 1
uvicorn app:app --host 0.0.0.0 --port 7860

# Terminal 2
python inference.py --all-tasks --server http://localhost:7860
```

### Option C — Docker
```bash
docker build -t cyber-defense-env .
docker run -p 7860:7860 cyber-defense-env

# With Anthropic API key passed into container
docker run -p 7860:7860 -e HF_TOKEN=hf_... cyber-defense-env
```

### Option D — Deploy to HuggingFace Spaces

This project is HuggingFace Spaces ready out of the box.

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Set **SDK** to **Docker** and **Hardware** to CPU Basic (free tier)
3. Clone your Space repo and push all project files:

```bash
git clone https://huggingface.co/spaces/<your-username>/<your-space-name>
cp -r cyber-defense-openenv/. <your-space-name>/
cd <your-space-name>
git add . && git commit -m "initial deploy" && git push
```

4. *(Optional)* Add your HuggingFace token as a Space Secret in **Settings → Variables and secrets**, name it `HF_TOKEN`
5. Once the build completes, the API and interactive docs are live at:

```
https://<your-username>-<your-space-name>.hf.space/docs
```

> The environment runs fully without an API key using the built-in rule-based fallback agent. The key is only needed to enable LLM-powered reasoning.

### Pre-submission validation
```bash
python tests/test_presubmission.py
```

---

## 🌐 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Health check |
| `GET`  | `/tasks` | List 3 tasks |
| `GET`  | `/actions` | List 6 actions with costs |
| `GET`  | `/observation_space` | Describe 8 observation features |
| `POST` | `/reset` | Start new episode → returns `session_id` |
| `POST` | `/step` | Take one action → returns obs + reward |
| `GET`  | `/state` | Current observation |
| `POST` | `/grade` | Grade completed episode → score 0.0–1.0 |

### POST /reset
```json
{ "task_id": "stop_exploit", "seed": 42 }
```

### POST /step
```json
{ "session_id": "...", "action": "block_ip", "reasoning": "High login failures detected" }
```

### POST /grade
```json
{ "session_id": "..." }
```

---

## 🧠 Environment Design

### Observation Space — 8 continuous features [0.0, 1.0]

| Feature | High value means... |
|---------|---------------------|
| `network_traffic_anomaly` | Active scanning or data transfer |
| `failed_login_rate` | Brute force / exploitation attempt |
| `suspicious_process_score` | Malware / persistence installed |
| `cpu_usage` | Heavy computation |
| `memory_usage` | Resident malware |
| `alert_level` | General threat activity |
| `bandwidth_usage` | Active data exfiltration |
| `time_step_norm` | Episode progress (0→1) |

> ⚠️ **Partial observability**: Gaussian noise (σ=0.15) is applied. The true attack stage is never directly visible — the agent must infer it.

### Action Space — 6 discrete actions

| Action | Cost | Best Against |
|--------|------|-------------|
| `do_nothing` | 0.00 | — |
| `monitor_traffic` | 0.05 | Any (halves noise) |
| `block_ip` | 0.10 | Recon, Exploit |
| `scan_system` | 0.20 | Persistence |
| `throttle_bandwidth` | 0.10 | Exfiltration |
| `isolate_subsystem` | 0.30 | Emergency (any stage) |

### Attack Kill Chain

```
Dormant ──► Recon ──► Exploit ──► Persist ──► Exfil
  0          1          2            3           4
           P=0.30     P=0.40       P=0.35    (terminal)
```

---

## 🏆 Reward Function

```
R = R_detection + R_stopping + R_health + R_efficiency
  − P_false_positive − P_resource_waste − P_missed_attack
```

| Component | Value | Condition |
|-----------|-------|-----------|
| `R_detection` | +1.0 | Monitoring active when stage advances |
| `R_stopping` early | +5 × (3−stage) | Stopped at recon/exploit (max +10) |
| `R_stopping` late | +2.0 | Stopped at persist/exfil |
| `R_health` | +0.5 × health | Per step |
| `R_efficiency` | +1.5 | Stopped cheaply (cost ≤ 0.10) |
| `P_false_positive` | −1.0 | Costly action when no attack |
| `P_resource_waste` | −0.5 × cost | Spent budget, attack still advanced |
| `P_missed_attack` | −0.6 to −2.0 | Did nothing while taking damage |

---

## 📋 Tasks

| Task | Difficulty | Max Steps | Success Threshold |
|------|-----------|-----------|------------------|
| `detect_recon` | Easy | 30 | 0.70 |
| `stop_exploit` | Medium | 40 | 0.60 |
| `prevent_exfil` | Hard | 50 | 0.50 |

---

## 🗂️ Project Structure

```
cyber-defense-env/
├── app.py                     # FastAPI OpenEnv server
├── inference.py               # LLM agent (standalone + HTTP modes)
├── config.py                  # API key + settings loader (.env support)
├── openenv.yaml               # OpenEnv configuration
├── Dockerfile                 # HuggingFace Spaces ready
├── .dockerignore              # Keeps Docker image lean
├── .gitignore
├── .env.example               # Copy to .env and add your API key
├── requirements.txt
├── README.md
├── LICENSE
├── env/
│   ├── __init__.py
│   ├── cyber_env.py           # Core environment logic
│   ├── attack_simulator.py    # Multi-stage stochastic attack model
│   └── reward.py              # Multi-objective reward function
├── tasks/
│   ├── __init__.py
│   └── graders.py             # Task graders returning score 0.0–1.0
└── tests/
    ├── __init__.py
    └── test_presubmission.py  # Full pre-submission validation suite
```

---

## 🔭 Why Different From CybORG / CyberBattleSim

| Feature | ACD-Env | CybORG | CyberBattleSim |
|---------|---------|--------|----------------|
| LLM-native HTTP API | ✅ | ❌ | ❌ |
| OpenEnv compatible | ✅ | ❌ | ❌ |
| Zero-dependency fallback | ✅ NumPy only | ❌ | ❌ |
| Task graders (0.0–1.0) | ✅ 3 tasks | ❌ | ❌ |
| HuggingFace Spaces ready | ✅ | ❌ | ❌ |
| Runs without API key | ✅ | N/A | N/A |

---

## ⚙️ Configuration

All settings are loaded from environment variables (highest priority) or a `.env` file in the project root.

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `https://api-inference.huggingface.co/v1/` | OpenAI-compatible API endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model to use via the API |
| `HF_TOKEN` | **no default** | HuggingFace token — enables LLM reasoning |
| `LOCAL_IMAGE_NAME` | *(empty)* | Optional — only for `from_docker_image()` |
| `HOST` | `0.0.0.0` | Server bind host |
| `PORT` | `7860` | Server port |
| `LOG_LEVEL` | `info` | Uvicorn log level |

**Setup:**
```bash
cp .env.example .env
# Edit .env and set HF_TOKEN=hf_...
```

**Docker runtime:**
```bash
docker run -p 7860:7860 \
  -e HF_TOKEN=hf_... \
  cyber-defense-env
```

**HuggingFace Spaces:** Add `HF_TOKEN` in Space Settings → Variables and secrets → Add secret.

> **Note:** The environment is fully functional without any API key. The rule-based fallback agent achieves 83%+ average score across all tasks.

---

## 📄 License

MIT
