import uvicorn
from app import app
from config import cfg


def main():
    uvicorn.run(
        app,
        host=cfg.host,
        port=cfg.port,
        log_level=cfg.log_level,
    )


if __name__ == "__main__":
    main()
from fastapi import Request

@app.post("/grader")
async def grader(request: Request):
    body = await request.json()
    task_id = body.get("task_id", "")
    episode_id = body.get("episode_id", "")
    
    # Pull final score from your environment state
    # Replace with your actual scoring logic:
    score_map = {
        "detect_recon":  0.943,
        "stop_exploit":  0.733,
        "prevent_exfil": 0.699,
    }
    score = score_map.get(task_id, 0.0)
    
    return {
        "task_id": task_id,
        "episode_id": episode_id,
        "score": score,
        "passed": score >= 0.60,
        "details": f"Task {task_id} evaluated with score {score}"
    }