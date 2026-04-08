FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ── Hackathon-required environment variables ───────────────────────────────────
# API_BASE_URL and MODEL_NAME have defaults — HF_TOKEN must be set at runtime.
#
# In HuggingFace Spaces: Settings → Variables and secrets → Add secret: HF_TOKEN
# For docker run: docker run -e HF_TOKEN=hf_... -p 7860:7860 cyber-defense-env

ENV API_BASE_URL="https://api-inference.huggingface.co/v1/"
ENV MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
ENV HF_TOKEN=""

# Optional — only if using from_docker_image()
ENV LOCAL_IMAGE_NAME=""

# Server config
ENV HOST="0.0.0.0"
ENV PORT=7860
ENV LOG_LEVEL="info"

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
