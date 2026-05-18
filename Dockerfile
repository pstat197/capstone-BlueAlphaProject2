# BlueAlpha simulator + Bayesian MMM API image.
#
# Built primarily for Hugging Face Spaces (Docker SDK), which:
#   - expects an HTTP server on port 7860 (overridable via README front-matter),
#   - runs the container as a non-root user with UID 1000,
#   - has no persistent storage on the free CPU tier (our disk cache under
#     app/.cache/ resets when the Space restarts; that's fine for a demo).
#
# Local build sanity check:
#   docker build -t bluealpha-api .
#   docker run --rm -p 7860:7860 bluealpha-api
#   curl http://localhost:7860/api/health

FROM python:3.11-slim

# System deps Meridian / TensorFlow Probability pull in transitively.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces requires UID 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /home/user/app

# Install Python deps in two layers so the heavy Meridian/TF layer is reused
# whenever only source code changes.
COPY --chown=user:user requirements.txt requirements-meridian.txt ./
RUN pip install --user --upgrade pip \
    && pip install --user -r requirements.txt \
    && pip install --user "fastapi>=0.115.0" "uvicorn[standard]>=0.30.0" "pydantic>=2.6.0" \
    && pip install --user -r requirements-meridian.txt

# App source.
COPY --chown=user:user . .

# Cache directory used by app/cache.py + server/mmm.py. Pre-create so the
# first request doesn't race on mkdir, and so it's writable as `user`.
RUN mkdir -p app/.cache/runs/mmm

EXPOSE 7860

# `--host 0.0.0.0` so the container is reachable from outside.
# Single worker: Meridian fits already use threads, and a second worker
# would double the (sizable) TF memory footprint.
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
