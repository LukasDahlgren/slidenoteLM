# syntax=docker/dockerfile:1

FROM node:22-slim AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build


FROM python:3.12-slim AS runtime
ARG TARGETARCH
ARG BWS_VERSION=2.1.0

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates curl unzip \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    ARCH="${TARGETARCH:-$(dpkg --print-architecture)}"; \
    case "$ARCH" in \
      amd64) BWS_ARCH="x86_64" ;; \
      arm64) BWS_ARCH="aarch64" ;; \
      *) echo "Unsupported architecture: $ARCH" >&2; exit 1 ;; \
    esac; \
    curl -fsSL "https://github.com/bitwarden/sdk-sm/releases/download/bws-v${BWS_VERSION}/bws-${BWS_ARCH}-unknown-linux-gnu-${BWS_VERSION}.zip" -o /tmp/bws.zip; \
    unzip -o /tmp/bws.zip -d /usr/local/bin; \
    chmod +x /usr/local/bin/bws; \
    rm -f /tmp/bws.zip; \
    bws --version

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend ./backend
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

EXPOSE 8000

CMD ["sh", "-c", "if [ -n \"${BWS_ACCESS_TOKEN:-}\" ]; then if [ -z \"${BWS_PROJECT_ID:-}\" ]; then echo 'ERROR: BWS_ACCESS_TOKEN is set but BWS_PROJECT_ID is missing. Set BWS_PROJECT_ID or unset BWS_ACCESS_TOKEN.' >&2; exit 1; fi; exec bws run --project-id \"$BWS_PROJECT_ID\" -- uvicorn backend.app.main:app --host 0.0.0.0 --port \"${PORT:-8000}\" --proxy-headers --forwarded-allow-ips='*'; else exec uvicorn backend.app.main:app --host 0.0.0.0 --port \"${PORT:-8000}\" --proxy-headers --forwarded-allow-ips='*'; fi"]
