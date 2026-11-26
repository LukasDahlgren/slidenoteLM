#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID=""
HOST_ADDRESS=""
PORT=""
NO_RELOAD=0
SKIP_BWS=0

usage() {
    cat <<'EOF'
Usage: ./scripts/run-backend.sh [options]

Options:
  --project-id <UUID>  Bitwarden Secrets Manager project UUID (defaults to BWS_PROJECT_ID env or .env)
  --host <HOST>        Bind address (defaults to BACKEND_HOST env/.env or 127.0.0.1)
  --port <PORT>        Bind port (defaults to BACKEND_PORT env/.env or 8000)
  --no-reload          Disable uvicorn --reload
  --skip-bws           Run uvicorn directly without Bitwarden Secrets Manager (uses .env / stub mode)
  -h, --help           Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --project-id)
            [[ $# -ge 2 ]] || { echo "Error: --project-id requires a value." >&2; exit 1; }
            PROJECT_ID="$2"
            shift 2
            ;;
        --host)
            [[ $# -ge 2 ]] || { echo "Error: --host requires a value." >&2; exit 1; }
            HOST_ADDRESS="$2"
            shift 2
            ;;
        --port)
            [[ $# -ge 2 ]] || { echo "Error: --port requires a value." >&2; exit 1; }
            PORT="$2"
            shift 2
            ;;
        --no-reload)
            NO_RELOAD=1
            shift
            ;;
        --skip-bws)
            SKIP_BWS=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option '$1'." >&2
            usage >&2
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"

trim() {
    local s="$1"
    s="${s%$'\r'}"
    s="${s#"${s%%[![:space:]]*}"}"
    s="${s%"${s##*[![:space:]]}"}"
    printf '%s' "$s"
}

get_dotenv_value() {
    local file_path="$1"
    local target_key="$2"
    [[ -f "$file_path" ]] || return 0

    local line trimmed key val
    while IFS= read -r line || [[ -n "$line" ]]; do
        trimmed="$(trim "$line")"
        [[ -z "$trimmed" || "$trimmed" == \#* ]] && continue
        [[ "$trimmed" == *=* ]] || continue
        key="$(trim "${trimmed%%=*}")"
        if [[ "$key" == "$target_key" ]]; then
            val="$(trim "${trimmed#*=}")"
            if [[ ${#val} -ge 2 ]]; then
                if [[ "$val" == \"*\" && "$val" == *\" ]] || [[ "$val" == \'*\' && "$val" == *\' ]]; then
                    val="${val:1:${#val}-2}"
                fi
            fi
            printf '%s' "$val"
            return 0
        fi
    done < "$file_path"
}

if [[ -z "$HOST_ADDRESS" ]]; then
    HOST_ADDRESS="${BACKEND_HOST:-}"
    if [[ -z "$HOST_ADDRESS" ]]; then
        HOST_ADDRESS="$(get_dotenv_value "$ENV_FILE" "BACKEND_HOST")"
    fi
    if [[ -z "$HOST_ADDRESS" ]]; then
        HOST_ADDRESS="127.0.0.1"
    fi
fi

if [[ -z "$PORT" ]]; then
    RAW_PORT="${BACKEND_PORT:-}"
    if [[ -z "$RAW_PORT" ]]; then
        RAW_PORT="$(get_dotenv_value "$ENV_FILE" "BACKEND_PORT")"
    fi
    if [[ "$RAW_PORT" =~ ^[1-9][0-9]*$ ]]; then
        PORT="$RAW_PORT"
    else
        PORT="8000"
    fi
elif [[ ! "$PORT" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: Invalid --port value '$PORT' (expected a positive integer)." >&2
    exit 1
fi

UVICORN_BIN="${REPO_ROOT}/.venv/bin/uvicorn"
if [[ ! -x "$UVICORN_BIN" ]]; then
    echo "Error: Virtual environment uvicorn not found at '${UVICORN_BIN}'." >&2
    echo "Create .venv and install dependencies first:" >&2
    echo "  python3 -m venv .venv && .venv/bin/pip install -r backend/requirements.txt" >&2
    exit 1
fi

UVICORN_ARGS=(
    "backend.app.main:app"
    "--host" "$HOST_ADDRESS"
    "--port" "$PORT"
)
if [[ "$NO_RELOAD" -eq 0 ]]; then
    UVICORN_ARGS+=("--reload")
fi

cd "$REPO_ROOT"

if [[ "$SKIP_BWS" -eq 1 ]]; then
    exec "$UVICORN_BIN" "${UVICORN_ARGS[@]}"
fi

if ! command -v bws >/dev/null 2>&1; then
    echo "Error: Bitwarden Secrets Manager CLI ('bws') not found on PATH." >&2
    echo "Install it from https://github.com/bitwarden/sdk-sm/releases or run with --skip-bws." >&2
    exit 1
fi

if [[ -n "$(get_dotenv_value "$ENV_FILE" "BWS_ACCESS_TOKEN")" ]]; then
    echo "Warning: BWS_ACCESS_TOKEN is present in .env. Remove it from .env so it is not stored in plaintext on disk." >&2
fi

BWS_TOKEN_PATTERN='^0\.[^.[:space:]:]+\.[^[:space:]:]+:[^[:space:]]+$'

if [[ -n "${BWS_ACCESS_TOKEN:-}" && ! "${BWS_ACCESS_TOKEN}" =~ $BWS_TOKEN_PATTERN ]]; then
    echo "Warning: Ignoring malformed BWS_ACCESS_TOKEN in current terminal session." >&2
    unset BWS_ACCESS_TOKEN
fi

if [[ -z "${BWS_ACCESS_TOKEN:-}" ]]; then
    IFS= read -r -s -p "Enter BWS_ACCESS_TOKEN (hidden): " token || true
    echo ""
    token="$(trim "${token:-}")"
    if [[ -z "$token" ]]; then
        echo "Error: BWS_ACCESS_TOKEN is required (or run with --skip-bws to use .env / stub mode)." >&2
        exit 1
    fi
    if [[ ! "$token" =~ $BWS_TOKEN_PATTERN ]]; then
        echo "Error: Invalid BWS_ACCESS_TOKEN format (expected '0.<uuid>.<secret>:<key>')." >&2
        exit 1
    fi
    export BWS_ACCESS_TOKEN="$token"
    unset token
fi

if [[ -z "$PROJECT_ID" ]]; then
    PROJECT_ID="${BWS_PROJECT_ID:-}"
fi
if [[ -z "$PROJECT_ID" ]]; then
    PROJECT_ID="$(get_dotenv_value "$ENV_FILE" "BWS_PROJECT_ID")"
fi
if [[ -z "$PROJECT_ID" ]]; then
    echo "Error: Bitwarden project ID not provided. Pass --project-id <UUID>, set BWS_PROJECT_ID, or add BWS_PROJECT_ID=<UUID> to .env." >&2
    exit 1
fi

exec bws run --project-id "$PROJECT_ID" -- "$UVICORN_BIN" "${UVICORN_ARGS[@]}"
