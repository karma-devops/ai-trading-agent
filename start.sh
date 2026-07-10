#!/bin/bash
# Gunicorn launcher used by Dockerfile and docker-compose.
# EasyPanel will run this automatically when the container starts.
cd "$(dirname "$0")"
export PORT="${PORT:-5000}"
exec gunicorn -w 1 -b "0.0.0.0:${PORT}" --timeout 120 --worker-class gthread --threads 4 --access-logfile - --error-logfile - wsgi:app
