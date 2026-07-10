#!/bin/bash
# Start the trading dashboard via Gunicorn for EasyPanel / production.
cd "$(dirname "$0")"
exec gunicorn -w 2 -b 0.0.0.0:5000 --timeout 120 --worker-class sync --access-logfile - --error-logfile - wsgi:app
