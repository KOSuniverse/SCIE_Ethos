# deployment/docker/gunicorn.conf.py
# Phase 6: Production Gunicorn Configuration

import multiprocessing
import os

# Server socket
bind = "0.0.0.0:8501"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
preload_app = True

# Restart workers after this many requests, with up to this much jitter
max_requests = 1000
max_requests_jitter = 100

# Timeout for handling requests
timeout = 120
keepalive = 5

# Logging
loglevel = os.getenv("LOG_LEVEL", "info")
accesslog = "-"
errorlog = "-"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "scie-ethos"

# Server mechanics
daemon = False
pidfile = "/tmp/gunicorn.pid"
user = "scie"
group = "scie"
tmp_upload_dir = None

# SSL (if needed)
keyfile = os.getenv("SSL_KEYFILE")
certfile = os.getenv("SSL_CERTFILE")

# Security
limit_request_line = 4096
limit_request_fields = 100
limit_request_field_size = 8190

# Performance tuning
worker_tmp_dir = "/dev/shm"

# Graceful shutdown
graceful_timeout = 30

# Development vs Production settings
if os.getenv("ENVIRONMENT") == "development":
    reload = True
    workers = 1
    loglevel = "debug"
else:
    reload = False
    
# Health check endpoint
def on_starting(server):
    server.log.info("Starting SCIE Ethos application server")

def on_reload(server):
    server.log.info("Reloading SCIE Ethos application server")

def when_ready(server):
    server.log.info("SCIE Ethos application server is ready")

def on_exit(server):
    server.log.info("Shutting down SCIE Ethos application server")
