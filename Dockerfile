# ============================================================
# 1. START WITH PYTHON
# ============================================================

# Use Python 3.12.
# "slim" means a smaller version of the Python image.
FROM python:3.12-slim


# ============================================================
# 2. SET OUR WORKING FOLDER
# ============================================================

# Create/use /app as our main folder inside the container.
# All the following commands will mainly work inside /app.
WORKDIR /app


# ============================================================
# 3. CREATE A NORMAL USER (created early, activated later)
# ============================================================

# Create a user called "appuser".
#
# We don't want our application to run as root
# because running as root is less secure.
#
# We create it now but don't switch to it yet (USER appuser
# comes later, after all files are copied in and owned).
RUN useradd --create-home appuser


# ============================================================
# 4. INSTALL SYSTEM DEPENDENCIES
# ============================================================

# Install curl, needed by the HEALTHCHECK below.
# python:3.12-slim does not include curl by default.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*


# ============================================================
# 5. INSTALL UV
# ============================================================

# Install "uv".
# uv is used to install and manage our Python dependencies.
RUN pip install uv


# ============================================================
# 6. COPY PROJECT DEPENDENCY FILES
# ============================================================

# Copy pyproject.toml from our computer into /app.
# This file tells uv what packages our project needs.
COPY pyproject.toml .

# Copy uv.lock into /app.
# uv.lock contains the exact versions of our dependencies.
COPY uv.lock* .


# ============================================================
# 7. INSTALL PROJECT DEPENDENCIES
# ============================================================

# Install the dependencies required by our application.
#
# --frozen:
# Use the versions already written in uv.lock.
# Do not change the lock file.
#
# --no-dev:
# Do not install development-only packages.
# This keeps the production image smaller.

# Force uv to use the Python already in this image (3.12)
# instead of downloading its own separate interpreter,
# which would live outside /app and cause permission
# issues once we switch to appuser.
ENV UV_PYTHON_INSTALL_DIR=/app/.uv-python

RUN uv sync --frozen --no-dev


# ============================================================
# 8. COPY OUR APPLICATION CODE
# ============================================================

# Copy our local "app" folder into the container.
#
# Computer:
# app/
#
# Container:
# /app/app/
#
# This brings our actual FastAPI code into the container.
COPY app/ app/


# ============================================================
# 9. HAND OWNERSHIP TO appuser AND SWITCH TO IT
# ============================================================

# Everything in /app (including the .venv created by uv sync,
# and the app/ code we just copied) currently belongs to root,
# since all the steps above ran as root.
#
# Give ownership to appuser now, then switch to it, so the
# application does not run as root.
RUN chown -R appuser /app
USER appuser


# ============================================================
# 10. DOCUMENT THE APPLICATION PORT
# ============================================================

# Our FastAPI application will listen on port 8000
# inside the container.
#
# IMPORTANT:
# EXPOSE does NOT publish the port to our computer.
# We normally publish it with:
#
# docker run -p 8000:8000 ...
#
EXPOSE 8000


# ============================================================
# 11. HEALTH CHECK
# ============================================================

# Docker will check whether our FastAPI application
# is still working.
#
# Every 30 seconds:
#   → Send a request to /health
#
# If it does not respond within 10 seconds:
#   → The check fails
#
# After 3 failed checks:
#   → Docker considers the container unhealthy
#
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1


# ============================================================
# 12. START THE FASTAPI APPLICATION
# ============================================================

# This command runs when the container starts.
#
# uv
#   → Use our uv environment
#
# run
#   → Run a command using the project environment
#
# uvicorn
#   → Start the FastAPI server
#
# app.main:app
#   → Go to:
#       app/main.py
#     and find:
#       app = FastAPI()
#
# --host 0.0.0.0
#   → Allow connections from outside the container
#
# --port 8000
#   → Run FastAPI on port 8000
#
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]