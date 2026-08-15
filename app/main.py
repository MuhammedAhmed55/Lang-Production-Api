# ============================================================
# main.py
# ============================================================
#
# PURPOSE OF THIS FILE:
# This is the MAIN file of our FastAPI application.
#
# It connects all the separate components together:
#
#     Frontend
#         ↓
#     FastAPI
#         ↓
#     Rate Limit
#         ↓
#     Security
#         ↓
#     Cache
#         ↓
#     AI Agent
#         ↓
#     Output Security
#         ↓
#     Cache + Metrics
#         ↓
#     Frontend
#
# The actual work is divided into separate files:
#
# config.py      → Application settings
# model.py       → Request/Response data models
# security.py    → Security checks
# cache.py       → Response caching
# monitoring.py  → Logs, metrics and request timing
# agent.py       → AI/LangGraph agent
#
# main.py simply coordinates all of them.
# ============================================================


# ============================================================
# 1. PYTHON / STANDARD LIBRARY IMPORTS
# ============================================================

import time
import os
import contextlib

# asynccontextmanager is used to run code when the FastAPI
# application STARTS and SHUTS DOWN.
from contextlib import asynccontextmanager


# ============================================================
# 2. THIRD-PARTY LIBRARY IMPORTS
# ============================================================

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

# SlowAPI is used for rate limiting.
# Example:
#     Maximum 20 requests per minute from one IP.
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# LangSmith is used to trace/monitor AI requests.
from langsmith import traceable

# Loads variables from the .env file.
from dotenv import load_dotenv


# ============================================================
# 3. OUR OWN PROJECT IMPORTS
# ============================================================

# Gets application settings such as:
# - environment
# - AI models
# - rate limit
# - cache TTL
from app.config import get_settings


# Pydantic models that define the structure of:
# - incoming chat requests
# - chat responses
# - health responses
# - metrics responses
from app.model import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    MetricsResponse,
)


# Security system.
# It checks user input and AI output.
#
# Example checks:
# - Prompt injection
# - PII / sensitive information
# - Unsafe input/output
from app.security import SecurityPipeline


# Cache system.
# It stores previous answers so we don't have to
# ask the AI the same question again.
from app.cache import ResponseCache


# Monitoring tools:
# - MetricsCollector → stores request statistics
# - get_logger       → application logs
# - RequestTimer     → measures request processing time
from app.monitoring import (
    MetricsCollector,
    get_logger,
    RequestTimer,
)


# ProductionAgent is responsible for actually talking
# to the AI models.
#
# It can use:
#     Primary model
#          ↓
#     Fallback model if primary fails
from app.agent import ProductionAgent


# ============================================================
# 4. LOAD ENVIRONMENT VARIABLES
# ============================================================

# Loads variables from .env
#
# Example:
#
# OPENAI_API_KEY=...
# LANGCHAIN_API_KEY=...
# PRIMARY_MODEL=...
#
# These values can then be accessed by our settings/config.
load_dotenv()


# ============================================================
# 5. GLOBAL COMPONENTS
# ============================================================
#
# We create these variables here, but some of them are
# initialized later when the application starts.
#
# Think of them as empty boxes that will later contain
# our application components.
# ============================================================

security: SecurityPipeline = None
cache: ResponseCache = None
agent: ProductionAgent = None


# Logger is available immediately.
logger = get_logger()

# Metrics collector keeps track of API statistics.
metrics = MetricsCollector()


# ============================================================
# 6. APPLICATION STARTUP / SHUTDOWN
# ============================================================
#
# This function runs:
#
#     When application starts → initialize everything
#
#     When application stops → clean up / log information
#
# This is the modern FastAPI lifespan pattern.
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global security, cache, metrics, agent

    # --------------------------------------------------------
    # GET APPLICATION SETTINGS
    # --------------------------------------------------------
    #
    # This reads our configuration from config.py.
    #
    # Example settings:
    #
    # app_env = development
    # primary_model = llama3.2
    # cache_ttl_seconds = 300
    # rate_limit = 20/minute
    #
    settings = get_settings()


    # --------------------------------------------------------
    # LOG THAT APPLICATION IS STARTING
    # --------------------------------------------------------

    logger.info(
        "Starting production API...",
        extra={
            "extra_data": {
                "environment": settings.app_env,
                "primary_model": settings.primary_model,
                "tracing_enabled": settings.langchain_tracing_v2,
            }
        },
    )


    # ========================================================
    # INITIALIZE ALL APPLICATION COMPONENTS
    # ========================================================

    # Create the security system.
    #
    # After this:
    #
    # security.check_input(...)
    # security.check_output(...)
    #
    # can be used.
    security = SecurityPipeline()


    # Create the cache system.
    #
    # ttl_seconds tells the cache how long an answer
    # should remain stored.
    cache = ResponseCache(
        ttl_seconds=settings.cache_ttl_seconds
    )


    # Create a fresh metrics collector.
    metrics = MetricsCollector()


    # Create the AI agent.
    #
    # The ProductionAgent is responsible for:
    #
    #     Primary model
    #          ↓
    #     Fallback model
    #
    agent = ProductionAgent()


    # Everything is ready.
    logger.info(
        "All components initialized. Ready to serve requests."
    )


    # ========================================================
    # YIELD = APPLICATION IS NOW RUNNING
    # ========================================================
    #
    # Everything before yield happens during STARTUP.
    #
    # The application now starts accepting requests.
    #
    yield


    # ========================================================
    # APPLICATION SHUTDOWN
    # ========================================================
    #
    # When the server is stopped, we reach here.
    # ========================================================

    logger.info(
        "Shutting down...",
        extra={
            "extra_data": metrics.summary
        }
    )


# ============================================================
# 7. RATE LIMITER
# ============================================================
#
# SlowAPI uses the client's IP address to identify the user.
#
# Example:
#
# User IP: 192.168.1.10
#
# If rate_limit = "20/minute":
#
#     20 requests → allowed
#     21st request → blocked
# ============================================================

limiter = Limiter(
    key_func=get_remote_address
)


# ============================================================
# 8. CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Production LangGraph API",

    description=(
        "A production-ready chat API "
        "with security, caching, and observability."
    ),

    version="1.0.0",

    # Tell FastAPI to use our startup/shutdown function.
    lifespan=lifespan,
)


# Store the rate limiter inside the FastAPI application.
app.state.limiter = limiter


# ============================================================
# 9. RATE LIMIT ERROR HANDLER
# ============================================================
#
# What happens when a user sends too many requests?
#
# Instead of returning a confusing error, we return:
#
# HTTP 429
#
# {
#     "error": "Rate limit exceeded..."
# }
# ============================================================

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(
    request: Request,
    exc: RateLimitExceeded
):

    # Log the IP and endpoint that exceeded the limit.
    logger.warning(
        "Rate limit exceeded",
        extra={
            "extra_data": {
                "client_ip": request.client.host,
                "path": request.url.path,
            }
        },
    )


    # Send a friendly response to the frontend.
    return JSONResponse(
        status_code=429,
        content={
            "error": (
                "Rate limit exceeded. "
                "Please try again later."
            )
        },
    )


# ============================================================
# 10. /chat ENDPOINT
# ============================================================
#
# THIS IS THE MAIN API ENDPOINT.
#
# Frontend sends:
#
#     POST /chat
#
# Example:
#
# {
#     "message": "What is Python?",
#     "thread_id": "123"
# }
#
#
# Then the following happens:
#
#     1. Rate limit
#     2. Input security
#     3. Cache check
#     4. AI Agent
#     5. Output security
#     6. Save to cache
#     7. Record metrics
#     8. Return response
# ============================================================

@app.post(
    "/chat",
    response_model=ChatResponse
)

# Apply rate limiting to this endpoint.
@limiter.limit(
    get_settings().rate_limit
)

# Trace this request in LangSmith.
@traceable(
    name="chat_endpoint"
)

async def chat(
    request: Request,
    body: ChatRequest
):

    """
    Main chat endpoint.
    """


    # ========================================================
    # START REQUEST TIMER
    # ========================================================
    #
    # This allows us to calculate:
    #
    # "How many milliseconds did this request take?"
    #
    with RequestTimer() as timer:

        # Stores any security warnings/notes.
        security_notes: list[str] = []


        # ====================================================
        # STEP 1 — SECURITY CHECK
        # ====================================================
        #
        # Take the message sent by the frontend and check it.
        #
        # Example:
        #
        # "What is Python?"
        #
        # Security system checks:
        #
        # - Prompt injection?
        # - PII?
        # - Dangerous input?
        #
        # It returns:
        #
        # is_allowed
        # cleaned_message
        # notes
        # ====================================================

        is_allowed, cleaned_message, notes = (
            security.check_input(body.message)
        )


        # Save security notes.
        security_notes.extend(notes)


        # ----------------------------------------------------
        # IF SECURITY REJECTS THE MESSAGE
        # ----------------------------------------------------

        if not is_allowed:

            # Log the blocked request.
            logger.warning(
                "Request blocked by security",
                extra={
                    "extra_data": {
                        "reason": notes,
                        "thread_id": body.thread_id,
                    }
                },
            )


            # Record this request as an error.
            metrics.record_request(
                latency_ms=0,
                error=True
            )


            # Tell the frontend that the request was blocked.
            raise HTTPException(
                status_code=400,
                detail=(
                    "Your message has been blocked "
                    "by our security filters."
                ),
            )


        # ====================================================
        # STEP 2 — CACHE CHECK
        # ====================================================
        #
        # Before calling the AI, ask:
        #
        # "Do we already have an answer for this question?"
        #
        # Example:
        #
        # User asks:
        #     "What is Python?"
        #
        # Cache:
        #     "What is Python?" → "Python is a programming..."
        #
        # If found:
        #     DON'T CALL AI
        #
        # If not found:
        #     Continue to AI agent.
        # ====================================================

        cached_response = cache.get(
            cleaned_message
        )


        # ----------------------------------------------------
        # CACHE HIT
        # ----------------------------------------------------

        if cached_response is not None:

            # Record that cache was used.
            metrics.record_request(
                latency_ms=0,
                cache_hit=True
            )


            # Log cache hit.
            logger.info(
                "Cache hit",
                extra={
                    "extra_data": {
                        "thread_id": body.thread_id,
                    }
                },
            )


            # Return cached answer immediately.
            #
            # AI model is NOT called.
            return ChatResponse(
                response=cached_response,
                thread_id=body.thread_id,
                model_used="cache",
                cached=True,
                processing_time_ms=0,
                security_notes=security_notes, 
            )


        # ====================================================
        # STEP 3 — CALL AI AGENT
        # ====================================================
        #
        # Cache did NOT contain the answer.
        #
        # So now we ask the ProductionAgent.
        #
        # ProductionAgent handles:
        #
        #     Primary Model
        #          ↓
        #     If primary fails
        #          ↓
        #     Fallback Model
        #
        # ====================================================

        try:

            result = agent.invoke(
                cleaned_message
            )


        # ----------------------------------------------------
        # IF AGENT FAILS
        # ----------------------------------------------------

        except Exception as e:

            # Log the actual error for developers.
            logger.error(
                f"Agent invocation failed "
                f"for thread_id: {body.thread_id}",

                extra={
                    "extra_data": {
                        "thread_id": body.thread_id,
                        "error": str(e),
                    }
                },
            )


            # Record failed request.
            metrics.record_request(
                latency_ms=0,
                error=True
            )


            # Don't expose internal error details
            # to the frontend.
            raise HTTPException(
                status_code=500,
                detail=(
                    "An error occurred "
                    "while processing your request."
                ),
            )


        # ====================================================
        # GET AI RESULT
        # ====================================================
        #
        # Agent returns something like:
        #
        # {
        #     "response": "Python is a programming language...",
        #     "model_used": "llama3.2"
        # }
        # ====================================================

        response_text = result["response"]

        model_used = result["model_used"]


        # ====================================================
        # STEP 4 — CHECK AI OUTPUT
        # ====================================================
        #
        # We don't blindly trust the AI's response.
        #
        # We run the response through our security system
        # again.
        #
        # This checks whether the AI output contains anything
        # that should be removed, masked, or warned about.
        # ====================================================

        validated_response, output_warnings = (
            security.check_output(
                response_text
            )
        )


        # Add output warnings to our security notes.
        security_notes.extend(
            output_warnings
        )


        # ====================================================
        # STEP 5 — SAVE RESPONSE IN CACHE
        # ====================================================
        #
        # Save:
        #
        #     question → answer
        #
        # So if the same question comes again,
        # we can return the cached answer instead of
        # calling the AI again.
        # ====================================================

        cache.set(
            cleaned_message,
            validated_response
        )


        # ====================================================
        # STEP 6 — CALCULATE TOKEN ESTIMATE
        # ====================================================
        #
        # Here we estimate token usage based on word count.
        #
        # This is NOT the exact token count.
        # It is just an approximate calculation.
        # ====================================================

        input_tokens = int(
            len(cleaned_message.split()) * 1.3
        )

        output_tokens = int(
            len(validated_response.split()) * 1.3
        )


        # ====================================================
        # STEP 7 — RECORD METRICS
        # ====================================================
        #
        # Save useful information about this request:
        #
        # - How long it took
        # - Input tokens
        # - Output tokens
        # - Was cache used?
        # - Did an error happen?
        # ====================================================

        metrics.record_request(
            latency_ms=timer.elapsed_ms,

            input_tokens=input_tokens,

            output_tokens=output_tokens,

            cache_hit=False
        )


        # ====================================================
        # LOG SECURITY NOTES IF THERE ARE ANY
        # ====================================================

        if security_notes:

            logger.warning(
                "Security notes for request",

                extra={
                    "extra_data": {
                        "thread_id": body.thread_id,
                        "notes": security_notes,
                    }
                },
            )


        # ====================================================
        # LOG SUCCESSFUL REQUEST
        # ====================================================

        logger.info(
            "Request processed successfully",

            extra={
                "extra_data": {
                    "thread_id": body.thread_id,
                    "model_used": model_used,
                    "latency_ms": timer.elapsed_ms,
                }
            },
        )


        # ====================================================
        # STEP 8 — SEND RESPONSE BACK TO FRONTEND
        # ====================================================
        #
        # Finally, the frontend receives the answer.
        #
        # Example:
        #
        # {
        #     "response": "Python is a programming language.",
        #     "thread_id": "123",
        #     "model_used": "llama3.2",
        #     "cached": false,
        #     "processing_time_ms": 532
        # }
        # ====================================================

        return ChatResponse(
            response=validated_response,

            thread_id=body.thread_id,

            model_used=model_used,

            cached=False,

            processing_time_ms=timer.elapsed_ms,
            security_notes=security_notes
        )


# ============================================================
# 11. /health ENDPOINT
# ============================================================
#
# This endpoint answers:
#
# "Is my application working properly?"
#
# Docker/Kubernetes/monitoring systems can call this.
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse
)

async def health():

    # Get current application settings.
    settings = get_settings()


    # Check whether all major components were initialized.
    checks = {

        # Is AI agent available?
        "agent": agent is not None,

        # Is security system available?
        "security": security is not None,

        # Is cache available?
        "cache": cache is not None,
    }


    # all() returns True only if every check is True.
    all_healthy = all(
        checks.values()
    )


    # Return health information.
    return HealthResponse(

        status=(
            "healthy"
            if all_healthy
            else "degraded"
        ),

        environment=settings.app_env,

        checks=checks,
    )


# ============================================================
# 12. /metrics ENDPOINT
# ============================================================
#
# This endpoint gives us application statistics.
#
# Example:
#
# - Total requests
# - Errors
# - Cache hits
# - Token usage
# - Latency
# ============================================================

@app.get(
    "/metrics",
    response_model=MetricsResponse
)

async def get_metrics():

    # Get current metrics summary.
    summary = metrics.summary


    # Convert the dictionary into our MetricsResponse model.
    return MetricsResponse(
        **summary
    )


# ============================================================
# 13. /cache/stats ENDPOINT
# ============================================================
#
# This endpoint gives statistics specifically about
# our cache.
#
# Example:
#
# - Cache hits
# - Cache misses
# - Cached items
# ============================================================

@app.get("/cache/stats")

async def cache_stats():

    # Return cache statistics.
    return cache.stats

"""
The flow is:

Frontend
   ↓
"What is Python?"
   ↓
FastAPI (main.py)
   ↓
1. Rate limit check
   ↓
2. Security check
   ├── Prompt injection?
   ├── Sensitive/PII information?
   └── Mask/block if needed
   ↓
3. Cache check
   ├── Found → return cached answer
   │
   └── Not found
          ↓
4. ProductionAgent (agent.py)
          ↓
      Primary model
          ↓
      If primary fails
          ↓
      Fallback model
          ↓
5. Check AI's output for security
          ↓
6. Save answer in cache
          ↓
7. Record metrics/logs
          ↓
8. Return answer
          ↓
Frontend

And yes, your last point is very important:

main.py is connecting all the separate files together.

You made separate files because each one has one specific responsibility:

main.py
   │
   ├── config.py
   │      └── Configuration/settings
   │
   ├── model.py
   │      └── Request/response structure
   │
   ├── security.py
   │      └── Security checks
   │
   ├── cache.py
   │      └── Cache system
   │
   ├── monitoring.py
   │      └── Logs, metrics, timer
   │
   └── agent.py
          └── AI / primary + fallback models

So you can think of `main.py as the coordinator/manager.

It basically says:
"""