"""
Production Monitoring & Logging

This file watches the AI system and records:
- Requests
- Errors
- Response time
- Token usage
- Cache performance
- Logs
"""

import logging
import json
import time
from datetime import datetime, timezone
from langchain_ollama import ChatOllama
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()


# ================================================================
# STRUCTURED LOGGING
# ================================================================

class JSONFormatter(logging.Formatter):
    """Formats logs into organized JSON."""

    def format(self, record):
        # Create structured log information
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }

        # Add extra information if provided
        if hasattr(record, "extra_data"):
            log_obj.update(record.extra_data)

        return json.dumps(log_obj)


def setup_logging():
    """Start the application's logging system."""

    logger = logging.getLogger("langgraph_app")
    logger.setLevel(logging.INFO)

    # Send logs to the console
    handler = logging.StreamHandler()

    # Use our JSON formatter
    handler.setFormatter(JSONFormatter())

    logger.addHandler(handler)

    return logger


# ================================================================
# METRICS COLLECTION
# ================================================================

class MetricsCollector:
    """
    Keeps track of AI system performance.

    Think of this as the application's scoreboard.
    """

    def __init__(self):

        # Counters for monitoring the application
        self.metrics = {
            "requests_total": 0,
            "errors_total": 0,
            "latency_sum": 0,
            "latency_count": 0,
            "tokens_input": 0,
            "tokens_output": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def record_request(
        self,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        error: bool = False,
        cache_hit: bool = False,
    ):
        """Record information about one AI request."""

        # Count the request
        self.metrics["requests_total"] += 1

        # Save response time
        self.metrics["latency_sum"] += latency_ms
        self.metrics["latency_count"] += 1

        # Save token usage
        self.metrics["tokens_input"] += input_tokens
        self.metrics["tokens_output"] += output_tokens

        # Count errors
        if error:
            self.metrics["errors_total"] += 1

        # Track cache usage
        if cache_hit:
            self.metrics["cache_hits"] += 1
        else:
            self.metrics["cache_misses"] += 1

    def get_summary(self) -> dict:
        """Return a simple summary of system performance."""

        # Calculate average response time
        avg_latency = (
            self.metrics["latency_sum"]
            / self.metrics["latency_count"]
            if self.metrics["latency_count"] > 0
            else 0
        )

        # Calculate error percentage
        error_rate = (
            self.metrics["errors_total"]
            / self.metrics["requests_total"]
            if self.metrics["requests_total"] > 0
            else 0
        )

        # Calculate cache hit percentage
        total_cache_requests = (
            self.metrics["cache_hits"]
            + self.metrics["cache_misses"]
        )

        cache_hit_rate = (
            self.metrics["cache_hits"]
            / total_cache_requests
            if total_cache_requests > 0
            else 0
        )

        # Return useful monitoring information
        return {
            "total_requests": self.metrics["requests_total"],
            "total_errors": self.metrics["errors_total"],
            "error_rate": f"{error_rate:.2%}",
            "avg_latency_ms": round(avg_latency, 2),
            "total_input_tokens": self.metrics["tokens_input"],
            "total_output_tokens": self.metrics["tokens_output"],
            "cache_hit_rate": f"{cache_hit_rate:.2%}",
        }


# ================================================================
# MONITORED AI MODEL
# ================================================================

class InstrumentedLLM:
    """
    AI model with monitoring attached.

    It calls the AI and records what happens.
    """

    def __init__(self):

        # Create the AI model
        self.llm = ChatOllama(
            model="llama3.2",
            temperature=0
        )

        # Create metrics tracker
        self.metrics = MetricsCollector()

        # Create logger
        self.logger = setup_logging()

    @traceable(name="instrumented_invoke")
    def invoke(self, query: str) -> str:
        """
        Send a question to the AI and monitor the request.
        """

        # Start timer
        start_time = time.time()

        try:

            # Send question to AI
            response = self.llm.invoke(query)

            # Get AI's answer
            result = response.content

            # Estimate token usage
            input_tokens = len(query.split()) * 4 // 3
            output_tokens = len(result.split()) * 4 // 3

            # Record request information
            self.metrics.record_request(
                latency_ms=(time.time() - start_time) * 1000,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                error=False,
                cache_hit=False,
            )

            # Write successful request to logs
            self.logger.info(
                "LLM request completed",
                extra={
                    "extra_data": {
                        "latency_ms": (
                            time.time() - start_time
                        ) * 1000,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    }
                },
            )

            return result

        except Exception as e:

            # Record failed request
            self.metrics.record_request(
                latency_ms=(time.time() - start_time) * 1000,
                input_tokens=0,
                output_tokens=0,
                error=True,
                cache_hit=False,
            )

            # Write error to logs
            self.logger.error(
                f"LLM request failed: {e}",
                extra={
                    "extra_data": {
                        "error": str(e)
                    }
                },
            )

            # Pass the error to the caller
            raise


# ================================================================
# MONITORING TEST
# ================================================================

def demo_monitoring():
    """Test the monitoring system."""

    # Create monitored AI
    llm = InstrumentedLLM()

    print("Monitoring Demo:\n")

    # Test questions
    queries = [
        "What is Python?",
        "Explain machine learning.",
        "What is 2 + 2?",
    ]

    # Send each question to the AI
    for query in queries:

        result = llm.invoke(query)

        print(
            f"Query: {query[:30]}... "
            f"-> {result[:30]}..."
        )

    # Display monitoring results
    print("\nMetrics Summary:")

    summary = llm.metrics.get_summary()

    for key, value in summary.items():
        print(f"  {key}: {value}")


# ================================================================
# RUN DEMO
# ================================================================

if __name__ == "__main__":
    # Start the monitoring test
    demo_monitoring()
