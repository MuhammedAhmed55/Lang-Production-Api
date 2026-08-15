"""
🧠 In one sentence
    "agents.py" receives the user's message, sends it to Ollama primary model, handles failures, tries a fallback model, and finally returns the answer or an error message.
"""


from typing import Optional, TypedDict
from typing_extensions import TypedDict , Annotated
from langgraph.graph import StateGraph , START , END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langsmith import traceable
from app.config import get_settings

settings = get_settings()

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    error: Optional[str]
    retry_count: int
    model_used: str

    """
    This keeps information about the current request.

    Think:

    AgentState
    │
    ├── messages      → User question + AI answers
    ├── error         → Did something go wrong?
    ├── retry_count   → How many times did we retry?
    └── model_used    → Which model answered?

    For example:

    User: "What is RAG?"
    retry_count: 0
    error: None
    model_used: ""

    As the request moves through the system, this state gets updated.
    """

class ProductionAgent: 
    """Main AI controller 🤖
    This is the main class.

    When you create it:
    agent = ProductionAgent()

    it:
    Gets your project settings
    Creates the Ollama models
    Gets retry settings
    Builds the LangGraph
"""
    def __init__(self):
        settings = get_settings()

        self.primary_llm = ChatOllama(
            model=settings.primary_model,
            temperature = 0,
            timeout = 30,
            max_retries = 0,
        )
        """
        This is your first AI model.

        The flow is:
        User
        ↓
        Primary Ollama model
        ↓
        Answer ✅

        For example:
        User: What is RAG?
            ↓
        llama3.2
            ↓
        RAG = Retrieval Augmented Generation
        """
        self.fallback_llm = ChatOllama(
            model=settings.fallback_model,
            temperature = 0,
            timeout = 30,
            max_retries = 0,
        )
        """
        This is your backup AI.

        Why?
        Because in production, things can fail.

        For example:
        Primary model
            ↓
        ❌ Error
            ↓
        Fallback model
            ↓
        ✅ Answer
        """
        self.max_retries = settings.max_retries
        self.graph = self._build_graph()

    def _build_graph(self):
        def process_message(state: AgentState) -> dict:
            """
            This function:
            response = self.primary_llm.invoke(state["messages"])
            sends the conversation to the primary Ollama model.

            If successful:
            Primary AI
            ↓
            ✅ Answer
            ↓
            model_used = "primary"

            If it fails:

            Primary AI
            ↓
            ❌ Error
            ↓
            retry_count + 1
            """
            try:
                response = self.primary_llm.invoke(state["messages"])
                return {
                    "messages": [response],
                    "error": None,
                    "model_used": "primary",
                }
            except Exception as e:
                return {
                    "error": str(e),
                    "retry_count": state["retry_count"] + 1,
                    "model_used": "",
                }
            
        def try_fallback(state: AgentState) -> dict:
            """
            If the primary AI fails, this function runs.
            response = self.fallback_llm.invoke(...)

            So:
            Primary Model
                ↓
            Failed ❌
                ↓
            Fallback Model
                ↓
            Success ✅

            That's very useful in production because one model/service failure doesn't necessarily kill the entire request.
            """
            try:
                response = self.fallback_llm.invoke(state["messages"])
                return {
                    "messages": [response],
                    "error": None,
                    "model_used": "fallback",
                }
            except Exception as e:
                return {
                    "error": str(e),
                    "model_used": "",
                }

        def handle_error(state: AgentState) -> dict:
            """What if both models fail?

            The user receives something like:
            "I'm sorry, but I encountered an error while processing your request. Please try again later."
            """
            return {
                "messages": [
                    AIMessage(content=(
                        "I'm sorry, but I encountered an error while processing your request. Please try again later."
                    ))
                ],
                "model_used": "error_handler",
            }

        def route_after_process(state: AgentState) -> str:
            """
            This is one of the most important parts.

            It decides:

            "Where should the request go next?"

            It checks:

            Did primary model work?
                    │
            ┌────┴────┐
            │         │
            YES       NO
            │         │
            ▼         ▼
            DONE     Retry available?
                        │
                    ┌────┴────┐
                YES        NO
                    │          │
                    ▼          ▼
                FALLBACK     ERROR

            So this function is basically a traffic controller for your AI workflow.
            """
            if state.get("error") is None:
                return "done"
            elif state["retry_count"] < self.max_retries:
                return "fallback"
            else:
                return "error"

        def route_after_fallback(state: AgentState) -> str:
            """After the fallback model runs, it asks:

            Did fallback work?
                │
            ┌───┴───┐
            YES      NO
            │        │
            ▼        ▼
            DONE    ERROR
            """
            if state.get("error") is None:
                return "done"
            else:
                return "error"

        # Build the graph
        graph = StateGraph(AgentState)

        graph.add_node("process", process_message)
        graph.add_node("fallback", try_fallback)
        graph.add_node("error", handle_error)

        graph.add_edge(START, "process")
        graph.add_conditional_edges(
            "process",
            route_after_process,
            {"done": END, "fallback": "fallback", "error": "error"},
        )

        graph.add_conditional_edges(
            "fallback",
            route_after_fallback,
            {"done": END, "error": "error"},
        )
        graph.add_edge("error", END)
        return graph.compile()

    @traceable(name="production_agent_invoke")
    def invoke(self, message: str) -> dict:
        """
        Invoke the agent with a user message.
        Returns: {"response": str, "model_used": str, "error": str | None}
        """
        result = self.graph.invoke({
            "messages": [HumanMessage(content=message)],
            "error": None,
            "retry_count": 0,
            "model_used": "",
        })
        return {
            "response": result["messages"][-1].content,
            "model_used": result.get("model_used", "unknown"),
            "error": result.get("error"),
        }
# ================================================================
# TEST THE AGENT DIRECTLY
# ================================================================

if __name__ == "__main__":
    """
    This block runs only when we execute this file directly.

    Example:
        python agents.py

    It will:
    1. Create the ProductionAgent.
    2. Send a test question to Ollama.
    3. Print the AI response.
    4. Show which model was used.
    5. Show any error.
    """

    print("\n🤖 Starting Production Agent...\n")

    # Create our AI agent.
    agent = ProductionAgent()

    # Test question.
    question = "What is RAG in AI?"

    print(f"👤 User: {question}")
    print("🧠 AI is thinking...\n")

    try:
        # Send the question through our LangGraph workflow.
        result = agent.invoke(question)

        # Print the final response.
        print("🤖 AI:", result["response"])

        # Show which model answered.
        print(f"\n📌 Model Used: {result['model_used']}")

        # Show error if something went wrong.
        print(f"❌ Error: {result['error']}")

    except Exception as e:
        # Catch unexpected errors while testing.
        print(f"\n❌ Something went wrong: {e}")