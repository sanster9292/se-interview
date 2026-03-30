import os
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from agent import build_agent
from pathlib import Path
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

# Phoenix tracing configuration - using local Phoenix server
tracer_provider = register(
    project_name="ai-travel-companion",
    endpoint="http://localhost:6006/v1/traces",
)

# Instrument LangChain
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

agent = build_agent()

app = FastAPI(
    title="LangGraph Agent API",
    description="A simple API for interacting with a LangGraph agent that can search the web",
    version="0.1.0",
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@app.get("/")
async def read_root():
    """Serve the main UI."""
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(content=html_path.read_text(), status_code=200)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Send a message to the agent and get a response."""
    result = agent.invoke({"messages": [HumanMessage(content=request.message)]})
    return ChatResponse(response=result["messages"][-1].content)


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}
