"""
ChromiumAI FastAPI Service
Phase 1: FastAPI MVP implementation with ACP Runtime
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import asyncio
import json
import time
from datetime import datetime

app = FastAPI(
    title="ChromiumAI API",
    description="AI-native browser API service - Phase 1 FastAPI Implementation",
    version="1.0.0"
)

# CORS for chrome://ai-browser/ integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["chrome://ai-browser", "http://localhost:3000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Anthropic API compatible models
class Message(BaseModel):
    role: str
    content: str

class AnthropicRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7
    system: Optional[str] = None

class AnthropicResponse(BaseModel):
    id: str
    type: str = "message"
    role: str = "assistant"
    content: List[Dict[str, Any]]
    model: str
    stop_reason: str = "end_turn"
    usage: Dict[str, int]

# Global ACP runtime (will be initialized)
acp_runtime = None

@app.on_event("startup")
async def startup_event():
    """Initialize ACP Runtime on startup"""
    global acp_runtime
    try:
        from acp_runtime import ACPRuntime
        acp_runtime = ACPRuntime()
        await acp_runtime.initialize()
        print("🚀 ChromiumAI API Service started - Ready for agent communication")
    except ImportError:
        print("⚠️  ACP Runtime not available - using mock implementation")
        from mock_acp_runtime import MockACPRuntime
        acp_runtime = MockACPRuntime()
        await acp_runtime.initialize()

@app.post("/v1/messages", response_model=AnthropicResponse)
async def anthropic_compatible_endpoint(request: AnthropicRequest):
    """
    Anthropic API compatible endpoint for seamless integration
    Environment variable ANTHROPIC_BASE_URL=http://localhost:3456 redirects here
    """
    try:
        # Route to ACP agent swarm
        result = await acp_runtime.process_request({
            "model": request.model,
            "messages": [msg.dict() for msg in request.messages],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "system": request.system,
            "timestamp": datetime.now().isoformat()
        })
        
        return AnthropicResponse(
            id=f"msg_{int(time.time())}",
            content=[{"type": "text", "text": result["content"]}],
            model=request.model,
            usage={"input_tokens": result.get("input_tokens", 0), 
                   "output_tokens": result.get("output_tokens", 0)}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent processing error: {str(e)}")

@app.get("/config")
async def get_configuration():
    """Get current AI configuration for WebUI"""
    return {
        "providers": acp_runtime.get_providers(),
        "agents": acp_runtime.get_agent_status(),
        "gaia_performance": acp_runtime.get_gaia_metrics(),
        "system_status": "operational",
        "phase": "fastapi_mvp",
        "version": "1.0.0"
    }

@app.post("/config")
async def update_configuration(config: Dict[str, Any]):
    """Update AI configuration from WebUI"""
    await acp_runtime.update_config(config)
    return {"status": "updated", "timestamp": datetime.now().isoformat()}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agents_active": len(acp_runtime.active_agents) if hasattr(acp_runtime, 'active_agents') else 0,
        "uptime": acp_runtime.get_uptime() if hasattr(acp_runtime, 'get_uptime') else "unknown",
        "version": "1.0.0",
        "phase": "fastapi_mvp"
    }

@app.get("/gaia/benchmark")
async def run_gaia_benchmark():
    """Run GAIA benchmark tests"""
    try:
        if hasattr(acp_runtime, 'run_gaia_benchmark'):
            results = await acp_runtime.run_gaia_benchmark()
            return {"status": "completed", "results": results}
        else:
            return {"status": "not_implemented", "message": "GAIA benchmark not available in current runtime"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GAIA benchmark error: {str(e)}")

if __name__ == "__main__":
    # Development server configuration
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=3456,
        reload=True,  # Enable reload for development
        log_level="info"
    )
