# 🚀 CHROMIUMAI: SINGLE SOURCE OF TRUTH (SSOT)
## Hybrid FastAPI → Rust Migration Strategy for AI-Native Browser

> **Version**: 1.0  
> **Status**: Implementation Ready  
> **Target**: GAIA Benchmark Leadership + Sponsorship-Ready MVP  
> **Timeline**: 14 weeks to production-ready demo

---

## 📋 EXECUTIVE SUMMARY

ChromiumAI is a true Chromium fork with invisible AI integration, designed to outperform Perplexity Comet through:
- **Phase 1**: FastAPI MVP (Weeks 1-6) - Rapid development & sponsorship demo
- **Phase 2**: Rust optimization (Weeks 7-10) - Performance critical paths
- **Phase 3**: Production integration (Weeks 11-14) - Full system optimization

**Key Innovation**: Invisible AI integration using API interception pattern, maintaining 100% Chromium compatibility while adding GAIA-optimized agent swarm capabilities.

---

## 🎯 CORE ARCHITECTURE PRINCIPLES

1. **Chromium Fork First**: True fork, not wrapper - looks and works exactly like Chromium
2. **Invisible Integration**: AI accessible via chrome://ai-browser/ WebUI + API interception
3. **GAIA Optimization**: Architecture specifically designed for GAIA benchmark excellence
4. **Hybrid Performance**: FastAPI for rapid development → Rust for performance optimization
5. **Maintainable Upgrades**: Automated Chromium upstream merge pipeline
6. **ACP Protocol**: IBM's Agent Communication Protocol for agent coordination

---

## 🏗️ SYSTEM ARCHITECTURE OVERVIEW

```
ChromiumAI Fork (True Chromium)
├── Standard Chromium (99% unchanged)
│   ├── Blink Rendering Engine
│   ├── V8 JavaScript Engine  
│   └── Network Stack
│
├── AI Integration Layer (1% addition)
│   ├── chrome://ai-browser/ WebUI
│   ├── API Interception Service (localhost:3456)
│   ├── ACP Agent Runtime
│   └── Configuration Manager
│
├── GAIA-Optimized Agent Swarm
│   ├── Planning Agent (ACP Orchestrator)
│   ├── Level 1 Agents (Quick Response)
│   ├── Level 2 Agents (Multi-Step Research)
│   └── Level 3 Agents (Complex Analysis)
│
└── Upgrade Compatibility System
    ├── Automated Merge Pipeline
    ├── Conflict Resolution
    └── Feature Flag Management
```

---

## 📁 PROJECT STRUCTURE (Implementation Ready)

```
chromiumai/
├── README.md
├── SSOT.md                          # This document
├── LICENSE
├── .github/
│   └── workflows/
│       ├── chromium-upgrade.yml     # Automated upstream merging
│       ├── gaia-benchmark.yml       # GAIA testing pipeline
│       └── build-test.yml           # Cross-platform builds
│
├── chromium/                        # Chromium fork (git subtree)
│   ├── src/
│   │   └── chrome/
│   │       └── browser/
│   │           └── ui/
│   │               └── webui/
│   │                   └── ai_browser/  # chrome://ai-browser/ implementation
│   └── build/                       # Build configuration
│
├── ai-integration/                  # AI layer (our additions)
│   ├── api-service/                 # FastAPI → Rust migration
│   │   ├── python/                  # Phase 1: FastAPI implementation
│   │   │   ├── main.py
│   │   │   ├── acp_runtime.py
│   │   │   ├── config_manager.py
│   │   │   └── requirements.txt
│   │   └── rust/                    # Phase 2: Rust optimization
│   │       ├── Cargo.toml
│   │       ├── src/
│   │       │   ├── main.rs
│   │       │   ├── api_server.rs
│   │       │   └── python_bridge.rs
│   │       └── build.rs
│   │
│   ├── webui/                       # React frontend for chrome://ai-browser/
│   │   ├── package.json
│   │   ├── src/
│   │   │   ├── App.tsx
│   │   │   ├── components/
│   │   │   │   ├── ChatInterface.tsx
│   │   │   │   ├── AgentStatus.tsx
│   │   │   │   ├── ProviderConfig.tsx
│   │   │   │   └── GAIABenchmark.tsx
│   │   │   └── services/
│   │   │       └── api.ts
│   │   └── public/
│   │
│   └── agents/                      # GAIA-optimized agent implementations
│       ├── __init__.py
│       ├── base/
│       │   ├── agent.py             # Base agent class
│       │   ├── acp_client.py        # ACP communication
│       │   └── gaia_optimizer.py    # GAIA-specific optimizations
│       ├── planning/
│       │   ├── planning_agent.py    # ACP orchestrator
│       │   └── prompts/
│       │       └── planning.yaml
│       ├── level1/                  # Quick response agents
│       │   ├── simple_reasoning.py
│       │   └── basic_tools.py
│       ├── level2/                  # Multi-step research agents
│       │   ├── research_coordinator.py
│       │   └── web_browser.py
│       └── level3/                  # Complex analysis agents
│           ├── deep_analyzer.py
│           └── multimodal.py
│
├── upgrade-tools/                   # Chromium upgrade automation
│   ├── merge-pipeline.sh
│   ├── conflict-resolver.py
│   └── build-tester.sh
│
├── build-system/                    # Cross-platform builds
│   ├── build.py                     # Main build script
│   ├── windows/
│   │   └── build-windows.bat
│   ├── macos/
│   │   └── build-macos.sh
│   └── linux/
│       └── build-linux.sh
│
├── configs/                         # Configuration files
│   ├── agents.yaml                  # Agent configurations
│   ├── providers.yaml               # AI provider settings
│   └── gaia.yaml                    # GAIA benchmark settings
│
├── scripts/                         # Development scripts
│   ├── setup-dev.sh                 # Development environment setup
│   ├── run-gaia-tests.py           # GAIA benchmark runner
│   └── performance-monitor.py       # Performance tracking
│
├── docs/                           # Documentation
│   ├── architecture.md
│   ├── development.md
│   ├── deployment.md
│   └── api.md
│
└── tests/                          # Test suites
    ├── unit/
    ├── integration/
    └── gaia/
        └── benchmark_suite.py
```

---

## ⚡ PHASE 1: FASTAPI MVP (WEEKS 1-6)

### Week 1-2: Foundation Setup

#### Chromium Fork Setup
```bash
# scripts/setup-chromium-fork.sh
#!/bin/bash
set -e

echo "Setting up ChromiumAI fork..."

# Clone Chromium
git clone https://chromium.googlesource.com/chromium/src.git chromium
cd chromium

# Add our AI integration points
mkdir -p src/chrome/browser/ui/webui/ai_browser

# Create minimal WebUI registration
cat > src/chrome/browser/ui/webui/ai_browser/ai_browser_ui.cc << 'EOF'
#include "chrome/browser/ui/webui/ai_browser/ai_browser_ui.h"
#include "content/public/browser/web_ui_data_source.h"
#include "chrome/grit/ai_browser_resources.h"

AIBrowserUI::AIBrowserUI(content::WebUI* web_ui) : content::WebUIController(web_ui) {
  content::WebUIDataSource* source = content::WebUIDataSource::Create("ai-browser");
  source->AddResourcePath("", IDR_AI_BROWSER_HTML);
  source->SetDefaultResource(IDR_AI_BROWSER_HTML);
  content::WebUIDataSource::Add(Profile::FromWebUI(web_ui), source);
}
EOF

echo "Chromium fork setup complete!"
```

#### FastAPI API Service
```python
# ai-integration/api-service/python/main.py
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
    description="AI-native browser API service",
    version="1.0.0"
)

# CORS for chrome://ai-browser/ integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["chrome://ai-browser", "http://localhost:3000"],
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
    global acp_runtime
    from acp_runtime import ACPRuntime
    acp_runtime = ACPRuntime()
    await acp_runtime.initialize()
    print("ChromiumAI API Service started - Ready for agent communication")

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
        "system_status": "operational"
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
        "agents_active": len(acp_runtime.active_agents),
        "uptime": acp_runtime.get_uptime(),
        "version": "1.0.0"
    }

if __name__ == "__main__":
    # Production-ready configuration
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=3456,
        workers=4,  # Multiple workers for performance
        loop="uvloop",  # High-performance event loop
        log_level="info"
    )
```

#### ACP Runtime Implementation
```python
# ai-integration/api-service/python/acp_runtime.py
import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import aiohttp
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ACPMessage:
    """IBM ACP Protocol compatible message"""
    id: str
    from_agent: str
    to_agent: str
    message_type: str  # 'request', 'response', 'tool_call', 'error'
    content: Any
    context: Dict[str, Any]
    timestamp: str
    gaia_level: Optional[int] = None

class ACPRuntime:
    """Agent Communication Protocol Runtime - IBM ACP Compatible"""
    
    def __init__(self):
        self.agents = {}
        self.message_queue = asyncio.Queue()
        self.active_agents = []
        self.providers = {}
        self.gaia_metrics = {
            "level1_accuracy": 0.0,
            "level2_accuracy": 0.0,
            "level3_accuracy": 0.0,
            "average_response_time": 0.0
        }
        self.start_time = datetime.now()
        
    async def initialize(self):
        """Initialize the ACP runtime with GAIA-optimized agents"""
        logger.info("Initializing ACP Runtime...")
        
        # Load agent configurations
        await self._load_agents()
        await self._load_providers()
        
        # Start message processing loop
        asyncio.create_task(self._process_message_queue())
        
        logger.info(f"ACP Runtime initialized with {len(self.agents)} agents")
    
    async def _load_agents(self):
        """Load GAIA-optimized agent swarm"""
        from agents.planning.planning_agent import PlanningAgent
        from agents.level1.simple_reasoning import SimpleReasoningAgent
        from agents.level2.research_coordinator import ResearchCoordinatorAgent
        from agents.level3.deep_analyzer import DeepAnalyzerAgent
        
        # Initialize agents with ACP communication
        self.agents = {
            "planning": PlanningAgent(acp_runtime=self),
            "simple_reasoning": SimpleReasoningAgent(acp_runtime=self),
            "research_coordinator": ResearchCoordinatorAgent(acp_runtime=self),
            "deep_analyzer": DeepAnalyzerAgent(acp_runtime=self)
        }
        
        self.active_agents = list(self.agents.keys())
    
    async def _load_providers(self):
        """Load AI provider configurations"""
        self.providers = {
            "anthropic": {
                "name": "Claude",
                "models": ["claude-3-sonnet", "claude-3-haiku"],
                "status": "active"
            },
            "openai": {
                "name": "OpenAI", 
                "models": ["gpt-4", "gpt-3.5-turbo"],
                "status": "active"
            }
        }
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main request processing - routes to appropriate GAIA-optimized agent
        """
        start_time = datetime.now()
        
        # Analyze request for GAIA level classification
        gaia_level = self._classify_gaia_level(request)
        
        # Create ACP message
        message = ACPMessage(
            id=str(uuid.uuid4()),
            from_agent="user",
            to_agent="planning",
            message_type="request",
            content=request,
            context={
                "gaia_level": gaia_level,
                "timestamp": start_time.isoformat(),
                "session_id": str(uuid.uuid4())
            },
            timestamp=start_time.isoformat(),
            gaia_level=gaia_level
        )
        
        # Route to planning agent
        result = await self.send_message(message)
        
        # Update performance metrics
        response_time = (datetime.now() - start_time).total_seconds()
        self._update_gaia_metrics(gaia_level, response_time)
        
        return {
            "content": result.content,
            "gaia_level": gaia_level,
            "response_time": response_time,
            "agent_chain": result.context.get("agent_chain", []),
            "input_tokens": len(str(request)) // 4,  # Rough estimation
            "output_tokens": len(str(result.content)) // 4
        }
    
    def _classify_gaia_level(self, request: Dict[str, Any]) -> int:
        """
        Classify request complexity for GAIA optimization
        Level 1: Simple reasoning, basic tool use
        Level 2: Multi-step reasoning, web browsing
        Level 3: Complex analysis, multi-modal
        """
        content = str(request.get("messages", "")).lower()
        
        # Simple heuristics for MVP (will be ML-based in production)
        if any(word in content for word in ["calculate", "simple", "what is", "define"]):
            return 1
        elif any(word in content for word in ["research", "find", "analyze", "compare"]):
            return 2
        else:
            return 3
    
    async def send_message(self, message: ACPMessage) -> ACPMessage:
        """Send ACP message to target agent"""
        if message.to_agent not in self.agents:
            raise ValueError(f"Agent {message.to_agent} not found")
        
        agent = self.agents[message.to_agent]
        response = await agent.process_message(message)
        return response
    
    async def _process_message_queue(self):
        """Background message processing loop"""
        while True:
            try:
                # Process queued messages (for future async agent communication)
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Message queue processing error: {e}")
    
    def _update_gaia_metrics(self, level: int, response_time: float):
        """Update GAIA performance metrics"""
        # Simplified metrics update (will be enhanced with actual GAIA scoring)
        self.gaia_metrics["average_response_time"] = (
            self.gaia_metrics["average_response_time"] * 0.9 + response_time * 0.1
        )
    
    def get_providers(self) -> Dict[str, Any]:
        """Get current provider configurations"""
        return self.providers
    
    def get_agent_status(self) -> List[Dict[str, Any]]:
        """Get current agent status"""
        return [
            {
                "name": name,
                "status": "active",
                "specialization": agent.specialization,
                "success_rate": getattr(agent, "success_rate", 0.95)
            }
            for name, agent in self.agents.items()
        ]
    
    def get_gaia_metrics(self) -> Dict[str, float]:
        """Get current GAIA benchmark metrics"""
        return self.gaia_metrics
    
    def get_uptime(self) -> str:
        """Get system uptime"""
        uptime = datetime.now() - self.start_time
        return str(uptime)
    
    async def update_config(self, config: Dict[str, Any]):
        """Update runtime configuration"""
        if "providers" in config:
            self.providers.update(config["providers"])
        
        logger.info("Configuration updated")
```

### Week 3-4: Agent Implementation

#### Base Agent Class
```python
# ai-integration/agents/base/agent.py
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ToolResult:
    """Consistent tool result format (from DeepResearchAgent pattern)"""
    output: Any
    error: Optional[str] = None

class BaseAgent(ABC):
    """Base agent class with ACP communication"""
    
    def __init__(self, name: str, specialization: str, acp_runtime=None):
        self.name = name
        self.specialization = specialization
        self.acp_runtime = acp_runtime
        self.success_rate = 0.95
        self.tools = []
        
    @abstractmethod
    async def process_message(self, message) -> Any:
        """Process incoming ACP message"""
        pass
    
    async def send_acp_message(self, to_agent: str, content: Any, message_type: str = "request"):
        """Send message via ACP protocol"""
        from acp_runtime import ACPMessage
        import uuid
        from datetime import datetime
        
        message = ACPMessage(
            id=str(uuid.uuid4()),
            from_agent=self.name,
            to_agent=to_agent,
            message_type=message_type,
            content=content,
            context={"agent_chain": [self.name]},
            timestamp=datetime.now().isoformat()
        )
        
        return await self.acp_runtime.send_message(message)
    
    def log_performance(self, task_type: str, success: bool, response_time: float):
        """Log performance metrics"""
        logger.info(f"{self.name} - {task_type}: {'SUCCESS' if success else 'FAILED'} ({response_time:.2f}s)")
```

#### Planning Agent (ACP Orchestrator)
```python
# ai-integration/agents/planning/planning_agent.py
from agents.base.agent import BaseAgent, ToolResult
import asyncio
import json
from datetime import datetime

class PlanningAgent(BaseAgent):
    """
    ACP Orchestrator - Routes tasks to specialized GAIA-optimized agents
    Based on IBM ACP specification for agent coordination
    """
    
    def __init__(self, acp_runtime=None):
        super().__init__("planning", "task_orchestration", acp_runtime)
        self.agent_routing = {
            1: "simple_reasoning",     # GAIA Level 1
            2: "research_coordinator", # GAIA Level 2  
            3: "deep_analyzer"        # GAIA Level 3
        }
    
    async def process_message(self, message) -> Any:
        """Process incoming requests and route to appropriate agents"""
        try:
            request_content = message.content
            gaia_level = message.gaia_level or 1
            
            # Determine routing strategy
            target_agent = self.agent_routing.get(gaia_level, "simple_reasoning")
            
            # Create execution plan
            plan = await self._create_execution_plan(request_content, gaia_level)
            
            # Execute plan
            result = await self._execute_plan(plan, target_agent)
            
            # Update context with agent chain
            result.context["agent_chain"] = message.context.get("agent_chain", []) + [self.name]
            
            return result
            
        except Exception as e:
            return message.__class__(
                id=message.id,
                from_agent=self.name,
                to_agent=message.from_agent,
                message_type="error",
                content=f"Planning error: {str(e)}",
                context=message.context,
                timestamp=datetime.now().isoformat()
            )
    
    async def _create_execution_plan(self, request: Dict[str, Any], gaia_level: int) -> Dict[str, Any]:
        """Create execution plan based on GAIA level and request complexity"""
        
        messages = request.get("messages", [])
        last_message = messages[-1]["content"] if messages else ""
        
        plan = {
            "gaia_level": gaia_level,
            "primary_agent": self.agent_routing[gaia_level],
            "steps": [],
            "estimated_time": self._estimate_time(gaia_level),
            "required_tools": self._identify_required_tools(last_message)
        }
        
        # GAIA Level-specific planning
        if gaia_level == 1:
            plan["steps"] = [
                {"action": "quick_reasoning", "agent": "simple_reasoning"},
                {"action": "generate_response", "agent": "simple_reasoning"}
            ]
        elif gaia_level == 2:
            plan["steps"] = [
                {"action": "analyze_query", "agent": "research_coordinator"},
                {"action": "web_search", "agent": "research_coordinator"},
                {"action": "synthesize_results", "agent": "research_coordinator"}
            ]
        else:  # Level 3
            plan["steps"] = [
                {"action": "deep_analysis", "agent": "deep_analyzer"},
                {"action": "multi_modal_processing", "agent": "deep_analyzer"},
                {"action": "complex_reasoning", "agent": "deep_analyzer"}
            ]
        
        return plan
    
    async def _execute_plan(self, plan: Dict[str, Any], target_agent: str):
        """Execute the created plan"""
        
        # Route to target agent
        response = await self.send_acp_message(
            to_agent=target_agent,
            content={
                "plan": plan,
                "original_request": plan
            },
            message_type="request"
        )
        
        return response
    
    def _estimate_time(self, gaia_level: int) -> float:
        """Estimate execution time based on GAIA level"""
        time_estimates = {1: 2.0, 2: 10.0, 3: 30.0}
        return time_estimates.get(gaia_level, 5.0)
    
    def _identify_required_tools(self, query: str) -> List[str]:
        """Identify tools needed for query (simple heuristics for MVP)"""
        tools = []
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["search", "find", "research"]):
            tools.append("web_searcher")
        if any(word in query_lower for word in ["calculate", "compute", "math"]):
            tools.append("calculator")
        if any(word in query_lower for word in ["image", "picture", "visual"]):
            tools.append("vision_processor")
            
        return tools or ["general_reasoning"]
```

### Week 5-6: WebUI Integration

#### React Frontend for chrome://ai-browser/
```typescript
// ai-integration/webui/src/App.tsx
import React, { useState, useEffect } from 'react';
import { ChatInterface } from './components/ChatInterface';
import { AgentStatus } from './components/AgentStatus';
import { ProviderConfig } from './components/ProviderConfig';
import { GAIABenchmark } from './components/GAIABenchmark';
import { ApiService } from './services/api';

interface AppState {
  activeTab: 'chat' | 'agents' | 'providers' | 'benchmark';
  config: any;
  agents: any[];
  gaiaMetrics: any;
}

const App: React.FC = () => {
  const [state, setState] = useState<AppState>({
    activeTab: 'chat',
    config: null,
    agents: [],
    gaiaMetrics: {}
  });

  useEffect(() => {
    loadConfiguration();
  }, []);

  const loadConfiguration = async () => {
    try {
      const config = await ApiService.getConfig();
      setState(prev => ({
        ...prev,
        config: config,
        agents: config.agents || [],
        gaiaMetrics: config.gaia_performance || {}
      }));
    } catch (error) {
      console.error('Failed to load configuration:', error);
    }
  };

  const handleTabChange = (tab: AppState['activeTab']) => {
    setState(prev => ({ ...prev, activeTab: tab }));
  };

  return (
    <div className="chromium-ai-app">
      <header className="app-header">
        <h1>ChromiumAI</h1>
        <nav className="tab-navigation">
          <button 
            className={state.activeTab === 'chat' ? 'active' : ''}
            onClick={() => handleTabChange('chat')}
          >
            AI Chat
          </button>
          <button 
            className={state.activeTab === 'agents' ? 'active' : ''}
            onClick={() => handleTabChange('agents')}
          >
            Agents ({state.agents.length})
          </button>
          <button 
            className={state.activeTab === 'providers' ? 'active' : ''}
            onClick={() => handleTabChange('providers')}
          >
            Providers
          </button>
          <button 
            className={state.activeTab === 'benchmark' ? 'active' : ''}
            onClick={() => handleTabChange('benchmark')}
          >
            GAIA Benchmark
          </button>
        </nav>
      </header>

      <main className="app-main">
        {state.activeTab === 'chat' && (
          <ChatInterface 
            onConfigChange={loadConfiguration}
          />
        )}
        
        {state.activeTab === 'agents' && (
          <AgentStatus 
            agents={state.agents}
            onRefresh={loadConfiguration}
          />
        )}
        
        {state.activeTab === 'providers' && (
          <ProviderConfig 
            config={state.config}
            onUpdate={loadConfiguration}
          />
        )}
        
        {state.activeTab === 'benchmark' && (
          <GAIABenchmark 
            metrics={state.gaiaMetrics}
            onRunTest={loadConfiguration}
          />
        )}
      </main>
    </div>
  );
};

export default App;
```

#### Chat Interface Component
```typescript
// ai-integration/webui/src/components/ChatInterface.tsx
import React, { useState, useRef, useEffect } from 'react';
import { ApiService } from '../services/api';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  gaiaLevel?: number;
  responseTime?: number;
}

interface ChatInterfaceProps {
  onConfigChange: () => void;
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({ onConfigChange }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState('claude-3-sonnet');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      role: 'user',
      content: input.trim(),
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await ApiService.sendMessage({
        model: selectedModel,
        messages: [
          ...messages.map(m => ({ role: m.role, content: m.content })),
          { role: userMessage.role, content: userMessage.content }
        ]
      });

      const assistantMessage: Message = {
        role: 'assistant',
        content: response.content[0]?.text || 'No response',
        timestamp: new Date().toISOString(),
        gaiaLevel: response.gaia_level,
        responseTime: response.response_time
      };

      setMessages(prev => [...prev, assistantMessage]);
      
    } catch (error) {
      const errorMessage: Message = {
        role: 'assistant',
        content: `Error: ${error.message}`,
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-interface">
      <div className="chat-header">
        <select 
          value={selectedModel} 
          onChange={(e) => setSelectedModel(e.target.value)}
          className="model-selector"
        >
          <option value="claude-3-sonnet">Claude 3 Sonnet</option>
          <option value="claude-3-haiku">Claude 3 Haiku</option>
          <option value="gpt-4">GPT-4</option>
          <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
        </select>
      </div>

      <div className="messages-container">
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.role}`}>
            <div className="message-content">
              {message.content}
            </div>
            <div className="message-meta">
              <span className="timestamp">
                {new Date(message.timestamp).toLocaleTimeString()}
              </span>
              {message.gaiaLevel && (
                <span className="gaia-level">
                  GAIA L{message.gaiaLevel}
                </span>
              )}
              {message.responseTime && (
                <span className="response-time">
                  {message.responseTime.toFixed(2)}s
                </span>
              )}
            </div>
          </div>
        ))}
        
        {isLoading && (
          <div className="message assistant loading">
            <div className="loading-indicator">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSubmit} className="input-form">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask ChromiumAI anything..."
          disabled={isLoading}
          className="message-input"
        />
        <button 
          type="submit" 
          disabled={isLoading || !input.trim()}
          className="send-button"
        >
          Send
        </button>
      </form>
    </div>
  );
};
```

This completes the first section of the SSOT document. Would you like me to continue with the remaining phases and implementation details?
