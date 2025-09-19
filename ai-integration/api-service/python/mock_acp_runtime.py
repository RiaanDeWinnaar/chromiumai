"""
Mock ACP Runtime for development and testing
Implements the ACP protocol interface without full agent implementation
"""
import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

class MockACPRuntime:
    """Mock ACP Runtime for development and testing"""
    
    def __init__(self):
        self.agents = {}
        self.active_agents = ["planning", "simple_reasoning", "research_coordinator", "deep_analyzer"]
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
        self.gaia_metrics = {
            "level1_accuracy": 0.95,
            "level2_accuracy": 0.90,
            "level3_accuracy": 0.80,
            "average_response_time": 0.5
        }
        self.start_time = datetime.now()
        
    async def initialize(self):
        """Initialize the mock ACP runtime"""
        print("🤖 Mock ACP Runtime initialized - Ready for development")
        
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mock request processing - simulates agent swarm behavior
        """
        messages = request.get("messages", [])
        last_message = messages[-1]["content"] if messages else "Hello"
        
        # Simulate GAIA level classification
        gaia_level = self._classify_gaia_level(last_message)
        
        # Mock response based on GAIA level
        if gaia_level == 1:
            response = f"Mock Level 1 Response: {last_message} (Simple reasoning)"
        elif gaia_level == 2:
            response = f"Mock Level 2 Response: {last_message} (Multi-step research)"
        else:
            response = f"Mock Level 3 Response: {last_message} (Complex analysis)"
        
        return {
            "content": response,
            "gaia_level": gaia_level,
            "response_time": 0.5,
            "agent_chain": ["planning", f"level{gaia_level}_agent"],
            "input_tokens": len(str(request)) // 4,
            "output_tokens": len(response) // 4
        }
    
    def _classify_gaia_level(self, content: str) -> int:
        """Mock GAIA level classification"""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ["calculate", "simple", "what is", "define"]):
            return 1
        elif any(word in content_lower for word in ["research", "find", "analyze", "compare"]):
            return 2
        else:
            return 3
    
    def get_providers(self) -> Dict[str, Any]:
        """Get current provider configurations"""
        return self.providers
    
    def get_agent_status(self) -> List[Dict[str, Any]]:
        """Get current agent status"""
        return [
            {
                "name": name,
                "status": "active",
                "specialization": f"level{name.split('_')[0] if '_' in name else 'planning'}_agent",
                "success_rate": 0.95
            }
            for name in self.active_agents
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
        print("🔧 Mock ACP Runtime configuration updated")
