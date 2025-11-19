
import asyncio
from core.system_context import SystemContext
from core.autonomous_optimizer import AutonomousOptimizer
from unittest.mock import MagicMock, AsyncMock

async def test_autonomous_capabilities():
    print("Testing Autonomous Mode Capabilities...")
    
    # 1. Test Guardrails
    print("\n1. Testing Safety Guardrails...")
    context = SystemContext()
    
    try:
        context.update_config("risk", "max_drawdown", 0.50) # 50% drawdown
        print("❌ FAILED: Guardrail did not stop dangerous drawdown!")
    except ValueError as e:
        print(f"✅ PASSED: Guardrail stopped dangerous drawdown: {e}")

    try:
        context.update_config("risk", "max_drawdown", 0.15) # 15% drawdown (safe)
        print(f"✅ PASSED: Allowed safe drawdown update. Value is now: {context.config['risk']['max_drawdown']}")
    except ValueError as e:
        print(f"❌ FAILED: Blocked safe drawdown update: {e}")

    # 2. Test Optimizer Logic
    print("\n2. Testing Autonomous Optimizer...")
    
    # Mock DeepSeek to return a specific suggestion
    mock_deepseek = MagicMock()
    mock_deepseek.optimize_system = AsyncMock(return_value={
        "suggestions": [
            {
                "category": "trading",
                "param": "leverage",
                "value": 2,
                "priority": "high",
                "recommendation": "Decrease leverage to 2x"
            }
        ]
    })
    
    optimizer = AutonomousOptimizer(context, mock_deepseek)
    
    # Force optimization (bypass time check)
    optimizer.last_optimization = datetime.now().replace(year=2000) 
    
    await optimizer.run_optimization_cycle()
    
    # Check if config was updated
    current_leverage = context.config.get("trading", {}).get("leverage")
    if current_leverage == 2:
        print("✅ PASSED: Optimizer successfully updated system config!")
    else:
        print(f"❌ FAILED: Optimizer did not update config. Current leverage: {current_leverage}")

from datetime import datetime
if __name__ == "__main__":
    asyncio.run(test_autonomous_capabilities())
