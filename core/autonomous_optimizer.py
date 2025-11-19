"""
Autonomous Optimizer Module
Enables the system to self-optimize by periodically analyzing performance
and applying DeepSeek's recommendations to system configuration.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any
from loguru import logger

class AutonomousOptimizer:
    """
    Manages the autonomous optimization loop.
    Connects system performance metrics with DeepSeek's reasoning capabilities
    to automatically adjust system parameters.
    """
    
    def __init__(self, system_context, deepseek_brain):
        """
        Initialize the optimizer.
        
        Args:
            system_context: The SystemContext instance to read metrics and update config
            deepseek_brain: The DeepSeekBrain instance to generate suggestions
        """
        self.context = system_context
        self.deepseek = deepseek_brain
        self.last_optimization = datetime.now()
        self.optimization_interval_hours = 1
        self.is_running = False

    async def start(self):
        """Start the autonomous optimization loop."""
        self.is_running = True
        logger.info("🧠 Autonomous Optimizer started")
        
        while self.is_running:
            try:
                # Run optimization cycle
                await self.run_optimization_cycle()
                
                # Wait for next interval (check every minute if interval passed)
                for _ in range(60):
                    if not self.is_running:
                        break
                    await asyncio.sleep(60)
                    
            except Exception as e:
                logger.error(f"Error in autonomous loop: {e}")
                await asyncio.sleep(60) # Wait a bit before retrying

    def stop(self):
        """Stop the autonomous optimization loop."""
        self.is_running = False
        logger.info("🧠 Autonomous Optimizer stopped")

    async def run_optimization_cycle(self):
        """
        Execute a single optimization cycle:
        1. Gather performance metrics
        2. Request analysis from DeepSeek
        3. Apply approved changes
        """
        # Check if enough time has passed
        time_since_last = (datetime.now() - self.last_optimization).total_seconds() / 3600
        if time_since_last < self.optimization_interval_hours:
            return

        logger.info("🧠 Starting autonomous optimization cycle...")
        
        # 1. Get performance data
        metrics = self.context.get_performance_summary()
        
        # Add current config to context so AI knows what to change
        current_config = {
            "risk": self.context.config.get("risk", {}),
            "trading": self.context.config.get("trading", {})
        }
        metrics["current_config"] = current_config
        
        # 2. Ask DeepSeek what to do
        # We use the existing optimize_system method
        optimization_result = await self.deepseek.optimize_system(metrics)
        
        if not optimization_result or "suggestions" not in optimization_result:
            logger.warning("No optimization suggestions received")
            return

        # 3. Apply the changes automatically
        changes_applied = 0
        
        for suggestion in optimization_result.get("suggestions", []):
            try:
                # Only apply High or Medium priority changes automatically
                priority = suggestion.get("priority", "low").lower()
                if priority not in ["high", "medium"]:
                    logger.info(f"Skipping low priority suggestion: {suggestion.get('recommendation')}")
                    continue
                    
                category = suggestion.get("category")
                # We expect the AI to provide 'param' and 'value' in the suggestion
                # If the format differs, we might need to parse 'recommendation' text
                # For this implementation, we assume the AI follows the JSON structure we requested
                
                # Note: The prompt in DeepSeekBrain._build_optimization_prompt might need 
                # to be updated to explicitly ask for 'param' and 'value' fields.
                # For now, we'll log the suggestion for manual review if fields are missing.
                
                param = suggestion.get("param")
                value = suggestion.get("value")
                
                if category and param and value is not None:
                    self.context.update_config(category, param, value)
                    changes_applied += 1
                else:
                    logger.warning(f"Cannot apply suggestion automatically (missing fields): {suggestion}")
                    
            except ValueError as ve:
                logger.warning(f"Guardrail prevented change: {ve}")
            except Exception as e:
                logger.error(f"Error applying suggestion: {e}")
                
        if changes_applied > 0:
            logger.info(f"✅ Applied {changes_applied} optimization changes")
            self.last_optimization = datetime.now()
        else:
            logger.info("No changes applied this cycle")
