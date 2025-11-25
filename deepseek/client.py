"""
DeepSeek Brain Client
Direct integration with DeepSeek AI for trading decisions.
OpenAI-compatible client for maximum flexibility.
Memory-optimized with M1MemoryManager integration.
"""

import os
import json
import asyncio
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
from loguru import logger
from dotenv import load_dotenv


class DeepSeekBrain:
    """
    DeepSeek AI Brain for the trading system.
    Handles all AI-related operations including:
    - Trading signal generation
    - System optimization
    - Interactive chat
    - risk assessment

    Memory-optimized with M1MemoryManager for M1 MacBook constraints.
    """

    def __init__(self, system_context, memory_manager=None):
        """
        Initialize DeepSeek Brain with system context and memory manager.

        Args:
            system_context: SystemContext instance for state awareness
            memory_manager: M1MemoryManager instance for memory optimization

        Raises:
            ValueError: If required environment variables are missing or invalid
        """
        self.system_context = system_context
        self.memory_manager = memory_manager

        # Load environment variables from .env file
        env_file = Path(__file__).parent.parent / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            logger.debug(f"Loaded environment from {env_file}")
        else:
            logger.warning(".env file not found - using system environment variables")

        # Load and validate environment variables
        self._load_and_validate_config()

        # Initialize OpenAI client with DeepSeek base URL
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_url
        )

        # Optional Reasoner client (separate endpoint/model)
        if self.reasoner_api_url:
            self.reasoner_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.reasoner_api_url
            )
        else:
            self.reasoner_client = self.client

        logger.info(f"DeepSeek Brain initialized with model: {self.model}")

    def _load_and_validate_config(self):
        """Load and validate all DeepSeek configuration from environment."""
        errors = []

        # Required variables
        required_vars = {
            'DEEPSEEK_API_KEY': str,
            'DEEPSEEK_API_URL': str,
            'DEEPSEEK_MODEL': str,
            'DEEPSEEK_MAX_TOKENS': int,
            'DEEPSEEK_TEMPERATURE': float,
        }

        config = {}
        for var_name, var_type in required_vars.items():
            value = os.getenv(var_name)
            if not value:
                errors.append(f"Missing required environment variable: {var_name}")
                continue

            # Check for placeholder values
            if value.startswith('your_') or value.startswith('replace_with_'):
                errors.append(f"Placeholder value detected for {var_name}. Please set a real value.")
                continue

            # Type conversion and validation
            try:
                if var_type == int:
                    converted = int(value)
                    if var_name == 'DEEPSEEK_MAX_TOKENS' and (converted <= 0 or converted > 32000):
                        errors.append(f"{var_name} must be between 1 and 32000, got {converted}")
                        continue
                    config[var_name] = converted
                elif var_type == float:
                    converted = float(value)
                    if var_name == 'DEEPSEEK_TEMPERATURE' and (converted < 0 or converted > 2):
                        errors.append(f"{var_name} must be between 0.0 and 2.0, got {converted}")
                        continue
                    config[var_name] = converted
                else:
                    config[var_name] = value
            except ValueError as e:
                errors.append(f"Invalid value for {var_name}: {value} - {str(e)}")
                continue

        # Additional validations
        if 'DEEPSEEK_API_KEY' in config:
            api_key = config['DEEPSEEK_API_KEY']
            if not api_key.startswith('sk-'):
                errors.append("DEEPSEEK_API_KEY should start with 'sk-'")

        if 'DEEPSEEK_API_URL' in config:
            api_url = config['DEEPSEEK_API_URL']
            if not api_url.startswith(('http://', 'https://')):
                errors.append("DEEPSEEK_API_URL must be a valid HTTP/HTTPS URL")

        # Optional reasoner config
        reasoner_api_url = os.getenv('DEEPSEEK_REASONER_API_URL')
        reasoner_model = os.getenv('DEEPSEEK_REASONER_MODEL')
        if reasoner_api_url and not reasoner_api_url.startswith(('http://', 'https://')):
            errors.append("DEEPSEEK_REASONER_API_URL must be a valid HTTP/HTTPS URL")

        if errors:
            error_msg = "DeepSeek configuration validation failed:\n  " + "\n  ".join(errors)
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Set validated configuration
        self.api_key = config['DEEPSEEK_API_KEY']
        self.api_url = config['DEEPSEEK_API_URL']
        self.model = config['DEEPSEEK_MODEL']
        self.max_tokens = config['DEEPSEEK_MAX_TOKENS']
        self.temperature = config['DEEPSEEK_TEMPERATURE']
        self.reasoner_api_url = reasoner_api_url or self.api_url
        self.reasoner_model = reasoner_model or 'deepseek-reasoner'

        logger.debug(
            f"DeepSeek configuration validated: "
            f"model={self.model}, max_tokens={self.max_tokens}, temperature={self.temperature}"
        )

    def _get_optimized_context(self) -> Dict[str, Any]:
        """
        Get memory-optimized context for DeepSeek analysis.

        Returns:
            Optimized context dictionary
        """
        if self.memory_manager:
            return self.memory_manager.optimize_context_for_deepseek(self.system_context)
        else:
            # Fallback to system context default method
            return self.system_context.get_context_for_deepseek()

    async def get_trading_signal(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate trading signal using DeepSeek analysis.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            market_data: Current market data
            features: Calculated technical features

        Returns:
            Dict containing signal decision and reasoning
        """
        try:
            # Get memory-optimized system context
            system_state = self._get_optimized_context()

            prompt = self._build_trading_prompt(
                symbol, market_data, features, system_state
            )

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            result = self._parse_trading_response(response)

            logger.info(
                f"DeepSeek signal for {symbol}: {result.get('action', 'HOLD')} "
                f"(confidence: {result.get('confidence', 0):.2f})"
            )

            return result

        except Exception as e:
            logger.error(f"Error getting trading signal: {e}")
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}",
                "error": True
            }

    async def optimize_system(self, performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggest system improvements based on performance.

        Args:
            performance_metrics: Current performance metrics

        Returns:
            Optimization suggestions
        """
        try:
            prompt = self._build_optimization_prompt(performance_metrics)

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            result = self._parse_optimization_response(response)

            logger.info(f"System optimization suggestions received")

            return result

        except Exception as e:
            logger.error(f"Error optimizing system: {e}")
            return {
                "suggestions": [],
                "reasoning": f"Error: {str(e)}",
                "error": True
            }

    async def run_reasoner(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """Invoke the DeepSeek Reasoner endpoint for complex reasoning tasks."""

        try:
            client = self.reasoner_client or self.client
            response = await client.chat.completions.create(
                model=self.reasoner_model,
                messages=messages,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature if temperature is not None else self.temperature
            )
            return {
                "content": response.choices[0].message.content,
                "usage": response.usage if hasattr(response, 'usage') else None
            }
        except Exception as exc:
            logger.error(f"Reasoner call failed: {exc}")
            return {
                "content": None,
                "error": True,
                "message": str(exc)
            }

    async def chat_interface(
        self,
        user_message: str,
        context: Dict[str, Any],
        message_type: str = "strategy"
    ) -> str:
        """
        Interactive chat with DeepSeek for strategy discussions.

        Args:
            user_message: User's message
            context: Current system context
            message_type: Type of message (strategy, system, market)

        Returns:
            AI response
        """
        try:
            prompt = self._build_chat_prompt(user_message, context, message_type)

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_chat_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            ai_response = response.choices[0].message.content

            # Update conversation memory
            self.system_context.add_conversation_message(
                user_message, ai_response, message_type
            )

            logger.info(f"Chat response: {ai_response[:100]}...")

            return ai_response

        except Exception as e:
            logger.error(f"Error in chat interface: {e}")
            return f"Sorry, I encountered an error: {str(e)}"

    async def assess_risk(self, risk_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess trade risk using DeepSeek analysis.

        Args:
            risk_context: Risk context data

        Returns:
            Risk assessment
        """
        try:
            prompt = self._build_risk_prompt(risk_context)

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_risk_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3  # Lower temperature for risk decisions
            )

            result = self._parse_risk_response(response)

            logger.info(f"Risk assessment: {result.get('approved', False)}")

            return result

        except Exception as e:
            logger.error(f"Error assessing risk: {e}")
            return {
                "approved": False,
                "reason": f"Error: {str(e)}",
                "error": True
            }

    async def optimize_position(
        self,
        signal: Dict[str, Any],
        system_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize position sizing and parameters.

        Args:
            signal: Trading signal
            system_state: Current system state

        Returns:
            Optimized position parameters
        """
        try:
            prompt = self._build_position_prompt(signal, system_state)

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_position_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.5
            )

            result = self._parse_position_response(response)

            logger.info(f"Position optimization: size={result.get('size', 0):.4f}")

            return result

        except Exception as e:
            logger.error(f"Error optimizing position: {e}")
            return {
                "size": signal.get("position_size", 0.02),
                "reasoning": f"Error: {str(e)}",
                "error": True
            }

    def _get_system_prompt(self) -> str:
        """Get system prompt for DeepSeek."""
        return """You are DeepSeek, an advanced AI trading system. You analyze market data,
make trading decisions, and optimize system performance. You have complete awareness
of the trading system state including positions, risk, and market conditions.

Your decisions should be:
- Data-driven based on provided features
- Risk-aware with capital preservation first
- Confident but not overconfident
- Clear and well-reasoned

Always respond in JSON format with clear reasoning."""

    def _get_chat_prompt(self) -> str:
        """Get chat-specific system prompt."""
        return """You are DeepSeek, an interactive AI trading assistant. You can discuss
trading strategies, system performance, market analysis, and answer questions about
the trading system. Be helpful, insightful, and educational.

Provide clear, actionable advice in a conversational tone."""

    def _get_risk_prompt(self) -> str:
        """Get risk assessment prompt."""
        return """You are a risk management specialist. Analyze the proposed trade and
assess if it meets risk criteria. Consider:
- Current portfolio exposure
- Market conditions
- Risk-reward ratio
- System drawdown limits

Approve trades only when risk is acceptable. Respond in JSON."""

    def _get_position_prompt(self) -> str:
        """Get position optimization prompt."""
        return """You are a position sizing expert. Optimize position size based on:
- Signal confidence
- Market volatility
- Current exposure
- Risk limits

Return optimal size as a decimal (e.g., 0.05 for 5%). Respond in JSON."""

    def _build_trading_prompt(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        features: Dict[str, Any],
        system_state: Dict[str, Any]
    ) -> str:
        """Build trading analysis prompt."""
        return f"""Analyze this trading opportunity for {symbol}:

CURRENT MARKET DATA:
- Price: ${market_data.get('price', 'N/A')}
- 24h Change: {market_data.get('change_24h', 'N/A')}%
- Sentiment: {system_state.get('sentiment', {}).get('value', 'N/A')} ({system_state.get('sentiment', {}).get('classification', 'N/A')})

TECHNICAL FEATURES:
- Market Regime: {features.get('market_regime', 'UNKNOWN')}
- Supertrend: {features.get('supertrend_trend', 'neutral')}
- Order Flow Imbalance: {features.get('order_flow_imbalance', 0):.3f}
- Liquidity Distance: {features.get('distance_to_zone_pct', 0):.3f}
- Timeframe Alignment: {features.get('timeframe_alignment', 0):.2f}

SYSTEM STATE:
- Active Positions: {len(system_state.get('active_positions', {}))}
- Total Exposure: {system_state.get('risk_exposure', 0):.2%}
- Current Drawdown: {system_state.get('current_drawdown', 0):.2%}

Provide your analysis in this JSON format:
{{
    "recommended_action": "ENTER_LONG|ENTER_SHORT|HOLD|CLOSE_POSITION",
    "confidence": 0.0-1.0,
    "reasoning": "Detailed explanation",
    "position_size": 0.0-1.0 (if applicable),
    "entry_conditions": ["condition1", "condition2"],
    "exit_strategy": {{
        "stop_loss": "price or method",
        "take_profit": "targets",
        "trailing_stop": true/false
    }}
}}"""

    def _build_optimization_prompt(self, performance_metrics: Dict[str, Any]) -> str:
        """Build system optimization prompt."""
        return f"""Analyze system performance and suggest improvements:

PERFORMANCE METRICS:
{json.dumps(performance_metrics, indent=2)}

Provide optimization suggestions in JSON:
{{
    "suggestions": [
        {{
            "category": "feature|risk|strategy|system",
            "recommendation": "what to change",
            "reasoning": "why this helps",
            "priority": "high|medium|low"
        }}
    ],
    "reasoning": "overall analysis"
}}"""

    def _build_chat_prompt(
        self,
        user_message: str,
        context: Dict[str, Any],
        message_type: str
    ) -> str:
        """Build chat prompt with conversation memory."""
        history_entries: List[str] = []

        # Chat-specific history from dashboard/chat interface
        for msg in context.get("conversation_history", [])[-10:]:
            role = msg.get('role', 'user')
            speaker = 'User' if role == 'user' else 'DeepSeek AI'
            timestamp = msg.get('timestamp', '')
            text = msg.get('text', '')
            history_entries.append(f"{timestamp} {speaker}: {text}")

        # System-level conversation memory (e.g., prior runs)
        system_state = context.get("system_state", {}) or {}
        system_memory = system_state.get("conversation_memory", [])
        for msg in system_memory[-5:]:
            ts = msg.get('timestamp', '')
            history_entries.append(f"{ts} User: {msg.get('user', '')}")
            history_entries.append(f"{ts} DeepSeek AI: {msg.get('ai', '')}")

        conversation_block = "No prior conversation available."
        if history_entries:
            conversation_block = "\n".join(history_entries)

        # Format context (exclude conversation_history since we're adding it separately)
        context_copy = {k: v for k, v in context.items() if k != "conversation_history"}
        context_str = json.dumps(context_copy, indent=2, default=str)

        return f"""Conversation History:
{conversation_block}

System Context:
{context_str}

User Message:
{user_message}

Respond helpfully as DeepSeek AI, taking the conversation into account."""

    def _build_risk_prompt(self, risk_context: Dict[str, Any]) -> str:
        """Build risk assessment prompt."""
        return f"""Assess risk for this trade:

{risk_context}

SENTIMENT: {self.system_context.sentiment.get('value', 'N/A')} ({self.system_context.sentiment.get('classification', 'N/A')})

Respond in JSON:
{{
    "approved": true/false,
    "adjusted_size": 0.0-1.0,
    "reasoning": "explanation",
    "risk_score": 0.0-1.0,
    "concerns": ["concern1", "concern2"]
}}"""

    def _build_position_prompt(
        self,
        signal: Dict[str, Any],
        system_state: Dict[str, Any]
    ) -> str:
        """Build position optimization prompt."""
        return f"""Optimize position for this signal:

SIGNAL: {json.dumps(signal, indent=2)}

SYSTEM STATE: {json.dumps(system_state, indent=2)}

Respond in JSON:
{{
    "size": 0.0-1.0,
    "reasoning": "explanation",
    "adjustments": ["adjustment1"],
    "risk_warnings": ["warning1"]
}}"""

    def _parse_trading_response(self, response) -> Dict[str, Any]:
        """Parse trading signal response."""
        try:
            content = response.choices[0].message.content
            # Try to extract JSON
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback parsing
                return {
                    "action": "HOLD",
                    "confidence": 0.5,
                    "reasoning": content,
                    "raw_response": content
                }
        except Exception as e:
            logger.error(f"Error parsing trading response: {e}")
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "reasoning": f"Parse error: {str(e)}"
            }

    def _parse_optimization_response(self, response) -> Dict[str, Any]:
        """Parse optimization response."""
        try:
            content = response.choices[0].message.content
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {"suggestions": [], "reasoning": content}
        except Exception as e:
            logger.error(f"Error parsing optimization response: {e}")
            return {"suggestions": [], "reasoning": f"Parse error: {str(e)}"}

    def _parse_risk_response(self, response) -> Dict[str, Any]:
        """Parse risk assessment response."""
        try:
            content = response.choices[0].message.content
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {"approved": False, "reasoning": content}
        except Exception as e:
            logger.error(f"Error parsing risk response: {e}")
            return {"approved": False, "reasoning": f"Parse error: {str(e)}"}

    def _parse_position_response(self, response) -> Dict[str, Any]:
        """Parse position optimization response."""
        try:
            content = response.choices[0].message.content
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {"size": 0.02, "reasoning": content}
        except Exception as e:
            logger.error(f"Error parsing position response: {e}")
            return {"size": 0.02, "reasoning": f"Parse error: {str(e)}"}
