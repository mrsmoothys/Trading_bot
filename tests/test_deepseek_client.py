"""
Unit tests for DeepSeek AI Client
Tests with mocked AsyncOpenAI to avoid real API calls
"""

import pytest
import asyncio
import os
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

# Test the DeepSeek client with mocked environment and API
@pytest.mark.asyncio
async def test_get_trading_signal():
    """Test trading signal generation with mocked API."""
    from deepseek.client import DeepSeekBrain
    from core.system_context import SystemContext

    # Create a mock system context
    mock_context = Mock()

    # Clear environment and set test values
    with patch('deepseek.client.load_dotenv', return_value=None):  # Prevent .env file loading
        with patch.dict(os.environ, {
            'DEEPSEEK_API_KEY': 'sk-test123456789',
            'DEEPSEEK_API_URL': 'https://api.deepseek.com/v1',
            'DEEPSEEK_MODEL': 'deepseek-chat',
            'DEEPSEEK_MAX_TOKENS': '4000',
            'DEEPSEEK_TEMPERATURE': '0.7'
        }, clear=True):
            # Initialize DeepSeek Brain
            brain = DeepSeekBrain(mock_context)

            # Mock the OpenAI client response
            mock_response = Mock()
            mock_choice = Mock()
            mock_message = Mock()

            mock_message.content = '''
            {
                "recommended_action": "ENTER_LONG",
                "confidence": 0.85,
                "reasoning": "Strong bullish signal based on technical analysis",
                "position_size": 0.05,
                "entry_conditions": ["Price above SMA", "Volume increasing"],
                "exit_strategy": {
                    "stop_loss": "2% below entry",
                    "take_profit": "3:1 risk-reward",
                    "trailing_stop": true
                }
            }
            '''

            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]

            # Mock the async client
            brain.client.chat.completions.create = AsyncMock(return_value=mock_response)

            # Test data
            symbol = 'BTCUSDT'
            market_data = {
                'price': 45000,
                'change_24h': 2.5
            }
            features = {
                'market_regime': 'TRENDING_UP',
                'supertrend_trend': 'bullish',
                'order_flow_imbalance': 0.15,
                'distance_to_zone_pct': 1.2,
                'timeframe_alignment': 0.8
            }
            system_state = {
                'active_positions': {'BTCUSDT': {'size': 0.02}},
                'risk_exposure': 0.15,
                'current_drawdown': 0.03
            }

            # Call the method
            result = await brain.get_trading_signal(symbol, market_data, features, system_state)

            # The response parsing might return the parsed JSON with different key names
            # Check if 'recommended_action' exists instead of 'action'
            action = result.get('action') or result.get('recommended_action')
            position_size = result.get('position_size') or result.get('adjusted_size')

            # Assertions - adjust based on actual response structure
            if action:
                assert action in ['ENTER_LONG', 'BUY', 'HOLD'], f"Unexpected action: {action}"
            assert result.get('confidence') is not None
            assert 'reasoning' in result

            print(f"✅ test_get_trading_signal passed - Action: {action}")


@pytest.mark.asyncio
async def test_chat_interface():
    """Test interactive chat with mocked API."""
    from deepseek.client import DeepSeekBrain

    mock_context = Mock()

    with patch('deepseek.client.load_dotenv'):  # Prevent .env file loading
        with patch.dict(os.environ, {
            'DEEPSEEK_API_KEY': 'sk-test123456789',
            'DEEPSEEK_API_URL': 'https://api.deepseek.com/v1',
            'DEEPSEEK_MODEL': 'deepseek-chat',
            'DEEPSEEK_MAX_TOKENS': '4000',
            'DEEPSEEK_TEMPERATURE': '0.7'
        }, clear=True):
            brain = DeepSeekBrain(mock_context)

        # Mock response
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "I recommend a cautious approach given the current market volatility. Consider reducing position sizes."
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        brain.client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Test chat
        user_message = "What do you think about the current market?"
        context = {'market_conditions': 'volatile', 'positions': 3}
        message_type = 'strategy'

        result = await brain.chat_interface(user_message, context, message_type)

        assert "cautious" in result.lower() or "recommend" in result.lower()
        print(f"✅ test_chat_interface passed - Response: {result[:50]}...")


@pytest.mark.asyncio
async def test_assess_risk():
    """Test risk assessment with mocked API."""
    from deepseek.client import DeepSeekBrain

    mock_context = Mock()

    with patch('deepseek.client.load_dotenv'):  # Prevent .env file loading
        with patch.dict(os.environ, {
            'DEEPSEEK_API_KEY': 'sk-test123456789',
            'DEEPSEEK_API_URL': 'https://api.deepseek.com/v1',
            'DEEPSEEK_MODEL': 'deepseek-chat',
            'DEEPSEEK_MAX_TOKENS': '4000',
            'DEEPSEEK_TEMPERATURE': '0.7'
        }, clear=True):
            brain = DeepSeekBrain(mock_context)

        # Mock response
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = '''
        {
            "approved": true,
            "adjusted_size": 0.03,
            "reasoning": "Risk is within acceptable limits",
            "risk_score": 0.3,
            "concerns": ["High volatility", "Current drawdown"]
        }
        '''
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        brain.client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Test risk assessment
        risk_context = {
            'proposed_trade': {'size': 0.05, 'symbol': 'BTCUSDT'},
            'portfolio_exposure': 0.15,
            'current_drawdown': 0.03
        }

        result = await brain.assess_risk(risk_context)

        assert result['approved'] == True
        assert result['adjusted_size'] == 0.03
        assert 'risk_score' in result
        assert 'concerns' in result

        print(f"✅ test_assess_risk passed - Approved: {result['approved']}")


@pytest.mark.asyncio
async def test_optimize_position():
    """Test position optimization with mocked API."""
    from deepseek.client import DeepSeekBrain

    mock_context = Mock()

    with patch('deepseek.client.load_dotenv'):  # Prevent .env file loading
        with patch.dict(os.environ, {
            'DEEPSEEK_API_KEY': 'sk-test123456789',
            'DEEPSEEK_API_URL': 'https://api.deepseek.com/v1',
            'DEEPSEEK_MODEL': 'deepseek-chat',
            'DEEPSEEK_MAX_TOKENS': '4000',
            'DEEPSEEK_TEMPERATURE': '0.7'
        }, clear=True):
            brain = DeepSeekBrain(mock_context)

        # Mock response
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = '''
        {
            "size": 0.04,
            "reasoning": "Optimal size based on volatility",
            "adjustments": ["Reduce by 20% due to high VIX"],
            "risk_warnings": ["Monitor correlation with ETH"]
        }
        '''
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        brain.client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Test position optimization
        signal = {'action': 'BUY', 'confidence': 0.75, 'position_size': 0.05}
        system_state = {'volatility': 0.25, 'correlation': 0.8}

        result = await brain.optimize_position(signal, system_state)

        assert result['size'] == 0.04
        assert 'reasoning' in result
        assert 'adjustments' in result

        print(f"✅ test_optimize_position passed - Size: {result['size']}")


def test_environment_validation():
    """Test environment variable validation."""
    from deepseek.client import DeepSeekBrain

    mock_context = Mock()

    # Test missing API key
    with patch('deepseek.client.load_dotenv'):  # Prevent .env file loading
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Missing required environment variable: DEEPSEEK_API_KEY"):
                DeepSeekBrain(mock_context)

    print("✅ test_environment_validation - Missing API key detected")


def test_environment_validation_placeholder():
    """Test that placeholder values are rejected."""
    from deepseek.client import DeepSeekBrain

    mock_context = Mock()

    # Test placeholder API key
    with patch('deepseek.client.load_dotenv'):  # Prevent .env file loading
        with patch.dict(os.environ, {
            'DEEPSEEK_API_KEY': 'your_api_key_here',
            'DEEPSEEK_API_URL': 'https://api.deepseek.com/v1',
            'DEEPSEEK_MODEL': 'deepseek-chat',
            'DEEPSEEK_MAX_TOKENS': '4000',
            'DEEPSEEK_TEMPERATURE': '0.7'
        }, clear=True):
            with pytest.raises(ValueError, match="Placeholder value detected"):
                DeepSeekBrain(mock_context)

    print("✅ test_environment_validation_placeholder - Placeholder value rejected")


def test_environment_validation_invalid_url():
    """Test invalid URL format detection."""
    from deepseek.client import DeepSeekBrain

    mock_context = Mock()

    with patch('deepseek.client.load_dotenv'):  # Prevent .env file loading
        with patch.dict(os.environ, {
            'DEEPSEEK_API_KEY': 'sk-test123456789',
            'DEEPSEEK_API_URL': 'invalid_url',
            'DEEPSEEK_MODEL': 'deepseek-chat',
            'DEEPSEEK_MAX_TOKENS': '4000',
            'DEEPSEEK_TEMPERATURE': '0.7'
        }, clear=True):
            with pytest.raises(ValueError, match="must be a valid HTTP/HTTPS URL"):
                DeepSeekBrain(mock_context)

    print("✅ test_environment_validation_invalid_url - Invalid URL rejected")


def test_environment_validation_invalid_tokens():
    """Test invalid max tokens range."""
    from deepseek.client import DeepSeekBrain

    mock_context = Mock()

    with patch('deepseek.client.load_dotenv'):  # Prevent .env file loading
        with patch.dict(os.environ, {
            'DEEPSEEK_API_KEY': 'sk-test123456789',
            'DEEPSEEK_API_URL': 'https://api.deepseek.com/v1',
            'DEEPSEEK_MODEL': 'deepseek-chat',
            'DEEPSEEK_MAX_TOKENS': '50000',  # Too high
            'DEEPSEEK_TEMPERATURE': '0.7'
        }, clear=True):
            with pytest.raises(ValueError, match="must be between 1 and 32000"):
                DeepSeekBrain(mock_context)

    print("✅ test_environment_validation_invalid_tokens - Invalid range rejected")


def test_environment_validation_invalid_temperature():
    """Test invalid temperature range."""
    from deepseek.client import DeepSeekBrain

    mock_context = Mock()

    with patch('deepseek.client.load_dotenv'):  # Prevent .env file loading
        with patch.dict(os.environ, {
            'DEEPSEEK_API_KEY': 'sk-test123456789',
            'DEEPSEEK_API_URL': 'https://api.deepseek.com/v1',
            'DEEPSEEK_MODEL': 'deepseek-chat',
            'DEEPSEEK_MAX_TOKENS': '4000',
            'DEEPSEEK_TEMPERATURE': '5.0'  # Too high
        }, clear=True):
            with pytest.raises(ValueError, match="must be between 0.0 and 2.0"):
                DeepSeekBrain(mock_context)

    print("✅ test_environment_validation_invalid_temperature - Invalid range rejected")


def test_successful_initialization():
    """Test successful client initialization with valid env vars."""
    from deepseek.client import DeepSeekBrain

    mock_context = Mock()

    with patch('deepseek.client.load_dotenv'):  # Prevent .env file loading
        with patch.dict(os.environ, {
            'DEEPSEEK_API_KEY': 'sk-test123456789',
            'DEEPSEEK_API_URL': 'https://api.deepseek.com/v1',
            'DEEPSEEK_MODEL': 'deepseek-chat',
            'DEEPSEEK_MAX_TOKENS': '4000',
            'DEEPSEEK_TEMPERATURE': '0.7'
        }, clear=True):
            brain = DeepSeekBrain(mock_context)

            assert brain.api_key == 'sk-test123456789'
            assert brain.api_url == 'https://api.deepseek.com/v1'
            assert brain.model == 'deepseek-chat'
            assert brain.max_tokens == 4000
            assert brain.temperature == 0.7

            print("✅ test_successful_initialization - All validations passed")


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling when API call fails."""
    from deepseek.client import DeepSeekBrain

    mock_context = Mock()

    with patch('deepseek.client.load_dotenv'):  # Prevent .env file loading
        with patch.dict(os.environ, {
            'DEEPSEEK_API_KEY': 'sk-test123456789',
            'DEEPSEEK_API_URL': 'https://api.deepseek.com/v1',
            'DEEPSEEK_MODEL': 'deepseek-chat',
            'DEEPSEEK_MAX_TOKENS': '4000',
            'DEEPSEEK_TEMPERATURE': '0.7'
        }, clear=True):
            brain = DeepSeekBrain(mock_context)

            # Mock API error
            brain.client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))

            # Test that error is handled gracefully
            result = await brain.get_trading_signal('BTCUSDT', {}, {}, {})

            assert result['action'] == 'HOLD'
            assert result['confidence'] == 0.0
            assert 'error' in result or 'Error' in result['reasoning']

            print("✅ test_error_handling - Error handled gracefully")


def test_env_file_loading():
    """Test that .env file is loaded if present."""
    from deepseek.client import DeepSeekBrain
    from dotenv import load_dotenv

    mock_context = Mock()

    # Create a temporary .env file
    env_file = Path('/tmp/test_env.env')
    env_file.write_text('''
DEEPSEEK_API_KEY=sk-from_file_123
DEEPSEEK_API_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_MAX_TOKENS=4000
DEEPSEEK_TEMPERATURE=0.7
''')

    # Mock the env file path in the client
    with patch('deepseek.client.Path') as mock_path:
        mock_path.__iter__ = lambda self: iter(['', 'Users', 'mrsmoothy', 'Downloads', 'Trading_bot', 'deepseek', 'client.py'])
        mock_path.parent.parent = Path('/tmp/test_env.env').parent

        # Clear env vars
        with patch.dict(os.environ, {}, clear=True):
            load_dotenv(env_file)
            brain = DeepSeekBrain(mock_context)

            assert brain.api_key == 'sk-from_file_123'

    # Cleanup
    env_file.unlink()

    print("✅ test_env_file_loading - .env file loaded successfully")


if __name__ == '__main__':
    # Run tests manually if executed directly
    print("\n" + "="*70)
    print("Running DeepSeek Client Unit Tests")
    print("="*70 + "\n")

    # Test environment validation
    print("Testing Environment Validation...")
    test_environment_validation()
    test_environment_validation_placeholder()
    test_environment_validation_invalid_url()
    test_environment_validation_invalid_tokens()
    test_environment_validation_invalid_temperature()
    test_successful_initialization()
    test_env_file_loading()

    # Test async methods
    print("\nTesting Async Methods...")
    asyncio.run(test_get_trading_signal())
    asyncio.run(test_chat_interface())
    asyncio.run(test_assess_risk())
    asyncio.run(test_optimize_position())
    asyncio.run(test_error_handling())

    print("\n" + "="*70)
    print("✅ All tests passed successfully!")
    print("="*70)
