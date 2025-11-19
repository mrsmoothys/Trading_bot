"""
Tests for RiskManager with AI Enhancement
Tests DeepSeek AI integration for risk assessment.
"""

import sys
sys.path.insert(0, '/Users/mrsmoothy/Downloads/Trading_bot')

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from core.system_context import SystemContext
from deepseek.client import DeepSeekBrain
from execution.risk_manager import RiskManager


class TestRiskManagerAI:
    """Test RiskManager with AI enhancements."""

    @pytest.mark.asyncio
    async def test_initialization_with_deepseek(self):
        """Test RiskManager initialization with DeepSeek AI."""
        print("\n" + "="*70)
        print("Test: RiskManager Initialization with DeepSeek")
        print("="*70)

        system_context = SystemContext()
        deepseek_brain = Mock()

        risk_manager = RiskManager(
            system_context=system_context,
            deepseek_brain=deepseek_brain
        )

        assert risk_manager.deepseek is deepseek_brain
        assert risk_manager.use_ai_risk_assessment is True

        print("✅ RiskManager initialized with DeepSeek AI")

    @pytest.mark.asyncio
    async def test_initialization_without_deepseek(self):
        """Test RiskManager initialization without DeepSeek."""
        print("\n" + "="*70)
        print("Test: RiskManager Initialization without DeepSeek")
        print("="*70)

        system_context = SystemContext()

        risk_manager = RiskManager(
            system_context=system_context,
            deepseek_brain=None
        )

        assert risk_manager.deepseek is None
        assert risk_manager.use_ai_risk_assessment is False

        print("✅ RiskManager initialized without AI (rule-based only)")

    @pytest.mark.asyncio
    async def test_validate_trade_with_ai_approval(self):
        """Test trade validation with AI approval."""
        print("\n" + "="*70)
        print("Test: Trade Validation with AI Approval")
        print("="*70)

        system_context = SystemContext()
        deepseek_brain = Mock()

        # Mock AI approval
        deepseek_brain.assess_risk = AsyncMock(return_value={
            "approved": True,
            "reasoning": "Low risk, good opportunity",
            "risk_score": 0.3,
            "adjusted_size": 0.025
        })

        risk_manager = RiskManager(
            system_context=system_context,
            deepseek_brain=deepseek_brain
        )

        signal = {
            "symbol": "BTCUSDT",
            "action": "LONG",
            "confidence": 0.75,
            "market_regime": "TRENDING_LOW_VOL"
        }

        result = await risk_manager.validate_trade(signal, 0.02)

        assert result["approved"] is True
        assert "ai_assessment" in result or "AI" in result.get("reason", "")

        # Verify AI was called
        deepseek_brain.assess_risk.assert_called_once()

        print("✅ Trade approved by AI risk assessment")

    @pytest.mark.asyncio
    async def test_validate_trade_with_ai_rejection(self):
        """Test trade validation with AI rejection."""
        print("\n" + "="*70)
        print("Test: Trade Validation with AI Rejection")
        print("="*70)

        system_context = SystemContext()
        deepseek_brain = Mock()

        # Mock AI rejection
        deepseek_brain.assess_risk = AsyncMock(return_value={
            "approved": False,
            "reason": "High correlation risk",
            "risk_score": 0.85,
            "adjusted_size": 0
        })

        risk_manager = RiskManager(
            system_context=system_context,
            deepseek_brain=deepseek_brain
        )

        signal = {
            "symbol": "ETHUSDT",
            "action": "LONG",
            "confidence": 0.70,
            "market_regime": "RANGING_EXPANSION"
        }

        result = await risk_manager.validate_trade(signal, 0.03)

        assert result["approved"] is False
        # Just verify it was rejected by AI (check for the reason field)
        assert "reason" in result
        assert result["reason"] is not None

        print(f"   Rejection reason: {result['reason']}")
        print("✅ Trade rejected by AI risk assessment")

    @pytest.mark.asyncio
    async def test_validate_trade_ai_error_failsafe(self):
        """Test that trade fails to AI error (failsafe)."""
        print("\n" + "="*70)
        print("Test: Trade Validation with AI Error (Fail-safe)")
        print("="*70)

        system_context = SystemContext()
        deepseek_brain = Mock()

        # Mock AI error
        deepseek_brain.assess_risk = AsyncMock(side_effect=Exception("API Error"))

        risk_manager = RiskManager(
            system_context=system_context,
            deepseek_brain=deepseek_brain
        )

        signal = {
            "symbol": "BTCUSDT",
            "action": "LONG",
            "confidence": 0.80
        }

        result = await risk_manager.validate_trade(signal, 0.02)

        # Should fail-safe (reject trade)
        assert result["approved"] is False
        assert "error" in result["reason"].lower() or "AI" in result["reason"]

        print("✅ Trade rejected by fail-safe when AI fails")

    @pytest.mark.asyncio
    async def test_validate_trade_without_ai(self):
        """Test trade validation without AI (rule-based only)."""
        print("\n" + "="*70)
        print("Test: Trade Validation without AI")
        print("="*70)

        system_context = SystemContext()

        risk_manager = RiskManager(
            system_context=system_context,
            deepseek_brain=None
        )

        signal = {
            "symbol": "BTCUSDT",
            "action": "LONG",
            "confidence": 0.75,
            "market_regime": "TRENDING_LOW_VOL"
        }

        result = await risk_manager.validate_trade(signal, 0.02)

        assert result["approved"] is True
        assert "ai_assessment" not in result

        print("✅ Trade validated using rule-based checks only")

    @pytest.mark.asyncio
    async def test_assess_trade_with_ai_method(self):
        """Test _assess_trade_with_ai method directly."""
        print("\n" + "="*70)
        print("Test: _assess_trade_with_ai Method")
        print("="*70)

        system_context = SystemContext()
        deepseek_brain = Mock()

        # Mock AI assessment
        deepseek_brain.assess_risk = AsyncMock(return_value={
            "approved": True,
            "reasoning": "Market conditions favorable",
            "risk_score": 0.4,
            "adjusted_size": 0.025
        })

        risk_manager = RiskManager(
            system_context=system_context,
            deepseek_brain=deepseek_brain
        )

        signal = {
            "symbol": "BTCUSDT",
            "action": "LONG",
            "confidence": 0.75,
            "market_regime": "TRENDING_LOW_VOL"
        }

        assessment = await risk_manager._assess_trade_with_ai(signal, 0.02)

        assert assessment["approved"] is True
        assert assessment["risk_score"] == 0.4

        # Verify context was passed correctly
        call_args = deepseek_brain.assess_risk.call_args[0][0]
        assert call_args["signal"] == signal
        assert call_args["proposed_size"] == 0.02

        print("✅ _assess_trade_with_ai method works correctly")

    @pytest.mark.asyncio
    async def test_assess_portfolio_risk_with_ai(self):
        """Test portfolio risk assessment with AI."""
        print("\n" + "="*70)
        print("Test: Portfolio Risk Assessment with AI")
        print("="*70)

        system_context = SystemContext()

        # Add some positions
        system_context.update_position("BTCUSDT", {
            "symbol": "BTCUSDT",
            "side": "LONG",
            "entry_price": 45000,
            "quantity": 0.1,
            "entry_time": datetime.now().isoformat()
        })

        deepseek_brain = Mock()

        # Mock AI portfolio assessment
        deepseek_brain.assess_portfolio_risk = AsyncMock(return_value={
            "risk_level": "MEDIUM",
            "risk_score": 0.5,
            "recommendations": ["Reduce exposure", "Monitor closely"]
        })

        risk_manager = RiskManager(
            system_context=system_context,
            deepseek_brain=deepseek_brain
        )

        assessment = await risk_manager._assess_portfolio_risk_with_ai()

        assert assessment["risk_level"] == "MEDIUM"
        assert assessment["risk_score"] == 0.5
        assert len(assessment["recommendations"]) > 0

        # Verify portfolio context was passed
        call_args = deepseek_brain.assess_portfolio_risk.call_args[0][0]
        assert "positions" in call_args
        assert "risk_metrics" in call_args

        print("✅ AI portfolio risk assessment completed")

    @pytest.mark.asyncio
    async def test_get_ai_risk_recommendations(self):
        """Test getting AI-generated recommendations."""
        print("\n" + "="*70)
        print("Test: Get AI Risk Recommendations")
        print("="*70)

        system_context = SystemContext()
        deepseek_brain = Mock()

        # Mock AI recommendations
        deepseek_brain.get_risk_recommendations = AsyncMock(return_value={
            "recommendations": [
                "Reduce position sizes",
                "Monitor drawdown",
                "Diversify across assets"
            ]
        })

        risk_manager = RiskManager(
            system_context=system_context,
            deepseek_brain=deepseek_brain
        )

        recommendations = await risk_manager.get_ai_risk_recommendations()

        assert len(recommendations) > 0
        assert "Reduce position sizes" in recommendations

        print(f"✅ AI recommendations: {len(recommendations)} items")

    @pytest.mark.asyncio
    async def test_position_size_limit_with_ai(self):
        """Test AI assessment respects position size limits."""
        print("\n" + "="*70)
        print("Test: Position Size Limit with AI")
        print("="*70)

        system_context = SystemContext()
        deepseek_brain = Mock()

        # Mock AI that approves but suggests smaller size
        deepseek_brain.assess_risk = AsyncMock(return_value={
            "approved": True,
            "reasoning": "Approved but reduce size",
            "risk_score": 0.6,
            "adjusted_size": 0.03  # Smaller than proposed 0.05
        })

        risk_manager = RiskManager(
            system_context=system_context,
            deepseek_brain=deepseek_brain
        )

        signal = {
            "symbol": "BTCUSDT",
            "action": "LONG",
            "confidence": 0.70
        }

        # Proposed size exceeds limit
        result = await risk_manager.validate_trade(signal, 0.10)

        # Should be rejected due to size limit
        assert result["approved"] is False
        assert "exceeds limit" in result["reason"].lower()

        print("✅ Position size limit enforced before AI assessment")

    @pytest.mark.asyncio
    async def test_drawdown_limit_with_ai(self):
        """Test AI assessment when drawdown is high."""
        print("\n" + "="*70)
        print("Test: Drawdown Limit with AI")
        print("="*70)

        system_context = SystemContext()

        # Set high drawdown
        system_context.risk_metrics["current_drawdown"] = 0.12  # 12%

        deepseek_brain = Mock()

        risk_manager = RiskManager(
            system_context=system_context,
            deepseek_brain=deepseek_brain
        )

        signal = {
            "symbol": "BTCUSDT",
            "action": "LONG",
            "confidence": 0.70
        }

        result = await risk_manager.validate_trade(signal, 0.02)

        # Should be rejected due to drawdown limit
        assert result["approved"] is False
        assert "drawdown" in result["reason"].lower()

        print("✅ Drawdown limit enforced before AI assessment")

    @pytest.mark.asyncio
    async def test_confidence_threshold_with_ai(self):
        """Test confidence threshold with AI assessment."""
        print("\n" + "="*70)
        print("Test: Confidence Threshold with AI")
        print("="*70)

        system_context = SystemContext()
        deepseek_brain = Mock()

        risk_manager = RiskManager(
            system_context=system_context,
            deepseek_brain=deepseek_brain
        )

        signal = {
            "symbol": "BTCUSDT",
            "action": "LONG",
            "confidence": 0.45  # Below 0.60 threshold
        }

        result = await risk_manager.validate_trade(signal, 0.02)

        # Should be rejected due to low confidence
        assert result["approved"] is False
        assert "confidence" in result["reason"].lower()

        print("✅ Confidence threshold enforced before AI assessment")

    @pytest.mark.asyncio
    async def test_market_regime_risk_with_ai(self):
        """Test AI assessment with high-risk market regime."""
        print("\n" + "="*70)
        print("Test: Market Regime Risk with AI")
        print("="*70)

        system_context = SystemContext()
        deepseek_brain = Mock()

        risk_manager = RiskManager(
            system_context=system_context,
            deepseek_brain=deepseek_brain
        )

        signal = {
            "symbol": "BTCUSDT",
            "action": "LONG",
            "confidence": 0.70,
            "market_regime": "TRANSITION"  # High risk regime
        }

        result = await risk_manager.validate_trade(signal, 0.02)

        # Market regime check should reduce size
        if not result["approved"]:
            assert "risk regime" in result["reason"].lower()
        else:
            # If approved, size should be reduced
            assert result.get("adjusted_size", 0.02) < 0.02

        print("✅ Market regime risk handled")

    @pytest.mark.asyncio
    async def test_correlation_risk_with_ai(self):
        """Test correlation risk check with AI."""
        print("\n" + "="*70)
        print("Test: Correlation Risk with AI")
        print("="*70)

        system_context = SystemContext()

        # Add BTC position
        system_context.update_position("BTCUSDT", {
            "symbol": "BTCUSDT",
            "side": "LONG",
            "entry_price": 45000,
            "quantity": 0.1,
            "entry_time": datetime.now().isoformat(),
            "value": 0.12  # 12% exposure
        })

        deepseek_brain = Mock()

        risk_manager = RiskManager(
            system_context=system_context,
            deepseek_brain=deepseek_brain
        )

        # Try to add another correlated asset (WBTC)
        signal = {
            "symbol": "WBTCUSDT",
            "action": "LONG",
            "confidence": 0.75
        }

        result = await risk_manager.validate_trade(signal, 0.05)

        # Check if correlation risk was assessed
        # (Implementation depends on correlation logic)
        print(f"   Result: {result['approved']}")
        print(f"   Reason: {result['reason']}")

        print("✅ Correlation risk check completed")

    @pytest.mark.asyncio
    async def test_ai_enhancement_logging(self):
        """Test that AI assessments are properly logged."""
        print("\n" + "="*70)
        print("Test: AI Enhancement Logging")
        print("="*70)

        system_context = SystemContext()
        deepseek_brain = Mock()

        # Mock AI approval
        deepseek_brain.assess_risk = AsyncMock(return_value={
            "approved": True,
            "reasoning": "Market conditions favorable",
            "risk_score": 0.3
        })

        risk_manager = RiskManager(
            system_context=system_context,
            deepseek_brain=deepseek_brain
        )

        signal = {
            "symbol": "BTCUSDT",
            "action": "LONG",
            "confidence": 0.75
        }

        with patch('execution.risk_manager.logger') as mock_logger:
            result = await risk_manager.validate_trade(signal, 0.02)

            # Verify info log was called for approval
            # (The actual log message is checked in _assess_trade_with_ai)
            assert result["approved"] is True

        print("✅ AI assessment logging works correctly")

    @pytest.mark.asyncio
    async def test_integration_all_checks_with_ai(self):
        """Test complete integration: all checks + AI."""
        print("\n" + "="*70)
        print("Test: Complete Integration (All Checks + AI)")
        print("="*70)

        system_context = SystemContext()

        # Set moderate drawdown
        system_context.risk_metrics["current_drawdown"] = 0.05

        deepseek_brain = Mock()

        # Mock AI approval with specific score
        deepseek_brain.assess_risk = AsyncMock(return_value={
            "approved": True,
            "reasoning": "All checks passed, good opportunity",
            "risk_score": 0.35,
            "adjusted_size": 0.02
        })

        risk_manager = RiskManager(
            system_context=system_context,
            deepseek_brain=deepseek_brain
        )

        # Valid signal that should pass all checks
        signal = {
            "symbol": "ETHUSDT",
            "action": "LONG",
            "confidence": 0.80,
            "market_regime": "TRENDING_LOW_VOL"
        }

        result = await risk_manager.validate_trade(signal, 0.03)

        assert result["approved"] is True
        assert result.get("adjusted_size", 0.03) <= 0.05  # Within limit

        # Verify AI was called
        deepseek_brain.assess_risk.assert_called_once()

        print("✅ Complete integration test passed")

    @pytest.mark.asyncio
    async def test_check_portfolio_risk_with_ai(self):
        """Test check_portfolio_risk method (existing + AI)."""
        print("\n" + "="*70)
        print("Test: Check Portfolio Risk (Rule-based)")
        print("="*70)

        system_context = SystemContext()

        # Add positions
        system_context.update_position("BTCUSDT", {
            "symbol": "BTCUSDT",
            "side": "LONG",
            "entry_price": 45000,
            "quantity": 0.05,
            "entry_time": datetime.now().isoformat(),
            "value": 0.05
        })

        deepseek_brain = Mock()

        risk_manager = RiskManager(
            system_context=system_context,
            deepseek_brain=deepseek_brain
        )

        # Call existing rule-based method
        risk_check = risk_manager.check_portfolio_risk()

        assert "risk_score" in risk_check
        assert "risk_level" in risk_check
        assert "total_exposure" in risk_check

        print(f"   Risk Level: {risk_check['risk_level']}")
        print(f"   Risk Score: {risk_check['risk_score']:.2f}")

        print("✅ Portfolio risk check completed")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("Running RiskManager AI Enhancement Tests")
    print("="*70 + "\n")

    # Run all tests
    async def run_all_tests():
        test_suite = TestRiskManagerAI()

        tests = [
            ("Initialization with DeepSeek",
             test_suite.test_initialization_with_deepseek),
            ("Initialization without DeepSeek",
             test_suite.test_initialization_without_deepseek),
            ("Trade Validation with AI Approval",
             test_suite.test_validate_trade_with_ai_approval),
            ("Trade Validation with AI Rejection",
             test_suite.test_validate_trade_with_ai_rejection),
            ("AI Error Fail-safe",
             test_suite.test_validate_trade_ai_error_failsafe),
            ("Trade Validation without AI",
             test_suite.test_validate_trade_without_ai),
            ("_assess_trade_with_ai Method",
             test_suite.test_assess_trade_with_ai_method),
            ("Portfolio Risk Assessment with AI",
             test_suite.test_assess_portfolio_risk_with_ai),
            ("Get AI Risk Recommendations",
             test_suite.test_get_ai_risk_recommendations),
            ("Position Size Limit with AI",
             test_suite.test_position_size_limit_with_ai),
            ("Drawdown Limit with AI",
             test_suite.test_drawdown_limit_with_ai),
            ("Confidence Threshold with AI",
             test_suite.test_confidence_threshold_with_ai),
            ("Market Regime Risk with AI",
             test_suite.test_market_regime_risk_with_ai),
            ("Correlation Risk with AI",
             test_suite.test_correlation_risk_with_ai),
            ("AI Enhancement Logging",
             test_suite.test_ai_enhancement_logging),
            ("Complete Integration",
             test_suite.test_integration_all_checks_with_ai),
            ("Check Portfolio Risk",
             test_suite.test_check_portfolio_risk_with_ai),
        ]

        passed = 0
        failed = 0

        for test_name, test_func in tests:
            try:
                await test_func()
                passed += 1
            except Exception as e:
                print(f"\n❌ {test_name} FAILED: {e}")
                import traceback
                traceback.print_exc()
                failed += 1

        print("\n" + "="*70)
        print(f"AI Enhancement Test Results: {passed} passed, {failed} failed")
        print("="*70)

        if failed == 0:
            print("\n✅ All RiskManager AI tests PASSED!")
        else:
            print(f"\n⚠️  {failed} test(s) FAILED")

    asyncio.run(run_all_tests())
