"""
Comprehensive DeepSeek Intelligence Test
Test DeepSeek's ability to create strategies, analyze market conditions, and optimize profitability
"""

import asyncio
import json
from deepseek.client import DeepSeekBrain
from core.system_context import SystemContext

async def test_deepseek_intelligence():
    """Test DeepSeek's strategic thinking and analysis capabilities."""
    
    print("\n" + "="*80)
    print("DEEPSEEK COMPREHENSIVE INTELLIGENCE TEST")
    print("="*80 + "\n")
    
    # Initialize
    ctx = SystemContext()
    brain = DeepSeekBrain(ctx)
    
    # Prepare realistic context
    context = {
        "current_positions": [],
        "total_exposure": 0.0,
        "current_drawdown": 0.02,  # 2% drawdown
        "market_regime": "VOLATILE",
        "recent_performance": {
            "win_rate": 0.58,
            "avg_return": 0.015,
            "sharpe_ratio": 1.2
        },
        "available_features": [
            "Supertrend", "Liquidity Zones", "Order Flow", 
            "Chandelier Exit", "RSI", "MACD", "Bollinger Bands"
        ]
    }
    
    # Test Questions (like a trading manager would ask)
    questions = [
        {
            "category": "Strategy Creation",
            "question": "Given the current volatile market regime and 2% drawdown, create a new momentum-based strategy for BTCUSDT. Include specific entry rules, exit rules, and risk management parameters. Make it actionable."
        },
        {
            "category": "Feature Optimization",
            "question": "We have Supertrend, Liquidity Zones, Order Flow, and Chandelier Exit features available. Which combination would work best in the current volatile market? Explain the synergy between them."
        },
        {
            "category": "Risk Assessment", 
            "question": "Our current win rate is 58% with a Sharpe ratio of 1.2. Should we increase position sizing or stay conservative? What's the optimal risk per trade given these metrics?"
        },
        {
            "category": "Market Analysis",
            "question": "BTCUSDT is in a volatile regime. What specific price action patterns should we watch for? Give me 3 concrete signals that would indicate a trend reversal."
        },
        {
            "category": "System Optimization",
            "question": "How can we improve our 58% win rate? Should we tighten entry conditions, adjust exit strategy, or change our feature set? Be specific with metrics."
        },
        {
            "category": "Strategy Comparison",
            "question": "Compare mean reversion vs momentum strategies for current market conditions. Which has better risk-adjusted returns in volatileregimes and why?"
        }
    ]
    
    results = []
    
    for i, test in enumerate(questions, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}/{len(questions)}: {test['category']}")
        print(f"{'='*80}")
        print(f"\nQUESTION:\n{test['question']}\n")
        print("Waiting for DeepSeek response...\n")
        
        try:
            # Call DeepSeek
            response = await brain.chat_interface(
                user_message=test['question'],
                context=context,
                message_type="strategy"
            )
            
            print(f"RESPONSE:\n{response}\n")
            
            # Analyze response quality
            word_count = len(response.split())
            has_specific_numbers = any(char.isdigit() for char in response)
            has_actionable_items = any(word in response.lower() for word in ['when', 'if', 'entry', 'exit', 'stop', 'target'])
            
            quality_score = {
                "length": "✅ Detailed" if word_count > 100 else "⚠️ Brief" if word_count > 50 else "❌ Too Short",
                "specificity": "✅ Has numbers/metrics" if has_specific_numbers else "❌ Too vague",
                "actionability": "✅ Actionable" if has_actionable_items else "❌ Not actionable"
            }
            
            results.append({
                "category": test['category'],
                "question": test['question'][:100] + "...",
                "response_length": word_count,
                "quality": quality_score,
                "response": response
            })
            
            print(f"\n📊 QUALITY ASSESSMENT:")
            for metric, score in quality_score.items():
                print(f"   {metric.capitalize()}: {score}")
                
        except Exception as e:
            print(f"❌ ERROR: {e}\n")
            results.append({
                "category": test['category'],
                "error": str(e)
            })
        
        # Small delay between questions
        await asyncio.sleep(2)
    
    # Final Summary
    print(f"\n\n{'='*80}")
    print("FINAL EVALUATION SUMMARY")
    print(f"{'='*80}\n")
    
    successful_tests = [r for r in results if 'response' in r]
    failed_tests = [r for r in results if 'error' in r]
    
    print(f"✅ Successful responses: {len(successful_tests)}/{len(questions)}")
    print(f"❌ Failed responses: {len(failed_tests)}/{len(questions)}\n")
    
    if successful_tests:
        avg_length = sum(r['response_length'] for r in successful_tests) / len(successful_tests)
        print(f"📝 Average response length: {avg_length:.0f} words\n")
        
        print("CATEGORY PERFORMANCE:")
        for result in successful_tests:
            print(f"\n  {result['category']}:")
            for metric, score in result['quality'].items():
                print(f"    - {metric}: {score}")
    
    # Overall Intelligence Rating
    if len(successful_tests) >= 5:
        if avg_length > 150 and all('✅' in str(r['quality']) for r in successful_tests):
            rating = "🌟🌟🌟🌟🌟 EXCELLENT - DeepSeek shows strong strategic intelligence"
        elif avg_length > 100:
            rating = "🌟🌟🌟🌟 GOOD - DeepSeek provides useful insights"
        else:
            rating = "🌟🌟🌟 ADEQUATE - DeepSeek responds but needs improvement"
    else:
        rating = "🌟🌟 POOR - Too many failures or connection issues"
    
    print(f"\n\n{'='*80}")
    print(f"OVERALL INTELLIGENCE RATING: {rating}")
    print(f"{'='*80}\n")
    
    # Save detailed results
    with open('deepseek_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("📁 Detailed results saved to: deepseek_test_results.json\n")

if __name__ == "__main__":
    asyncio.run(test_deepseek_intelligence())
