"""
Unit tests for signal database operations.
Tests the DatabaseManager class and signal persistence functionality.
"""

import sys
sys.path.insert(0, '/Users/mrsmoothy/Downloads/Trading_bot')

import pytest
import asyncio
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from sqlalchemy import text

from ops.db import DatabaseManager, SignalRecord, get_database, close_database


class TestDatabaseManager:
    """Test DatabaseManager functionality."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        # Cleanup
        if os.path.exists(db_path):
            os.remove(db_path)

    @pytest.mark.asyncio
    async def test_database_initialization(self, temp_db_path):
        """Test database initialization and table creation."""
        db = DatabaseManager(db_path=temp_db_path)

        # Verify engine was created
        assert db.engine is not None
        assert 'sqlite' in str(db.engine.url)

        # Verify tables were created
        with db.get_session() as session:
            # Check if signals table exists by querying it
            result = session.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='signals'"))
            assert result.fetchone() is not None

        print("✅ test_database_initialization passed")

    @pytest.mark.asyncio
    async def test_save_and_retrieve_signal(self, temp_db_path):
        """Test saving and retrieving a signal."""
        db = DatabaseManager(db_path=temp_db_path)

        # Create a test signal
        test_signal = {
            'symbol': 'BTCUSDT',
            'action': 'LONG',
            'confidence': 0.75,
            'position_size': 0.02,
            'reasoning': 'Strong bullish signal',
            'deepseek_confidence': 0.80,
            'feature_confidence': 0.70,
            'current_price': 45000.0,
            'timestamp': datetime.now().isoformat(),
            'market_regime': 'TRENDING_LOW_VOL',
            'risk_adjustment': {'volatility_adj': 1.0},
            'volatility_percentile': 0.45,
            'trend_strength': 0.65,
            'entry_conditions': {'preferred_entry': 44900.0},
            'exit_strategy': {'stop_loss': 44000.0},
            'feature_highlights': ['Near liquidity zone', 'Strong trend'],
            'is_tradeable': True,
            'signal_strength': 'MEDIUM'
        }

        # Save signal
        signal_id = await db.save_signal(test_signal)
        assert signal_id is not None
        assert signal_id > 0

        # Retrieve signal
        retrieved_signal = await db.get_signal(signal_id)
        assert retrieved_signal is not None
        assert retrieved_signal['symbol'] == 'BTCUSDT'
        assert retrieved_signal['action'] == 'LONG'
        assert retrieved_signal['confidence'] == 0.75

        print("✅ test_save_and_retrieve_signal passed")

    @pytest.mark.asyncio
    async def test_get_signals_with_filters(self, temp_db_path):
        """Test retrieving signals with various filters."""
        db = DatabaseManager(db_path=temp_db_path)

        # Save multiple test signals
        signals = []
        for i in range(5):
            signal = {
                'symbol': f'SYMBOL{i}',
                'action': 'LONG' if i % 2 == 0 else 'SHORT',
                'confidence': 0.5 + (i * 0.1),
                'position_size': 0.02,
                'reasoning': f'Test signal {i}',
                'deepseek_confidence': 0.6,
                'feature_confidence': 0.6,
                'current_price': 45000.0,
                'timestamp': datetime.now().isoformat(),
            }
            signal_id = await db.save_signal(signal)
            signals.append(signal_id)

        # Test filtering by symbol
        results = await db.get_signals(symbol='SYMBOL0', limit=10)
        assert len(results) >= 1

        # Test filtering by confidence
        results = await db.get_signals(min_confidence=0.7, limit=10)
        assert len(results) >= 1
        for signal in results:
            assert signal['confidence'] >= 0.7

        # Test limit
        results = await db.get_signals(limit=2)
        assert len(results) <= 2

        print("✅ test_get_signals_with_filters passed")

    @pytest.mark.asyncio
    async def test_get_latest_signal(self, temp_db_path):
        """Test getting the latest signal for a symbol."""
        db = DatabaseManager(db_path=temp_db_path)

        # Save multiple signals for same symbol
        for i in range(3):
            signal = {
                'symbol': 'BTCUSDT',
                'action': 'LONG',
                'confidence': 0.7,
                'position_size': 0.02,
                'reasoning': f'Signal {i}',
                'deepseek_confidence': 0.7,
                'feature_confidence': 0.7,
                'current_price': 45000.0 + i * 100,
                'timestamp': (datetime.now() + timedelta(seconds=i)).isoformat(),
            }
            await db.save_signal(signal)

        # Get latest signal
        latest = await db.get_latest_signal('BTCUSDT')
        assert latest is not None
        assert latest['symbol'] == 'BTCUSDT'

        print("✅ test_get_latest_signal passed")

    @pytest.mark.asyncio
    async def test_signal_statistics(self, temp_db_path):
        """Test signal statistics calculation."""
        db = DatabaseManager(db_path=temp_db_path)

        # Save test signals
        for i in range(5):
            signal = {
                'symbol': 'BTCUSDT',
                'action': 'LONG',
                'confidence': 0.5 + (i * 0.1),
                'position_size': 0.02,
                'reasoning': f'Test {i}',
                'deepseek_confidence': 0.6,
                'feature_confidence': 0.6,
                'current_price': 45000.0,
                'timestamp': datetime.now().isoformat(),
                'is_tradeable': i % 2 == 0,
                'is_executed': i < 2,
            }
            await db.save_signal(signal)

        # Get statistics
        stats = await db.get_signal_statistics(days=30)
        assert 'total_signals' in stats
        assert stats['total_signals'] == 5
        assert 'avg_confidence' in stats
        assert 'tradeable_count' in stats
        assert 'executed_count' in stats

        print("✅ test_signal_statistics passed")

    @pytest.mark.asyncio
    async def test_update_signal_execution(self, temp_db_path):
        """Test updating signal execution status."""
        db = DatabaseManager(db_path=temp_db_path)

        # Save a signal
        signal = {
            'symbol': 'BTCUSDT',
            'action': 'LONG',
            'confidence': 0.75,
            'position_size': 0.02,
            'reasoning': 'Test',
            'deepseek_confidence': 0.75,
            'feature_confidence': 0.75,
            'current_price': 45000.0,
            'timestamp': datetime.now().isoformat(),
            'is_executed': False,
        }
        signal_id = await db.save_signal(signal)

        # Update execution status
        success = await db.update_signal_execution(signal_id, True)
        assert success is True

        # Verify update
        retrieved = await db.get_signal(signal_id)
        assert retrieved['is_executed'] is True

        print("✅ test_update_signal_execution passed")

    @pytest.mark.asyncio
    async def test_date_filtering(self, temp_db_path):
        """Test filtering signals by date range."""
        db = DatabaseManager(temp_db_path)

        # Save signals with different timestamps
        now = datetime.now()
        for i in range(3):
            signal = {
                'symbol': 'BTCUSDT',
                'action': 'LONG',
                'confidence': 0.7,
                'position_size': 0.02,
                'reasoning': f'Test {i}',
                'deepseek_confidence': 0.7,
                'feature_confidence': 0.7,
                'current_price': 45000.0,
                'timestamp': (now + timedelta(days=i)).isoformat(),
            }
            await db.save_signal(signal)

        # Filter by date range
        start_date = now + timedelta(days=0)
        end_date = now + timedelta(days=1)
        results = await db.get_signals(
            start_date=start_date,
            end_date=end_date
        )
        assert len(results) >= 1

        print("✅ test_date_filtering passed")

    @pytest.mark.asyncio
    async def test_get_signal_statistics_date_range(self, temp_db_path):
        """Test statistics with date filtering."""
        db = DatabaseManager(temp_db_path)

        # Save signal from 45 days ago (should be excluded)
        old_signal = {
            'symbol': 'BTCUSDT',
            'action': 'LONG',
            'confidence': 0.7,
            'position_size': 0.02,
            'reasoning': 'Old signal',
            'deepseek_confidence': 0.7,
            'feature_confidence': 0.7,
            'current_price': 45000.0,
            'timestamp': (datetime.now() - timedelta(days=45)).isoformat(),
        }
        await db.save_signal(old_signal)

        # Save recent signal
        recent_signal = {
            'symbol': 'BTCUSDT',
            'action': 'LONG',
            'confidence': 0.7,
            'position_size': 0.02,
            'reasoning': 'Recent signal',
            'deepseek_confidence': 0.7,
            'feature_confidence': 0.7,
            'current_price': 45000.0,
            'timestamp': datetime.now().isoformat(),
        }
        await db.save_signal(recent_signal)

        # Get statistics for 30 days (should only include recent signal)
        stats = await db.get_signal_statistics(days=30)
        assert stats['total_signals'] == 1
        assert 'Recent signal' in stats.get('reasoning_breakdown', {})

        print("✅ test_get_signal_statistics_date_range passed")

    @pytest.mark.asyncio
    async def test_delete_old_signals(self, temp_db_path):
        """Test deleting old signals."""
        db = DatabaseManager(temp_db_path)

        # Save old signals
        for i in range(5):
            signal = {
                'symbol': 'BTCUSDT',
                'action': 'LONG',
                'confidence': 0.7,
                'position_size': 0.02,
                'reasoning': f'Old signal {i}',
                'deepseek_confidence': 0.7,
                'feature_confidence': 0.7,
                'current_price': 45000.0,
                'timestamp': (datetime.now() - timedelta(days=100)).isoformat(),
            }
            await db.save_signal(signal)

        # Delete old signals (keep 90 days)
        deleted_count = await db.delete_old_signals(days_to_keep=90)
        assert deleted_count == 5

        print("✅ test_delete_old_signals passed")

    @pytest.mark.asyncio
    async def test_global_database_instance(self):
        """Test global database instance management."""
        # Get database instance
        db1 = get_database()
        assert db1 is not None

        # Get again - should be same instance
        db2 = get_database()
        assert db1 is db2

        # Close database
        await close_database()

        # Should get new instance after close
        db3 = get_database()
        assert db3 is not db1

        # Cleanup
        await close_database()

        print("✅ test_global_database_instance passed")


class TestSignalRecord:
    """Test SignalRecord model."""

    @pytest.mark.asyncio
    async def test_signal_record_to_dict(self):
        """Test conversion of SignalRecord to dictionary."""
        record = SignalRecord(
            symbol='BTCUSDT',
            action='LONG',
            confidence=0.75,
            position_size=0.02,
            reasoning='Test signal',
            deepseek_confidence=0.80,
            feature_confidence=0.70,
            current_price=45000.0,
            timestamp=datetime.now(),
            market_regime='TRENDING',
            risk_adjustment='{"vol_adj": 1.0}',
            volatility_percentile=0.45,
            trend_strength=0.65,
            entry_conditions='{"entry": 44900.0}',
            exit_strategy='{"stop": 44000.0}',
            feature_highlights='["Near zone", "Strong trend"]',
            is_tradeable=True,
            is_executed=False,
            signal_strength='MEDIUM',
            timeframe='1h'
        )

        result = record.to_dict()

        assert result['symbol'] == 'BTCUSDT'
        assert result['action'] == 'LONG'
        assert result['confidence'] == 0.75
        assert result['market_regime'] == 'TRENDING'
        assert result['is_tradeable'] is True
        assert result['is_executed'] is False
        assert isinstance(result['timestamp'], str)
        assert isinstance(result['risk_adjustment'], dict)

        print("✅ test_signal_record_to_dict passed")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("Running Signal Database Tests")
    print("="*70 + "\n")

    # Test DatabaseManager
    import tempfile
    import os

    async def run_tests():
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            # Create an instance and run tests
            db = DatabaseManager(db_path=db_path)

            # Test basic operations
            signal = {
                'symbol': 'BTCUSDT',
                'action': 'LONG',
                'confidence': 0.75,
                'position_size': 0.02,
                'reasoning': 'Test signal',
                'deepseek_confidence': 0.80,
                'feature_confidence': 0.70,
                'current_price': 45000.0,
                'timestamp': datetime.now().isoformat(),
            }

            signal_id = await db.save_signal(signal)
            print(f"✅ Saved signal with ID: {signal_id}")

            retrieved = await db.get_signal(signal_id)
            print(f"✅ Retrieved signal: {retrieved['symbol']} {retrieved['action']}")

            stats = await db.get_signal_statistics(days=30)
            print(f"✅ Signal statistics: {stats['total_signals']} total signals")

        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    asyncio.run(run_tests())

    print("\n" + "="*70)
    print("✅ All signal database tests completed successfully!")
    print("="*70)
