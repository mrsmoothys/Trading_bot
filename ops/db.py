"""
Database operations for the trading bot.
Handles signal persistence and retrieval using SQLite with SQLAlchemy.
"""

import json
import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Text, DateTime, Boolean, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from loguru import logger

# Database base class
Base = declarative_base()


class SignalRecord(Base):
    """Signal database record."""
    __tablename__ = 'signals'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Core signal data
    symbol = Column(String(20), nullable=False, index=True)
    action = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    position_size = Column(Float, nullable=False)

    # Reasoning and analysis
    reasoning = Column(Text, nullable=True)
    deepseek_confidence = Column(Float, nullable=False)
    feature_confidence = Column(Float, nullable=False)

    # Price and timing
    current_price = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.now)

    # Market context
    market_regime = Column(String(100), nullable=True)
    risk_adjustment = Column(Text, nullable=True)  # JSON string
    volatility_percentile = Column(Float, nullable=True)
    trend_strength = Column(Float, nullable=True)

    # Entry/Exit conditions
    entry_conditions = Column(Text, nullable=True)  # JSON string
    exit_strategy = Column(Text, nullable=True)  # JSON string

    # Feature highlights
    feature_highlights = Column(Text, nullable=True)  # JSON string

    # Signal metadata
    is_tradeable = Column(Boolean, default=False, index=True)
    is_executed = Column(Boolean, default=False, index=True)
    signal_strength = Column(String(20), nullable=True)

    # Additional context
    timeframe = Column(String(10), nullable=True, index=True)

    # Indexes for common queries
    __table_args__ = (
        Index('idx_signal_search', 'symbol', 'timestamp'),
        Index('idx_confidence', 'confidence'),
        Index('idx_tradeable', 'is_tradeable', 'is_executed'),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert database record to dictionary."""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'action': self.action,
            'confidence': self.confidence,
            'position_size': self.position_size,
            'reasoning': self.reasoning,
            'deepseek_confidence': self.deepseek_confidence,
            'feature_confidence': self.feature_confidence,
            'current_price': self.current_price,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'market_regime': self.market_regime,
            'risk_adjustment': json.loads(self.risk_adjustment) if self.risk_adjustment else None,
            'volatility_percentile': self.volatility_percentile,
            'trend_strength': self.trend_strength,
            'entry_conditions': json.loads(self.entry_conditions) if self.entry_conditions else None,
            'exit_strategy': json.loads(self.exit_strategy) if self.exit_strategy else None,
            'feature_highlights': json.loads(self.feature_highlights) if self.feature_highlights else None,
            'is_tradeable': self.is_tradeable,
            'is_executed': self.is_executed,
            'signal_strength': self.signal_strength,
            'timeframe': self.timeframe,
        }


class DatabaseManager:
    """
    Database manager for signal persistence.
    Handles SQLite database with connection pooling.
    """

    def __init__(self, db_path: str = "data/trading_signals.db"):
        """
        Initialize database manager.

        Args:
            db_path: Path to SQLite database file
        """
        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Create engine with optimizations for SQLite
        self.engine = create_engine(
            f'sqlite:///{db_path}',
            poolclass=StaticPool,
            connect_args={
                "check_same_thread": False,
                "timeout": 30
            },
            echo=False,  # Set to True for SQL debugging
            pool_pre_ping=True,
            pool_recycle=3600,
        )

        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )

        # Create tables
        self.create_tables()

        logger.info(f"Database initialized: {db_path}")

    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created/verified")

    @contextmanager
    def get_session(self) -> Session:
        """
        Get database session with automatic cleanup.

        Yields:
            SQLAlchemy session
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

    async def save_signal(self, signal: Dict[str, Any]) -> int:
        """
        Save signal to database.

        Args:
            signal: Signal dictionary to save

        Returns:
            ID of saved signal
        """
        try:
            # Extract fields from signal
            record = SignalRecord(
                symbol=signal.get('symbol', ''),
                action=signal.get('action', 'HOLD'),
                confidence=float(signal.get('confidence', 0.0)),
                position_size=float(signal.get('position_size', 0.0)),
                reasoning=signal.get('reasoning', ''),
                deepseek_confidence=float(signal.get('deepseek_confidence', 0.0)),
                feature_confidence=float(signal.get('feature_confidence', 0.0)),
                current_price=float(signal.get('current_price', 0.0)),
                timestamp=datetime.fromisoformat(signal.get('timestamp', datetime.now().isoformat())),
                market_regime=signal.get('market_regime'),
                risk_adjustment=json.dumps(signal.get('risk_adjustment', {})),
                volatility_percentile=signal.get('volatility_percentile'),
                trend_strength=signal.get('trend_strength'),
                entry_conditions=json.dumps(signal.get('entry_conditions', {})),
                exit_strategy=json.dumps(signal.get('exit_strategy', {})),
                feature_highlights=json.dumps(signal.get('feature_highlights', [])),
                is_tradeable=signal.get('is_tradeable', False),
                is_executed=signal.get('is_executed', False),
                signal_strength=signal.get('signal_strength'),
                timeframe=signal.get('timeframe'),
            )

            # Save to database
            with self.get_session() as session:
                session.add(record)
                session.flush()  # Get the ID without committing
                signal_id = record.id

            logger.info(f"Signal saved to database (ID: {signal_id}, {signal['symbol']} {signal['action']})")
            return signal_id

        except Exception as e:
            logger.error(f"Error saving signal to database: {e}")
            raise

    async def get_signal(self, signal_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve signal by ID.

        Args:
            signal_id: Signal ID

        Returns:
            Signal dictionary or None
        """
        try:
            with self.get_session() as session:
                record = session.query(SignalRecord).filter(SignalRecord.id == signal_id).first()
                if record:
                    return record.to_dict()
                return None
        except Exception as e:
            logger.error(f"Error retrieving signal {signal_id}: {e}")
            return None

    async def get_signals(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        min_confidence: Optional[float] = None,
        only_tradeable: bool = False,
        only_executed: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve signals with filters.

        Args:
            symbol: Filter by symbol
            limit: Maximum number of records
            offset: Offset for pagination
            start_date: Start date filter
            end_date: End date filter
            min_confidence: Minimum confidence filter
            only_tradeable: Only tradeable signals
            only_executed: Only executed signals

        Returns:
            List of signal dictionaries
        """
        try:
            query = self.SessionLocal().query(SignalRecord)

            # Apply filters
            if symbol:
                query = query.filter(SignalRecord.symbol == symbol)

            if start_date:
                query = query.filter(SignalRecord.timestamp >= start_date)

            if end_date:
                query = query.filter(SignalRecord.timestamp <= end_date)

            if min_confidence:
                query = query.filter(SignalRecord.confidence >= min_confidence)

            if only_tradeable:
                query = query.filter(SignalRecord.is_tradeable == True)

            if only_executed:
                query = query.filter(SignalRecord.is_executed == True)

            # Apply ordering and pagination
            query = query.order_by(SignalRecord.timestamp.desc()).offset(offset).limit(limit)

            # Execute query
            with self.get_session() as session:
                records = query.all()
                return [record.to_dict() for record in records]

        except Exception as e:
            logger.error(f"Error retrieving signals: {e}")
            return []

    async def get_latest_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get latest signal for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Latest signal or None
        """
        try:
            with self.get_session() as session:
                record = (
                    session.query(SignalRecord)
                    .filter(SignalRecord.symbol == symbol)
                    .order_by(SignalRecord.timestamp.desc())
                    .first()
                )
                if record:
                    return record.to_dict()
                return None
        except Exception as e:
            logger.error(f"Error retrieving latest signal for {symbol}: {e}")
            return None

    async def get_signal_statistics(
        self,
        symbol: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get signal statistics.

        Args:
            symbol: Filter by symbol (None for all)
            days: Number of days to analyze

        Returns:
            Statistics dictionary
        """
        try:
            start_date = datetime.now() - timedelta(days=days)

            with self.get_session() as session:
                query = session.query(SignalRecord).filter(
                    SignalRecord.timestamp >= start_date
                )

                if symbol:
                    query = query.filter(SignalRecord.symbol == symbol)

                records = query.all()

                if not records:
                    return {
                        'total_signals': 0,
                        'avg_confidence': 0.0,
                        'tradeable_count': 0,
                        'executed_count': 0
                    }

                # Calculate statistics
                total_signals = len(records)
                avg_confidence = sum(r.confidence for r in records) / total_signals
                tradeable_count = sum(1 for r in records if r.is_tradeable)
                executed_count = sum(1 for r in records if r.is_executed)

                # Action breakdown
                action_breakdown = {}
                for record in records:
                    action = record.action
                    action_breakdown[action] = action_breakdown.get(action, 0) + 1

                # Reasoning breakdown
                reasoning_breakdown = {}
                for record in records:
                    reason = record.reasoning or 'Unknown'
                    reasoning_breakdown[reason] = reasoning_breakdown.get(reason, 0) + 1

                return {
                    'total_signals': total_signals,
                    'avg_confidence': round(avg_confidence, 3),
                    'tradeable_count': tradeable_count,
                    'executed_count': executed_count,
                    'tradeable_percentage': round((tradeable_count / total_signals) * 100, 2),
                    'executed_percentage': round((executed_count / total_signals) * 100, 2),
                    'action_breakdown': action_breakdown,
                    'reasoning_breakdown': reasoning_breakdown,
                    'date_range_days': days
                }

        except Exception as e:
            logger.error(f"Error calculating signal statistics: {e}")
            return {}

    async def update_signal_execution(
        self,
        signal_id: int,
        is_executed: bool
    ) -> bool:
        """
        Update signal execution status.

        Args:
            signal_id: Signal ID
            is_executed: Whether signal was executed

        Returns:
            Success status
        """
        try:
            with self.get_session() as session:
                record = session.query(SignalRecord).filter(
                    SignalRecord.id == signal_id
                ).first()

                if record:
                    record.is_executed = is_executed
                    logger.info(f"Updated signal {signal_id} execution status: {is_executed}")
                    return True
                return False

        except Exception as e:
            logger.error(f"Error updating signal execution status: {e}")
            return False

    async def delete_old_signals(self, days_to_keep: int = 90) -> int:
        """
        Delete signals older than specified days.

        Args:
            days_to_keep: Number of days to keep

        Returns:
            Number of deleted records
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)

            with self.get_session() as session:
                deleted_count = session.query(SignalRecord).filter(
                    SignalRecord.timestamp < cutoff_date
                ).delete(synchronize_session=False)

            logger.info(f"Deleted {deleted_count} signals older than {days_to_keep} days")
            return deleted_count

        except Exception as e:
            logger.error(f"Error deleting old signals: {e}")
            return 0

    async def close(self):
        """Close database connections."""
        self.engine.dispose()
        logger.info("Database connections closed")


# Global database manager instance
_db_manager = None


def get_database() -> DatabaseManager:
    """
    Get global database manager instance.

    Returns:
        DatabaseManager instance
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


async def close_database():
    """Close global database manager."""
    global _db_manager
    if _db_manager:
        await _db_manager.close()
        _db_manager = None
