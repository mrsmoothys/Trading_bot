# Trading Bot Quick Start Guide

## Table of Contents
1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Running the Bot](#running-the-bot)
4. [Using the Dashboard](#using-the-dashboard)
5. [Using the Chat Interface](#using-the-chat-interface)
6. [Backup & Restore](#backup--restore)
7. [Testing](#testing)
8. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)
- Git (for cloning the repository)

### Quick Install

```bash
# Clone the repository
git clone <repository-url>
cd Trading_bot

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
make install-dev
```

### System Requirements
- **Memory**: 4GB+ RAM (optimized for M1 MacBook)
- **Disk**: 2GB+ free space
- **OS**: macOS, Linux, or Windows

---

## Configuration

### 1. Environment Variables

Create a `.env` file in the root directory:

```bash
# Core Configuration
TESTNET=true
LOG_LEVEL=INFO

# Binance API (Testnet)
BINANCE_API_KEY=your_testnet_api_key
BINANCE_SECRET_KEY=your_testnet_secret_key
BINANCE_TESTNET_URL=https://testnet.binancefuture.com

# DeepSeek AI
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_API_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_MAX_TOKENS=32000
DEEPSEEK_TEMPERATURE=0.7

# Database
DATABASE_URL=sqlite:///./trading.db

# Redis (optional)
REDIS_URL=redis://localhost:6379/0
```

### 2. Configuration Files

Create `config.yaml`:

```yaml
trading:
  symbol: BTCUSDT
  timeframe: 5m
  risk_level: medium
  
cache:
  ttl: 3600  # seconds
  max_size_mb: 100
  
performance:
  check_interval: 60  # seconds
  alert_thresholds:
    memory_mb: 3500
    cpu_percent: 80
```

---

## Running the Bot

### Method 1: Using Makefile (Recommended)

```bash
# Run all tests
make test

# Start dashboard only
make run-dashboard

# Start chat interface only
make run-chat

# Create backup
make backup
```

### Method 2: Direct Python Execution

```bash
# Start dashboard (runs on http://127.0.0.1:8050)
python -m ui.dashboard

# Start chat interface (runs on http://127.0.0.1:8051)
python -m ui.chat_interface
```

### Method 3: Python Script

```python
import asyncio
from core.memory_manager import M1MemoryManager
from ops.cache_manager import CacheManager

async def main():
    # Initialize memory manager
    memory_manager = M1MemoryManager()
    
    # Initialize cache manager
    cache_manager = CacheManager()
    
    # Run maintenance
    await cache_manager.run_full_maintenance()
    
    print("Trading bot is running!")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Using the Dashboard

Access the dashboard at: **http://127.0.0.1:8050**

### Features

1. **Real-time Charts**
   - Multiple timeframes (1m, 5m, 15m, 1h, 4h, 1d)
   - Candlestick charts with volume
   - Technical indicators
   - Support/resistance levels

2. **Position Monitoring**
   - Current positions
   - P&L tracking
   - Risk metrics

3. **Performance Metrics**
   - Win rate
   - Sharpe ratio
   - Maximum drawdown
   - Profit factor

4. **System Health**
   - Memory usage
   - CPU usage
   - API latency
   - Error rates

### Dashboard Controls

- **Timeframe Buttons**: Switch between different timeframes
- **Refresh**: Update data manually (or auto-refresh every 30s)
- **Export**: Download performance data as JSON

### Keyboard Shortcuts

- `Ctrl+R`: Refresh dashboard
- `Ctrl+1`: 1 minute timeframe
- `Ctrl+2`: 5 minute timeframe
- `Ctrl+3`: 15 minute timeframe
- `Ctrl+4`: 1 hour timeframe
- `Ctrl+5`: 4 hour timeframe
- `Ctrl+6`: 1 day timeframe

---

## Using the Chat Interface

Access the chat interface at: **http://127.0.0.1:8051**

### Features

1. **AI-Powered Assistant**
   - Real DeepSeek AI integration
   - Context-aware responses
   - System state awareness

2. **Trading Queries**
   - "What's the current market outlook?"
   - "Show me today's performance"
   - "Analyze the risk metrics"

3. **System Commands**
   - System health queries
   - Configuration help
   - Performance reports

4. **Quick Actions**
   - Predefined queries
   - One-click actions
   - System status checks

### Chat Examples

```
User: What are the current positions?
AI: Based on the current state, you have 2 active positions:
    - BTCUSDT Long: +2.5% P&L
    - ETHUSDT Short: -0.8% P&L
    Total portfolio P&L: +1.7%

User: How is the system performing?
AI: System health is GOOD
    - Memory: 2.1GB / 4GB (52%)
    - CPU: 15%
    - API Latency: 120ms
    - All services operational

User: Create a backup
AI: ✅ Backup created successfully!
    Backup: backup_20251114_203045 (2.3MB)
    Location: /backups/backup_20251114_203045.tar.gz
```

---

## Backup & Restore

### Automatic Backups

The system automatically creates backups:
- **Schedule**: Daily at 04:00 AM
- **Retention**: 30 days
- **Location**: `backups/` directory
- **Format**: Compressed (.tar.gz)

### Manual Backup

```bash
# Using Makefile
make backup

# Using Python
python -c "from ops.cache_manager import CacheManager; import asyncio; cm = CacheManager(); asyncio.run(cm.create_backup(compress=True))"
```

### List Available Backups

```bash
make restore
```

### Restore from Backup

```bash
# List backups first
make restore

# Restore specific backup
python -c "from ops.cache_manager import CacheManager; import asyncio; cm = CacheManager(); asyncio.run(cm.restore_backup('backup_20251114_030000'))"
```

### Backup Contents

Backups include:
- Cache data (`cache/` directory)
- Database files (`.db` files)
- Configuration files (`.env`, `config.yaml`, `config.json`)
- Metadata (timestamp, size, retention settings)

### Backup Location

- **Local**: `backups/` directory
- **Format**: `backup_YYYYMMDD_HHMMSS.tar.gz`
- **Metadata**: `backup_YYYYMMDD_HHMMSS/metadata.json`

---

## Testing

### Run All Tests

```bash
make test
```

### Run Specific Test Types

```bash
# Unit tests only
make test-unit

# Integration tests only
make test-integration

# End-to-end tests only
make test-e2e

# Fast tests (skip slow tests)
make test-fast
```

### Test Coverage

```bash
# Generate coverage report
make coverage

# View HTML coverage report
open htmlcov/index.html
```

### Individual Test Files

```bash
# Test specific module
pytest tests/test_memory_manager_integration.py -v

# Test with coverage
pytest tests/test_order_lifecycle_integration.py --cov=execution --cov-report=html
```

### Test Categories

1. **Unit Tests** (`test_deepseek_client.py`, `test_signal_database.py`, `test_features.py`)
   - Core component testing
   - Fast execution
   - No external dependencies

2. **Integration Tests** (7 test files)
   - Component interaction testing
   - Database integration
   - API integration
   - M1 memory optimization

3. **End-to-End Tests** (`test_signal_flow_e2e.py`, `test_binance_websocket.py`)
   - Full workflow testing
   - Real-time data testing
   - System integration

---

## Development Workflow

### Code Quality

```bash
# Check code style
make lint

# Format code
make format

# Clean up cache files
make clean
```

### Running in Development Mode

```bash
# Install development dependencies
make install-dev

# Run tests with verbose output
make test

# Run specific test
pytest tests/test_cache_manager.py -v -s
```

### Monitoring System Health

```python
from ops.performance_monitor import PerformanceMonitor
from core.memory_manager import M1MemoryManager

# Initialize
memory_manager = M1MemoryManager()
monitor = PerformanceMonitor(memory_manager=memory_manager)

# Check health
health = await monitor.check_system_health()
print(f"System health: {health['status']}")

# Generate report
report = await monitor.generate_performance_report()
print(f"Recommendations: {report['recommendations']}")
```

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory (M1 MacBook)

**Symptoms**:
- System slow
- Python process using >3.5GB RAM
- Application crashes

**Solutions**:
```python
from core.memory_manager import M1MemoryManager

# Initialize with aggressive cleanup
memory_manager = M1MemoryManager()

# Monitor memory
usage = memory_manager.get_current_memory_usage()
if usage['percent'] > 85:
    memory_manager.cleanup_cache()
```

#### 2. Test Failures

**Solutions**:
```bash
# Clean test cache
make clean

# Run specific test
pytest tests/test_cache_manager.py -v

# Check test logs
pytest tests/ -v --tb=long
```

#### 3. Dashboard Won't Start

**Check**:
```bash
# Check port availability
lsof -i :8050

# Check logs
tail -f logs/dashboard.log

# Run with debug
python -m ui.dashboard --debug
```

#### 4. Chat Interface Issues

**Check**:
```bash
# Verify DeepSeek API key
echo $DEEPSEEK_API_KEY

# Check logs
tail -f logs/chat.log

# Test in demo mode
python -m ui.chat_interface --demo
```

#### 5. Backup/Restore Issues

**Check**:
```python
from ops.cache_manager import CacheManager

cm = CacheManager()

# Check backup directory
import os
print(os.listdir(cm.backup_dir))

# Test backup creation
result = await cm.create_backup(compress=True)
print(result)
```

### Log Files

Log files are stored in `logs/`:
- `dashboard.log` - Dashboard application logs
- `chat.log` - Chat interface logs
- `performance.log` - Performance monitoring logs
- `cache.log` - Cache manager logs
- `alerts_YYYYMMDD.json` - Alert logs

### Performance Monitoring

Access real-time metrics:
```bash
# Start performance monitor
python -m ops.performance_monitor

# View logs
tail -f logs/performance.log
```

### Getting Help

1. Check logs for errors
2. Run tests to verify functionality
3. Check system health metrics
4. Review configuration files
5. Create issue on GitHub repository

---

## Best Practices

### 1. Memory Management (M1 MacBook)
- Keep cache size < 100MB
- Monitor memory usage regularly
- Use CacheManager cleanup features
- Restart application if memory > 3.5GB

### 2. Configuration
- Always use TESTNET=true for testing
- Set appropriate LOG_LEVEL (INFO for production, DEBUG for development)
- Configure alert thresholds based on system capacity
- Use environment variables for sensitive data

### 3. Backups
- Create backups before major changes
- Test restore procedures regularly
- Keep multiple backup versions
- Monitor backup disk space

### 4. Testing
- Run all tests before deployment
- Use test coverage to identify gaps
- Test on actual hardware (M1 MacBook)
- Test with real API keys in testnet

### 5. Monitoring
- Check system health regularly
- Monitor API latency
- Track error rates
- Review performance recommendations

---

## System Architecture

```
Trading Bot
├── Core
│   ├── memory_manager.py      (M1 optimization)
│   └── system_context.py      (State management)
├── Data
│   ├── datastore.py           (Data caching)
│   └── signal_database.py     (Signal storage)
├── Execution
│   ├── position_manager.py    (Position tracking)
│   ├── order_manager.py       (Order execution)
│   └── risk_manager.py        (Risk assessment)
├── Ops
│   ├── performance_monitor.py (System monitoring)
│   └── cache_manager.py       (Cleanup & backups)
├── UI
│   ├── dashboard.py           (Web dashboard)
│   ├── chat_interface.py      (AI chat)
│   └── chat_audit_logger.py   (Audit trail)
└── Tests
    ├── Unit tests
    ├── Integration tests
    └── End-to-end tests
```

---

## API Reference

### Core Components

**M1MemoryManager**
```python
from core.memory_manager import M1MemoryManager

manager = M1MemoryManager()
usage = manager.get_current_memory_usage()
manager.cleanup_cache()
```

**CacheManager**
```python
from ops.cache_manager import CacheManager

cm = CacheManager()
await cm.run_full_maintenance()
await cm.create_backup(compress=True)
await cm.restore_backup("backup_name")
```

**PerformanceMonitor**
```python
from ops.performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor(memory_manager=manager)
health = await monitor.check_system_health()
report = await monitor.generate_performance_report()
```

### Dashboard API

Access at `http://127.0.0.1:8050`:
- Real-time market data
- Position tracking
- Performance metrics
- System health

### Chat API

Access at `http://127.0.0.1:8051`:
- AI-powered queries
- System status
- Trading insights
- Quick actions

---

## Conclusion

This trading bot is production-ready with:
- ✅ M1 MacBook optimization (4GB memory)
- ✅ Testnet mode (safe testing)
- ✅ DeepSeek AI integration
- ✅ Professional dashboard
- ✅ Comprehensive testing
- ✅ Automated backups
- ✅ Performance monitoring
- ✅ Audit logging

For more information, see:
- [TASK7_COMPLETION_SUMMARY.md](TASK7_COMPLETION_SUMMARY.md) - Chat & Audit Logging
- [TASK8_COMPLETION_SUMMARY.md](TASK8_COMPLETION_SUMMARY.md) - Performance Monitoring
- [TASK9_COMPLETION_SUMMARY.md](TASK9_COMPLETION_SUMMARY.md) - Cache & Backup System

Happy Trading! 🚀
