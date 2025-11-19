# Production Runbook
## DeepSeek Trading System - Operations Manual

### Table of Contents
1. [System Overview](#system-overview)
2. [Demo Procedures](#demo-procedures)
3. [Monitoring & Alerts](#monitoring--alerts)
4. [On-Call Procedures](#on-call-procedures)
5. [Emergency Response](#emergency-response)
6. [Rollback Procedures](#rollback-procedures)
7. [Maintenance](#maintenance)
8. [Troubleshooting](#troubleshooting)

---

## System Overview

### Architecture
- **Frontend**: Dash/Plotly dashboard (port 8050)
- **Backend**: Python async trading system
- **Database**: SQLite (data/trading_signals.db)
- **Cache**: In-memory + Parquet files (data/cache/)
- **Data Source**: Binance WebSocket + REST API
- **Backup**: Automated compression (backups/)

### Key Components
1. **DataStore** (`core/data/data_store.py`) - Market data caching with 30-day persistence
2. **BinanceStream** (`core/data/binance_stream.py`) - Real-time WebSocket streaming
3. **DeepSeekBrain** (`deepseek/client.py`) - AI trading decisions
4. **SignalGenerator** (`core/signal_generator.py`) - Signal processing
5. **CacheManager** (`ops/cache_manager.py`) - Automated backups & cleanup

### Critical Paths
- WebSocket → DataStore → Dashboard (real-time data)
- Database ← SignalGenerator (trade records)
- CacheManager → Backups (data protection)

---

## Demo Procedures

### Phase 2: Real-Time Data Integration

#### Prerequisites
- [ ] All tests passing (`make test-fast`)
- [ ] DataStore initialized (`data/cache/` exists)
- [ ] Database accessible (`data/trading_signals.db`)
- [ ] Dashboard running (`make dashboard`)

#### Demo Steps

**Step 1: Verify Real-Time Data Flow**
```bash
# Start the dashboard
make dashboard

# In a separate terminal, verify WebSocket connection
python -c "from core.data.binance_stream import BinanceStream; import asyncio; \
async def handler(sym, intv, kline): print(f'{sym} {intv}: {kline[\"c\"]}'); \
asyncio.run(BinanceStream().stream_klines('BTCUSDT', '1m', handler))"
```

**Expected Output:**
- Dashboard loads at http://localhost:8050
- Chart shows live BTCUSDT price data
- Candles updating in real-time
- No errors in logs

**Step 2: Verify Caching Layer**
```bash
# Check DataStore cache
ls -lh data/cache/
# Should show recent Parquet files with timestamp

# Check cache statistics
python -c "from core.data.data_store import DataStore; ds = DataStore(); \
print(ds.get_cache_stats())"
```

**Expected Output:**
- Multiple Parquet files (BTCUSDT_1m_history.parquet, etc.)
- Cache hit rate > 50%
- Memory usage < 100MB

**Step 3: Verify Database Integration**
```bash
# Check database
sqlite3 data/trading_signals.db ".tables"
sqlite3 data/trading_signals.db "SELECT COUNT(*) FROM signals;"

# Compute success metrics
python scripts/compute_success_metrics.py
```

**Expected Output:**
- `signals` table exists
- Trade records present
- Metrics computed (win rate, Sharpe ratio, etc.)

### Phase 3: Production Features

#### Prerequisites
- [ ] Phase 2 demo completed successfully
- [ ] Backup system tested (`make backup`)
- [ ] Feature toggles functional

#### Demo Steps

**Step 1: Feature Toggle System**
```bash
# Start dashboard
make dashboard

# Navigate to http://localhost:8050
# In "Chart Overlays" section:
# 1. Enable Liquidity Zones ✓
# 2. Enable Supertrend ✓
# 3. Enable Chandelier Exit ✓
# 4. Enable Order Flow ✓
# 5. Enable Market Regime ✓
# 6. Enable Timeframe Alignment ✓
```

**Expected Output:**
- Chart overlays appear instantly
- Visual indicators update
- No performance degradation
- Smooth timeframe switching (1m, 5m, 15m, 1h, 4h, 1d)

**Step 2: Memory Optimization**
```bash
# Check memory usage
python -c "import psutil; import os; print(f'Memory: {psutil.Process(os.getpid()).memory_info().rss/1024/1024:.1f}MB')"

# Trigger cache cleanup
python -c "from ops.cache_manager import CacheManager; import asyncio; \
cm = CacheManager(); asyncio.run(cm.cleanup_old_cache())"
```

**Expected Output:**
- Memory usage stays under 2GB
- Cache cleanup removes old files
- No OOM errors

**Step 3: Backup & Recovery**
```bash
# Create backup
make backup

# Verify backup
ls -lh backups/
# Should show timestamped .tar.gz file

# Test restore (dry run)
python -c "from ops.cache_manager import CacheManager; import asyncio; \
cm = CacheManager(); print('Restore ready:', 'backup_' in str(list(cm.backup_dir.glob('backup_*'))))"
```

**Expected Output:**
- Compressed backup created
- Backup size < 10MB
- Restore function operational

---

## Monitoring & Alerts

### Key Metrics to Monitor

**System Health**
- Memory usage: < 2GB (WARNING > 3GB, CRITICAL > 4GB)
- CPU usage: < 50% average (WARNING > 70%, CRITICAL > 90%)
- Disk space: < 80% (WARNING > 85%, CRITICAL > 90%)
- Cache hit rate: > 50% (WARNING < 30%, CRITICAL < 10%)

**Data Pipeline**
- WebSocket connection: Active
- DataStore cache: Populated
- Database writes: < 1s latency
- Dashboard response time: < 2s
- Feature telemetry table populated (latency/memory per indicator)

**Trading Performance**
- Signal generation rate: 1-5 per hour
- Win rate: Track trend
- Drawdown: < 20% (WARNING > 25%, CRITICAL > 35%)
- Sharpe ratio: > 1.0 (WARNING < 0.5)
- Market regime: Should rarely remain UNKNOWN (fall back to last known)

### Telemetry Guidance
- Use the dashboard’s “Feature Performance Telemetry” card to identify expensive overlays.
- Regime recommendation text suggests which overlays to enable; disable others if latency spikes.

### Log Locations
```
logs/
├── trading.log          # Main trading activity
├── system.log           # System events
├── cache_manager.log    # Backup/cleanup operations
└── chat_history.log     # Dashboard interactions
```

### Log Monitoring Commands
```bash
# Watch live logs
tail -f logs/trading.log

# Check for errors
grep -i error logs/*.log | tail -20

# Monitor WebSocket status
grep -i websocket logs/system.log | tail -10
```

---

## On-Call Procedures

### Response Priority Matrix

| Severity | Description | Response Time | Example |
|----------|-------------|---------------|---------|
| **P1 - Critical** | System down, data loss, trading halted | 15 minutes | Dashboard won't load, WebSocket disconnected |
| **P2 - High** | Degraded performance, partial functionality | 1 hour | Cache not updating, slow response |
| **P3 - Medium** | Non-critical issues, cosmetic bugs | 4 hours | Chart overlay glitch, minor UI issue |
| **P4 - Low** | Feature requests, documentation | 24 hours | New indicator request |

### Standard Response Procedure

**For All Incidents:**

1. **Acknowledge** (within response time SLA)
   - Check alert source
   - Acknowledge in monitoring system
   - Create incident ticket

2. **Assess** (within 5 minutes of ack)
   - Identify affected components
   - Determine severity
   - Check recent deployments

3. **Communicate** (immediately for P1/P2)
   - Notify stakeholders
   - Post updates every 30 minutes
   - Document in incident log

4. **Resolve** (per severity SLA)
   - Follow runbook procedures
   - Test fix in staging if available
   - Deploy fix with approval

5. **Verify** (post-resolution)
   - Confirm full recovery
   - Monitor for 30 minutes
   - Check related metrics

6. **Document** (within 24 hours)
   - Update runbook if needed
   - Create post-mortem for P1/P2
   - Archive incident ticket

### Common Issue Playbooks

#### Issue: WebSocket Disconnection
**Symptoms:**
- No new price data
- Stale timestamps on dashboard
- WebSocket reconnect messages in logs

**Diagnosis:**
```bash
# Check WebSocket status
grep -i "connection" logs/system.log | tail -5

# Verify Binance API status
curl -s https://api.binance.com/api/v3/ping

# Check network connectivity
ping fstream.binance.com
```

**Resolution:**
1. Wait 30 seconds (auto-reconnect should occur)
2. If persists, restart WebSocket stream:
   ```bash
   pkill -f "binance_stream"
   python -m core.data.binance_stream
   ```
3. Clear cache and reload dashboard
4. Verify data flow returns to normal

**Prevention:**
- Monitor connection status
- Set up alerts for disconnections
- Review network stability

#### Issue: High Memory Usage
**Symptoms:**
- System slowdown
- Out of memory errors
- Swap usage high

**Diagnosis:**
```bash
# Check memory usage
python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"

# Find memory leaks
python -c "from ops.cache_manager import CacheManager; \
cm = CacheManager(); print(cm.get_statistics())"

# Check cache size
du -sh data/cache/
```

**Resolution:**
1. Trigger emergency cache cleanup:
   ```bash
   python -c "from ops.cache_manager import CacheManager; \
   import asyncio; cm = CacheManager(); \
   asyncio.run(cm.cleanup_old_cache())"
   ```
2. Restart dashboard to clear memory
3. Verify memory returns to normal
4. Investigate root cause (likely data accumulation)

**Prevention:**
- Daily automated cleanup via Cron
- Weekly backup/rotate cycle
- Monitor cache growth trends

#### Issue: Database Lock/Performance
**Symptoms:**
- Signals not being saved
- "database is locked" errors
- Slow database queries

**Diagnosis:**
```bash
# Check database integrity
sqlite3 data/trading_signals.db "PRAGMA integrity_check;"

# View active connections
lsof data/trading_signals.db

# Check database size
du -sh data/trading_signals.db
```

**Resolution:**
1. Check for zombie processes holding locks
2. Wait for long-running queries to complete
3. If necessary, restart application
4. Run database maintenance:
   ```bash
   sqlite3 data/trading_signals.db "VACUUM;"
   ```

**Prevention:**
- Optimize database operations
- Use connection pooling if needed
- Regular VACUUM maintenance
- Monitor query performance

---

## Emergency Response

### Emergency Stop Procedures

**System-Wide Emergency Stop:**
```bash
# Stop all trading immediately
echo "EMERGENCY STOP TRIGGERED" | tee -a logs/emergency.log

# Kill all trading processes
pkill -f "trading" || true
pkill -f "signal_generator" || true
pkill -f "deepseek" || true

# Stop dashboard
pkill -f "dashboard" || true

# Create immediate backup
python -c "from ops.cache_manager import CacheManager; \
import asyncio; cm = CacheManager(); \
asyncio.run(cm.create_backup(compress=True))"

# Log the incident
echo "Emergency stop at $(date)" >> logs/emergency.log
```

### Data Recovery Procedures

**From Automated Backup:**
```bash
# List available backups
ls -lt backups/*.tar.gz | head -5

# Restore from latest backup
BACKUP_NAME="backup_YYYYMMDD_HHMMSS"  # Use actual filename
python -c "from ops.cache_manager import CacheManager; \
import asyncio; cm = CacheManager(); \
asyncio.run(cm.restore_backup('$BACKUP_NAME'))"

# Verify restoration
python scripts/compute_success_metrics.py

# Restart services
make dashboard
make test-fast
```

**Manual Database Recovery:**
```bash
# If database is corrupted
cp data/trading_signals.db data/trading_signals.db.backup

# Restore from last known good
# (assuming backup was created before corruption)
sqlite3 data/trading_signals.db < last_good_backup.sql

# Verify data integrity
sqlite3 data/trading_signals.db "PRAGMA integrity_check;"
```

### System Recovery Checklist

- [ ] System stopped (if emergency)
- [ ] Issue diagnosed
- [ ] Data backup verified
- [ ] Fix tested (if applicable)
- [ ] Services restarted
- [ ] Data flow verified
- [ ] Dashboard accessible
- [ ] Metrics normal
- [ ] Stakeholders notified
- [ ] Incident documented

---

## Rollback Procedures

### Deployment Rollback

**If Recent Code Changes Caused Issues:**

```bash
# Stop current system
make stop  # or pkill -f trading system processes

# Revert to last known good state
git log --oneline -10
git checkout <last_good-commit-hash>

# Restore data from backup
make restore BACKUP=<latest-backup-name>

# Restart services
make dashboard
make test-fast

# Verify rollback
curl -s http://localhost:8050 | grep -q "DeepSeek Trading Dashboard" && \
  echo "Dashboard OK" || echo "Dashboard FAILED"
```

### Database Rollback

**If Database Migration Failed:**

```bash
# Create immediate backup
cp data/trading_signals.db data/trading_signals.db.emergency

# Rollback migration
# (Assumes migration created backup automatically)
if [ -f data/trading_signals.db.pre_migration ]; then
    cp data/trading_signals.db.pre_migration data/trading_signals.db
    echo "Rolled back database migration"
else
    echo "No pre-migration backup found!"
fi

# Verify rollback
python scripts/compute_success_metrics.py
```

### Configuration Rollback

**If Config Changes Caused Issues:**

```bash
# Restore config from backup
cp backups/<backup-name>/config/* . 2>/dev/null || \
  cp config.yaml.backup config.yaml 2>/dev/null || \
  echo "No config backup found!"

# Reload configuration
# (Restart application to pick up new config)
```

### Rollback Verification

**After Any Rollback:**
```bash
# 1. Verify services start
make test-fast
# Should show all tests passing

# 2. Verify data integrity
python scripts/compute_success_metrics.py
# Should compute metrics without errors

# 3. Verify dashboard
curl -s http://localhost:8050
# Should return HTML

# 4. Verify WebSocket stream
python -c "from core.data.binance_stream import BinanceStream; \
import asyncio; asyncio.run(BinanceStream().stream_klines('BTCUSDT', '1m', lambda s,i,k: print('OK')))" | head -1
# Should show 'OK'

# 5. Check logs for errors
grep -i error logs/*.log | tail -20
# Should show no new errors
```

---

## Maintenance

### Daily Maintenance
**Automated via Cron:**
```bash
# Run daily at 3 AM
0 3 * * * cd /Users/mrsmoothy/Downloads/Trading_bot && \
  python -c "from ops.cache_manager import CacheManager; \
  import asyncio; asyncio.run(CacheManager().run_full_maintenance())"
```

**Manual Daily Check:**
```bash
# Check system health
python -c "from ops.cache_manager import CacheManager; \
cm = CacheManager(); print(cm.get_statistics())"

# Review logs
tail -50 logs/trading.log | grep -i error
```

### Weekly Maintenance
```bash
# Full system backup
make backup

# Database optimization
sqlite3 data/trading_signals.db "VACUUM;"

# Clean old logs
find logs/ -name "*.log" -mtime +30 -delete

# Review performance metrics
python scripts/compute_success_metrics.py > reports/weekly_metrics.txt
```

### Monthly Maintenance
```bash
# Full system test
make test-all

# Review and rotate backups
ls -lt backups/*.tar.gz
# Keep last 30 days, remove older

# Update documentation
# Review runbook for updates

# Performance tuning review
# Analyze metrics trends
```

### Database Maintenance
```bash
# Weekly database maintenance
sqlite3 data/trading_signals.db <<EOF
VACUUM;
ANALYZE;
PRAGMA optimize;
EOF

# Check database size
du -sh data/trading_signals.db

# Archive old records (optional)
# sqlite3 data/trading_signals.db "DELETE FROM signals WHERE created_at < date('now', '-90 days');"
```

---

## Troubleshooting

### Diagnostic Commands

**System Health Check:**
```bash
#!/bin/bash
echo "=== SYSTEM HEALTH CHECK ==="
echo "Memory: $(python -c "import psutil; print(f'{psutil.virtual_memory().percent}%')")"
echo "CPU: $(python -c "import psutil; print(f'{psutil.cpu_percent()}%')")"
echo "Disk: $(df -h . | tail -1 | awk '{print $5}')"
echo "Cache: $(du -sh data/cache/ | cut -f1)"
echo "Database: $(du -sh data/trading_signals.db | cut -f1)"
echo "Backups: $(ls -1 backups/*.tar.gz 2>/dev/null | wc -l) files"
echo "=== END CHECK ==="
```

**Test All Components:**
```bash
# Quick test suite
make test-fast

# Manual component tests
python -c "from core.data.data_store import DataStore; ds = DataStore(); print('DataStore:', ds.get_cache_stats())"
python -c "from ops.cache_manager import CacheManager; cm = CacheManager(); print('CacheManager:', cm.get_statistics())"
python scripts/compute_success_metrics.py
```

### Common Issues & Solutions

**Issue: Dashboard Blank/White Screen**
- **Cause**: JavaScript error, wrong port, or dashboard not running
- **Fix**:
  ```bash
  # Check if dashboard is running
  lsof -i :8050

  # Check logs
  tail -50 logs/system.log | grep -i dashboard

  # Restart dashboard
  make dashboard
  ```

**Issue: No Data on Charts**
- **Cause**: WebSocket not connected, cache empty, or DataStore issue
- **Fix**:
  ```bash
  # Check cache
  python -c "from core.data.data_store import DataStore; ds = DataStore(); \
  print(ds.get_cached_symbols())"

  # Force refresh cache
  rm -f data/cache/*.parquet
  # Restart dashboard to reload
  ```

**Issue: High CPU Usage**
- **Cause**: Infinite loop, too many threads, or inefficient code
- **Fix**:
  ```bash
  # Find CPU-intensive processes
  top -o %CPU

  # Check for runaway threads
  python -c "import threading; print(f'Threads: {threading.active_count()}')"

  # Restart system if needed
  pkill -f trading
  make start
  ```

**Issue: Database Errors**
- **Cause**: Corruption, locks, or schema mismatch
- **Fix**:
  ```bash
  # Check integrity
  sqlite3 data/trading_signals.db "PRAGMA integrity_check;"

  # Check schema
  sqlite3 data/trading_signals.db ".schema signals"

  # If corrupted, restore from backup
  make restore BACKUP=<backup-name>
  ```

**Issue: Telemetry Table Empty**
- **Cause**: Trading loop hasn’t run since startup; feature metrics only record when signals are generated.
- **Fix**:
  ```bash
  # Kick off trading loop
  python main.py
  # or trigger a manual signal generation test
  pytest tests/test_signal_flow_e2e.py -k generate
  ```
  Ensure `logs/trading.log` shows feature calculations; refresh dashboard to load metrics.

### Getting Help

**Internal Resources:**
- Runbook: This document
- Documentation: `/docs/` directory
- Logs: `/logs/` directory
- Test suite: `make test-fast`

**External Resources:**
- Project repository: [GitHub URL]
- Monitoring dashboard: http://localhost:8050
- API documentation: [URL if applicable]

**Contact Information:**
- On-call engineer: [Contact Info]
- Tech lead: [Contact Info]
- Emergency escalation: [Contact Info]

---

## Appendix

### File Locations
```
/Users/mrsmoothy/Downloads/Trading_bot/
├── data/
│   ├── trading_signals.db       # Main database
│   └── cache/                   # DataStore cache (Parquet files)
├── logs/                        # Application logs
├── backups/                     # Automated backups
├── core/                        # Core trading system
├── ui/                          # Dashboard
└── scripts/                     # Utility scripts
```

### Important Files
- `config.yaml` - System configuration
- `.env` - Environment variables
- `logs/trading.log` - Main application log
- `logs/system.log` - System events log
- `backups/` - Backup archive directory

### Quick Reference

**Start System:**
```bash
make dashboard  # Start dashboard
```

**Stop System:**
```bash
pkill -f trading  # Kill trading processes
pkill -f dashboard  # Kill dashboard
```

**Emergency Stop:**
```bash
make emergency-stop  # Complete system halt
```

**Backup:**
```bash
make backup  # Create compressed backup
```

**Test:**
```bash
make test-fast  # Run quick tests
make test-all   # Run full test suite
```

**Monitor:**
```bash
tail -f logs/trading.log  # Watch live logs
```

---

**Document Version:** 1.0
**Last Updated:** 2025-11-15
**Owner:** Trading System Team
**Review Schedule:** Monthly

For updates or corrections to this runbook, submit a PR with changes.
