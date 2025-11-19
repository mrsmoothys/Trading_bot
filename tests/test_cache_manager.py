"""
Test CacheManager - Automated Cleanup and Backup System
Quick validation of comprehensive cache management functionality.
"""

import asyncio
import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import json
import gzip
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ops.cache_manager import CacheManager
from loguru import logger


@pytest.mark.asyncio
async def test_cache_manager():
    """Test comprehensive cache management functionality."""
    logger.info("="*70)
    logger.info("Testing CacheManager - Automated Cleanup & Backup System")
    logger.info("="*70)

    # Create temporary directories for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        backup_dir = Path(tmpdir) / "backups"
        logs_dir = Path(tmpdir) / "logs"

        # Initialize cache manager with test directories
        cache_manager = CacheManager(
            cache_dir=str(cache_dir),
            backup_dir=str(backup_dir),
            logs_dir=str(logs_dir),
            retention_days=1,  # Short retention for testing
            backup_retention_days=1
        )

        logger.info(f"Test directories created: {tmpdir}")

        # Test 1: Cache cleanup with old files
        logger.info("\n[Test 1] Testing cache cleanup with old files...")

        # Create old cache files (older than retention)
        old_cache_file = cache_dir / "old_cache.json"
        old_cache_file.write_text('{"old": "data"}')

        # Create recent cache file
        recent_cache_file = cache_dir / "recent_cache.json"
        recent_cache_file.write_text('{"recent": "data"}')

        # Make old file truly old
        import time
        import os
        old_time = time.time() - (2 * 24 * 3600)  # 2 days ago
        os.utime(old_cache_file, (old_time, old_time))

        logger.info(f"  Created: {old_cache_file.name} (old)")
        logger.info(f"  Created: {recent_cache_file.name} (recent)")

        # Run cleanup
        result = await cache_manager.cleanup_old_cache()

        # Verify cleanup
        assert result['status'] == 'SUCCESS', f"Cleanup failed: {result}"
        assert result['files_deleted'] == 1, f"Expected 1 file deleted, got {result['files_deleted']}"
        assert not old_cache_file.exists(), "Old cache file should be deleted"
        assert recent_cache_file.exists(), "Recent cache file should NOT be deleted"

        logger.info(f"✅ Cache cleanup successful: {result['files_deleted']} old file(s) deleted")

        # Test 2: Log cleanup with old log files
        logger.info("\n[Test 2] Testing log cleanup with old log files...")

        # Create old log files
        old_log_file = logs_dir / "old_log.log"
        old_log_file.write_text("Old log entry\n")

        # Create recent log file
        recent_log_file = logs_dir / "recent_log.log"
        recent_log_file.write_text("Recent log entry\n")

        # Make old log file old
        os.utime(old_log_file, (old_time, old_time))

        logger.info(f"  Created: {old_log_file.name} (old)")
        logger.info(f"  Created: {recent_log_file.name} (recent)")

        # Run cleanup
        result = await cache_manager.cleanup_old_logs()

        # Verify cleanup
        assert result['status'] == 'SUCCESS', f"Log cleanup failed: {result}"
        assert result['files_deleted'] == 1, f"Expected 1 log file deleted, got {result['files_deleted']}"
        assert not old_log_file.exists(), "Old log file should be deleted"
        assert recent_log_file.exists(), "Recent log file should NOT be deleted"

        logger.info(f"✅ Log cleanup successful: {result['files_deleted']} old file(s) deleted")

        # Test 3: Backup creation with compression
        logger.info("\n[Test 3] Testing backup creation with compression...")

        # Create test cache data
        cache_manager.cache_dir.mkdir(parents=True, exist_ok=True)
        test_cache = cache_manager.cache_dir / "test_data.json"
        test_cache.write_text('{"test": "cache_data", "timestamp": "2025-11-14"}')

        # Create test database file
        test_db = Path(tmpdir) / "test.db"
        test_db.write_text("fake database content")

        # Create test config file
        test_config = Path(tmpdir) / ".env"
        test_config.write_text("TEST_CONFIG=value\nTEST_KEY=secret")

        logger.info(f"  Created cache file: {test_cache.name}")
        logger.info(f"  Created database: {test_db.name}")
        logger.info(f"  Created config: {test_config.name}")

        # Run backup
        result = await cache_manager.create_backup(compress=True)

        # Verify backup
        assert result['status'] == 'SUCCESS', f"Backup failed: {result}"
        assert result['compressed'] == True, "Backup should be compressed"

        # Check for compressed backup file
        backup_files = list(cache_manager.backup_dir.glob("backup_*.tar.gz"))
        assert len(backup_files) > 0, "No compressed backup created"

        logger.info(f"✅ Backup created: {result['backup_name']} ({result['size_mb']:.2f}MB)")

        # Test 4: Backup statistics tracking
        logger.info("\n[Test 4] Testing backup statistics tracking...")

        stats = cache_manager.get_statistics()
        logger.info(f"  Backup stats: {stats['backup_stats']}")

        assert stats['backup_stats']['total_backups'] > 0, "Should have created backups"
        assert stats['backup_stats']['last_backup'] is not None, "Should track last backup time"
        assert stats['backup_stats']['last_backup_size_mb'] > 0, "Should track backup size"

        logger.info(f"✅ Statistics tracking working: {stats['backup_stats']['total_backups']} total backup(s)")

        # Test 5: Restore backup
        logger.info("\n[Test 5] Testing backup restoration...")

        backup_name = result['backup_name']

        # Clear cache directory first
        for item in cache_manager.cache_dir.glob('*'):
            item.unlink()

        # Restore from backup
        result = await cache_manager.restore_backup(backup_name)

        # Verify restoration
        assert result['status'] == 'SUCCESS', f"Restore failed: {result}"

        # Check if cache was restored
        restored_cache = cache_manager.cache_dir / "test_data.json"
        assert restored_cache.exists(), "Cache file should be restored"
        restored_data = json.loads(restored_cache.read_text())
        assert 'test' in restored_data, "Restored data should contain test content"

        logger.info(f"✅ Backup restored successfully: {backup_name}")

        # Test 6: Cleanup old backups
        logger.info("\n[Test 6] Testing old backup cleanup...")

        # Create a backup that's older than retention
        # For testing, we'll just create one and run cleanup
        initial_backups = len(list(cache_manager.backup_dir.glob("backup_*")))

        result = await cache_manager.cleanup_old_backups()

        logger.info(f"  Initial backups: {initial_backups}")
        logger.info(f"  After cleanup: {len(list(cache_manager.backup_dir.glob('backup_*')))}")

        logger.info(f"✅ Backup cleanup executed: {result['status']}")

        # Test 7: Full maintenance cycle
        logger.info("\n[Test 7] Testing full maintenance cycle...")

        # Create some old files for cleanup
        old_cache_file2 = cache_dir / "old_cache_2.json"
        old_cache_file2.write_text('{"old": "data2"}')
        os.utime(old_cache_file2, (old_time, old_time))

        old_log_file2 = logs_dir / "old_log_2.log"
        old_log_file2.write_text("Old log entry 2\n")
        os.utime(old_log_file2, (old_time, old_time))

        # Run full maintenance
        result = await cache_manager.run_full_maintenance()

        # Verify full maintenance
        assert result['summary']['all_successful'], "Full maintenance should succeed"
        assert result['cache_cleanup']['status'] == 'SUCCESS', "Cache cleanup in maintenance should succeed"
        assert result['backup']['status'] == 'SUCCESS', "Backup in maintenance should succeed"

        logger.info(f"✅ Full maintenance completed successfully")
        logger.info(f"  Total space freed: {result['summary']['total_space_freed_mb']:.2f}MB")

        # Test 8: Get statistics
        logger.info("\n[Test 8] Testing statistics retrieval...")

        stats = cache_manager.get_statistics()

        logger.info(f"  Cache size: {stats['cache_size_mb']:.2f}MB")
        logger.info(f"  Backup size: {stats['backup_size_mb']:.2f}MB")
        logger.info(f"  Logs size: {stats['logs_size_mb']:.2f}MB")
        logger.info(f"  Total size: {stats['total_size_mb']:.2f}MB")

        assert 'cache_size_mb' in stats, "Should include cache size"
        assert 'backup_size_mb' in stats, "Should include backup size"
        assert 'logs_size_mb' in stats, "Should include logs size"
        assert 'total_size_mb' in stats, "Should include total size"

        logger.info(f"✅ Statistics retrieval working correctly")

        # Test 9: Error handling
        logger.info("\n[Test 9] Testing error handling...")

        # Test with non-existent backup restoration
        result = await cache_manager.restore_backup("non_existent_backup_12345")

        assert result['status'] == 'ERROR', "Should return error for non-existent backup"
        assert 'error' in result, "Error message should be included"

        logger.info(f"✅ Error handling working correctly")

        logger.info("\n" + "="*70)
        logger.info("✅ All CacheManager tests passed!")
        logger.info("="*70)

        return True


async def main():
    """Run all tests."""
    try:
        success = await test_cache_manager()
        if success:
            logger.success("CacheManager tests completed successfully!")
            return 0
        else:
            logger.error("CacheManager tests failed")
            return 1
    except Exception as e:
        logger.error(f"Test error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
