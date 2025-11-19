"""
Cache Manager - Automated Cleanup and Backup System
Handles cache pruning, data retention, and automated backups with scheduling.
"""

import asyncio
import os
import shutil
import json
import gzip
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from loguru import logger
import schedule
import time


class CacheManager:
    """
    Automated cache cleanup and backup manager.
    Handles data retention policies, cache pruning, and scheduled backups.
    """

    def __init__(
        self,
        cache_dir: str = "cache",
        backup_dir: str = "backups",
        logs_dir: str = "logs",
        retention_days: int = 7,
        backup_retention_days: int = 30
    ):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory for cache files
            backup_dir: Directory for backups
            logs_dir: Directory for log files
            retention_days: Days to retain cache data
            backup_retention_days: Days to retain backups
        """
        self.cache_dir = Path(cache_dir)
        self.backup_dir = Path(backup_dir)
        self.logs_dir = Path(logs_dir)
        self.retention_days = retention_days
        self.backup_retention_days = backup_retention_days

        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Cleanup statistics
        self.cleanup_stats = {
            'last_cleanup': None,
            'total_cleaned': 0,
            'space_freed_mb': 0,
            'files_deleted': 0
        }

        # Backup statistics
        self.backup_stats = {
            'last_backup': None,
            'total_backups': 0,
            'total_backup_size_mb': 0,
            'last_backup_size_mb': 0
        }

        logger.info(f"CacheManager initialized: retention={retention_days}d, backup_retention={backup_retention_days}d")

    async def cleanup_old_cache(self) -> Dict[str, Any]:
        """
        Clean up old cache files based on retention policy.

        Returns:
            Cleanup statistics
        """
        logger.info("Starting cache cleanup...")

        cutoff_time = datetime.now() - timedelta(days=self.retention_days)
        files_deleted = 0
        space_freed = 0

        try:
            # Clean cache directory
            for file_path in self.cache_dir.rglob('*'):
                if file_path.is_file():
                    try:
                        file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)

                        if file_mtime < cutoff_time:
                            file_size = file_path.stat().st_size
                            file_path.unlink()
                            files_deleted += 1
                            space_freed += file_size
                            logger.debug(f"Deleted old cache file: {file_path}")

                    except Exception as e:
                        logger.warning(f"Error deleting {file_path}: {e}")

            space_freed_mb = space_freed / 1024 / 1024

            # Update statistics
            self.cleanup_stats.update({
                'last_cleanup': datetime.now().isoformat(),
                'total_cleaned': self.cleanup_stats['total_cleaned'] + files_deleted,
                'space_freed_mb': self.cleanup_stats['space_freed_mb'] + space_freed_mb,
                'files_deleted': files_deleted
            })

            logger.info(f"Cache cleanup completed: {files_deleted} files deleted, {space_freed_mb:.2f}MB freed")

            return {
                'status': 'SUCCESS',
                'files_deleted': files_deleted,
                'space_freed_mb': space_freed_mb,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
            return {
                'status': 'ERROR',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    async def cleanup_old_logs(self) -> Dict[str, Any]:
        """
        Clean up old log files based on retention policy.

        Returns:
            Cleanup statistics
        """
        logger.info("Starting log cleanup...")

        cutoff_time = datetime.now() - timedelta(days=self.retention_days)
        files_deleted = 0
        space_freed = 0

        try:
            # Clean logs directory
            for file_path in self.logs_dir.rglob('*.log'):
                if file_path.is_file():
                    try:
                        file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)

                        if file_mtime < cutoff_time:
                            file_size = file_path.stat().st_size
                            file_path.unlink()
                            files_deleted += 1
                            space_freed += file_size
                            logger.debug(f"Deleted old log file: {file_path}")

                    except Exception as e:
                        logger.warning(f"Error deleting {file_path}: {e}")

            space_freed_mb = space_freed / 1024 / 1024

            logger.info(f"Log cleanup completed: {files_deleted} files deleted, {space_freed_mb:.2f}MB freed")

            return {
                'status': 'SUCCESS',
                'files_deleted': files_deleted,
                'space_freed_mb': space_freed_mb,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error during log cleanup: {e}")
            return {
                'status': 'ERROR',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    async def create_backup(self, compress: bool = True) -> Dict[str, Any]:
        """
        Create backup of important data.

        Args:
            compress: Whether to compress backup files

        Returns:
            Backup statistics
        """
        logger.info("Starting backup creation...")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"backup_{timestamp}"
        backup_path = self.backup_dir / backup_name

        try:
            # Create backup directory
            backup_path.mkdir(parents=True, exist_ok=True)

            total_size = 0

            # Backup cache directory (important data only)
            cache_backup = backup_path / "cache"
            if self.cache_dir.exists():
                shutil.copytree(self.cache_dir, cache_backup, dirs_exist_ok=True)
                total_size += self._get_directory_size(cache_backup)

            # Backup database files if they exist
            db_files = list(Path('.').glob('*.db'))
            if db_files:
                db_backup = backup_path / "databases"
                db_backup.mkdir(exist_ok=True)
                for db_file in db_files:
                    shutil.copy2(db_file, db_backup / db_file.name)
                    total_size += db_file.stat().st_size

            # Backup important config files
            config_backup = backup_path / "config"
            config_backup.mkdir(exist_ok=True)

            config_files = ['.env', 'config.yaml', 'config.json']
            for config_file in config_files:
                config_path = Path(config_file)
                if config_path.exists():
                    shutil.copy2(config_path, config_backup / config_path.name)
                    total_size += config_path.stat().st_size

            # Create backup metadata
            metadata = {
                'timestamp': timestamp,
                'backup_name': backup_name,
                'created_at': datetime.now().isoformat(),
                'total_size_bytes': total_size,
                'compressed': compress,
                'retention_days': self.backup_retention_days
            }

            metadata_file = backup_path / 'metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Compress if requested
            if compress:
                compressed_file = self.backup_dir / f"{backup_name}.tar.gz"
                shutil.make_archive(
                    str(self.backup_dir / backup_name),
                    'gztar',
                    backup_path
                )

                # Remove uncompressed backup
                shutil.rmtree(backup_path)

                total_size = compressed_file.stat().st_size
                logger.info(f"Backup compressed to: {compressed_file}")

            total_size_mb = total_size / 1024 / 1024

            # Update statistics
            self.backup_stats.update({
                'last_backup': datetime.now().isoformat(),
                'total_backups': self.backup_stats['total_backups'] + 1,
                'total_backup_size_mb': self.backup_stats['total_backup_size_mb'] + total_size_mb,
                'last_backup_size_mb': total_size_mb
            })

            logger.info(f"Backup created successfully: {backup_name} ({total_size_mb:.2f}MB)")

            return {
                'status': 'SUCCESS',
                'backup_name': backup_name,
                'size_mb': total_size_mb,
                'compressed': compress,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error creating backup: {e}")

            # Cleanup failed backup
            if backup_path.exists():
                shutil.rmtree(backup_path)

            return {
                'status': 'ERROR',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    async def cleanup_old_backups(self) -> Dict[str, Any]:
        """
        Clean up old backups based on retention policy.

        Returns:
            Cleanup statistics
        """
        logger.info("Starting backup cleanup...")

        cutoff_time = datetime.now() - timedelta(days=self.backup_retention_days)
        backups_deleted = 0
        space_freed = 0

        try:
            # Clean backup directory
            for backup_item in self.backup_dir.iterdir():
                try:
                    # Check both directories and compressed files
                    if backup_item.is_file() and backup_item.suffix in ['.gz', '.tar', '.zip']:
                        backup_mtime = datetime.fromtimestamp(backup_item.stat().st_mtime)

                        if backup_mtime < cutoff_time:
                            backup_size = backup_item.stat().st_size
                            backup_item.unlink()
                            backups_deleted += 1
                            space_freed += backup_size
                            logger.debug(f"Deleted old backup: {backup_item}")

                    elif backup_item.is_dir():
                        # Check metadata file for timestamp
                        metadata_file = backup_item / 'metadata.json'
                        if metadata_file.exists():
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)

                            backup_time = datetime.fromisoformat(metadata['created_at'])

                            if backup_time < cutoff_time:
                                backup_size = self._get_directory_size(backup_item)
                                shutil.rmtree(backup_item)
                                backups_deleted += 1
                                space_freed += backup_size
                                logger.debug(f"Deleted old backup directory: {backup_item}")

                except Exception as e:
                    logger.warning(f"Error processing backup {backup_item}: {e}")

            space_freed_mb = space_freed / 1024 / 1024

            logger.info(f"Backup cleanup completed: {backups_deleted} backups deleted, {space_freed_mb:.2f}MB freed")

            return {
                'status': 'SUCCESS',
                'backups_deleted': backups_deleted,
                'space_freed_mb': space_freed_mb,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error during backup cleanup: {e}")
            return {
                'status': 'ERROR',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _get_directory_size(self, directory: Path) -> int:
        """
        Get total size of directory in bytes.

        Args:
            directory: Directory path

        Returns:
            Size in bytes
        """
        total_size = 0
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                try:
                    total_size += file_path.stat().st_size
                except:
                    pass
        return total_size

    async def run_full_maintenance(self) -> Dict[str, Any]:
        """
        Run complete maintenance: cleanup + backup.

        Returns:
            Maintenance statistics
        """
        logger.info("="*70)
        logger.info("Starting full maintenance cycle")
        logger.info("="*70)

        results = {
            'timestamp': datetime.now().isoformat(),
            'cache_cleanup': await self.cleanup_old_cache(),
            'log_cleanup': await self.cleanup_old_logs(),
            'backup': await self.create_backup(compress=True),
            'backup_cleanup': await self.cleanup_old_backups()
        }

        # Calculate totals
        total_space_freed = (
            results['cache_cleanup'].get('space_freed_mb', 0) +
            results['log_cleanup'].get('space_freed_mb', 0) +
            results['backup_cleanup'].get('space_freed_mb', 0)
        )

        results['summary'] = {
            'total_space_freed_mb': total_space_freed,
            'backup_created': results['backup']['status'] == 'SUCCESS',
            'backup_size_mb': results['backup'].get('size_mb', 0),
            'all_successful': all(
                r.get('status') == 'SUCCESS'
                for r in [results['cache_cleanup'], results['log_cleanup'],
                         results['backup'], results['backup_cleanup']]
            )
        }

        logger.info("="*70)
        logger.info(f"Maintenance completed: {total_space_freed:.2f}MB freed")
        logger.info("="*70)

        return results

    def schedule_maintenance(
        self,
        cleanup_time: str = "03:00",
        backup_time: str = "04:00",
        cleanup_days: List[str] = None
    ):
        """
        Schedule automated maintenance tasks.

        Args:
            cleanup_time: Time to run cleanup (HH:MM format)
            backup_time: Time to run backup (HH:MM format)
            cleanup_days: Days to run cleanup (None = every day)
        """
        if cleanup_days is None:
            cleanup_days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']

        # Schedule cache cleanup
        for day in cleanup_days:
            getattr(schedule.every(), day.lower()).at(cleanup_time).do(
                self._run_async, self.cleanup_old_cache()
            )
            getattr(schedule.every(), day.lower()).at(cleanup_time).do(
                self._run_async, self.cleanup_old_logs()
            )

        # Schedule backup
        schedule.every().day.at(backup_time).do(
            self._run_async, self.create_backup(compress=True)
        )

        # Schedule old backup cleanup (weekly)
        schedule.every().monday.at(backup_time).do(
            self._run_async, self.cleanup_old_backups()
        )

        logger.info(f"Maintenance scheduled: cleanup at {cleanup_time}, backup at {backup_time}")
        logger.info(f"Cleanup days: {', '.join(cleanup_days)}")

    def _run_async(self, coro):
        """Run async coroutine in sync context."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(coro)
            loop.close()
            return result
        except Exception as e:
            logger.error(f"Error running scheduled task: {e}")

    async def maintenance_loop(self, interval_hours: int = 24):
        """
        Continuous maintenance loop.

        Args:
            interval_hours: Hours between maintenance cycles
        """
        logger.info(f"Starting maintenance loop (every {interval_hours}h)")

        while True:
            try:
                await self.run_full_maintenance()
                await asyncio.sleep(interval_hours * 3600)

            except Exception as e:
                logger.error(f"Error in maintenance loop: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour on error

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get cache and backup statistics.

        Returns:
            Statistics dictionary
        """
        return {
            'cleanup_stats': self.cleanup_stats,
            'backup_stats': self.backup_stats,
            'cache_size_mb': self._get_directory_size(self.cache_dir) / 1024 / 1024,
            'backup_size_mb': self._get_directory_size(self.backup_dir) / 1024 / 1024,
            'logs_size_mb': self._get_directory_size(self.logs_dir) / 1024 / 1024,
            'total_size_mb': (
                self._get_directory_size(self.cache_dir) +
                self._get_directory_size(self.backup_dir) +
                self._get_directory_size(self.logs_dir)
            ) / 1024 / 1024
        }

    async def restore_backup(self, backup_name: str) -> Dict[str, Any]:
        """
        Restore from a backup.

        Args:
            backup_name: Name of backup to restore

        Returns:
            Restore status
        """
        logger.info(f"Restoring backup: {backup_name}")

        try:
            # Find backup
            backup_path = self.backup_dir / backup_name
            compressed_backup = self.backup_dir / f"{backup_name}.tar.gz"

            if compressed_backup.exists():
                # Extract compressed backup
                shutil.unpack_archive(compressed_backup, backup_path)
                logger.info(f"Extracted compressed backup: {compressed_backup}")

            if not backup_path.exists():
                raise FileNotFoundError(f"Backup not found: {backup_name}")

            # Read metadata
            metadata_file = backup_path / 'metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Backup metadata: {metadata}")

            # Restore cache
            cache_backup = backup_path / "cache"
            if cache_backup.exists():
                # Backup current cache first
                if self.cache_dir.exists():
                    backup_current = self.cache_dir.parent / f"cache_before_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    shutil.move(self.cache_dir, backup_current)
                    logger.info(f"Backed up current cache to: {backup_current}")

                # Restore cache
                shutil.copytree(cache_backup, self.cache_dir)
                logger.info("Cache restored successfully")

            # Restore databases
            db_backup = backup_path / "databases"
            if db_backup.exists():
                for db_file in db_backup.glob('*.db'):
                    shutil.copy2(db_file, Path('.') / db_file.name)
                    logger.info(f"Restored database: {db_file.name}")

            # Restore config files
            config_backup = backup_path / "config"
            if config_backup.exists():
                for config_file in config_backup.iterdir():
                    if config_file.is_file():
                        shutil.copy2(config_file, Path('.') / config_file.name)
                        logger.info(f"Restored config: {config_file.name}")

            logger.info(f"Backup restored successfully: {backup_name}")

            return {
                'status': 'SUCCESS',
                'backup_name': backup_name,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error restoring backup: {e}")
            return {
                'status': 'ERROR',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
