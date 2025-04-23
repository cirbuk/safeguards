"""SQLite-based metrics storage implementation."""

import sqlite3
from datetime import datetime

from safeguards.base.monitoring import MetricsStorage, ResourceMetrics


class SQLiteMetricsStorage(MetricsStorage):
    """Store metrics in SQLite database."""

    def __init__(self, db_path: str = "metrics.db"):
        """Initialize SQLite storage.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metrics (
                    timestamp DATETIME NOT NULL,
                    cpu_percent REAL NOT NULL,
                    memory_percent REAL NOT NULL,
                    disk_percent REAL NOT NULL,
                    network_mbps REAL NOT NULL,
                    process_count INTEGER NOT NULL,
                    open_files INTEGER NOT NULL
                )
            """,
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_timestamp ON metrics(timestamp)",
            )

    def store_metrics(self, metrics: ResourceMetrics) -> None:
        """Store resource metrics in database.

        Args:
            metrics: Resource metrics to store
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO metrics (
                    timestamp,
                    cpu_percent,
                    memory_percent,
                    disk_percent,
                    network_mbps,
                    process_count,
                    open_files
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metrics.timestamp,
                    metrics.cpu_percent,
                    metrics.memory_percent,
                    metrics.disk_percent,
                    metrics.network_mbps,
                    metrics.process_count,
                    metrics.open_files,
                ),
            )

    def get_metrics(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> list[ResourceMetrics]:
        """Retrieve metrics for a time range.

        Args:
            start_time: Start of time range
            end_time: End of time range

        Returns:
            List of metrics in time range
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM metrics
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp ASC
                """,
                (start_time, end_time),
            )

            return [
                ResourceMetrics(
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    cpu_percent=row["cpu_percent"],
                    memory_percent=row["memory_percent"],
                    disk_percent=row["disk_percent"],
                    network_mbps=row["network_mbps"],
                    process_count=row["process_count"],
                    open_files=row["open_files"],
                )
                for row in cursor
            ]

    def cleanup_old_metrics(self, older_than: datetime) -> None:
        """Remove metrics older than specified time.

        Args:
            older_than: Remove metrics older than this time
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM metrics WHERE timestamp < ?",
                (older_than,),
            )
