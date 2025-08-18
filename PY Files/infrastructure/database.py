# PY Files/infrastructure/database.py
"""
Phase 6: Database Management System
Provides PostgreSQL integration with connection pooling, performance monitoring, and enterprise features.
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
from contextlib import contextmanager
import threading
import queue
from pathlib import Path

# Try to import optional dependencies
try:
    import psycopg2
    from psycopg2.pool import SimpleConnectionPool, ThreadedConnectionPool
    from psycopg2.extras import RealDictCursor, Json
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Local imports
try:
    from constants import PROJECT_ROOT
    from path_utils import get_project_paths
except ImportError:
    PROJECT_ROOT = "/Project_Root"

class DatabaseManager:
    """Enterprise database manager with connection pooling and performance monitoring."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._load_config()
        self.pool = None
        self.performance_metrics = {}
        self.query_history = []
        self.connection_stats = {
            "total_connections": 0,
            "active_connections": 0,
            "failed_connections": 0,
            "connection_errors": []
        }
        
        # Performance monitoring
        self.query_times = []
        self.slow_query_threshold = self.config.get("slow_query_threshold", 1.0)
        self.max_query_history = self.config.get("max_query_history", 1000)
        
        # Initialize connection pool
        if POSTGRES_AVAILABLE:
            self._init_connection_pool()
        else:
            logging.warning("PostgreSQL not available. Database operations will be simulated.")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load database configuration."""
        try:
            config_path = Path(PROJECT_ROOT) / "configs" / "database.yaml"
            if config_path.exists():
                import yaml
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
        except Exception as e:
            logging.warning(f"Could not load database config: {e}")
        
        # Default configuration
        return {
            "host": os.environ.get("DB_HOST", "localhost"),
            "port": int(os.environ.get("DB_PORT", "5432")),
            "database": os.environ.get("DB_NAME", "scie_ethos"),
            "user": os.environ.get("DB_USER", "postgres"),
            "password": os.environ.get("DB_PASSWORD", ""),
            "pool_size": int(os.environ.get("DB_POOL_SIZE", "10")),
            "max_overflow": int(os.environ.get("DB_MAX_OVERFLOW", "20")),
            "pool_timeout": int(os.environ.get("DB_POOL_TIMEOUT", "30")),
            "slow_query_threshold": float(os.environ.get("DB_SLOW_QUERY_THRESHOLD", "1.0")),
            "max_query_history": int(os.environ.get("DB_MAX_QUERY_HISTORY", "1000")),
            "ssl_mode": os.environ.get("DB_SSL_MODE", "prefer")
        }
    
    def _init_connection_pool(self):
        """Initialize connection pool."""
        try:
            if self.config.get("use_threaded_pool", True):
                self.pool = ThreadedConnectionPool(
                    minconn=1,
                    maxconn=self.config["pool_size"],
                    host=self.config["host"],
                    port=self.config["port"],
                    database=self.config["database"],
                    user=self.config["user"],
                    password=self.config["password"],
                    sslmode=self.config["ssl_mode"]
                )
            else:
                self.pool = SimpleConnectionPool(
                    minconn=1,
                    maxconn=self.config["pool_size"],
                    host=self.config["host"],
                    port=self.config["port"],
                    database=self.config["database"],
                    user=self.config["user"],
                    password=self.config["password"],
                    sslmode=self.config["ssl_mode"]
                )
            
            # Test connection
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT version()")
                    version = cur.fetchone()[0]
                    logging.info(f"Connected to PostgreSQL: {version}")
                    
        except Exception as e:
            logging.error(f"Failed to initialize connection pool: {e}")
            self.connection_stats["connection_errors"].append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            })
    
    @contextmanager
    def _get_connection(self):
        """Get a connection from the pool with error handling."""
        conn = None
        start_time = time.time()
        
        try:
            if self.pool:
                conn = self.pool.getconn()
                self.connection_stats["total_connections"] += 1
                self.connection_stats["active_connections"] += 1
                yield conn
            else:
                # Fallback for development/testing
                yield None
                
        except Exception as e:
            self.connection_stats["failed_connections"] += 1
            self.connection_stats["connection_errors"].append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            })
            logging.error(f"Database connection error: {e}")
            raise
            
        finally:
            if conn and self.pool:
                self.pool.putconn(conn)
                self.connection_stats["active_connections"] -= 1
            
            # Log connection time
            connection_time = time.time() - start_time
            if connection_time > 1.0:  # Log slow connections
                logging.warning(f"Slow database connection: {connection_time:.2f}s")
    
    def execute_query(self, query: str, params: List[Any] = None, 
                     return_dict: bool = True) -> List[Dict[str, Any]]:
        """Execute a query and return results."""
        start_time = time.time()
        results = []
        
        try:
            with self._get_connection() as conn:
                if not conn:
                    # Fallback for development
                    return self._simulate_query(query, params)
                
                cursor_class = RealDictCursor if return_dict else None
                with conn.cursor(cursor_factory=cursor_class) as cur:
                    cur.execute(query, params or ())
                    
                    if cur.description:  # SELECT query
                        if return_dict:
                            results = [dict(row) for row in cur.fetchall()]
                        else:
                            results = cur.fetchall()
                    else:  # INSERT/UPDATE/DELETE query
                        results = {"affected_rows": cur.rowcount}
                    
                    # Commit if this was a write operation
                    if not cur.description:
                        conn.commit()
            
            # Log query performance
            execution_time = time.time() - start_time
            self._log_query_performance(query, execution_time, params)
            
            return results
            
        except Exception as e:
            logging.error(f"Query execution failed: {e}")
            logging.error(f"Query: {query}")
            logging.error(f"Params: {params}")
            raise
    
    def execute_transaction(self, queries: List[Tuple[str, List[Any]]]) -> bool:
        """Execute multiple queries in a transaction."""
        try:
            with self._get_connection() as conn:
                if not conn:
                    # Fallback for development
                    return self._simulate_transaction(queries)
                
                with conn.cursor() as cur:
                    for query, params in queries:
                        cur.execute(query, params or ())
                    
                    conn.commit()
                    return True
                    
        except Exception as e:
            logging.error(f"Transaction failed: {e}")
            return False
    
    def batch_insert(self, table: str, columns: List[str], 
                    data: List[List[Any]], batch_size: int = 1000) -> int:
        """Insert data in batches for better performance."""
        if not data:
            return 0
        
        total_inserted = 0
        placeholders = ",".join(["%s"] * len(columns))
        query = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})"
        
        try:
            with self._get_connection() as conn:
                if not conn:
                    # Fallback for development
                    return self._simulate_batch_insert(table, columns, data)
                
                with conn.cursor() as cur:
                    for i in range(0, len(data), batch_size):
                        batch = data[i:i + batch_size]
                        cur.executemany(query, batch)
                        total_inserted += len(batch)
                    
                    conn.commit()
                    return total_inserted
                    
        except Exception as e:
            logging.error(f"Batch insert failed: {e}")
            raise
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get database performance metrics."""
        metrics = {
            "connection_stats": self.connection_stats.copy(),
            "query_performance": {
                "total_queries": len(self.query_history),
                "slow_queries": len([q for q in self.query_history if q["execution_time"] > self.slow_query_threshold]),
                "average_execution_time": 0.0,
                "max_execution_time": 0.0,
                "min_execution_time": float('inf')
            },
            "pool_status": {
                "pool_size": self.config["pool_size"],
                "active_connections": self.connection_stats["active_connections"],
                "available_connections": self.config["pool_size"] - self.connection_stats["active_connections"]
            }
        }
        
        if self.query_history:
            execution_times = [q["execution_time"] for q in self.query_history]
            metrics["query_performance"]["average_execution_time"] = sum(execution_times) / len(execution_times)
            metrics["query_performance"]["max_execution_time"] = max(execution_times)
            metrics["query_performance"]["min_execution_time"] = min(execution_times)
        
        return metrics
    
    def get_slow_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the slowest queries for performance analysis."""
        slow_queries = [q for q in self.query_history if q["execution_time"] > self.slow_query_threshold]
        slow_queries.sort(key=lambda x: x["execution_time"], reverse=True)
        return slow_queries[:limit]
    
    def get_query_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent query history."""
        return self.query_history[-limit:]
    
    def _log_query_performance(self, query: str, execution_time: float, params: List[Any]):
        """Log query performance metrics."""
        query_info = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "params": params,
            "execution_time": execution_time,
            "slow_query": execution_time > self.slow_query_threshold
        }
        
        self.query_history.append(query_info)
        
        # Maintain query history size
        if len(self.query_history) > self.max_query_history:
            self.query_history.pop(0)
        
        # Log slow queries
        if execution_time > self.slow_query_threshold:
            logging.warning(f"Slow query detected: {execution_time:.2f}s - {query[:100]}...")
    
    def _simulate_query(self, query: str, params: List[Any]) -> List[Dict[str, Any]]:
        """Simulate query execution for development/testing."""
        start_time = time.time()
        logging.info(f"Simulating query: {query}")
        
        # Simulate different query types
        if "SELECT" in query.upper():
            if "user_sessions" in query:
                result = [{"user_id": 123, "session_data": "test_data"}]
            elif "performance_metrics" in query:
                result = [{"metric": "cpu_usage", "value": 75.5}]
            else:
                result = [{"result": "simulated_data"}]
        else:
            result = {"affected_rows": 1}
        
        # Log performance for simulation mode
        execution_time = time.time() - start_time
        self._log_query_performance(query, execution_time, params)
        
        return result
    
    def _simulate_transaction(self, queries: List[Tuple[str, List[Any]]]) -> bool:
        """Simulate transaction execution for development/testing."""
        logging.info(f"Simulating transaction with {len(queries)} queries")
        return True
    
    def _simulate_batch_insert(self, table: str, columns: List[str], 
                              data: List[List[Any]]) -> int:
        """Simulate batch insert for development/testing."""
        logging.info(f"Simulating batch insert into {table}: {len(data)} rows")
        return len(data)
    
    def health_check(self) -> Dict[str, Any]:
        """Perform database health check."""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "checks": {}
        }
        
        try:
            # Test connection
            with self._get_connection() as conn:
                if conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                        health_status["checks"]["connection"] = "healthy"
                else:
                    health_status["checks"]["connection"] = "simulated"
            
            # Check pool status
            if self.pool:
                health_status["checks"]["pool"] = "healthy"
            else:
                health_status["checks"]["pool"] = "not_initialized"
            
            # Check performance
            metrics = self.get_performance_metrics()
            if metrics["query_performance"]["slow_queries"] > 10:
                health_status["status"] = "degraded"
                health_status["checks"]["performance"] = "degraded"
            else:
                health_status["checks"]["performance"] = "healthy"
                
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["checks"]["error"] = str(e)
        
        return health_status
    
    def close(self):
        """Close the database manager and connection pool."""
        if self.pool:
            self.pool.closeall()
            logging.info("Database connection pool closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class DatabaseMigrator:
    """Database schema migration and version management."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.migrations_dir = Path(PROJECT_ROOT) / "migrations"
        self.migrations_table = "schema_migrations"
    
    def create_migrations_table(self):
        """Create the migrations tracking table."""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.migrations_table} (
            id SERIAL PRIMARY KEY,
            version VARCHAR(255) NOT NULL UNIQUE,
            name VARCHAR(255) NOT NULL,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            checksum VARCHAR(64) NOT NULL,
            execution_time FLOAT
        );
        """
        
        try:
            self.db_manager.execute_query(create_table_sql)
            logging.info("Migrations table created successfully")
        except Exception as e:
            logging.error(f"Failed to create migrations table: {e}")
    
    def get_applied_migrations(self) -> List[str]:
        """Get list of applied migration versions."""
        try:
            query = f"SELECT version FROM {self.migrations_table} ORDER BY applied_at"
            results = self.db_manager.execute_query(query)
            return [row["version"] for row in results]
        except Exception as e:
            logging.error(f"Failed to get applied migrations: {e}")
            return []
    
    def apply_migration(self, migration_file: Path) -> bool:
        """Apply a single migration file."""
        try:
            with open(migration_file, 'r') as f:
                migration_sql = f.read()
            
            # Calculate checksum
            import hashlib
            checksum = hashlib.md5(migration_sql.encode()).hexdigest()
            
            # Extract version and name from filename
            version = migration_file.stem.split('_')[0]
            name = migration_file.stem.replace(f"{version}_", "")
            
            start_time = time.time()
            
            # Execute migration
            self.db_manager.execute_query(migration_sql)
            
            execution_time = time.time() - start_time
            
            # Record migration
            insert_sql = f"""
            INSERT INTO {self.migrations_table} (version, name, checksum, execution_time)
            VALUES (%s, %s, %s, %s)
            """
            self.db_manager.execute_query(insert_sql, [version, name, checksum, execution_time])
            
            logging.info(f"Migration {version} applied successfully in {execution_time:.2f}s")
            return True
            
        except Exception as e:
            logging.error(f"Failed to apply migration {migration_file}: {e}")
            return False
    
    def run_migrations(self) -> Dict[str, Any]:
        """Run all pending migrations."""
        self.create_migrations_table()
        
        applied_migrations = self.get_applied_migrations()
        pending_migrations = []
        
        # Find pending migrations
        for migration_file in sorted(self.migrations_dir.glob("*.sql")):
            version = migration_file.stem.split('_')[0]
            if version not in applied_migrations:
                pending_migrations.append(migration_file)
        
        results = {
            "total_migrations": len(pending_migrations),
            "applied_migrations": [],
            "failed_migrations": [],
            "execution_time": 0.0
        }
        
        start_time = time.time()
        
        for migration_file in pending_migrations:
            if self.apply_migration(migration_file):
                results["applied_migrations"].append(migration_file.name)
            else:
                results["failed_migrations"].append(migration_file.name)
        
        results["execution_time"] = time.time() - start_time
        
        logging.info(f"Migration run completed: {len(results['applied_migrations'])} applied, "
                    f"{len(results['failed_migrations'])} failed")
        
        return results


# Factory function for easy instantiation
def create_database_manager(config: Dict[str, Any] = None) -> DatabaseManager:
    """Create a database manager instance."""
    return DatabaseManager(config)


# Example usage
if __name__ == "__main__":
    # Test database manager
    db = DatabaseManager()
    
    try:
        # Test health check
        health = db.health_check()
        print(f"Database health: {health['status']}")
        
        # Test query execution
        results = db.execute_query("SELECT version()")
        print(f"PostgreSQL version: {results[0]['version'] if results else 'Unknown'}")
        
        # Get performance metrics
        metrics = db.get_performance_metrics()
        print(f"Performance metrics: {metrics}")
        
    finally:
        db.close()
