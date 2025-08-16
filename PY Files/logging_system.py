# PY Files/logging_system.py
"""
Phase 5: Comprehensive Logging System
Implements S3 turn logging, retention policy management, and advanced analytics.
"""

import os
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import hashlib

import streamlit as st

# Try to import optional dependencies
try:
    import boto3
    from botocore.exceptions import ClientError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False

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

class TurnLogger:
    """Comprehensive turn-by-turn logging system with S3 integration."""
    
    def __init__(self, project_root: str = None):
        self.project_root = project_root or PROJECT_ROOT
        self.session_id = str(uuid.uuid4())
        self.turn_count = 0
        
        # Load retention policy
        self.retention_policy = self._load_retention_policy()
        
        # Initialize S3 client if available
        self.s3_client = None
        if S3_AVAILABLE:
            self.s3_client = self._init_s3_client()
    
    def _load_retention_policy(self) -> Dict[str, Any]:
        """Load retention policy configuration."""
        try:
            policy_path = Path(self.project_root) / "configs" / "retention_policy.yaml"
            if policy_path.exists():
                import yaml
                with open(policy_path, 'r') as f:
                    return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load retention policy: {e}")
        
        # Default retention policy
        return {
            "s3": {
                "logs": {"prefix": "project-root/logs/", "retention_days": 365},
                "exports": {"prefix": "project-root/exports/", "retention_days": 180}
            },
            "dropbox": {
                "exports": {"base_path": "/Apps/Ethos LLM/Project_Root/03_Summaries"}
            }
        }
    
    def _init_s3_client(self):
        """Initialize S3 client from Streamlit secrets or environment."""
        try:
            # Try Streamlit secrets first
            if hasattr(st, 'secrets'):
                return boto3.client(
                    's3',
                    region_name=st.secrets.get("AWS_DEFAULT_REGION"),
                    aws_access_key_id=st.secrets.get("AWS_ACCESS_KEY_ID"),
                    aws_secret_access_key=st.secrets.get("AWS_SECRET_ACCESS_KEY"),
                )
            else:
                # Fall back to environment variables
                return boto3.client(
                    's3',
                    region_name=os.environ.get("AWS_DEFAULT_REGION"),
                    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
                )
        except Exception as e:
            print(f"Warning: Could not initialize S3 client: {e}")
            return None
    
    def log_turn(
        self,
        question: str,
        intent: str,
        sources: List[str],
        confidence: float,
        model_used: str = "unknown",
        tokens: int = 0,
        cost: float = 0.0,
        service_level: float = 0.95,
        sub_skill: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        similarities: Optional[List[float]] = None,
        missing_fields: Optional[List[str]] = None,
        export_refs: Optional[List[str]] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Log a complete turn according to s3_turn_log.schema.json.
        
        Returns the logged turn data.
        """
        self.turn_count += 1
        
        # Calculate z-score from service level
        z_score = self._calculate_z_score(service_level)
        
        # Create turn log entry
        turn_log = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "session_id": self.session_id,
            "turn_number": self.turn_count,
            "question": question,
            "intent": intent,
            "sub_skill": sub_skill,
            "filters": filters or {},
            "sources": sources,
            "similarities": similarities or [],
            "confidence": confidence,
            "model_used": model_used,
            "tokens": tokens,
            "cost": cost,
            "service_level": service_level,
            "z_score": z_score,
            "missing_fields": missing_fields or [],
            "export_refs": export_refs or [],
            "additional_metadata": additional_metadata or {}
        }
        
        # Validate against schema
        self._validate_turn_log(turn_log)
        
        # Store locally and in S3
        self._store_turn_log(turn_log)
        
        return turn_log
    
    def _calculate_z_score(self, service_level: float) -> float:
        """Calculate z-score from service level."""
        service_level_map = {
            0.90: 1.645,
            0.95: 1.960,
            0.975: 2.241,
            0.99: 2.576
        }
        return service_level_map.get(service_level, 1.960)
    
    def _validate_turn_log(self, turn_log: Dict[str, Any]):
        """Validate turn log against schema requirements."""
        required_fields = ["ts", "question", "intent", "sources", "confidence"]
        for field in required_fields:
            if field not in turn_log:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate data types
        if not isinstance(turn_log["confidence"], (int, float)):
            raise ValueError("confidence must be numeric")
        if not isinstance(turn_log["sources"], list):
            raise ValueError("sources must be a list")
    
    def _store_turn_log(self, turn_log: Dict[str, Any]):
        """Store turn log locally and in S3."""
        # Store locally
        self._store_local_log(turn_log)
        
        # Store in S3 if available
        if self.s3_client:
            self._store_s3_log(turn_log)
    
    def _store_local_log(self, turn_log: Dict[str, Any]):
        """Store turn log locally."""
        try:
            logs_dir = Path(self.project_root) / "06_Logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            
            # Create daily log file
            today = datetime.now().strftime("%Y-%m-%d")
            log_file = logs_dir / f"turn_logs_{today}.jsonl"
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(turn_log, ensure_ascii=False) + '\n')
                
        except Exception as e:
            print(f"Warning: Could not store local log: {e}")
    
    def _store_s3_log(self, turn_log: Dict[str, Any]):
        """Store turn log in S3."""
        try:
            if not self.s3_client:
                return
            
            # Get S3 configuration
            bucket = st.secrets.get("S3_BUCKET") if hasattr(st, 'secrets') else os.environ.get("S3_BUCKET")
            if not bucket:
                return
            
            # Create S3 key based on retention policy
            prefix = self.retention_policy["s3"]["logs"]["prefix"]
            timestamp = datetime.now()
            s3_key = f"{prefix}{timestamp.strftime('%Y/%m/%d')}/turn_log_{self.session_id}_{self.turn_count}.json"
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=bucket,
                Key=s3_key,
                Body=json.dumps(turn_log, ensure_ascii=False),
                ContentType="application/json",
                Metadata={
                    "session_id": self.session_id,
                    "turn_number": str(self.turn_count),
                    "intent": turn_log["intent"],
                    "confidence": str(turn_log["confidence"])
                }
            )
            
        except Exception as e:
            print(f"Warning: Could not store S3 log: {e}")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the current session."""
        return {
            "session_id": self.session_id,
            "turn_count": self.turn_count,
            "started_at": datetime.utcnow().isoformat() + "Z"
        }
    
    def get_confidence_trends(self, window_size: int = 10) -> Dict[str, Any]:
        """Get confidence trend analysis for recent turns."""
        # This would typically query the stored logs
        # For now, return placeholder data
        return {
            "window_size": window_size,
            "average_confidence": 0.0,
            "confidence_trend": "stable",
            "low_confidence_count": 0
        }

class RetentionManager:
    """Manages data retention according to policy."""
    
    def __init__(self, project_root: str = None):
        self.project_root = project_root or PROJECT_ROOT
        self.retention_policy = self._load_retention_policy()
        self.s3_client = self._init_s3_client() if S3_AVAILABLE else None
    
    def _load_retention_policy(self) -> Dict[str, Any]:
        """Load retention policy configuration."""
        try:
            policy_path = Path(self.project_root) / "configs" / "retention_policy.yaml"
            if policy_path.exists():
                import yaml
                with open(policy_path, 'r') as f:
                    return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load retention policy: {e}")
        
        return {
            "s3": {
                "logs": {"prefix": "project-root/logs/", "retention_days": 365},
                "exports": {"prefix": "project-root/exports/", "retention_days": 180}
            },
            "dropbox": {
                "exports": {"base_path": "/Apps/Ethos LLM/Project_Root/03_Summaries"}
            }
        }
    
    def _init_s3_client(self):
        """Initialize S3 client."""
        try:
            if hasattr(st, 'secrets'):
                return boto3.client(
                    's3',
                    region_name=st.secrets.get("AWS_DEFAULT_REGION"),
                    aws_access_key_id=st.secrets.get("AWS_ACCESS_KEY_ID"),
                    aws_secret_access_key=st.secrets.get("AWS_SECRET_ACCESS_KEY"),
                )
            else:
                return boto3.client(
                    's3',
                    region_name=os.environ.get("AWS_DEFAULT_REGION"),
                    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
                )
        except Exception as e:
            print(f"Warning: Could not initialize S3 client: {e}")
            return None
    
    def cleanup_expired_data(self) -> Dict[str, int]:
        """Clean up expired data according to retention policy."""
        results = {"logs_cleaned": 0, "exports_cleaned": 0, "errors": 0}
        
        if not self.s3_client:
            return results
        
        try:
            # Clean up expired logs
            logs_cleaned = self._cleanup_expired_logs()
            results["logs_cleaned"] = logs_cleaned
            
            # Clean up expired exports
            exports_cleaned = self._cleanup_expired_exports()
            results["exports_cleaned"] = exports_cleaned
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
            results["errors"] += 1
        
        return results
    
    def _cleanup_expired_logs(self) -> int:
        """Clean up expired log files."""
        if not self.s3_client:
            return 0
        
        try:
            bucket = st.secrets.get("S3_BUCKET") if hasattr(st, 'secrets') else os.environ.get("S3_BUCKET")
            if not bucket:
                return 0
            
            prefix = self.retention_policy["s3"]["logs"]["prefix"]
            retention_days = self.retention_policy["s3"]["logs"]["retention_days"]
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            # List objects in logs prefix
            response = self.s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix
            )
            
            cleaned_count = 0
            for obj in response.get('Contents', []):
                if obj['LastModified'].replace(tzinfo=None) < cutoff_date:
                    self.s3_client.delete_object(Bucket=bucket, Key=obj['Key'])
                    cleaned_count += 1
            
            return cleaned_count
            
        except Exception as e:
            print(f"Error cleaning up logs: {e}")
            return 0
    
    def _cleanup_expired_exports(self) -> int:
        """Clean up expired export files."""
        if not self.s3_client:
            return 0
        
        try:
            bucket = st.secrets.get("S3_BUCKET") if hasattr(st, 'secrets') else os.environ.get("S3_BUCKET")
            if not bucket:
                return 0
            
            prefix = self.retention_policy["s3"]["exports"]["prefix"]
            retention_days = self.retention_policy["s3"]["exports"]["retention_days"]
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            # List objects in exports prefix
            response = self.s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix
            )
            
            cleaned_count = 0
            for obj in response.get('Contents', []):
                if obj['LastModified'].replace(tzinfo=None) < cutoff_date:
                    self.s3_client.delete_object(Bucket=bucket, Key=obj['Key'])
                    cleaned_count += 1
            
            return cleaned_count
            
        except Exception as e:
            print(f"Error cleaning up exports: {e}")
            return 0
    
    def get_retention_summary(self) -> Dict[str, Any]:
        """Get summary of retention policy and current status."""
        return {
            "policy": self.retention_policy,
            "last_cleanup": datetime.utcnow().isoformat() + "Z",
            "next_cleanup_recommended": (datetime.now() + timedelta(days=1)).isoformat() + "Z"
        }

class AnalyticsEngine:
    """Advanced analytics and monitoring for the system."""
    
    def __init__(self, project_root: str = None):
        self.project_root = project_root or PROJECT_ROOT
        self.logs_dir = Path(self.project_root) / "06_Logs"
    
    def analyze_confidence_trends(self, days: int = 30) -> Dict[str, Any]:
        """Analyze confidence trends over specified period."""
        try:
            if not PANDAS_AVAILABLE:
                return {"error": "pandas not available"}
            
            # Collect log data from local files
            logs_data = self._collect_logs_data(days)
            
            if not logs_data:
                return {"error": "No log data available"}
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(logs_data)
            
            # Calculate trends
            trends = {
                "total_turns": len(df),
                "average_confidence": df["confidence"].mean(),
                "confidence_std": df["confidence"].std(),
                "low_confidence_rate": (df["confidence"] < 0.7).mean(),
                "intent_distribution": df["intent"].value_counts().to_dict(),
                "model_usage": df["model_used"].value_counts().to_dict(),
                "daily_trends": self._calculate_daily_trends(df)
            }
            
            return trends
            
        except Exception as e:
            return {"error": str(e)}
    
    def _collect_logs_data(self, days: int) -> List[Dict[str, Any]]:
        """Collect log data from local files."""
        logs_data = []
        cutoff_date = datetime.now() - timedelta(days=days)
        
        try:
            if not self.logs_dir.exists():
                return logs_data
            
            # Scan log files
            for log_file in self.logs_dir.glob("turn_logs_*.jsonl"):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                log_entry = json.loads(line)
                                log_time = datetime.fromisoformat(log_entry["ts"].replace("Z", "+00:00"))
                                if log_time >= cutoff_date:
                                    logs_data.append(log_entry)
                except Exception as e:
                    print(f"Error reading log file {log_file}: {e}")
                    continue
            
        except Exception as e:
            print(f"Error collecting logs data: {e}")
        
        return logs_data
    
    def _calculate_daily_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate daily confidence trends."""
        try:
            # Convert timestamp to date
            df["date"] = pd.to_datetime(df["ts"]).dt.date
            
            # Group by date and calculate daily stats
            daily_stats = df.groupby("date").agg({
                "confidence": ["mean", "std", "count"],
                "tokens": "sum",
                "cost": "sum"
            }).round(4)
            
            return daily_stats.to_dict()
            
        except Exception as e:
            return {"error": str(e)}
    
    def generate_performance_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        trends = self.analyze_confidence_trends(days)
        
        if "error" in trends:
            return trends
        
        # Calculate additional metrics
        report = {
            "period_days": days,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "trends": trends,
            "recommendations": self._generate_recommendations(trends),
            "alerts": self._generate_alerts(trends)
        }
        
        return report
    
    def _generate_recommendations(self, trends: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on trends."""
        recommendations = []
        
        if trends.get("low_confidence_rate", 0) > 0.3:
            recommendations.append("High rate of low confidence responses - consider improving source quality")
        
        if trends.get("average_confidence", 0) < 0.7:
            recommendations.append("Average confidence below threshold - review model selection and prompts")
        
        if trends.get("total_turns", 0) < 10:
            recommendations.append("Limited data for analysis - continue logging for better insights")
        
        return recommendations
    
    def _generate_alerts(self, trends: Dict[str, Any]) -> List[str]:
        """Generate alerts based on trends."""
        alerts = []
        
        if trends.get("low_confidence_rate", 0) > 0.5:
            alerts.append("CRITICAL: More than 50% of responses have low confidence")
        
        if trends.get("average_confidence", 0) < 0.5:
            alerts.append("WARNING: Average confidence below 0.5")
        
        return alerts
