# PY Files/phase5_governance/query_logger.py
# Phase 5A: Query logs JSONL (user, intent, sources, confidence, tokens, $)

from __future__ import annotations
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import uuid

class QueryLogger:
    """
    Phase 5A: Comprehensive query logging system.
    Logs all user interactions with intent, sources, confidence, tokens, and costs.
    """
    
    def __init__(self, log_path: str = "04_Data/04_Metadata/query_log.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Token cost mapping (approximate costs per 1K tokens)
        self.token_costs = {
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002}
        }
    
    def log_query(
        self,
        user_id: str,
        query: str,
        intent: str,
        sources: List[Dict[str, Any]],
        confidence_score: float,
        confidence_badge: str,
        model_used: str = "gpt-4o-mini",
        input_tokens: int = 0,
        output_tokens: int = 0,
        response_time_ms: int = 0,
        artifacts_created: List[str] = None,
        kb_sources_used: List[str] = None,
        error: str = None
    ) -> str:
        """
        Log a complete query interaction.
        
        Returns:
            str: Unique log entry ID
        """
        
        # Generate unique log ID
        log_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Calculate costs
        cost_data = self._calculate_costs(model_used, input_tokens, output_tokens)
        
        # Extract source metadata
        source_summary = self._summarize_sources(sources, kb_sources_used)
        
        # Create log entry
        log_entry = {
            "log_id": log_id,
            "timestamp": timestamp,
            "user_id": user_id,
            "query": {
                "text": query,
                "intent": intent,
                "length": len(query)
            },
            "response": {
                "confidence_score": confidence_score,
                "confidence_badge": confidence_badge,
                "response_time_ms": response_time_ms,
                "error": error
            },
            "sources": source_summary,
            "model": {
                "name": model_used,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            },
            "costs": cost_data,
            "artifacts": artifacts_created or [],
            "metadata": {
                "session_id": self._get_session_id(user_id),
                "phase": "5A_logging"
            }
        }
        
        # Write to JSONL file
        self._write_log_entry(log_entry)
        
        return log_id
    
    def _calculate_costs(self, model: str, input_tokens: int, output_tokens: int) -> Dict[str, float]:
        """Calculate token costs for the query."""
        
        if model not in self.token_costs:
            model = "gpt-4o-mini"  # Default fallback
        
        costs = self.token_costs[model]
        
        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]
        total_cost = input_cost + output_cost
        
        return {
            "input_cost_usd": round(input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(total_cost, 6),
            "cost_per_query": round(total_cost, 6)
        }
    
    def _summarize_sources(
        self, 
        sources: List[Dict[str, Any]], 
        kb_sources: List[str] = None
    ) -> Dict[str, Any]:
        """Summarize source information for logging."""
        
        summary = {
            "data_sources": [],
            "kb_sources": kb_sources or [],
            "source_count": len(sources) if sources else 0,
            "kb_source_count": len(kb_sources) if kb_sources else 0
        }
        
        if sources:
            for source in sources:
                if isinstance(source, dict):
                    summary["data_sources"].append({
                        "type": source.get("type", "unknown"),
                        "path": source.get("path", "unknown"),
                        "sheet": source.get("sheet", ""),
                        "rows": source.get("rows", 0)
                    })
                else:
                    summary["data_sources"].append({"type": "string", "content": str(source)[:100]})
        
        return summary
    
    def _get_session_id(self, user_id: str) -> str:
        """Get or create session ID for user."""
        # Simple session management - could be enhanced with actual session tracking
        return f"session_{user_id}_{datetime.now().strftime('%Y%m%d')}"
    
    def _write_log_entry(self, log_entry: Dict[str, Any]):
        """Write log entry to JSONL file."""
        
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"Failed to write log entry: {e}")
    
    def get_usage_stats(self, days: int = 30) -> Dict[str, Any]:
        """
        Get usage statistics for the last N days.
        """
        
        if not self.log_path.exists():
            return {"error": "No log file found"}
        
        # Read and parse log entries
        entries = []
        cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
        
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            entry = json.loads(line)
                            entry_time = datetime.fromisoformat(entry["timestamp"]).timestamp()
                            if entry_time >= cutoff_date:
                                entries.append(entry)
                        except (json.JSONDecodeError, KeyError, ValueError):
                            continue
        except FileNotFoundError:
            return {"error": "Log file not found"}
        
        if not entries:
            return {"message": f"No entries found in the last {days} days"}
        
        # Calculate statistics
        stats = self._calculate_usage_stats(entries)
        stats["period_days"] = days
        stats["total_entries"] = len(entries)
        
        return stats
    
    def _calculate_usage_stats(self, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive usage statistics."""
        
        # Initialize counters
        total_queries = len(entries)
        total_cost = 0
        total_tokens = 0
        intent_counts = {}
        model_counts = {}
        confidence_counts = {"High": 0, "Medium": 0, "Low": 0}
        error_count = 0
        avg_response_time = 0
        unique_users = set()
        
        # Process each entry
        for entry in entries:
            # User tracking
            unique_users.add(entry.get("user_id", "unknown"))
            
            # Intent tracking
            intent = entry.get("query", {}).get("intent", "unknown")
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
            
            # Model tracking
            model = entry.get("model", {}).get("name", "unknown")
            model_counts[model] = model_counts.get(model, 0) + 1
            
            # Cost tracking
            cost = entry.get("costs", {}).get("total_cost_usd", 0)
            total_cost += cost
            
            # Token tracking
            tokens = entry.get("model", {}).get("total_tokens", 0)
            total_tokens += tokens
            
            # Confidence tracking
            confidence = entry.get("response", {}).get("confidence_badge", "Unknown")
            if confidence in confidence_counts:
                confidence_counts[confidence] += 1
            
            # Error tracking
            if entry.get("response", {}).get("error"):
                error_count += 1
            
            # Response time tracking
            response_time = entry.get("response", {}).get("response_time_ms", 0)
            avg_response_time += response_time
        
        # Calculate averages
        avg_response_time = avg_response_time / total_queries if total_queries > 0 else 0
        avg_cost_per_query = total_cost / total_queries if total_queries > 0 else 0
        avg_tokens_per_query = total_tokens / total_queries if total_queries > 0 else 0
        
        return {
            "summary": {
                "total_queries": total_queries,
                "unique_users": len(unique_users),
                "total_cost_usd": round(total_cost, 4),
                "total_tokens": total_tokens,
                "error_rate": round(error_count / total_queries * 100, 2) if total_queries > 0 else 0,
                "avg_response_time_ms": round(avg_response_time, 2),
                "avg_cost_per_query": round(avg_cost_per_query, 6),
                "avg_tokens_per_query": round(avg_tokens_per_query, 1)
            },
            "breakdown": {
                "by_intent": dict(sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)),
                "by_model": dict(sorted(model_counts.items(), key=lambda x: x[1], reverse=True)),
                "by_confidence": confidence_counts
            },
            "top_intents": list(dict(sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)).keys())[:5],
            "most_used_model": max(model_counts.items(), key=lambda x: x[1])[0] if model_counts else "none"
        }
    
    def get_recent_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent queries."""
        
        if not self.log_path.exists():
            return []
        
        entries = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            entry = json.loads(line)
                            entries.append(entry)
                        except json.JSONDecodeError:
                            continue
        except FileNotFoundError:
            return []
        
        # Sort by timestamp and return most recent
        entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return entries[:limit]
    
    def export_usage_report(self, days: int = 30, output_path: str = None) -> str:
        """Export usage statistics to JSON file."""
        
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"04_Data/04_Metadata/usage_report_{timestamp}.json"
        
        stats = self.get_usage_stats(days)
        recent_queries = self.get_recent_queries(20)
        
        report = {
            "report_generated": datetime.now().isoformat(),
            "period_days": days,
            "usage_statistics": stats,
            "recent_queries_sample": [
                {
                    "timestamp": q.get("timestamp"),
                    "user_id": q.get("user_id"),
                    "intent": q.get("query", {}).get("intent"),
                    "confidence": q.get("response", {}).get("confidence_badge"),
                    "cost": q.get("costs", {}).get("total_cost_usd"),
                    "tokens": q.get("model", {}).get("total_tokens")
                }
                for q in recent_queries[:10]
            ]
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        
        return str(output_path)
