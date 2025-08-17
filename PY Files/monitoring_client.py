"""
Phase 6: Monitoring Client
Provides a monitoring client with no-op fallback for production environments.
"""

import logging
from typing import Any, Dict, Optional

class MonitoringClient:
    """Monitoring client with no-op fallback for production environments."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize monitoring client with fallback to no-op mode."""
        self.enabled = True
        self.config = config or {}
        
        try:
            # Try to initialize real monitoring client
            self._init_real_client()
        except Exception as e:
            logging.warning(f"Failed to initialize monitoring client: {e}. Using no-op mode.")
            self.enabled = False
    
    def _init_real_client(self):
        """Initialize the real monitoring client."""
        # This would typically initialize New Relic, Datadog, Prometheus, etc.
        # For now, we'll just set enabled to True if no exceptions occur
        pass
    
    def emit(self, metric: str, value: Any, **labels) -> bool:
        """Emit a metric with labels."""
        if not self.enabled:
            return True
        
        try:
            # Try to emit the real metric
            self._emit_real_metric(metric, value, **labels)
            return True
        except Exception as e:
            logging.error(f"Failed to emit metric {metric}: {e}")
            return False
    
    def _emit_real_metric(self, metric: str, value: Any, **labels):
        """Emit a metric to the real monitoring system."""
        # This would typically send to New Relic, Datadog, Prometheus, etc.
        # For now, we'll just log it
        logging.info(f"MONITORING: {metric}={value} labels={labels}")
    
    def increment(self, metric: str, value: int = 1, **labels) -> bool:
        """Increment a counter metric."""
        return self.emit(metric, value, **labels)
    
    def gauge(self, metric: str, value: float, **labels) -> bool:
        """Set a gauge metric."""
        return self.emit(metric, value, **labels)
    
    def histogram(self, metric: str, value: float, **labels) -> bool:
        """Record a histogram metric."""
        return self.emit(metric, value, **labels)
    
    def timing(self, metric: str, value: float, **labels) -> bool:
        """Record a timing metric."""
        return self.emit(metric, value, **labels)
    
    def health_check(self) -> Dict[str, Any]:
        """Get monitoring client health status."""
        return {
            "enabled": self.enabled,
            "status": "healthy" if self.enabled else "no_op_mode"
        }


# Factory function for easy instantiation
def create_monitoring_client(config: Dict[str, Any] = None) -> MonitoringClient:
    """Create a monitoring client instance."""
    return MonitoringClient(config)


# Example usage
if __name__ == "__main__":
    # Test monitoring client
    client = MonitoringClient()
    
    # Test metrics
    client.emit("test.counter", 1, service="test")
    client.increment("test.increment", 5, service="test")
    client.gauge("test.gauge", 42.5, service="test")
    client.timing("test.timing", 0.123, service="test")
    
    # Check health
    health = client.health_check()
    print(f"Monitoring client health: {health}")
