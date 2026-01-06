"""Metrics service for tracking API performance.

Singleton service to track inference calls and latency metrics.
"""

import threading
import time
from typing import Dict


class MetricsService:
    """Singleton service for tracking API metrics.
    
    Thread-safe counter and latency tracking for inference calls.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Create singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(MetricsService, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize metrics counters."""
        if self._initialized:
            return
        
        self._lock = threading.Lock()
        self._inference_count = 0
        self._total_latency_ms = 0.0
        self._min_latency_ms = float('inf')
        self._max_latency_ms = 0.0
        self._initialized = True
    
    def record_inference(self, latency_ms: float) -> None:
        """Record an inference call with its latency.
        
        Args:
            latency_ms: Latency in milliseconds
        """
        with self._lock:
            self._inference_count += 1
            self._total_latency_ms += latency_ms
            
            if latency_ms < self._min_latency_ms:
                self._min_latency_ms = latency_ms
            
            if latency_ms > self._max_latency_ms:
                self._max_latency_ms = latency_ms
    
    def get_metrics(self) -> Dict:
        """Get current metrics.
        
        Returns:
            Dictionary with metrics including:
            - inference_count: Total number of inference calls
            - average_latency_ms: Average latency in milliseconds
            - min_latency_ms: Minimum latency observed
            - max_latency_ms: Maximum latency observed
        """
        with self._lock:
            avg_latency = (
                self._total_latency_ms / self._inference_count
                if self._inference_count > 0
                else 0.0
            )
            
            return {
                "inference_count": self._inference_count,
                "average_latency_ms": round(avg_latency, 2),
                "min_latency_ms": round(self._min_latency_ms, 2) if self._min_latency_ms != float('inf') else 0.0,
                "max_latency_ms": round(self._max_latency_ms, 2),
            }
    
    def reset(self) -> None:
        """Reset all metrics (useful for testing)."""
        with self._lock:
            self._inference_count = 0
            self._total_latency_ms = 0.0
            self._min_latency_ms = float('inf')
            self._max_latency_ms = 0.0


# Global singleton instance
metrics_service = MetricsService()

