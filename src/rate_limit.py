# rate_limit.py
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, Tuple
import time

# Rate limiting configuration
RATE_LIMIT_REQUESTS = 100  # Number of requests
RATE_LIMIT_WINDOW = 60  # Time window in seconds

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware that limits requests per IP address.
    Uses a sliding window algorithm.
    """
    
    def __init__(self, app, requests_per_minute: int = RATE_LIMIT_REQUESTS):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.window_seconds = RATE_LIMIT_WINDOW
        # Store request timestamps per IP: {ip: [timestamp1, timestamp2, ...]}
        self.request_history: Dict[str, list] = defaultdict(list)
        self._lock = defaultdict(lambda: False)  # Simple lock per IP
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks and management endpoints
        skip_paths = ["/health", "/docs", "/openapi.json", "/redoc", "/api/registry/stats", "/api/registry/cleanup"]
        if any(request.url.path.startswith(path) for path in skip_paths):
            return await call_next(request)
        
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Clean old entries (older than window)
        current_time = time.time()
        cutoff_time = current_time - self.window_seconds
        
        # Clean up old entries for this IP
        if client_ip in self.request_history:
            self.request_history[client_ip] = [
                ts for ts in self.request_history[client_ip] 
                if ts > cutoff_time
            ]
        
        # Check rate limit
        request_count = len(self.request_history[client_ip])
        
        if request_count >= self.requests_per_minute:
            # Rate limit exceeded
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Maximum {self.requests_per_minute} requests per {self.window_seconds} seconds."
            )
        
        # Record this request
        self.request_history[client_ip].append(current_time)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = self.requests_per_minute - len(self.request_history[client_ip])
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
        response.headers["X-RateLimit-Reset"] = str(int(current_time + self.window_seconds))
        
        return response

