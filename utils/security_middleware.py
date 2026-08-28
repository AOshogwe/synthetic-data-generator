# utils/security_middleware.py - Security Middleware and Rate Limiting
import time
import hashlib
import secrets
from collections import defaultdict, deque
from functools import wraps
from flask import request, jsonify, g
from werkzeug.exceptions import TooManyRequests
import logging

logger = logging.getLogger(__name__)

class SecurityMiddleware:
    """Security middleware for Flask applications"""
    
    def __init__(self, app=None):
        self.app = app
        self.rate_limiters = defaultdict(lambda: deque())
        self.blocked_ips = {}  # ip -> unblock_at (epoch seconds), not a bare set
        self.block_duration = 3600  # 1 hour -- blocks used to be permanent until restart
        self.failed_attempts = defaultdict(int)
        
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize security middleware with Flask app"""
        self.app = app

        # Trust exactly one hop of X-Forwarded-For/X-Proto (Railway's own
        # reverse proxy) so request.remote_addr is the real client IP,
        # instead of hand-parsing attacker-controlled headers ourselves.
        from werkzeug.middleware.proxy_fix import ProxyFix
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)

        app.before_request(self.before_request)
        app.after_request(self.after_request)
        
        # Security headers
        @app.after_request
        def add_security_headers(response):
            # Prevent clickjacking
            response.headers['X-Frame-Options'] = 'DENY'
            
            # Prevent MIME type sniffing
            response.headers['X-Content-Type-Options'] = 'nosniff'
            
            # XSS protection
            response.headers['X-XSS-Protection'] = '1; mode=block'
            
            # Referrer policy
            response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
            
            # Content Security Policy
            response.headers['Content-Security-Policy'] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self' https: data:; "
                "connect-src 'self';"
            )
            
            return response
    
    def before_request(self):
        """Security checks before processing request"""
        client_ip = self.get_client_ip()
        
        # Check if IP is blocked (with expiry -- a block used to be
        # permanent until process restart, which risks permanently locking
        # out a shared/NAT'd IP over a handful of retries)
        unblock_at = self.blocked_ips.get(client_ip)
        if unblock_at is not None:
            if time.time() < unblock_at:
                logger.warning(f"Blocked IP attempted access: {client_ip}")
                return jsonify({'error': 'Access denied'}), 403
            else:
                del self.blocked_ips[client_ip]
                self.failed_attempts[client_ip] = 0
        
        # Rate limiting
        if not self.rate_limit_check(client_ip):
            self.failed_attempts[client_ip] += 1
            
            # Block IP after too many failures
            if self.failed_attempts[client_ip] > 50:
                self.blocked_ips[client_ip] = time.time() + self.block_duration
                logger.warning(f"IP blocked for {self.block_duration}s due to rate limit violations: {client_ip}")
            
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            raise TooManyRequests("Rate limit exceeded. Please try again later.")
        
        # Validate request size against the app's own configured limit,
        # rather than a second, different hardcoded number (this used to
        # say 200MB while app.py's MAX_CONTENT_LENGTH defaults to 100MB, so
        # Flask's own limit always fired first and this check never did).
        max_size = self.app.config.get('MAX_CONTENT_LENGTH') if self.app else None
        if max_size and request.content_length and request.content_length > max_size:
            return jsonify({'error': 'Request too large'}), 413
        
        # Basic request validation
        if request.method in ['POST', 'PUT', 'PATCH']:
            if not self.validate_request_content():
                return jsonify({'error': 'Invalid request content'}), 400
    
    def after_request(self, response):
        """Security processing after request"""
        # Reset failed attempts on successful requests
        if response.status_code < 400:
            client_ip = self.get_client_ip()
            if client_ip in self.failed_attempts:
                self.failed_attempts[client_ip] = max(0, self.failed_attempts[client_ip] - 1)
        
        return response
    
    def get_client_ip(self):
        """Get client IP address safely -- relies on ProxyFix (set up in
        init_app) to have already resolved this from a trusted proxy hop,
        rather than reading attacker-controlled headers directly."""
        return request.remote_addr or 'unknown'
    
    def rate_limit_check(self, client_ip, max_requests=100, time_window=300):
        """
        Rate limiting check
        max_requests: Maximum requests allowed
        time_window: Time window in seconds (default: 5 minutes)
        """
        current_time = time.time()
        
        # Clean old entries
        requests = self.rate_limiters[client_ip]
        while requests and current_time - requests[0] > time_window:
            requests.popleft()
        
        # Check if limit exceeded
        if len(requests) >= max_requests:
            return False
        
        # Add current request
        requests.append(current_time)
        return True
    
    def validate_request_content(self):
        """Validate request content for security issues"""
        try:
            # Check for common injection patterns
            if request.is_json:
                data = request.get_json()
                if self.contains_suspicious_patterns(str(data)):
                    return False
            
            # Check form data
            for key, value in request.form.items():
                if self.contains_suspicious_patterns(value):
                    return False
            
            # Check query parameters
            for key, value in request.args.items():
                if self.contains_suspicious_patterns(value):
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating request content: {e}")
            return False
    
    def contains_suspicious_patterns(self, text):
        """Check for suspicious patterns in text"""
        if not isinstance(text, str):
            return False
        
        text_lower = text.lower()
        
        # SQL injection patterns
        sql_patterns = [
            'union select', 'drop table', 'delete from', 'insert into',
            'update set', 'exec(', 'execute(', 'sp_', 'xp_',
            "'; --", "' or '1'='1", "' or 1=1", "admin' --"
        ]
        
        # XSS patterns
        xss_patterns = [
            '<script', '</script>', 'javascript:', 'onload=',
            'onerror=', 'onclick=', 'onmouseover=', 'onfocus=',
            'eval(', 'alert(', 'document.cookie', 'window.location'
        ]
        
        # Path traversal
        path_patterns = [
            '../', '..\\', '/etc/passwd', '\\windows\\system32',
            '%2e%2e%2f', '%2e%2e\\', '....//....'
        ]
        
        all_patterns = sql_patterns + xss_patterns + path_patterns
        
        return any(pattern in text_lower for pattern in all_patterns)
    
    def require_api_key(self, f):
        """Decorator to require API key authentication"""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            api_key = request.headers.get('X-API-Key')
            
            if not api_key:
                return jsonify({'error': 'API key required'}), 401
            
            # Validate API key (implement your own validation logic)
            if not self.validate_api_key(api_key):
                return jsonify({'error': 'Invalid API key'}), 401
            
            return f(*args, **kwargs)
        
        return decorated_function
    
    def validate_api_key(self, api_key):
        """Validate API key - implement your own logic"""
        # This is a placeholder - implement proper API key validation
        # You might check against database, JWT tokens, etc.
        return len(api_key) >= 32 and api_key.isalnum()
    
    def generate_csrf_token(self):
        """Generate CSRF token"""
        return secrets.token_urlsafe(32)
    
    def validate_csrf_token(self, token):
        """Validate CSRF token"""
        # This is a simplified version - implement proper CSRF protection
        # You might store tokens in session, database, etc.
        return isinstance(token, str) and len(token) == 43  # urlsafe_32 length

class InputSanitizer:
    """Input sanitization utilities"""
    
    @staticmethod
    def sanitize_filename(filename):
        """Sanitize filename to prevent path traversal"""
        import os
        import re
        
        if not filename:
            return ""
        
        # Remove path components
        filename = os.path.basename(filename)
        
        # Remove dangerous characters
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        
        # Remove control characters
        filename = ''.join(char for char in filename if ord(char) >= 32)
        
        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:250] + ext
        
        return filename
    
    @staticmethod
    def sanitize_sql_input(text):
        """Basic SQL input sanitization"""
        if not isinstance(text, str):
            return text
        
        # Remove common SQL injection patterns
        dangerous_chars = ["'", '"', ';', '--', '/*', '*/', 'xp_', 'sp_']
        
        for char in dangerous_chars:
            text = text.replace(char, '')
        
        return text
    
    @staticmethod
    def sanitize_html_input(text):
        """Basic HTML input sanitization"""
        if not isinstance(text, str):
            return text
        
        # Replace dangerous HTML characters
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        text = text.replace('"', '&quot;')
        text = text.replace("'", '&#x27;')
        text = text.replace('&', '&amp;')
        
        return text