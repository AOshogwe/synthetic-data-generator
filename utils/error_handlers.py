# utils/error_handlers.py - Comprehensive Error Handling System
import logging
import traceback
import json
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from flask import request, jsonify
from werkzeug.exceptions import HTTPException
import uuid

logger = logging.getLogger(__name__)

class ErrorHandler:
    """Centralized error handling system"""
    
    def __init__(self, app=None):
        self.app = app
        self.error_log = []
        self.max_error_log_size = 1000
        
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize error handlers with Flask app"""
        self.app = app
        
        # Register error handlers
        app.errorhandler(400)(self.handle_bad_request)
        app.errorhandler(401)(self.handle_unauthorized)
        app.errorhandler(403)(self.handle_forbidden)
        app.errorhandler(404)(self.handle_not_found)
        app.errorhandler(413)(self.handle_payload_too_large)
        app.errorhandler(429)(self.handle_too_many_requests)
        app.errorhandler(500)(self.handle_internal_server_error)
        app.errorhandler(Exception)(self.handle_generic_exception)
    
    def create_error_response(
        self, 
        error_code: str, 
        message: str, 
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ) -> Tuple[Dict[str, Any], int]:
        """Create standardized error response"""
        
        if not request_id:
            request_id = str(uuid.uuid4())[:8]
        
        error_response = {
            'error': {
                'code': error_code,
                'message': message,
                'request_id': request_id,
                'timestamp': datetime.utcnow().isoformat(),
                'status_code': status_code
            }
        }
        
        if details:
            error_response['error']['details'] = details
        
        # Log error
        self.log_error(error_code, message, status_code, details, request_id)
        
        return error_response, status_code
    
    def log_error(
        self, 
        error_code: str, 
        message: str, 
        status_code: int,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ):
        """Log error with context information"""
        
        error_entry = {
            'request_id': request_id or str(uuid.uuid4())[:8],
            'error_code': error_code,
            'message': message,
            'status_code': status_code,
            'timestamp': datetime.utcnow().isoformat(),
            'endpoint': request.endpoint if request else None,
            'method': request.method if request else None,
            'url': request.url if request else None,
            'user_agent': request.headers.get('User-Agent') if request else None,
            'ip_address': request.environ.get('REMOTE_ADDR') if request else None,
            'details': details
        }
        
        # Add to error log (with size limit)
        self.error_log.append(error_entry)
        if len(self.error_log) > self.max_error_log_size:
            self.error_log.pop(0)
        
        # Log to application logger
        log_message = f"[{request_id}] {error_code}: {message}"
        if status_code >= 500:
            logger.error(log_message, extra=error_entry)
        elif status_code >= 400:
            logger.warning(log_message, extra=error_entry)
        else:
            logger.info(log_message, extra=error_entry)
    
    def handle_bad_request(self, error):
        """Handle 400 Bad Request errors"""
        return self.create_error_response(
            'BAD_REQUEST',
            'The request was invalid or malformed',
            400,
            {'original_error': str(error)}
        )
    
    def handle_unauthorized(self, error):
        """Handle 401 Unauthorized errors"""
        return self.create_error_response(
            'UNAUTHORIZED',
            'Authentication required',
            401,
            {'original_error': str(error)}
        )
    
    def handle_forbidden(self, error):
        """Handle 403 Forbidden errors"""
        return self.create_error_response(
            'FORBIDDEN',
            'Access denied',
            403,
            {'original_error': str(error)}
        )
    
    def handle_not_found(self, error):
        """Handle 404 Not Found errors"""
        return self.create_error_response(
            'NOT_FOUND',
            'The requested resource was not found',
            404,
            {'endpoint': request.endpoint if request else None}
        )
    
    def handle_payload_too_large(self, error):
        """Handle 413 Payload Too Large errors"""
        return self.create_error_response(
            'PAYLOAD_TOO_LARGE',
            'The uploaded file or request is too large',
            413,
            {'max_size': '100MB'}
        )
    
    def handle_too_many_requests(self, error):
        """Handle 429 Too Many Requests errors"""
        return self.create_error_response(
            'RATE_LIMIT_EXCEEDED',
            'Too many requests. Please try again later',
            429,
            {'retry_after': '300 seconds'}
        )
    
    def handle_internal_server_error(self, error):
        """Handle 500 Internal Server Error"""
        request_id = str(uuid.uuid4())[:8]
        
        # Log full stack trace for internal errors
        logger.error(
            f"[{request_id}] Internal server error: {str(error)}\n"
            f"Stack trace: {traceback.format_exc()}"
        )
        
        return self.create_error_response(
            'INTERNAL_SERVER_ERROR',
            'An unexpected error occurred. Please try again later',
            500,
            request_id=request_id
        )
    
    def handle_generic_exception(self, error):
        """Handle generic exceptions"""
        request_id = str(uuid.uuid4())[:8]
        
        # Don't handle HTTP exceptions here - let specific handlers deal with them
        if isinstance(error, HTTPException):
            raise error
        
        # Log full stack trace for unexpected errors
        logger.error(
            f"[{request_id}] Unhandled exception: {str(error)}\n"
            f"Stack trace: {traceback.format_exc()}"
        )
        
        return self.create_error_response(
            'UNEXPECTED_ERROR',
            'An unexpected error occurred',
            500,
            {'error_type': type(error).__name__},
            request_id=request_id
        )
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        if not self.error_log:
            return {'total_errors': 0, 'error_breakdown': {}}
        
        error_breakdown = {}
        for error in self.error_log:
            code = error['error_code']
            if code in error_breakdown:
                error_breakdown[code] += 1
            else:
                error_breakdown[code] = 1
        
        return {
            'total_errors': len(self.error_log),
            'error_breakdown': error_breakdown,
            'recent_errors': self.error_log[-10:]  # Last 10 errors
        }

class BusinessLogicError(Exception):
    """Custom exception for business logic errors"""
    
    def __init__(self, message: str, error_code: str = 'BUSINESS_LOGIC_ERROR', details: Optional[Dict] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

class ValidationError(BusinessLogicError):
    """Custom exception for validation errors"""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        details = {}
        if field:
            details['field'] = field
        if value is not None:
            details['value'] = str(value)
        
        super().__init__(message, 'VALIDATION_ERROR', details)

class DataProcessingError(BusinessLogicError):
    """Custom exception for data processing errors"""
    
    def __init__(self, message: str, stage: Optional[str] = None, table: Optional[str] = None):
        details = {}
        if stage:
            details['stage'] = stage
        if table:
            details['table'] = table
        
        super().__init__(message, 'DATA_PROCESSING_ERROR', details)

class SecurityError(BusinessLogicError):
    """Custom exception for security-related errors"""
    
    def __init__(self, message: str, security_check: Optional[str] = None):
        details = {}
        if security_check:
            details['security_check'] = security_check
        
        super().__init__(message, 'SECURITY_ERROR', details)

def handle_business_logic_error(error_handler: ErrorHandler, error: BusinessLogicError) -> Tuple[Dict[str, Any], int]:
    """Handle business logic errors"""
    return error_handler.create_error_response(
        error.error_code,
        error.message,
        400,  # Business logic errors are usually client errors
        error.details
    )

def safe_execute(func, *args, error_handler: Optional[ErrorHandler] = None, **kwargs):
    """Safely execute a function with comprehensive error handling"""
    try:
        return func(*args, **kwargs)
    except BusinessLogicError as e:
        if error_handler:
            return handle_business_logic_error(error_handler, e)
        else:
            raise
    except Exception as e:
        error_msg = f"Unexpected error in {func.__name__}: {str(e)}"
        logger.error(error_msg + f"\nStack trace: {traceback.format_exc()}")
        
        if error_handler:
            return error_handler.create_error_response(
                'EXECUTION_ERROR',
                error_msg,
                500,
                {'function': func.__name__, 'error_type': type(e).__name__}
            )
        else:
            raise

# Context manager for safe operations
class SafeOperation:
    """Context manager for safe operations with automatic error handling"""
    
    def __init__(self, operation_name: str, error_handler: Optional[ErrorHandler] = None):
        self.operation_name = operation_name
        self.error_handler = error_handler
        self.start_time = None
        self.success = False
    
    def __enter__(self):
        self.start_time = datetime.utcnow()
        logger.info(f"Starting operation: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.utcnow() - self.start_time).total_seconds()
        
        if exc_type is None:
            self.success = True
            logger.info(f"Operation completed successfully: {self.operation_name} ({duration:.2f}s)")
        else:
            logger.error(
                f"Operation failed: {self.operation_name} ({duration:.2f}s)\n"
                f"Error: {exc_val}\n"
                f"Stack trace: {traceback.format_exc()}"
            )
            
            if self.error_handler and isinstance(exc_val, BusinessLogicError):
                # Handle business logic errors gracefully
                return True  # Suppress the exception
        
        return False  # Let other exceptions propagate