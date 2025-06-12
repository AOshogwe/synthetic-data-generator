# monitoring.py - Performance Monitoring and Analytics
import time
import psutil
import logging
from functools import wraps
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import json
from collections import defaultdict, deque
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Collect and track performance metrics"""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.generation_times = deque(maxlen=100)  # Keep last 100 generation times
        self.memory_usage = deque(maxlen=100)
        self.lock = threading.Lock()

    def record_request(self, endpoint: str, duration: float, status_code: int):
        """Record API request metrics"""
        with self.lock:
            self.request_count += 1
            if status_code >= 400:
                self.error_count += 1

            self.metrics['requests'].append({
                'endpoint': endpoint,
                'duration': duration,
                'status_code': status_code,
                'timestamp': datetime.now().isoformat()
            })

    def record_generation(self, table_name: str, rows: int, duration: float, method: str):
        """Record synthetic data generation metrics"""
        with self.lock:
            self.generation_times.append(duration)

            self.metrics['generations'].append({
                'table_name': table_name,
                'rows': rows,
                'duration': duration,
                'method': method,
                'rows_per_second': rows / duration if duration > 0 else 0,
                'timestamp': datetime.now().isoformat()
            })

    def record_memory_usage(self):
        """Record current memory usage"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024

            with self.lock:
                self.memory_usage.append({
                    'memory_mb': memory_mb,
                    'timestamp': datetime.now().isoformat()
                })
        except Exception as e:
            logger.warning(f"Could not record memory usage: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        with self.lock:
            uptime = time.time() - self.start_time

            # Calculate averages
            avg_generation_time = sum(self.generation_times) / len(
                self.generation_times) if self.generation_times else 0
            avg_memory = sum(m['memory_mb'] for m in self.memory_usage) / len(
                self.memory_usage) if self.memory_usage else 0

            # Error rate
            error_rate = (self.error_count / self.request_count * 100) if self.request_count > 0 else 0

            return {
                'uptime_seconds': uptime,
                'total_requests': self.request_count,
                'error_count': self.error_count,
                'error_rate_percent': error_rate,
                'average_generation_time': avg_generation_time,
                'average_memory_mb': avg_memory,
                'total_generations': len(self.metrics['generations']),
                'requests_per_minute': (self.request_count / uptime * 60) if uptime > 0 else 0
            }


# Global metrics instance
performance_metrics = PerformanceMetrics()


def monitor_performance(endpoint_name: str = None):
    """Decorator to monitor function performance"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                # Record successful execution
                if endpoint_name:
                    performance_metrics.record_request(endpoint_name, duration, 200)

                return result

            except Exception as e:
                duration = time.time() - start_time

                # Record failed execution
                if endpoint_name:
                    performance_metrics.record_request(endpoint_name, duration, 500)

                raise e

        return wrapper

    return decorator


class SystemMonitor:
    """Monitor system resources and health"""

    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.monitoring = False
        self.monitor_thread = None
        self.alerts = []

        # Thresholds
        self.memory_threshold = 80  # Percent
        self.cpu_threshold = 80  # Percent
        self.disk_threshold = 90  # Percent

    def start_monitoring(self):
        """Start background monitoring"""
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("System monitoring started")

    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("System monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                self._check_system_health()
                performance_metrics.record_memory_usage()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.check_interval)

    def _check_system_health(self):
        """Check system health metrics"""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            if memory.percent > self.memory_threshold:
                self._add_alert('high_memory', f"Memory usage: {memory.percent:.1f}%")

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > self.cpu_threshold:
                self._add_alert('high_cpu', f"CPU usage: {cpu_percent:.1f}%")

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            if disk_percent > self.disk_threshold:
                self._add_alert('high_disk', f"Disk usage: {disk_percent:.1f}%")

        except Exception as e:
            logger.error(f"Error checking system health: {e}")

    def _add_alert(self, alert_type: str, message: str):
        """Add alert if not already present"""
        # Check if this alert was recently added
        recent_alerts = [a for a in self.alerts[-10:] if a['type'] == alert_type]
        if not recent_alerts:
            alert = {
                'type': alert_type,
                'message': message,
                'timestamp': datetime.now().isoformat()
            }
            self.alerts.append(alert)
            logger.warning(f"System alert: {message}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                'memory': {
                    'total_gb': memory.total / (1024 ** 3),
                    'available_gb': memory.available / (1024 ** 3),
                    'used_percent': memory.percent
                },
                'cpu': {
                    'cores': psutil.cpu_count(),
                    'usage_percent': psutil.cpu_percent(interval=1)
                },
                'disk': {
                    'total_gb': disk.total / (1024 ** 3),
                    'free_gb': disk.free / (1024 ** 3),
                    'used_percent': (disk.used / disk.total) * 100
                },
                'alerts': self.alerts[-5:],  # Last 5 alerts
                'uptime': time.time() - performance_metrics.start_time
            }
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}


class DataProcessingProfiler:
    """Profile data processing operations"""

    def __init__(self):
        self.profiles = []

    def profile_operation(self, operation_name: str):
        """Decorator to profile data processing operations"""

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = self._get_memory_usage()

                try:
                    result = func(*args, **kwargs)

                    end_time = time.time()
                    end_memory = self._get_memory_usage()

                    # Extract data size information if available
                    data_size = self._extract_data_size(args, kwargs, result)

                    profile = {
                        'operation': operation_name,
                        'duration': end_time - start_time,
                        'memory_before_mb': start_memory,
                        'memory_after_mb': end_memory,
                        'memory_delta_mb': end_memory - start_memory,
                        'data_size': data_size,
                        'timestamp': datetime.now().isoformat(),
                        'success': True
                    }

                    self.profiles.append(profile)

                    # Log performance info
                    logger.info(f"Profiled {operation_name}: {profile['duration']:.2f}s, "
                                f"{profile['memory_delta_mb']:.1f}MB memory delta")

                    return result

                except Exception as e:
                    end_time = time.time()
                    end_memory = self._get_memory_usage()

                    profile = {
                        'operation': operation_name,
                        'duration': end_time - start_time,
                        'memory_before_mb': start_memory,
                        'memory_after_mb': end_memory,
                        'memory_delta_mb': end_memory - start_memory,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat(),
                        'success': False
                    }

                    self.profiles.append(profile)
                    raise

            return wrapper

        return decorator

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 ** 2)
        except:
            return 0.0

    def _extract_data_size(self, args, kwargs, result) -> Dict[str, Any]:
        """Extract data size information from function arguments and result"""
        size_info = {}

        # Check for pandas DataFrames in arguments
        for i, arg in enumerate(args):
            if hasattr(arg, 'shape'):  # Likely a DataFrame or numpy array
                size_info[f'arg_{i}_shape'] = arg.shape
                if hasattr(arg, 'memory_usage'):
                    size_info[f'arg_{i}_memory_mb'] = arg.memory_usage(deep=True).sum() / (1024 ** 2)

        # Check result
        if hasattr(result, 'shape'):
            size_info['result_shape'] = result.shape
            if hasattr(result, 'memory_usage'):
                size_info['result_memory_mb'] = result.memory_usage(deep=True).sum() / (1024 ** 2)

        return size_info

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        if not self.profiles:
            return {'message': 'No profiling data available'}

        # Aggregate by operation
        operation_stats = defaultdict(list)
        for profile in self.profiles:
            operation_stats[profile['operation']].append(profile)

        report = {
            'total_operations': len(self.profiles),
            'successful_operations': sum(1 for p in self.profiles if p['success']),
            'failed_operations': sum(1 for p in self.profiles if not p['success']),
            'operation_summary': {}
        }

        for operation, profiles in operation_stats.items():
            successful_profiles = [p for p in profiles if p['success']]

            if successful_profiles:
                durations = [p['duration'] for p in successful_profiles]
                memory_deltas = [p['memory_delta_mb'] for p in successful_profiles]

                report['operation_summary'][operation] = {
                    'count': len(profiles),
                    'success_rate': len(successful_profiles) / len(profiles),
                    'avg_duration': sum(durations) / len(durations),
                    'max_duration': max(durations),
                    'min_duration': min(durations),
                    'avg_memory_delta_mb': sum(memory_deltas) / len(memory_deltas),
                    'max_memory_delta_mb': max(memory_deltas)
                }

        return report


class AnalyticsCollector:
    """Collect analytics and usage patterns"""

    def __init__(self):
        self.events = []
        self.session_data = {}
        self.lock = threading.Lock()

    def track_event(self, event_type: str, properties: Dict[str, Any] = None):
        """Track an analytics event"""
        with self.lock:
            event = {
                'type': event_type,
                'properties': properties or {},
                'timestamp': datetime.now().isoformat(),
                'session_id': properties.get('session_id') if properties else None
            }
            self.events.append(event)

    def track_data_upload(self, file_count: int, total_size_mb: float, file_types: List[str], session_id: str):
        """Track data upload event"""
        self.track_event('data_upload', {
            'file_count': file_count,
            'total_size_mb': total_size_mb,
            'file_types': file_types,
            'session_id': session_id
        })

    def track_generation(self, method: str, rows: int, columns: int, duration: float, session_id: str):
        """Track synthetic data generation event"""
        self.track_event('data_generation', {
            'method': method,
            'rows': rows,
            'columns': columns,
            'duration': duration,
            'rows_per_second': rows / duration if duration > 0 else 0,
            'session_id': session_id
        })

    def track_export(self, format_type: str, tables: int, total_rows: int, session_id: str):
        """Track data export event"""
        self.track_event('data_export', {
            'format': format_type,
            'tables': tables,
            'total_rows': total_rows,
            'session_id': session_id
        })

    def get_usage_analytics(self) -> Dict[str, Any]:
        """Get usage analytics summary"""
        with self.lock:
            if not self.events:
                return {'message': 'No analytics data available'}

            # Count events by type
            event_counts = defaultdict(int)
            for event in self.events:
                event_counts[event['type']] += 1

            # Get unique sessions
            sessions = set()
            for event in self.events:
                if event.get('session_id'):
                    sessions.add(event['session_id'])

            # Calculate time range
            timestamps = [datetime.fromisoformat(event['timestamp']) for event in self.events]
            time_range = max(timestamps) - min(timestamps) if timestamps else timedelta(0)

            return {
                'total_events': len(self.events),
                'event_types': dict(event_counts),
                'unique_sessions': len(sessions),
                'time_range_hours': time_range.total_seconds() / 3600,
                'events_per_hour': len(self.events) / (
                            time_range.total_seconds() / 3600) if time_range.total_seconds() > 0 else 0
            }

    def export_analytics(self, file_path: str):
        """Export analytics data to file"""
        with self.lock:
            analytics_data = {
                'events': self.events,
                'summary': self.get_usage_analytics(),
                'exported_at': datetime.now().isoformat()
            }

            with open(file_path, 'w') as f:
                json.dump(analytics_data, f, indent=2, default=str)


# Global instances
system_monitor = SystemMonitor()
profiler = DataProcessingProfiler()
analytics = AnalyticsCollector()


# Flask integration functions
def init_monitoring(app):
    """Initialize monitoring for Flask app"""

    @app.before_request
    def before_request():
        """Record request start time"""
        from flask import g
        g.start_time = time.time()

    @app.after_request
    def after_request(response):
        """Record request metrics"""
        from flask import g, request

        if hasattr(g, 'start_time'):
            duration = time.time() - g.start_time
            performance_metrics.record_request(
                endpoint=request.endpoint or 'unknown',
                duration=duration,
                status_code=response.status_code
            )

        return response

    # Add monitoring endpoints
    @app.route('/api/metrics')
    def get_metrics():
        """Get performance metrics"""
        return {
            'performance': performance_metrics.get_summary(),
            'system': system_monitor.get_system_status(),
            'analytics': analytics.get_usage_analytics()
        }

    @app.route('/api/health/detailed')
    def detailed_health():
        """Detailed health check with metrics"""
        status = system_monitor.get_system_status()
        metrics = performance_metrics.get_summary()

        # Determine overall health
        health_status = 'healthy'

        if status.get('memory', {}).get('used_percent', 0) > 90:
            health_status = 'degraded'

        if metrics.get('error_rate_percent', 0) > 10:
            health_status = 'unhealthy'

        return {
            'status': health_status,
            'timestamp': datetime.now().isoformat(),
            'uptime': metrics.get('uptime_seconds', 0),
            'system': status,
            'performance': metrics
        }

    # Start system monitoring
    system_monitor.start_monitoring()

    logger.info("Monitoring initialized for Flask app")


def cleanup_monitoring():
    """Cleanup monitoring resources"""
    system_monitor.stop_monitoring()
    logger.info("Monitoring cleanup completed")