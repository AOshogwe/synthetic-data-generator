# config.py - Configuration Management System
import os
from typing import Dict, Any, Optional
import json
from pathlib import Path


class ConfigManager:
    """Centralized configuration management"""

    def __init__(self):
        self.config = {}
        self.load_default_config()
        self.load_environment_config()

    def load_default_config(self):
        """Load default configuration values"""
        self.config = {
            # Flask Configuration
            'FLASK': {
                'SECRET_KEY': 'dev-secret-key-change-in-production',
                'DEBUG': False,
                'TESTING': False,
                'MAX_CONTENT_LENGTH': 100 * 1024 * 1024,  # 100MB
                'UPLOAD_FOLDER': 'uploads',
                'OUTPUT_FOLDER': 'outputs',
                'ALLOWED_EXTENSIONS': ['csv', 'xlsx', 'json'],
                'SESSION_TIMEOUT': 3600,  # 1 hour
            },

            # CORS Configuration
            'CORS': {
                'ORIGINS': '*',
                'METHODS': ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
                'ALLOW_HEADERS': ['Content-Type', 'Authorization'],
                'EXPOSE_HEADERS': ['Content-Disposition'],
                'SUPPORTS_CREDENTIALS': False
            },

            # Logging Configuration
            'LOGGING': {
                'LEVEL': 'INFO',
                'FORMAT': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'FILE': 'app.log',
                'MAX_BYTES': 10 * 1024 * 1024,  # 10MB
                'BACKUP_COUNT': 5,
                'CONSOLE_OUTPUT': True
            },

            # Data Processing Configuration
            'DATA_PROCESSING': {
                'MAX_COLUMNS': 1000,
                'MAX_ROWS_PREVIEW': 1000,
                'CHUNK_SIZE': 10000,
                'MEMORY_THRESHOLD': 500 * 1024 * 1024,  # 500MB
                'TEMP_FILE_CLEANUP': True,
                'AUTO_DETECT_ENCODING': True,
                'SAMPLE_SIZE_FOR_INFERENCE': 10000
            },

            # Synthetic Data Generation
            'GENERATION': {
                'DEFAULT_METHOD': 'auto',
                'MAX_SYNTHETIC_ROWS': 1000000,
                'DEFAULT_PERTURBATION_FACTOR': 0.2,
                'MIN_SAMPLES_FOR_STATISTICAL': 100,
                'CORRELATION_THRESHOLD': 0.7,
                'TEMPORAL_CONSISTENCY_THRESHOLD': 0.8,
                'PRIVACY_SCORE_THRESHOLD': 0.7
            },

            # Security Configuration
            'SECURITY': {
                'RATE_LIMIT': {
                    'ENABLED': True,
                    'REQUESTS_PER_MINUTE': 100,
                    'BURST_LIMIT': 200
                },
                'FILE_VALIDATION': {
                    'SCAN_FOR_MALWARE': False,
                    'CHECK_FILE_HEADERS': True,
                    'QUARANTINE_SUSPICIOUS': True
                },
                'SESSION_SECURITY': {
                    'SECURE_COOKIES': True,
                    'HTTPONLY_COOKIES': True,
                    'SAMESITE': 'Lax'
                }
            },

            # Performance Configuration
            'PERFORMANCE': {
                'WORKERS': 4,
                'TIMEOUT': 300,
                'KEEP_ALIVE': 2,
                'MAX_REQUESTS': 1000,
                'MAX_REQUESTS_JITTER': 50,
                'PRELOAD_APP': True,
                'WORKER_CLASS': 'sync',
                'ENABLE_CACHING': True,
                'CACHE_TIMEOUT': 300
            },

            # Monitoring Configuration
            'MONITORING': {
                'HEALTH_CHECK_INTERVAL': 30,
                'METRICS_ENABLED': True,
                'PERFORMANCE_TRACKING': True,
                'ERROR_REPORTING': True,
                'USAGE_ANALYTICS': False
            },

            # Feature Flags
            'FEATURES': {
                'ADVANCED_PRIVACY_METHODS': True,
                'REAL_TIME_PROGRESS': True,
                'BATCH_PROCESSING': True,
                'API_VERSIONING': True,
                'EXPORT_FORMATS': ['csv', 'excel', 'json', 'parquet'],
                'DATABASE_CONNECTORS': ['postgresql', 'mysql', 'sqlite'],
                'CLOUD_STORAGE': False,
                'NOTIFICATION_SYSTEM': False
            },

            # Development Configuration
            'DEVELOPMENT': {
                'HOT_RELOAD': False,
                'DEBUG_TOOLBAR': False,
                'PROFILING': False,
                'MOCK_DATA': False,
                'API_DOCUMENTATION': True
            }
        }

    def load_environment_config(self):
        """Load configuration from environment variables"""
        # Flask settings
        if os.getenv('SECRET_KEY'):
            self.config['FLASK']['SECRET_KEY'] = os.getenv('SECRET_KEY')

        if os.getenv('FLASK_DEBUG'):
            self.config['FLASK']['DEBUG'] = os.getenv('FLASK_DEBUG').lower() == 'true'

        if os.getenv('MAX_CONTENT_LENGTH'):
            self.config['FLASK']['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH'))

        # CORS settings
        if os.getenv('CORS_ORIGINS'):
            origins = os.getenv('CORS_ORIGINS')
            self.config['CORS']['ORIGINS'] = origins.split(',') if ',' in origins else origins

        # Logging settings
        if os.getenv('LOG_LEVEL'):
            self.config['LOGGING']['LEVEL'] = os.getenv('LOG_LEVEL')

        # Performance settings
        if os.getenv('WORKERS'):
            self.config['PERFORMANCE']['WORKERS'] = int(os.getenv('WORKERS'))

        if os.getenv('TIMEOUT'):
            self.config['PERFORMANCE']['TIMEOUT'] = int(os.getenv('TIMEOUT'))

        # Database settings
        if os.getenv('DATABASE_URL'):
            self.config['DATABASE'] = {
                'URL': os.getenv('DATABASE_URL'),
                'POOL_SIZE': int(os.getenv('DB_POOL_SIZE', 10)),
                'MAX_OVERFLOW': int(os.getenv('DB_MAX_OVERFLOW', 20)),
                'POOL_TIMEOUT': int(os.getenv('DB_POOL_TIMEOUT', 30))
            }

        # Cloud storage settings
        if os.getenv('CLOUD_STORAGE_BUCKET'):
            self.config['CLOUD_STORAGE'] = {
                'PROVIDER': os.getenv('CLOUD_PROVIDER', 'aws'),
                'BUCKET': os.getenv('CLOUD_STORAGE_BUCKET'),
                'ACCESS_KEY': os.getenv('CLOUD_ACCESS_KEY'),
                'SECRET_KEY': os.getenv('CLOUD_SECRET_KEY'),
                'REGION': os.getenv('CLOUD_REGION', 'us-east-1')
            }

    def get(self, path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        try:
            keys = path.split('.')
            value = self.config

            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default

            return value
        except Exception:
            return default

    def set(self, path: str, value: Any) -> None:
        """Set configuration value using dot notation"""
        keys = path.split('.')
        config = self.config

        # Navigate to the parent dictionary
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        # Set the final value
        config[keys[-1]] = value

    def update(self, new_config: Dict[str, Any]) -> None:
        """Update configuration with new values"""

        def deep_update(base_dict: dict, update_dict: dict) -> dict:
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
            return base_dict

        deep_update(self.config, new_config)

    def load_from_file(self, file_path: str) -> bool:
        """Load configuration from JSON file"""
        try:
            path = Path(file_path)
            if path.exists() and path.suffix.lower() == '.json':
                with open(path, 'r') as f:
                    file_config = json.load(f)
                    self.update(file_config)
                return True
        except Exception as e:
            print(f"Error loading config from file {file_path}: {e}")
        return False

    def save_to_file(self, file_path: str) -> bool:
        """Save current configuration to JSON file"""
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, 'w') as f:
                json.dump(self.config, f, indent=2, default=str)
            return True
        except Exception as e:
            print(f"Error saving config to file {file_path}: {e}")
        return False

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []

        # Validate required settings
        if not self.get('FLASK.SECRET_KEY') or self.get('FLASK.SECRET_KEY') == 'dev-secret-key-change-in-production':
            if not self.get('FLASK.DEBUG'):
                issues.append("SECRET_KEY should be set to a secure value in production")

        # Validate file size limits
        max_content_length = self.get('FLASK.MAX_CONTENT_LENGTH')
        if max_content_length and max_content_length > 1024 * 1024 * 1024:  # 1GB
            issues.append("MAX_CONTENT_LENGTH is very large and may cause memory issues")

        # Validate worker configuration
        workers = self.get('PERFORMANCE.WORKERS')
        if workers and workers > 16:
            issues.append("High number of workers may cause resource issues")

        # Validate directories
        upload_folder = self.get('FLASK.UPLOAD_FOLDER')
        output_folder = self.get('FLASK.OUTPUT_FOLDER')

        for folder_name, folder_path in [('UPLOAD_FOLDER', upload_folder), ('OUTPUT_FOLDER', output_folder)]:
            if folder_path:
                try:
                    Path(folder_path).mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    issues.append(f"Cannot create {folder_name} directory {folder_path}: {e}")

        return issues

    def get_flask_config(self) -> Dict[str, Any]:
        """Get Flask-specific configuration"""
        flask_config = {}

        # Map our config to Flask config names
        mapping = {
            'SECRET_KEY': 'FLASK.SECRET_KEY',
            'DEBUG': 'FLASK.DEBUG',
            'TESTING': 'FLASK.TESTING',
            'MAX_CONTENT_LENGTH': 'FLASK.MAX_CONTENT_LENGTH',
            'UPLOAD_FOLDER': 'FLASK.UPLOAD_FOLDER',
            'OUTPUT_FOLDER': 'FLASK.OUTPUT_FOLDER'
        }

        for flask_key, config_path in mapping.items():
            value = self.get(config_path)
            if value is not None:
                flask_config[flask_key] = value

        return flask_config

    def get_cors_config(self) -> Dict[str, Any]:
        """Get CORS-specific configuration"""
        return self.get('CORS', {})

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging-specific configuration"""
        return self.get('LOGGING', {})

    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.get('FLASK.DEBUG', False) or os.getenv('FLASK_ENV') == 'development'

    def is_production(self) -> bool:
        """Check if running in production mode"""
        return not self.is_development()

    def get_feature_flag(self, feature: str) -> bool:
        """Get feature flag value"""
        return self.get(f'FEATURES.{feature}', False)

    def __str__(self) -> str:
        """String representation (excluding sensitive data)"""
        safe_config = self.config.copy()

        # Remove sensitive information
        if 'FLASK' in safe_config and 'SECRET_KEY' in safe_config['FLASK']:
            safe_config['FLASK']['SECRET_KEY'] = '***HIDDEN***'

        if 'DATABASE' in safe_config:
            for key in ['URL', 'PASSWORD']:
                if key in safe_config['DATABASE']:
                    safe_config['DATABASE'][key] = '***HIDDEN***'

        if 'CLOUD_STORAGE' in safe_config:
            for key in ['ACCESS_KEY', 'SECRET_KEY']:
                if key in safe_config['CLOUD_STORAGE']:
                    safe_config['CLOUD_STORAGE'][key] = '***HIDDEN***'

        return json.dumps(safe_config, indent=2, default=str)


# Global configuration instance
config_manager = ConfigManager()


# Convenience functions
def get_config(path: str, default: Any = None) -> Any:
    """Get configuration value"""
    return config_manager.get(path, default)


def set_config(path: str, value: Any) -> None:
    """Set configuration value"""
    config_manager.set(path, value)


def load_config_file(file_path: str) -> bool:
    """Load configuration from file"""
    return config_manager.load_from_file(file_path)


def is_feature_enabled(feature: str) -> bool:
    """Check if feature is enabled"""
    return config_manager.get_feature_flag(feature)


def validate_config() -> List[str]:
    """Validate current configuration"""
    return config_manager.validate()