# startup.py - Startup validation and system checks
import os
import sys
import subprocess
import importlib
import logging
from pathlib import Path
import requests
import time
from datetime import datetime

# Configure logging only if not already configured
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SystemValidator:
    """Comprehensive system validation and startup checks"""

    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = 0

    def run_all_checks(self):
        """Run all system validation checks"""
        logger.info("🚀 Starting Synthetic Data Generator System Validation")
        logger.info("=" * 60)

        # Core system checks
        self.check_python_version()
        self.check_dependencies()
        self.check_file_structure()
        self.check_environment_variables()

        # Application checks
        self.check_configuration()
        self.check_directories()
        self.check_permissions()

        # Optional checks
        self.check_memory_requirements()
        self.check_network_connectivity()

        # Security checks
        self.check_security_settings()

        # Print summary
        self.print_summary()

        return self.checks_failed == 0

    def check_python_version(self):
        """Validate Python version"""
        logger.info("🐍 Checking Python version...")

        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"

        if version.major == 3 and version.minor >= 9:
            logger.info(f"✅ Python version {version_str} is supported")
            self.checks_passed += 1
        elif version.major == 3 and version.minor >= 7:
            logger.warning(f"⚠️  Python version {version_str} is acceptable but not optimal")
            logger.warning("   Recommended: Python 3.9+")
            self.warnings += 1
        else:
            logger.error(f"❌ Python version {version_str} is not supported")
            logger.error("   Minimum required: Python 3.7")
            self.checks_failed += 1

    def check_dependencies(self):
        """Check if all required packages are installed"""
        logger.info("📦 Checking dependencies...")

        # Core dependencies
        core_deps = [
            'flask', 'pandas', 'numpy', 'sklearn', 'scipy'
        ]

        # Optional dependencies
        optional_deps = [
            'openpyxl', 'sqlalchemy', 'chardet', 'requests'
        ]

        missing_core = []
        missing_optional = []

        for dep in core_deps:
            try:
                importlib.import_module(dep)
            except ImportError:
                missing_core.append(dep)

        for dep in optional_deps:
            try:
                importlib.import_module(dep)
            except ImportError:
                missing_optional.append(dep)

        if not missing_core:
            logger.info("✅ All core dependencies are installed")
            self.checks_passed += 1
        else:
            logger.error(f"❌ Missing core dependencies: {', '.join(missing_core)}")
            logger.error("   Run: pip install -r requirements.txt")
            self.checks_failed += 1

        if missing_optional:
            logger.warning(f"⚠️  Missing optional dependencies: {', '.join(missing_optional)}")
            logger.warning("   Some features may not be available")
            self.warnings += 1
        else:
            logger.info("✅ All optional dependencies are installed")

    def check_file_structure(self):
        """Validate required files and directories exist"""
        logger.info("📁 Checking file structure...")

        required_files = [
            'app.py',
            'synthetic_data_pipeline.py',
            'config.py',
            'requirements.txt'
        ]

        optional_files = [
            'Dockerfile',
            'railway.json',
            '.env.example',
            'test_basic.py'
        ]

        missing_required = []
        missing_optional = []

        for file in required_files:
            if not Path(file).exists():
                missing_required.append(file)

        for file in optional_files:
            if not Path(file).exists():
                missing_optional.append(file)

        if not missing_required:
            logger.info("✅ All required files are present")
            self.checks_passed += 1
        else:
            logger.error(f"❌ Missing required files: {', '.join(missing_required)}")
            self.checks_failed += 1

        if missing_optional:
            logger.warning(f"⚠️  Missing optional files: {', '.join(missing_optional)}")
            self.warnings += 1

    def check_environment_variables(self):
        """Check environment variables"""
        logger.info("🔧 Checking environment variables...")

        # Check for production-critical variables
        if os.getenv('FLASK_ENV') == 'production':
            secret_key = os.getenv('SECRET_KEY')
            if not secret_key or secret_key == 'dev-secret-key-change-in-production':
                logger.error("❌ SECRET_KEY must be set for production")
                self.checks_failed += 1
            else:
                logger.info("✅ SECRET_KEY is properly configured")
                self.checks_passed += 1
        else:
            logger.info("ℹ️  Development mode - SECRET_KEY check skipped")

        # Check other environment variables
        env_vars = {
            'PORT': os.getenv('PORT', '5000'),
            'FLASK_ENV': os.getenv('FLASK_ENV', 'development'),
            'LOG_LEVEL': os.getenv('LOG_LEVEL', 'INFO'),
            'MAX_CONTENT_LENGTH': os.getenv('MAX_CONTENT_LENGTH', '100MB')
        }

        logger.info("Environment configuration:")
        for key, value in env_vars.items():
            logger.info(f"  {key}: {value}")

    def check_configuration(self):
        """Validate application configuration"""
        logger.info("⚙️  Checking application configuration...")

        try:
            from config import config_manager

            # Validate configuration
            issues = config_manager.validate()

            if not issues:
                logger.info("✅ Application configuration is valid")
                self.checks_passed += 1
            else:
                logger.warning("⚠️  Configuration issues found:")
                for issue in issues:
                    logger.warning(f"   - {issue}")
                self.warnings += len(issues)

        except Exception as e:
            logger.error(f"❌ Error loading configuration: {e}")
            self.checks_failed += 1

    def check_directories(self):
        """Ensure required directories exist and are writable"""
        logger.info("📂 Checking directories...")

        directories = ['uploads', 'outputs', 'logs']

        for directory in directories:
            try:
                Path(directory).mkdir(exist_ok=True)

                # Test write permissions
                test_file = Path(directory) / '.test_write'
                test_file.write_text('test')
                test_file.unlink()

                logger.info(f"✅ Directory '{directory}' is accessible and writable")

            except Exception as e:
                logger.error(f"❌ Cannot create/write to directory '{directory}': {e}")
                self.checks_failed += 1

        if self.checks_failed == 0:
            self.checks_passed += 1

    def check_permissions(self):
        """Check file and directory permissions"""
        logger.info("🔐 Checking permissions...")

        # Check if running as root (not recommended)
        if os.getuid() == 0 if hasattr(os, 'getuid') else False:
            logger.warning("⚠️  Running as root is not recommended for security")
            self.warnings += 1

        # Check file permissions for sensitive files
        sensitive_files = ['.env', 'config.py']

        for file in sensitive_files:
            if Path(file).exists():
                stat = Path(file).stat()
                # Check if file is readable by others (Unix-like systems)
                if hasattr(stat, 'st_mode') and stat.st_mode & 0o044:
                    logger.warning(f"⚠️  File '{file}' is readable by others")
                    self.warnings += 1

        logger.info("✅ Permission checks completed")
        self.checks_passed += 1

    def check_memory_requirements(self):
        """Check available memory"""
        logger.info("💾 Checking memory requirements...")

        try:
            import psutil

            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024 ** 3)

            if available_gb >= 2:
                logger.info(f"✅ Sufficient memory available: {available_gb:.1f}GB")
                self.checks_passed += 1
            elif available_gb >= 1:
                logger.warning(f"⚠️  Low memory available: {available_gb:.1f}GB")
                logger.warning("   Consider upgrading for better performance")
                self.warnings += 1
            else:
                logger.error(f"❌ Insufficient memory: {available_gb:.1f}GB")
                logger.error("   Minimum recommended: 1GB")
                self.checks_failed += 1

        except ImportError:
            logger.info("ℹ️  psutil not available - memory check skipped")
        except Exception as e:
            logger.warning(f"⚠️  Could not check memory: {e}")
            self.warnings += 1

    def check_network_connectivity(self):
        """Check network connectivity (optional)"""
        logger.info("🌐 Checking network connectivity...")

        test_urls = [
            'https://httpbin.org/status/200',
            'https://api.github.com',
        ]

        for url in test_urls:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    logger.info(f"✅ Network connectivity verified: {url}")
                    self.checks_passed += 1
                    break
            except Exception:
                continue
        else:
            logger.warning("⚠️  Network connectivity could not be verified")
            logger.warning("   Some features may not work properly")
            self.warnings += 1

    def check_security_settings(self):
        """Check security-related settings"""
        logger.info("🛡️  Checking security settings...")

        security_checks = 0

        # Check if HTTPS is enforced (in production)
        if os.getenv('FLASK_ENV') == 'production':
            if not os.getenv('FORCE_HTTPS'):
                logger.warning("⚠️  HTTPS enforcement not configured")
                self.warnings += 1
            else:
                security_checks += 1

        # Check CORS settings
        cors_origins = os.getenv('CORS_ORIGINS', '*')
        if cors_origins == '*' and os.getenv('FLASK_ENV') == 'production':
            logger.warning("⚠️  CORS allows all origins in production")
            logger.warning("   Consider restricting to specific domains")
            self.warnings += 1
        else:
            security_checks += 1

        # Check for debug mode in production
        if os.getenv('FLASK_ENV') == 'production' and os.getenv('FLASK_DEBUG') == 'True':
            logger.error("❌ Debug mode enabled in production")
            self.checks_failed += 1
        else:
            security_checks += 1

        if security_checks > 0:
            logger.info(f"✅ Security checks passed: {security_checks}")
            self.checks_passed += 1

    def print_summary(self):
        """Print validation summary"""
        logger.info("=" * 60)
        logger.info("📊 VALIDATION SUMMARY")
        logger.info("=" * 60)

        total_checks = self.checks_passed + self.checks_failed

        if self.checks_failed == 0:
            logger.info("🎉 ALL CHECKS PASSED!")
            logger.info(f"✅ Passed: {self.checks_passed}")
            if self.warnings > 0:
                logger.info(f"⚠️  Warnings: {self.warnings}")
            logger.info("")
            logger.info("🚀 System is ready for deployment!")

        else:
            logger.error("❌ VALIDATION FAILED!")
            logger.error(f"❌ Failed: {self.checks_failed}")
            logger.info(f"✅ Passed: {self.checks_passed}")
            if self.warnings > 0:
                logger.info(f"⚠️  Warnings: {self.warnings}")
            logger.error("")
            logger.error("🛠️  Please fix the issues above before proceeding")


def create_sample_data():
    """Create sample data files for testing"""
    logger.info("📄 Creating sample data files...")

    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta

    np.random.seed(42)

    # Sample healthcare data
    n_patients = 1000

    healthcare_data = pd.DataFrame({
        'patient_id': range(1, n_patients + 1),
        'first_name': [f'Patient_{i}' for i in range(1, n_patients + 1)],
        'last_name': [f'Lastname_{i}' for i in range(1, n_patients + 1)],
        'age': np.random.randint(18, 85, n_patients),
        'gender': np.random.choice(['M', 'F'], n_patients),
        'admission_date': pd.date_range('2023-01-01', periods=n_patients, freq='H'),
        'discharge_date': pd.date_range('2023-01-01', periods=n_patients, freq='H') + pd.Timedelta(days=3),
        'diagnosis': np.random.choice(['Diabetes', 'Hypertension', 'Heart Disease', 'Cancer'], n_patients),
        'treatment_cost': np.random.normal(5000, 2000, n_patients),
        'insurance_type': np.random.choice(['Private', 'Medicare', 'Medicaid', 'Uninsured'], n_patients),
        'zip_code': np.random.randint(10000, 99999, n_patients),
        'email': [f'patient{i}@hospital.com' for i in range(1, n_patients + 1)],
        'phone': [f'555-{np.random.randint(100, 999)}-{np.random.randint(1000, 9999)}' for _ in range(n_patients)]
    })

    # Sample employee data
    n_employees = 500

    employee_data = pd.DataFrame({
        'employee_id': range(1, n_employees + 1),
        'name': [f'Employee {i}' for i in range(1, n_employees + 1)],
        'department': np.random.choice(['IT', 'Sales', 'Marketing', 'HR', 'Finance'], n_employees),
        'position': np.random.choice(['Manager', 'Senior', 'Junior', 'Intern'], n_employees),
        'salary': np.random.normal(60000, 20000, n_employees),
        'hire_date': pd.date_range('2020-01-01', periods=n_employees, freq='D'),
        'birth_date': pd.date_range('1960-01-01', '2000-01-01', periods=n_employees),
        'address': [f'{np.random.randint(100, 999)} Main St, City, State {np.random.randint(10000, 99999)}'
                    for _ in range(n_employees)],
        'is_active': np.random.choice([True, False], n_employees, p=[0.9, 0.1])
    })

    # Sample sales data
    n_sales = 2000

    sales_data = pd.DataFrame({
        'sale_id': range(1, n_sales + 1),
        'customer_name': [f'Customer {i}' for i in range(1, n_sales + 1)],
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], n_sales),
        'sale_amount': np.random.exponential(100, n_sales),
        'sale_date': pd.date_range('2023-01-01', periods=n_sales, freq='H'),
        'salesperson_id': np.random.randint(1, 51, n_sales),
        'customer_age': np.random.randint(18, 80, n_sales),
        'customer_city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston'], n_sales),
        'payment_method': np.random.choice(['Credit Card', 'Cash', 'Debit Card', 'PayPal'], n_sales)
    })

    # Create sample_data directory
    sample_dir = Path('sample_data')
    sample_dir.mkdir(exist_ok=True)

    # Save sample files
    healthcare_data.to_csv(sample_dir / 'healthcare_sample.csv', index=False)
    employee_data.to_csv(sample_dir / 'employee_sample.csv', index=False)
    sales_data.to_csv(sample_dir / 'sales_sample.csv', index=False)

    # Create Excel file with multiple sheets
    with pd.ExcelWriter(sample_dir / 'multi_sheet_sample.xlsx') as writer:
        healthcare_data.head(100).to_excel(writer, sheet_name='Healthcare', index=False)
        employee_data.head(100).to_excel(writer, sheet_name='Employees', index=False)
        sales_data.head(100).to_excel(writer, sheet_name='Sales', index=False)

    logger.info(f"✅ Sample data files created in '{sample_dir}':")
    logger.info(f"   - healthcare_sample.csv ({len(healthcare_data)} rows)")
    logger.info(f"   - employee_sample.csv ({len(employee_data)} rows)")
    logger.info(f"   - sales_sample.csv ({len(sales_data)} rows)")
    logger.info(f"   - multi_sheet_sample.xlsx (combined)")


def test_basic_functionality():
    """Test basic application functionality"""
    logger.info("🧪 Testing basic functionality...")

    try:
        # Test imports
        from synthetic_data_pipeline import SyntheticDataGenerator
        from data_processing import DataProcessor
        from config import config_manager

        logger.info("✅ All modules import successfully")

        # Test configuration
        config = config_manager.get('FLASK.SECRET_KEY')
        if config:
            logger.info("✅ Configuration system working")

        # Test data processor
        processor = DataProcessor()
        logger.info("✅ DataProcessor initialized")

        # Test generator
        generator = SyntheticDataGenerator()
        logger.info("✅ SyntheticDataGenerator initialized")

        logger.info("✅ Basic functionality test passed")
        return True

    except Exception as e:
        logger.error(f"❌ Basic functionality test failed: {e}")
        return False


def main():
    """Main startup validation function"""
    print("🔮 Synthetic Data Generator - System Validation")
    print("=" * 60)

    validator = SystemValidator()

    # Run validation
    validation_passed = validator.run_all_checks()

    if validation_passed:
        # Create sample data if requested
        print("\n" + "=" * 60)
        create_sample = input("📄 Create sample data files for testing? (y/N): ").lower().strip()
        if create_sample.startswith('y'):
            create_sample_data()

        # Test basic functionality
        print("\n" + "=" * 60)
        test_functionality = input("🧪 Run basic functionality tests? (y/N): ").lower().strip()
        if test_functionality.startswith('y'):
            test_basic_functionality()

        print("\n" + "=" * 60)
        print("🚀 System validation completed successfully!")
        print("💡 Next steps:")
        print("   1. python app.py                    # Start the application")
        print("   2. python test_basic.py             # Run basic tests")
        print("   3. railway up                       # Deploy to Railway")
        print("   4. Open http://localhost:5000       # Access the web interface")
        print("=" * 60)

        return True
    else:
        print("\n" + "=" * 60)
        print("❌ System validation failed!")
        print("🛠️  Please fix the issues above and run this script again.")
        print("=" * 60)

        return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)