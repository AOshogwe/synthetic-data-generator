# tests/test_synthetic_data_generator.py
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
import io
from datetime import datetime, timedelta

# Import our modules
import sys

sys.path.append('..')

from synthetic_data_pipeline import SyntheticDataGenerator
from data_processing import DataProcessor
from config import ConfigManager


class TestSyntheticDataGenerator:
    """Test suite for SyntheticDataGenerator"""

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample dataframe for testing"""
        np.random.seed(42)
        return pd.DataFrame({
            'id': range(1, 101),
            'name': [f'Person_{i}' for i in range(1, 101)],
            'age': np.random.randint(18, 80, 100),
            'salary': np.random.normal(50000, 15000, 100),
            'department': np.random.choice(['IT', 'Sales', 'Marketing', 'HR'], 100),
            'hire_date': pd.date_range('2020-01-01', periods=100, freq='D'),
            'email': [f'person{i}@company.com' for i in range(1, 101)],
            'is_active': np.random.choice([True, False], 100)
        })

    @pytest.fixture
    def generator(self):
        """Create a SyntheticDataGenerator instance"""
        return SyntheticDataGenerator()

    @pytest.fixture
    def temp_csv_file(self, sample_dataframe):
        """Create a temporary CSV file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_dataframe.to_csv(f.name, index=False)
            yield f.name
        os.unlink(f.name)

    def test_init(self, generator):
        """Test generator initialization"""
        assert hasattr(generator, 'original_data')
        assert hasattr(generator, 'synthetic_data')
        assert hasattr(generator, 'schema')
        assert hasattr(generator, 'config')
        assert isinstance(generator.original_data, dict)
        assert isinstance(generator.synthetic_data, dict)
        assert isinstance(generator.schema, dict)
        assert isinstance(generator.config, dict)

    def test_load_data_from_files(self, generator, temp_csv_file):
        """Test loading data from files"""
        files = [{'name': 'test.csv', 'path': temp_csv_file}]
        success = generator.load_data_from_files(files)

        assert success is True
        assert 'test' in generator.original_data
        assert len(generator.original_data['test']) == 100
        assert len(generator.schema) > 0
        assert 'test' in generator.schema

    def test_schema_inference(self, generator, sample_dataframe):
        """Test schema inference"""
        generator.original_data = {'test_table': sample_dataframe}
        generator._infer_schema()

        schema = generator.schema['test_table']
        assert 'columns' in schema

        # Check specific column types
        columns = schema['columns']
        assert columns['id']['type'] == 'numeric'
        assert columns['name']['subtype'] == 'name'
        assert columns['age']['subtype'] == 'age'
        assert columns['email']['subtype'] == 'email'
        assert columns['hire_date']['type'] == 'datetime'

    def test_configure_generation(self, generator):
        """Test generation configuration"""
        config = {
            'generation_method': 'perturbation',
            'perturbation_factor': 0.3,
            'anonymize_names': True
        }

        generator.configure_generation(config)
        assert generator.config == config

    def test_generate_synthetic_data_perturbation(self, generator, sample_dataframe):
        """Test synthetic data generation with perturbation method"""
        generator.original_data = {'test_table': sample_dataframe}
        generator._infer_schema()
        generator.configure_generation({
            'generation_method': 'perturbation',
            'perturbation_factor': 0.2,
            'data_size': {'type': 'same'}
        })

        success = generator.generate_synthetic_data()
        assert success is True
        assert 'test_table' in generator.synthetic_data

        synthetic_df = generator.synthetic_data['test_table']
        assert len(synthetic_df) == len(sample_dataframe)
        assert list(synthetic_df.columns) == list(sample_dataframe.columns)

    def test_generate_synthetic_data_auto(self, generator, sample_dataframe):
        """Test synthetic data generation with auto method"""
        generator.original_data = {'test_table': sample_dataframe}
        generator._infer_schema()
        generator.configure_generation({
            'generation_method': 'auto',
            'data_size': {'type': 'percentage', 'value': 80}
        })

        success = generator.generate_synthetic_data()
        assert success is True

        synthetic_df = generator.synthetic_data['test_table']
        assert len(synthetic_df) == 80  # 80% of original

    def test_name_anonymization(self, generator, sample_dataframe):
        """Test name anonymization"""
        generator.original_data = {'test_table': sample_dataframe}
        generator._infer_schema()
        generator.configure_generation({
            'generation_method': 'perturbation',
            'anonymize_names': True,
            'name_method': 'synthetic'
        })

        success = generator.generate_synthetic_data()
        assert success is True

        synthetic_df = generator.synthetic_data['test_table']
        original_names = set(sample_dataframe['name'].values)
        synthetic_names = set(synthetic_df['name'].values)

        # Names should be different
        assert len(original_names.intersection(synthetic_names)) < len(original_names) * 0.1

    def test_age_grouping(self, generator, sample_dataframe):
        """Test age grouping functionality"""
        generator.original_data = {'test_table': sample_dataframe}
        generator._infer_schema()
        generator.configure_generation({
            'generation_method': 'perturbation',
            'apply_age_grouping': True,
            'age_grouping_method': '10-year'
        })

        success = generator.generate_synthetic_data()
        assert success is True

        synthetic_df = generator.synthetic_data['test_table']
        # Age should be converted to categorical ranges
        assert synthetic_df['age'].dtype == 'category' or synthetic_df['age'].dtype == 'object'

    def test_evaluation_metrics(self, generator, sample_dataframe):
        """Test synthetic data evaluation"""
        generator.original_data = {'test_table': sample_dataframe}
        generator._infer_schema()
        generator.configure_generation({'generation_method': 'perturbation'})
        generator.generate_synthetic_data()

        evaluation_results = generator.evaluate_synthetic_data()

        assert 'test_table' in evaluation_results
        metrics = evaluation_results['test_table']

        assert 'statistical_similarity' in metrics
        assert 'privacy_score' in metrics
        assert 'utility_score' in metrics
        assert 'overall_score' in metrics

        # Scores should be between 0 and 1
        for score_name in ['statistical_similarity', 'privacy_score', 'utility_score', 'overall_score']:
            score = metrics[score_name]
            assert 0 <= score <= 1, f"{score_name} should be between 0 and 1, got {score}"

    def test_temporal_relationship_detection(self, generator):
        """Test temporal relationship detection"""
        # Create dataframe with temporal relationships
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'admission_date': dates,
            'discharge_date': dates + pd.Timedelta(days=3),  # Always 3 days later
            'patient_id': range(100)
        })

        generator.original_data = {'hospital': df}
        generator._infer_schema()

        schema = generator.schema['hospital']
        temporal_rels = schema['relationships']['temporal']

        assert len(temporal_rels) > 0
        rel = temporal_rels[0]
        assert rel['from'] == 'admission_date'
        assert rel['to'] == 'discharge_date'
        assert rel['consistency'] > 0.9

    def test_error_handling_empty_data(self, generator):
        """Test error handling for empty data"""
        with pytest.raises(ValueError, match="No original data available"):
            generator.generate_synthetic_data()

    def test_error_handling_invalid_file(self, generator):
        """Test error handling for invalid files"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("invalid content")
            invalid_file = f.name

        try:
            files = [{'name': 'invalid.txt', 'path': invalid_file}]
            with pytest.raises(Exception):
                generator.load_data_from_files(files)
        finally:
            os.unlink(invalid_file)

    def test_large_dataset_handling(self, generator):
        """Test handling of large datasets"""
        # Create a larger dataset
        np.random.seed(42)
        large_df = pd.DataFrame({
            'id': range(10000),
            'value': np.random.normal(0, 1, 10000)
        })

        generator.original_data = {'large_table': large_df}
        generator._infer_schema()
        generator.configure_generation({'generation_method': 'auto'})

        success = generator.generate_synthetic_data()
        assert success is True
        assert len(generator.synthetic_data['large_table']) == 10000


class TestDataProcessor:
    """Test suite for DataProcessor"""

    @pytest.fixture
    def processor(self):
        """Create a DataProcessor instance"""
        return DataProcessor()

    @pytest.fixture
    def sample_csv_content(self):
        """Sample CSV content for testing"""
        return """id,name,age,email
1,John Doe,25,john@example.com
2,Jane Smith,30,jane@example.com
3,Bob Johnson,35,bob@example.com"""

    def test_detect_encoding(self, processor):
        """Test encoding detection"""
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.csv', delete=False) as f:
            f.write("test,data\n1,test")
            temp_file = f.name

        try:
            encoding = processor.detect_encoding(temp_file)
            assert encoding in ['utf-8', 'ascii']
        finally:
            os.unlink(temp_file)

    def test_load_csv_file(self, processor, sample_csv_content):
        """Test CSV file loading"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(sample_csv_content)
            temp_file = f.name

        try:
            df = processor.load_csv_file(temp_file)
            assert len(df) == 3
            assert list(df.columns) == ['id', 'name', 'age', 'email']
            assert df['id'].dtype in [np.int64, np.float64]
        finally:
            os.unlink(temp_file)

    def test_data_validation(self, processor):
        """Test data validation"""
        df = pd.DataFrame({
            'valid_col': [1, 2, 3, 4, 5],
            'missing_col': [1, None, None, None, None],
            'duplicate_col': [1, 1, 1, 1, 1]
        })

        validation_result = processor.validate_data(df)

        assert 'valid' in validation_result
        assert 'issues' in validation_result
        assert 'warnings' in validation_result
        assert 'statistics' in validation_result

        # Should have warnings about missing data
        warnings = validation_result['warnings']
        assert any('missing' in warning.lower() for warning in warnings)

    def test_data_profiling(self, processor):
        """Test data profiling"""
        df = pd.DataFrame({
            'numeric_col': [1, 2, 3, 4, 5],
            'text_col': ['a', 'b', 'c', 'd', 'e'],
            'date_col': pd.date_range('2023-01-01', periods=5)
        })

        profile = processor.generate_data_profile(df, 'test_table')

        assert 'table_name' in profile
        assert 'basic_info' in profile
        assert 'column_profiles' in profile
        assert 'data_quality' in profile

        # Check column profiles
        assert 'numeric_col' in profile['column_profiles']
        assert 'text_col' in profile['column_profiles']
        assert 'date_col' in profile['column_profiles']

        # Check numeric column profile
        numeric_profile = profile['column_profiles']['numeric_col']
        assert numeric_profile['type'] == 'numeric'
        assert 'mean' in numeric_profile
        assert 'std' in numeric_profile


class TestConfigManager:
    """Test suite for ConfigManager"""

    @pytest.fixture
    def config_manager(self):
        """Create a ConfigManager instance"""
        return ConfigManager()

    def test_initialization(self, config_manager):
        """Test config manager initialization"""
        assert hasattr(config_manager, 'config')
        assert isinstance(config_manager.config, dict)
        assert 'FLASK' in config_manager.config
        assert 'LOGGING' in config_manager.config

    def test_get_config(self, config_manager):
        """Test getting configuration values"""
        # Test existing config
        value = config_manager.get('FLASK.DEBUG')
        assert value is not None

        # Test non-existing config with default
        value = config_manager.get('NONEXISTENT.KEY', 'default')
        assert value == 'default'

    def test_set_config(self, config_manager):
        """Test setting configuration values"""
        config_manager.set('TEST.VALUE', 'test_value')
        assert config_manager.get('TEST.VALUE') == 'test_value'

    def test_config_validation(self, config_manager):
        """Test configuration validation"""
        issues = config_manager.validate()
        assert isinstance(issues, list)

    def test_flask_config_generation(self, config_manager):
        """Test Flask configuration generation"""
        flask_config = config_manager.get_flask_config()
        assert isinstance(flask_config, dict)
        assert 'SECRET_KEY' in flask_config


# Integration Tests
class TestIntegration:
    """Integration tests for the complete pipeline"""

    def test_full_pipeline_csv(self):
        """Test complete pipeline with CSV data"""
        # Create sample data
        df = pd.DataFrame({
            'id': range(1, 51),
            'name': [f'Person_{i}' for i in range(1, 51)],
            'age': np.random.randint(18, 80, 50),
            'salary': np.random.normal(50000, 15000, 50)
        })

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_file = f.name

        try:
            # Initialize generator
            generator = SyntheticDataGenerator()

            # Load data
            files = [{'name': 'test.csv', 'path': temp_file}]
            success = generator.load_data_from_files(files)
            assert success

            # Configure
            config = {
                'generation_method': 'perturbation',
                'anonymize_names': True,
                'data_size': {'type': 'same'}
            }
            generator.configure_generation(config)

            # Generate
            success = generator.generate_synthetic_data()
            assert success

            # Evaluate
            evaluation = generator.evaluate_synthetic_data()
            assert 'test' in evaluation

            # Check that synthetic data exists and has reasonable properties
            synthetic_df = generator.synthetic_data['test']
            assert len(synthetic_df) == len(df)
            assert list(synthetic_df.columns) == list(df.columns)

        finally:
            os.unlink(temp_file)

    def test_full_pipeline_with_relationships(self):
        """Test pipeline with temporal relationships"""
        # Create data with temporal relationships
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        df = pd.DataFrame({
            'patient_id': range(1, 31),
            'admission_date': dates,
            'discharge_date': dates + pd.Timedelta(days=2),
            'diagnosis': np.random.choice(['A', 'B', 'C'], 30)
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_file = f.name

        try:
            generator = SyntheticDataGenerator()
            files = [{'name': 'hospital.csv', 'path': temp_file}]
            generator.load_data_from_files(files)

            # Check that temporal relationships were detected
            schema = generator.schema['hospital']
            temporal_rels = schema['relationships']['temporal']
            assert len(temporal_rels) > 0

            # Generate with relationship preservation
            config = {
                'generation_method': 'perturbation',
                'preserve_temporal': True
            }
            generator.configure_generation(config)
            generator.generate_synthetic_data()

            # Check that relationships are preserved
            synthetic_df = generator.synthetic_data['hospital']
            admission_dates = pd.to_datetime(synthetic_df['admission_date'])
            discharge_dates = pd.to_datetime(synthetic_df['discharge_date'])

            # All discharge dates should be after admission dates
            violations = (discharge_dates < admission_dates).sum()
            assert violations < len(synthetic_df) * 0.1  # Less than 10% violations

        finally:
            os.unlink(temp_file)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


# Test data fixtures
@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data"""
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])