# pipeline/schema_manager.py - Schema Detection and Management
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class SchemaManager:
    """Handles schema detection and management for datasets"""
    
    def __init__(self):
        self.schema = {}
        self.column_profiles = {}
    
    def infer_schema(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Infer comprehensive schema from datasets"""
        schema = {}
        
        for table_name, df in data.items():
            logger.info(f"Inferring schema for table: {table_name}")
            table_schema = self._infer_table_schema(df)
            schema[table_name] = table_schema
        
        self.schema = schema
        return schema
    
    def _infer_table_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Infer schema for a single table"""
        table_schema = {
            'columns': {},
            'row_count': len(df),
            'column_count': len(df.columns),
            'primary_key_candidates': [],
            'foreign_key_candidates': [],
            'constraints': {}
        }
        
        for column in df.columns:
            column_profile = self._profile_column(df[column])
            table_schema['columns'][column] = column_profile
            
            # Check for primary key characteristics
            if column_profile['unique_ratio'] >= 0.95 and column_profile['null_ratio'] == 0:
                table_schema['primary_key_candidates'].append(column)
        
        return table_schema
    
    def _profile_column(self, series: pd.Series) -> Dict[str, Any]:
        """Generate detailed profile for a column"""
        profile = {
            'name': series.name,
            'dtype': str(series.dtype),
            'null_count': series.isnull().sum(),
            'null_ratio': series.isnull().sum() / len(series),
            'unique_count': series.nunique(),
            'unique_ratio': series.nunique() / len(series),
            'data_type': self._infer_data_type(series),
            'patterns': [],
            'constraints': {}
        }
        
        # Add type-specific profiling
        if profile['data_type'] == 'numeric':
            profile.update(self._profile_numeric_column(series))
        elif profile['data_type'] == 'categorical':
            profile.update(self._profile_categorical_column(series))
        elif profile['data_type'] == 'datetime':
            profile.update(self._profile_datetime_column(series))
        elif profile['data_type'] == 'text':
            profile.update(self._profile_text_column(series))
        
        return profile
    
    def _infer_data_type(self, series: pd.Series) -> str:
        """Infer semantic data type beyond pandas dtype"""
        if pd.api.types.is_numeric_dtype(series):
            return 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(series):
            return 'datetime'
        elif pd.api.types.is_categorical_dtype(series) or series.nunique() / len(series) < 0.1:
            return 'categorical'
        elif pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            # Check for specific patterns
            if self._is_email_column(series):
                return 'email'
            elif self._is_phone_column(series):
                return 'phone'
            elif self._is_address_column(series):
                return 'address'
            elif self._is_datetime_string(series):
                return 'datetime'
            else:
                return 'text'
        else:
            return 'unknown'
    
    def _profile_numeric_column(self, series: pd.Series) -> Dict[str, Any]:
        """Profile numeric column"""
        numeric_series = pd.to_numeric(series, errors='coerce')
        return {
            'min_value': numeric_series.min(),
            'max_value': numeric_series.max(),
            'mean': numeric_series.mean(),
            'median': numeric_series.median(),
            'std': numeric_series.std(),
            'is_integer': numeric_series.dtype in ['int64', 'int32', 'int16', 'int8'],
            'has_negatives': (numeric_series < 0).any(),
            'zero_count': (numeric_series == 0).sum()
        }
    
    def _profile_categorical_column(self, series: pd.Series) -> Dict[str, Any]:
        """Profile categorical column"""
        value_counts = series.value_counts()
        return {
            'categories': value_counts.index.tolist()[:20],  # Top 20
            'category_counts': value_counts.head(20).to_dict(),
            'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
            'least_frequent': value_counts.index[-1] if len(value_counts) > 0 else None,
            'is_binary': len(value_counts) == 2
        }
    
    def _profile_datetime_column(self, series: pd.Series) -> Dict[str, Any]:
        """Profile datetime column"""
        dt_series = pd.to_datetime(series, errors='coerce')
        return {
            'min_date': dt_series.min(),
            'max_date': dt_series.max(),
            'date_range_days': (dt_series.max() - dt_series.min()).days if dt_series.min() and dt_series.max() else 0,
            'has_time_component': dt_series.dt.time.nunique() > 1,
            'common_formats': []  # Could detect common date formats
        }
    
    def _profile_text_column(self, series: pd.Series) -> Dict[str, Any]:
        """Profile text column"""
        lengths = series.astype(str).str.len()
        return {
            'min_length': lengths.min(),
            'max_length': lengths.max(),
            'avg_length': lengths.mean(),
            'common_patterns': self._detect_text_patterns(series),
            'contains_special_chars': series.astype(str).str.contains(r'[^a-zA-Z0-9\s]').any(),
            'is_mixed_case': (series.astype(str).str.islower() | series.astype(str).str.isupper()).sum() < len(series)
        }
    
    def _is_email_column(self, series: pd.Series) -> bool:
        """Check if column contains email addresses"""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        sample = series.dropna().astype(str).sample(min(100, len(series)))
        matches = sample.str.match(email_pattern).sum()
        return matches / len(sample) > 0.8
    
    def _is_phone_column(self, series: pd.Series) -> bool:
        """Check if column contains phone numbers"""
        phone_patterns = [
            r'^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$',  # US format
            r'^\+?[0-9\s\-\(\)]{10,15}$'  # International format
        ]
        sample = series.dropna().astype(str).sample(min(100, len(series)))
        for pattern in phone_patterns:
            matches = sample.str.match(pattern).sum()
            if matches / len(sample) > 0.7:
                return True
        return False
    
    def _is_address_column(self, series: pd.Series) -> bool:
        """Check if column contains addresses"""
        address_keywords = ['street', 'st', 'avenue', 'ave', 'road', 'rd', 'drive', 'dr', 'lane', 'ln']
        column_name = series.name.lower()
        
        # Check column name
        if any(keyword in column_name for keyword in ['address', 'addr', 'location']):
            return True
        
        # Check content
        sample = series.dropna().astype(str).sample(min(50, len(series)))
        has_keywords = sum(any(keyword in text.lower() for keyword in address_keywords) for text in sample)
        return has_keywords / len(sample) > 0.3
    
    def _is_datetime_string(self, series: pd.Series) -> bool:
        """Check if string column contains datetime values"""
        sample = series.dropna().sample(min(50, len(series)))
        try:
            parsed = pd.to_datetime(sample, errors='coerce')
            return parsed.notna().sum() / len(sample) > 0.8
        except:
            return False
    
    def _detect_text_patterns(self, series: pd.Series) -> List[str]:
        """Detect common patterns in text data"""
        patterns = []
        sample = series.dropna().astype(str).sample(min(100, len(series)))
        
        # Common patterns to check
        pattern_checks = [
            (r'^\d+$', 'numeric_string'),
            (r'^[A-Z]{2,3}\d+$', 'code_pattern'),
            (r'^\d{4}-\d{2}-\d{2}$', 'date_string'),
            (r'^\d+\.\d+$', 'decimal_string'),
            (r'^[A-Za-z\s]+$', 'alpha_only'),
            (r'^[A-Z][a-z\s]+$', 'title_case'),
        ]
        
        for pattern, name in pattern_checks:
            matches = sample.str.match(pattern).sum()
            if matches / len(sample) > 0.7:
                patterns.append(name)
        
        return patterns
    
    def get_column_constraints(self, table_name: str, column_name: str) -> Dict[str, Any]:
        """Get constraints for a specific column"""
        if table_name not in self.schema:
            return {}
        
        column_info = self.schema[table_name]['columns'].get(column_name, {})
        constraints = {}
        
        # Generate constraints based on column profile
        if column_info.get('data_type') == 'numeric':
            constraints['min_value'] = column_info.get('min_value')
            constraints['max_value'] = column_info.get('max_value')
            constraints['data_type'] = 'numeric'
        elif column_info.get('data_type') == 'categorical':
            constraints['allowed_values'] = column_info.get('categories', [])
            constraints['data_type'] = 'categorical'
        elif column_info.get('data_type') == 'datetime':
            constraints['min_date'] = column_info.get('min_date')
            constraints['max_date'] = column_info.get('max_date')
            constraints['data_type'] = 'datetime'
        
        if column_info.get('null_ratio', 1) == 0:
            constraints['nullable'] = False
        
        return constraints