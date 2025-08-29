# pipeline/data_preprocessor.py - Data Preprocessing and Cleaning
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Handles data preprocessing, cleaning, and transformation"""
    
    def __init__(self):
        self.preprocessing_stats = {}
        self.transformations_applied = {}
    
    def preprocess_data(self, data: Dict[str, pd.DataFrame], schema: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Apply comprehensive preprocessing to all tables"""
        preprocessed_data = {}
        
        for table_name, df in data.items():
            logger.info(f"Preprocessing table: {table_name}")
            preprocessed_df = self._preprocess_table(df, schema.get(table_name, {}))
            preprocessed_data[table_name] = preprocessed_df
            
        return preprocessed_data
    
    def _preprocess_table(self, df: pd.DataFrame, table_schema: Dict[str, Any]) -> pd.DataFrame:
        """Preprocess a single table"""
        df_processed = df.copy()
        table_stats = {
            'original_shape': df.shape,
            'transformations': []
        }
        
        # 1. Handle missing values
        df_processed = self._handle_missing_values(df_processed, table_schema)
        table_stats['transformations'].append('missing_values')
        
        # 2. Fix data types
        df_processed = self._fix_data_types(df_processed, table_schema)
        table_stats['transformations'].append('data_types')
        
        # 3. Remove duplicates
        original_len = len(df_processed)
        df_processed = df_processed.drop_duplicates()
        duplicates_removed = original_len - len(df_processed)
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate rows")
            table_stats['transformations'].append('duplicates')
        
        # 4. Handle outliers
        df_processed = self._handle_outliers(df_processed, table_schema)
        table_stats['transformations'].append('outliers')
        
        # 5. Standardize text data
        df_processed = self._standardize_text_data(df_processed, table_schema)
        table_stats['transformations'].append('text_standardization')
        
        # 6. Validate date formats
        df_processed = self._validate_dates(df_processed, table_schema)
        table_stats['transformations'].append('date_validation')
        
        table_stats['final_shape'] = df_processed.shape
        table_stats['rows_removed'] = df.shape[0] - df_processed.shape[0]
        
        self.preprocessing_stats[table_schema.get('name', 'unknown')] = table_stats
        
        logger.info(f"Preprocessing completed. Shape: {df.shape} -> {df_processed.shape}")
        return df_processed
    
    def _handle_missing_values(self, df: pd.DataFrame, schema: Dict[str, Any]) -> pd.DataFrame:
        """Handle missing values based on column types and patterns"""
        df_processed = df.copy()
        columns_info = schema.get('columns', {})
        
        for column in df.columns:
            column_info = columns_info.get(column, {})
            null_ratio = column_info.get('null_ratio', 0)
            data_type = column_info.get('data_type', 'unknown')
            
            if null_ratio > 0.5:  # Too many missing values
                logger.warning(f"Column {column} has {null_ratio:.2%} missing values - considering removal")
                continue
            
            if null_ratio > 0:
                if data_type == 'numeric':
                    # Fill with median for numeric
                    median_val = df_processed[column].median()
                    df_processed[column].fillna(median_val, inplace=True)
                    logger.debug(f"Filled {column} nulls with median: {median_val}")
                    
                elif data_type == 'categorical':
                    # Fill with mode for categorical
                    mode_val = df_processed[column].mode().iloc[0] if len(df_processed[column].mode()) > 0 else 'Unknown'
                    df_processed[column].fillna(mode_val, inplace=True)
                    logger.debug(f"Filled {column} nulls with mode: {mode_val}")
                    
                elif data_type == 'datetime':
                    # Forward fill for dates
                    df_processed[column].fillna(method='ffill', inplace=True)
                    
                else:  # text or unknown
                    df_processed[column].fillna('', inplace=True)
        
        return df_processed
    
    def _fix_data_types(self, df: pd.DataFrame, schema: Dict[str, Any]) -> pd.DataFrame:
        """Fix and optimize data types"""
        df_processed = df.copy()
        columns_info = schema.get('columns', {})
        
        for column in df.columns:
            column_info = columns_info.get(column, {})
            data_type = column_info.get('data_type', 'unknown')
            
            try:
                if data_type == 'numeric':
                    if column_info.get('is_integer', False):
                        # Try to convert to appropriate integer type
                        min_val = column_info.get('min_value', 0)
                        max_val = column_info.get('max_value', 0)
                        
                        if min_val >= 0 and max_val <= 255:
                            df_processed[column] = df_processed[column].astype('uint8')
                        elif min_val >= -128 and max_val <= 127:
                            df_processed[column] = df_processed[column].astype('int8')
                        elif min_val >= 0 and max_val <= 65535:
                            df_processed[column] = df_processed[column].astype('uint16')
                        elif min_val >= -32768 and max_val <= 32767:
                            df_processed[column] = df_processed[column].astype('int16')
                        else:
                            df_processed[column] = df_processed[column].astype('int32')
                    else:
                        df_processed[column] = pd.to_numeric(df_processed[column], errors='coerce')
                
                elif data_type == 'datetime':
                    df_processed[column] = pd.to_datetime(df_processed[column], errors='coerce')
                
                elif data_type == 'categorical':
                    unique_count = column_info.get('unique_count', 0)
                    if unique_count < 100:  # Convert to category for memory efficiency
                        df_processed[column] = df_processed[column].astype('category')
                
            except Exception as e:
                logger.warning(f"Could not convert column {column} to {data_type}: {e}")
        
        return df_processed
    
    def _handle_outliers(self, df: pd.DataFrame, schema: Dict[str, Any]) -> pd.DataFrame:
        """Handle outliers in numeric columns"""
        df_processed = df.copy()
        columns_info = schema.get('columns', {})
        
        for column in df.columns:
            column_info = columns_info.get(column, {})
            
            if column_info.get('data_type') == 'numeric' and pd.api.types.is_numeric_dtype(df_processed[column]):
                # Use IQR method for outlier detection
                Q1 = df_processed[column].quantile(0.25)
                Q3 = df_processed[column].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((df_processed[column] < lower_bound) | (df_processed[column] > upper_bound))
                outlier_count = outliers.sum()
                
                if outlier_count > 0 and outlier_count < len(df_processed) * 0.05:  # Less than 5% outliers
                    # Cap outliers instead of removing them
                    df_processed.loc[df_processed[column] < lower_bound, column] = lower_bound
                    df_processed.loc[df_processed[column] > upper_bound, column] = upper_bound
                    logger.info(f"Capped {outlier_count} outliers in column {column}")
        
        return df_processed
    
    def _standardize_text_data(self, df: pd.DataFrame, schema: Dict[str, Any]) -> pd.DataFrame:
        """Standardize text data"""
        df_processed = df.copy()
        columns_info = schema.get('columns', {})
        
        for column in df.columns:
            column_info = columns_info.get(column, {})
            
            if column_info.get('data_type') in ['text', 'categorical'] and pd.api.types.is_string_dtype(df_processed[column]):
                # Remove leading/trailing whitespace
                df_processed[column] = df_processed[column].astype(str).str.strip()
                
                # Standardize case for categorical data
                if column_info.get('data_type') == 'categorical':
                    unique_count = column_info.get('unique_count', 0)
                    if unique_count < 50:  # Small number of categories
                        df_processed[column] = df_processed[column].str.title()
        
        return df_processed
    
    def _validate_dates(self, df: pd.DataFrame, schema: Dict[str, Any]) -> pd.DataFrame:
        """Validate and fix date columns"""
        df_processed = df.copy()
        columns_info = schema.get('columns', {})
        
        for column in df.columns:
            column_info = columns_info.get(column, {})
            
            if column_info.get('data_type') == 'datetime':
                try:
                    # Convert to datetime if not already
                    if not pd.api.types.is_datetime64_any_dtype(df_processed[column]):
                        df_processed[column] = pd.to_datetime(df_processed[column], errors='coerce')
                    
                    # Check for unreasonable dates
                    min_reasonable_date = pd.Timestamp('1900-01-01')
                    max_reasonable_date = pd.Timestamp('2100-01-01')
                    
                    invalid_dates = ((df_processed[column] < min_reasonable_date) | 
                                   (df_processed[column] > max_reasonable_date))
                    
                    if invalid_dates.any():
                        logger.warning(f"Found {invalid_dates.sum()} unreasonable dates in {column}")
                        df_processed.loc[invalid_dates, column] = pd.NaT
                
                except Exception as e:
                    logger.warning(f"Error validating dates in column {column}: {e}")
        
        return df_processed
    
    def detect_relationships(self, data: Dict[str, pd.DataFrame]) -> Dict[str, List[Dict[str, Any]]]:
        """Detect relationships between tables"""
        relationships = {}
        
        table_names = list(data.keys())
        
        for i, table1 in enumerate(table_names):
            relationships[table1] = []
            
            for j, table2 in enumerate(table_names):
                if i != j:
                    # Look for potential foreign key relationships
                    potential_fks = self._find_foreign_keys(data[table1], data[table2], table1, table2)
                    relationships[table1].extend(potential_fks)
        
        return relationships
    
    def _find_foreign_keys(self, df1: pd.DataFrame, df2: pd.DataFrame, table1: str, table2: str) -> List[Dict[str, Any]]:
        """Find potential foreign key relationships between two tables"""
        potential_fks = []
        
        for col1 in df1.columns:
            for col2 in df2.columns:
                # Skip if data types don't match
                if df1[col1].dtype != df2[col2].dtype:
                    continue
                
                # Check for overlap in values
                set1 = set(df1[col1].dropna().unique())
                set2 = set(df2[col2].dropna().unique())
                
                if len(set1) == 0 or len(set2) == 0:
                    continue
                
                overlap = len(set1.intersection(set2))
                overlap_ratio = overlap / min(len(set1), len(set2))
                
                # If significant overlap, potential FK relationship
                if overlap_ratio > 0.8 and overlap > 10:
                    potential_fks.append({
                        'source_table': table1,
                        'source_column': col1,
                        'target_table': table2,
                        'target_column': col2,
                        'overlap_ratio': overlap_ratio,
                        'overlap_count': overlap
                    })
        
        return potential_fks
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get summary of preprocessing operations"""
        return {
            'tables_processed': len(self.preprocessing_stats),
            'stats_by_table': self.preprocessing_stats,
            'total_transformations': sum(len(stats['transformations']) for stats in self.preprocessing_stats.values())
        }