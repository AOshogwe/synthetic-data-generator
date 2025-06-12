# data_processing.py - Advanced Data Processing Module
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import re
from datetime import datetime
import chardet
import io
from pathlib import Path
import zipfile
import gzip
from config import get_config

logger = logging.getLogger(__name__)


class DataProcessor:
    """Advanced data processing and validation"""

    def __init__(self):
        self.max_columns = get_config('DATA_PROCESSING.MAX_COLUMNS', 1000)
        self.max_preview_rows = get_config('DATA_PROCESSING.MAX_ROWS_PREVIEW', 1000)
        self.chunk_size = get_config('DATA_PROCESSING.CHUNK_SIZE', 10000)
        self.memory_threshold = get_config('DATA_PROCESSING.MEMORY_THRESHOLD', 500 * 1024 * 1024)

    def detect_encoding(self, file_path: str) -> str:
        """Detect file encoding"""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB
                result = chardet.detect(raw_data)
                encoding = result['encoding'] if result['confidence'] > 0.7 else 'utf-8'
                logger.info(f"Detected encoding: {encoding} (confidence: {result['confidence']})")
                return encoding
        except Exception as e:
            logger.warning(f"Error detecting encoding: {e}, defaulting to utf-8")
            return 'utf-8'

    def load_csv_file(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load CSV file with advanced error handling and optimization"""
        try:
            # Detect encoding if not specified
            encoding = kwargs.get('encoding')
            if not encoding and get_config('DATA_PROCESSING.AUTO_DETECT_ENCODING', True):
                encoding = self.detect_encoding(file_path)

            # Default parameters for robust CSV loading
            csv_params = {
                'encoding': encoding or 'utf-8',
                'low_memory': False,
                'skipinitialspace': True,
                'na_values': ['', 'NULL', 'null', 'None', 'none', 'N/A', 'n/a', '#N/A', 'NaN', 'nan'],
                'keep_default_na': True,
                'error_bad_lines': False,
                'warn_bad_lines': True
            }

            # Update with any user-provided parameters
            csv_params.update(kwargs)

            # Check file size for memory management
            file_size = Path(file_path).stat().st_size

            if file_size > self.memory_threshold:
                logger.info(f"Large file detected ({file_size} bytes), using chunked loading")
                return self._load_large_csv(file_path, csv_params)
            else:
                df = pd.read_csv(file_path, **csv_params)
                return self._clean_dataframe(df)

        except UnicodeDecodeError as e:
            logger.warning(f"Unicode error with {encoding}, trying with latin-1")
            csv_params['encoding'] = 'latin-1'
            df = pd.read_csv(file_path, **csv_params)
            return self._clean_dataframe(df)

        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {e}")
            raise

    def _load_large_csv(self, file_path: str, csv_params: dict) -> pd.DataFrame:
        """Load large CSV files in chunks"""
        chunks = []
        total_rows = 0

        try:
            chunk_iter = pd.read_csv(file_path, chunksize=self.chunk_size, **csv_params)

            for i, chunk in enumerate(chunk_iter):
                if len(chunk) == 0:
                    continue

                # Clean each chunk
                chunk = self._clean_dataframe(chunk)
                chunks.append(chunk)
                total_rows += len(chunk)

                # Limit total rows for very large files
                max_rows = get_config('GENERATION.MAX_SYNTHETIC_ROWS', 1000000)
                if total_rows >= max_rows:
                    logger.warning(f"Limiting rows to {max_rows} for memory management")
                    break

                if i % 10 == 0:  # Log progress every 10 chunks
                    logger.info(f"Processed {i + 1} chunks, {total_rows} total rows")

            if chunks:
                df = pd.concat(chunks, ignore_index=True)
                logger.info(f"Loaded large CSV: {len(df)} rows, {len(df.columns)} columns")
                return df
            else:
                raise ValueError("No valid data found in file")

        except Exception as e:
            logger.error(f"Error loading large CSV: {e}")
            raise

    def load_excel_file(self, file_path: str, **kwargs) -> Dict[str, pd.DataFrame]:
        """Load Excel file with multiple sheets support"""
        try:
            # Load all sheets by default
            sheet_name = kwargs.get('sheet_name', None)

            excel_params = {
                'na_values': ['', 'NULL', 'null', 'None', 'none', 'N/A', 'n/a', '#N/A', 'NaN', 'nan'],
                'keep_default_na': True
            }
            excel_params.update(kwargs)

            if sheet_name is None:
                # Load all sheets
                xl_file = pd.ExcelFile(file_path)
                dataframes = {}

                for sheet in xl_file.sheet_names:
                    try:
                        df = pd.read_excel(file_path, sheet_name=sheet, **excel_params)
                        if not df.empty:
                            dataframes[sheet] = self._clean_dataframe(df)
                    except Exception as e:
                        logger.warning(f"Error loading sheet '{sheet}': {e}")
                        continue

                return dataframes
            else:
                # Load specific sheet
                df = pd.read_excel(file_path, **excel_params)
                return {sheet_name: self._clean_dataframe(df)}

        except Exception as e:
            logger.error(f"Error loading Excel file {file_path}: {e}")
            raise

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate dataframe"""
        if df.empty:
            return df

        # Clean column names
        df.columns = df.columns.astype(str)
        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.replace('\n', ' ')
        df.columns = df.columns.str.replace('\r', ' ')

        # Remove completely empty rows and columns
        df = df.dropna(how='all')  # Remove rows where all values are NaN
        df = df.loc[:, ~df.isnull().all()]  # Remove columns where all values are NaN

        # Limit number of columns
        if len(df.columns) > self.max_columns:
            logger.warning(f"Too many columns ({len(df.columns)}), limiting to {self.max_columns}")
            df = df.iloc[:, :self.max_columns]

        # Convert object columns that should be numeric
        df = self._auto_convert_types(df)

        return df

    def _auto_convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Automatically convert column types"""
        for column in df.columns:
            if df[column].dtype == 'object':
                # Try to convert to numeric
                if self._is_numeric_column(df[column]):
                    df[column] = pd.to_numeric(df[column], errors='coerce')

                # Try to convert to datetime
                elif self._is_datetime_column(df[column]):
                    df[column] = pd.to_datetime(df[column], errors='coerce', infer_datetime_format=True)

        return df

    def _is_numeric_column(self, series: pd.Series) -> bool:
        """Check if column should be numeric"""
        # Sample non-null values
        sample = series.dropna().head(100)
        if len(sample) == 0:
            return False

        # Check if majority can be converted to numeric
        numeric_count = 0
        for value in sample:
            try:
                float(str(value).replace(',', '').replace('$', '').replace('%', ''))
                numeric_count += 1
            except (ValueError, TypeError):
                continue

        return numeric_count / len(sample) > 0.8

    def _is_datetime_column(self, series: pd.Series) -> bool:
        """Check if column should be datetime"""
        # Look for date-like patterns in column name
        column_name = series.name.lower()
        date_keywords = ['date', 'time', 'created', 'updated', 'modified', 'timestamp']

        if any(keyword in column_name for keyword in date_keywords):
            return True

        # Sample non-null values and try to parse as dates
        sample = series.dropna().head(50)
        if len(sample) == 0:
            return False

        date_count = 0
        for value in sample:
            try:
                pd.to_datetime(str(value), infer_datetime_format=True)
                date_count += 1
            except (ValueError, TypeError):
                continue

        return date_count / len(sample) > 0.7

    def validate_data(self, df: pd.DataFrame, table_name: str = "") -> Dict[str, Any]:
        """Comprehensive data validation"""
        validation_result = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'statistics': {}
        }

        # Basic validation
        if df.empty:
            validation_result['valid'] = False
            validation_result['issues'].append("Dataframe is empty")
            return validation_result

        # Check for reasonable size
        if len(df) > get_config('GENERATION.MAX_SYNTHETIC_ROWS', 1000000):
            validation_result['warnings'].append(f"Large dataset ({len(df)} rows) may impact performance")

        # Check for too many columns
        if len(df.columns) > self.max_columns:
            validation_result['warnings'].append(f"Many columns ({len(df.columns)}) detected")

        # Analyze data quality
        stats = self._calculate_data_statistics(df)
        validation_result['statistics'] = stats

        # Check for potential issues
        if stats['missing_percentage'] > 50:
            validation_result['warnings'].append(f"High missing data percentage: {stats['missing_percentage']:.1f}%")

        if stats['duplicate_percentage'] > 20:
            validation_result['warnings'].append(
                f"High duplicate rows percentage: {stats['duplicate_percentage']:.1f}%")

        # Check for suspicious columns
        suspicious_columns = self._detect_suspicious_columns(df)
        if suspicious_columns:
            validation_result['warnings'].extend([f"Suspicious column detected: {col}" for col in suspicious_columns])

        return validation_result

    def _calculate_data_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive data statistics"""
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()

        stats = {
            'rows': len(df),
            'columns': len(df.columns),
            'missing_cells': missing_cells,
            'missing_percentage': (missing_cells / total_cells * 100) if total_cells > 0 else 0,
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_percentage': (df.duplicated().sum() / len(df) * 100) if len(df) > 0 else 0,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'column_types': df.dtypes.value_counts().to_dict()
        }

        # Numeric statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats['numeric_columns'] = len(numeric_cols)
            stats['numeric_stats'] = {
                'mean_values': df[numeric_cols].mean().to_dict(),
                'std_values': df[numeric_cols].std().to_dict(),
                'outlier_counts': {}
            }

            # Detect outliers using IQR method
            for col in numeric_cols:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                outliers = ((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))).sum()
                stats['numeric_stats']['outlier_counts'][col] = outliers

        # Categorical statistics
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            stats['categorical_columns'] = len(categorical_cols)
            stats['categorical_stats'] = {}

            for col in categorical_cols:
                unique_count = df[col].nunique()
                stats['categorical_stats'][col] = {
                    'unique_values': unique_count,
                    'cardinality': unique_count / len(df) if len(df) > 0 else 0
                }

        return stats

    def _detect_suspicious_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect potentially problematic columns"""
        suspicious = []

        for column in df.columns:
            # Check for columns with all identical values
            if df[column].nunique() == 1:
                suspicious.append(f"{column} (all identical values)")

            # Check for columns with mostly missing data
            if df[column].isnull().sum() / len(df) > 0.95:
                suspicious.append(f"{column} (mostly missing)")

            # Check for potential encoding issues
            if df[column].dtype == 'object':
                sample_values = df[column].dropna().head(10).astype(str)
                for value in sample_values:
                    if any(char in value for char in ['�', '\ufffd']):
                        suspicious.append(f"{column} (potential encoding issues)")
                        break

        return suspicious

    def generate_data_profile(self, df: pd.DataFrame, table_name: str = "") -> Dict[str, Any]:
        """Generate comprehensive data profile"""
        profile = {
            'table_name': table_name,
            'generated_at': datetime.now().isoformat(),
            'basic_info': {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
            },
            'column_profiles': {},
            'data_quality': self.validate_data(df, table_name),
            'relationships': self._detect_column_relationships(df)
        }

        # Profile each column
        for column in df.columns:
            profile['column_profiles'][column] = self._profile_column(df, column)

        return profile

    def _profile_column(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Generate detailed profile for a single column"""
        series = df[column]

        profile = {
            'name': column,
            'dtype': str(series.dtype),
            'count': series.count(),
            'null_count': series.isnull().sum(),
            'null_percentage': (series.isnull().sum() / len(series) * 100) if len(series) > 0 else 0,
            'unique_count': series.nunique(),
            'unique_percentage': (series.nunique() / len(series) * 100) if len(series) > 0 else 0,
            'memory_usage': series.memory_usage(deep=True)
        }

        # Type-specific analysis
        if pd.api.types.is_numeric_dtype(series):
            profile.update(self._profile_numeric_column(series))
        elif pd.api.types.is_datetime64_any_dtype(series):
            profile.update(self._profile_datetime_column(series))
        else:
            profile.update(self._profile_categorical_column(series))

        # Pattern analysis for string columns
        if series.dtype == 'object':
            profile['patterns'] = self._analyze_patterns(series)

        return profile

    def _profile_numeric_column(self, series: pd.Series) -> Dict[str, Any]:
        """Profile numeric column"""
        return {
            'type': 'numeric',
            'min': series.min(),
            'max': series.max(),
            'mean': series.mean(),
            'median': series.median(),
            'std': series.std(),
            'variance': series.var(),
            'skewness': series.skew(),
            'kurtosis': series.kurtosis(),
            'quantiles': {
                '25%': series.quantile(0.25),
                '50%': series.quantile(0.50),
                '75%': series.quantile(0.75),
                '95%': series.quantile(0.95),
                '99%': series.quantile(0.99)
            },
            'zeros_count': (series == 0).sum(),
            'negative_count': (series < 0).sum(),
            'infinite_count': np.isinf(series).sum()
        }

    def _profile_datetime_column(self, series: pd.Series) -> Dict[str, Any]:
        """Profile datetime column"""
        valid_dates = series.dropna()

        if len(valid_dates) == 0:
            return {'type': 'datetime', 'min_date': None, 'max_date': None}

        return {
            'type': 'datetime',
            'min_date': valid_dates.min(),
            'max_date': valid_dates.max(),
            'date_range_days': (valid_dates.max() - valid_dates.min()).days,
            'most_common_year': valid_dates.dt.year.mode().iloc[0] if len(valid_dates.dt.year.mode()) > 0 else None,
            'most_common_month': valid_dates.dt.month.mode().iloc[0] if len(valid_dates.dt.month.mode()) > 0 else None,
            'weekday_distribution': valid_dates.dt.dayofweek.value_counts().to_dict()
        }

    def _profile_categorical_column(self, series: pd.Series) -> Dict[str, Any]:
        """Profile categorical column"""
        value_counts = series.value_counts()

        profile = {
            'type': 'categorical',
            'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
            'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
            'least_frequent': value_counts.index[-1] if len(value_counts) > 0 else None,
            'least_frequent_count': value_counts.iloc[-1] if len(value_counts) > 0 else 0,
            'top_values': value_counts.head(10).to_dict()
        }

        # Calculate entropy (diversity measure)
        if len(value_counts) > 0:
            probabilities = value_counts / value_counts.sum()
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            profile['entropy'] = entropy

        return profile

    def _analyze_patterns(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze patterns in string columns"""
        patterns = {
            'email_like': 0,
            'phone_like': 0,
            'url_like': 0,
            'date_like': 0,
            'numeric_like': 0,
            'common_patterns': {}
        }

        # Sample for performance
        sample = series.dropna().head(1000)

        # Email pattern
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        patterns['email_like'] = sum(1 for val in sample if email_pattern.match(str(val)))

        # Phone pattern (simple)
        phone_pattern = re.compile(r'[\d\-\(\)\+\s]{10,}')
        patterns['phone_like'] = sum(1 for val in sample if phone_pattern.match(str(val)))

        # URL pattern
        url_pattern = re.compile(r'^https?://')
        patterns['url_like'] = sum(1 for val in sample if url_pattern.match(str(val)))

        # Extract common patterns
        pattern_counts = {}
        for val in sample:
            # Create a pattern by replacing digits with 'D' and letters with 'L'
            pattern = re.sub(r'\d', 'D', str(val))
            pattern = re.sub(r'[a-zA-Z]', 'L', pattern)
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        # Get top 5 patterns
        patterns['common_patterns'] = dict(sorted(pattern_counts.items(),
                                                  key=lambda x: x[1], reverse=True)[:5])

        return patterns

    def _detect_column_relationships(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect relationships between columns"""
        relationships = {
            'high_correlation_pairs': [],
            'potential_foreign_keys': [],
            'hierarchical_relationships': [],
            'temporal_sequences': []
        }

        # Correlation analysis for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()

            # Find high correlation pairs
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:  # High correlation threshold
                        relationships['high_correlation_pairs'].append({
                            'column1': corr_matrix.columns[i],
                            'column2': corr_matrix.columns[j],
                            'correlation': corr_val
                        })

        # Detect potential foreign key relationships
        for col in df.columns:
            if col.lower().endswith('_id') or col.lower().endswith('id'):
                unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
                if 0.1 < unique_ratio < 0.9:  # Potential foreign key
                    relationships['potential_foreign_keys'].append({
                        'column': col,
                        'unique_ratio': unique_ratio,
                        'unique_count': df[col].nunique()
                    })

        return relationships

    def extract_sample_data(self, df: pd.DataFrame, n_samples: int = None) -> Dict[str, Any]:
        """Extract sample data for preview"""
        if n_samples is None:
            n_samples = min(self.max_preview_rows, len(df))

        # Take a stratified sample if possible
        if len(df) > n_samples:
            sample_df = df.sample(n_samples, random_state=42)
        else:
            sample_df = df.copy()

        return {
            'data': sample_df.fillna('').to_dict('records'),
            'columns': list(df.columns),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'shape': df.shape,
            'sample_size': len(sample_df)
        }