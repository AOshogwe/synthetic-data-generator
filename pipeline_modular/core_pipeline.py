# pipeline/core_pipeline.py - Main Pipeline Orchestrator
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import os
import json
import logging
import traceback
from datetime import datetime

from .database_connector import DatabaseConnector
from .schema_manager import SchemaManager
from .data_preprocessor import DataPreprocessor

# Import existing modules
from preprocessing.cleaning import DataCleaningPipeline
from stats.relationships import RelationshipDiscoverer
from models.generation_engine import SyntheticGenerationEngine
from evaluation.statistical_evaluator import SyntheticDataEvaluator
from export.validator import DataValidator
from export.adapters import ExportAdapter
from models.address_synthesis import AddressSynthesizer

try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
    import usaddress
    GIS_AVAILABLE = True
except ImportError:
    GIS_AVAILABLE = False

logger = logging.getLogger(__name__)

class SyntheticDataPipeline:
    """
    Main pipeline orchestrator for synthetic data generation system
    Refactored into modular components for better maintainability
    """

    def __init__(self, config_path: Optional[str] = None):
        # Initialize core components
        self.db_connector = DatabaseConnector()
        self.schema_manager = SchemaManager()
        self.data_preprocessor = DataPreprocessor()
        
        # Initialize legacy components (to be refactored)
        self.cleaner = None
        self.relationship_discoverer = None
        self.generator = None
        self.evaluator = None
        self.validator = None
        self.exporter = None
        self.address_synthesizer = None

        # Pipeline state
        self.original_data = {}
        self.processed_data = {}
        self.synthetic_data = {}
        self.schema = {}
        self.relationships = {}
        self.evaluation_results = None
        
        # Configuration
        self.config = {}
        
        # Load configuration if provided
        if config_path:
            self.load_config(config_path)

        # Set up logging
        self._setup_logging()
        
        logger.info("SyntheticDataPipeline initialized successfully")

    def load_config(self, config_path: str) -> bool:
        """Load configuration from file"""
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)

            # Load schema if provided
            if 'schema' in self.config:
                self.schema = self.config['schema']

            logger.info(f"Configuration loaded from {config_path}")
            return True

        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return False

    def _setup_logging(self):
        """Set up logging configuration"""
        if not logging.getLogger().hasHandlers():
            log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            logging.basicConfig(
                level=logging.INFO,
                format=log_format,
                handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler('synthetic_data.log')
                ]
            )

    def connect_database(self, connection_string: str, tables: List[str] = None, chunk_size: int = 10000) -> bool:
        """Connect to database and load data"""
        try:
            # Connect to database
            if not self.db_connector.connect(connection_string, chunk_size):
                return False
            
            # Load table data
            self.original_data = self.db_connector.load_tables(tables, chunk_size)
            
            if not self.original_data:
                logger.error("No data loaded from database")
                return False
            
            # Infer schema if not already loaded
            if not self.schema:
                self.schema = self.schema_manager.infer_schema(self.original_data)
            
            logger.info(f"Successfully loaded {len(self.original_data)} tables from database")
            return True

        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            return False

    def load_files(self, file_paths: Dict[str, str]) -> bool:
        """Load data from files"""
        try:
            for table_name, file_path in file_paths.items():
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                elif file_path.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(file_path)
                elif file_path.endswith('.json'):
                    df = pd.read_json(file_path)
                else:
                    logger.warning(f"Unsupported file format: {file_path}")
                    continue
                
                self.original_data[table_name] = df
                logger.info(f"Loaded {table_name}: {len(df)} rows, {len(df.columns)} columns")
            
            # Infer schema
            if not self.schema:
                self.schema = self.schema_manager.infer_schema(self.original_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading files: {e}")
            return False

    def preprocess_data(self) -> bool:
        """Preprocess and clean the loaded data"""
        try:
            if not self.original_data:
                logger.error("No data to preprocess")
                return False
            
            logger.info("Starting data preprocessing")
            
            # Use the new preprocessor
            self.processed_data = self.data_preprocessor.preprocess_data(
                self.original_data, 
                self.schema
            )
            
            # Detect relationships
            self.relationships = self.data_preprocessor.detect_relationships(
                self.processed_data
            )
            
            logger.info("Data preprocessing completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            return False

    def generate_synthetic_data(
        self, 
        model_type: str = "StatisticalGenerator",
        num_samples: int = 1000,
        privacy_level: str = "medium",
        **kwargs
    ) -> bool:
        """Generate synthetic data using specified model"""
        try:
            if not self.processed_data:
                logger.error("No processed data available for synthesis")
                return False
            
            # Initialize generator if not already done
            if not self.generator:
                self.generator = SyntheticGenerationEngine()
            
            logger.info(f"Starting synthetic data generation with {model_type}")
            
            # Generate synthetic data for each table
            self.synthetic_data = {}
            for table_name, df in self.processed_data.items():
                try:
                    result = self.generator.generate_synthetic_data(
                        df,
                        model_type=model_type,
                        num_samples=num_samples,
                        privacy_level=privacy_level,
                        **kwargs
                    )
                    
                    if result.get('success', False):
                        self.synthetic_data[table_name] = result['synthetic_data']
                        logger.info(f"Generated {len(self.synthetic_data[table_name])} synthetic records for {table_name}")
                    else:
                        logger.warning(f"Failed to generate synthetic data for {table_name}: {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    logger.error(f"Error generating synthetic data for {table_name}: {e}")
                    continue
            
            if self.synthetic_data:
                logger.info("Synthetic data generation completed successfully")
                return True
            else:
                logger.error("No synthetic data was generated")
                return False
                
        except Exception as e:
            logger.error(f"Error during synthetic data generation: {e}")
            return False

    def evaluate_synthetic_data(self) -> bool:
        """Evaluate the quality of synthetic data"""
        try:
            if not self.synthetic_data or not self.processed_data:
                logger.error("No synthetic data or original data available for evaluation")
                return False
            
            # Initialize evaluator if not already done
            if not self.evaluator:
                self.evaluator = SyntheticDataEvaluator()
            
            logger.info("Starting synthetic data evaluation")
            
            evaluation_results = {}
            
            for table_name in self.synthetic_data.keys():
                if table_name not in self.processed_data:
                    continue
                
                try:
                    original_df = self.processed_data[table_name]
                    synthetic_df = self.synthetic_data[table_name]
                    
                    # Comprehensive evaluation
                    table_evaluation = self.evaluator.comprehensive_evaluation(
                        original_df, 
                        synthetic_df
                    )
                    
                    evaluation_results[table_name] = table_evaluation
                    logger.info(f"Evaluation completed for {table_name}")
                    
                except Exception as e:
                    logger.error(f"Error evaluating {table_name}: {e}")
                    continue
            
            self.evaluation_results = evaluation_results
            
            if evaluation_results:
                logger.info("Synthetic data evaluation completed successfully")
                return True
            else:
                logger.error("No evaluation results generated")
                return False
                
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return False

    def export_data(self, output_format: str = 'csv', output_dir: str = 'outputs') -> bool:
        """Export synthetic data in specified format"""
        try:
            if not self.synthetic_data:
                logger.error("No synthetic data to export")
                return False
            
            # Initialize exporter if not already done
            if not self.exporter:
                self.exporter = ExportAdapter()
            
            logger.info(f"Exporting synthetic data in {output_format} format")
            
            os.makedirs(output_dir, exist_ok=True)
            
            for table_name, df in self.synthetic_data.items():
                try:
                    output_path = os.path.join(
                        output_dir, 
                        f"{table_name}_synthetic.{output_format}"
                    )
                    
                    if output_format.lower() == 'csv':
                        df.to_csv(output_path, index=False)
                    elif output_format.lower() in ['xlsx', 'excel']:
                        df.to_excel(output_path, index=False)
                    elif output_format.lower() == 'json':
                        df.to_json(output_path, orient='records')
                    else:
                        logger.warning(f"Unsupported output format: {output_format}")
                        continue
                    
                    logger.info(f"Exported {table_name} to {output_path}")
                    
                except Exception as e:
                    logger.error(f"Error exporting {table_name}: {e}")
                    continue
            
            # Export evaluation results if available
            if self.evaluation_results:
                eval_path = os.path.join(output_dir, 'evaluation_results.json')
                with open(eval_path, 'w') as f:
                    json.dump(self.evaluation_results, f, indent=2, default=str)
                logger.info(f"Exported evaluation results to {eval_path}")
            
            logger.info("Data export completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during data export: {e}")
            return False

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            'has_original_data': bool(self.original_data),
            'has_processed_data': bool(self.processed_data),
            'has_synthetic_data': bool(self.synthetic_data),
            'has_schema': bool(self.schema),
            'has_relationships': bool(self.relationships),
            'has_evaluation': bool(self.evaluation_results),
            'table_count': len(self.original_data),
            'database_connected': self.db_connector.connected if hasattr(self.db_connector, 'connected') else False
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive pipeline summary"""
        summary = {
            'pipeline_status': self.get_pipeline_status(),
            'data_summary': {},
            'schema_summary': {},
            'evaluation_summary': {}
        }
        
        # Data summary
        if self.original_data:
            summary['data_summary'] = {
                table: {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'memory_usage': df.memory_usage(deep=True).sum()
                }
                for table, df in self.original_data.items()
            }
        
        # Schema summary
        if self.schema:
            summary['schema_summary'] = {
                table: {
                    'column_count': info.get('column_count', 0),
                    'row_count': info.get('row_count', 0),
                    'primary_key_candidates': len(info.get('primary_key_candidates', [])),
                    'foreign_key_candidates': len(info.get('foreign_key_candidates', []))
                }
                for table, info in self.schema.items()
            }
        
        # Evaluation summary
        if self.evaluation_results:
            summary['evaluation_summary'] = {
                table: {
                    'overall_score': results.get('overall_score', 0),
                    'privacy_score': results.get('privacy_score', 0),
                    'utility_score': results.get('utility_score', 0)
                }
                for table, results in self.evaluation_results.items()
            }
        
        return summary

    def cleanup(self):
        """Clean up resources"""
        try:
            if self.db_connector:
                self.db_connector.close()
            logger.info("Pipeline cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()

# Legacy compatibility - import the main class at package level
# This allows existing code to continue working: from pipeline import SyntheticDataPipeline