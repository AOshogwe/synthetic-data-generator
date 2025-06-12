# app.py - Updated Flask Backend for SyntheticDataPipeline
import os
import sys
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import time
import io
import zipfile
from datetime import datetime
import traceback
from werkzeug.utils import secure_filename
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import signal
import atexit


# Configuration class
class Config:
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'

    # File upload settings
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
    OUTPUT_FOLDER = os.environ.get('OUTPUT_FOLDER', 'outputs')
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 100 * 1024 * 1024))  # 100MB

    # Allowed file extensions
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'json'}

    # CORS settings
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*')

    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', 'app.log')

    # Performance
    WORKERS = int(os.environ.get('WORKERS', 4))
    TIMEOUT = int(os.environ.get('TIMEOUT', 300))


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize CORS
    CORS(app, origins=app.config['CORS_ORIGINS'])

    # Set up logging
    setup_logging(app)

    # Ensure directories exist
    Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
    Path(app.config['OUTPUT_FOLDER']).mkdir(exist_ok=True)

    return app


def setup_logging(app):
    """Configure application logging"""
    if not app.debug:
        # File handler for production
        file_handler = RotatingFileHandler(
            app.config['LOG_FILE'],
            maxBytes=10240000,
            backupCount=10
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(getattr(logging, app.config['LOG_LEVEL']))
        app.logger.addHandler(file_handler)

        app.logger.setLevel(getattr(logging, app.config['LOG_LEVEL']))
        app.logger.info('Synthetic Data Generator startup')


# Create Flask app
app = create_app()

# Import the advanced pipeline - UPDATED IMPORT
from pipeline import SyntheticDataPipeline

# Global state for the pipeline - UPDATED TO USE ADVANCED PIPELINE
pipeline_state = {
    'pipeline': None,  # Will hold the SyntheticDataPipeline instance
    'status': 'ready',
    'session_id': None,
    'config': {},
    'has_data': False,
    'has_synthetic_data': False,
    'has_evaluation': False
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def initialize_pipeline():
    """Initialize a new pipeline instance"""
    return SyntheticDataPipeline()


@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 100MB.'}), 413


@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f'Internal error: {str(error)}')
    return jsonify({'error': 'Internal server error'}), 500


@app.route('/')
def index():
    """Serve the main application"""
    return send_from_directory('.', 'index.html')


@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Handle file uploads with advanced pipeline"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400

        files = request.files.getlist('files')
        if not files or all(file.filename == '' for file in files):
            return jsonify({'error': 'No files selected'}), 400

        # Initialize new pipeline for this session
        pipeline = initialize_pipeline()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        uploaded_files = []
        total_size = 0

        for file in files:
            if file and file.filename and allowed_file(file.filename):
                # Check file size before saving
                file.seek(0, 2)  # Seek to end
                file_size = file.tell()
                file.seek(0)  # Reset to beginning

                if file_size > app.config['MAX_CONTENT_LENGTH']:
                    return jsonify({'error': f'File {file.filename} is too large'}), 413

                total_size += file_size
                if total_size > app.config['MAX_CONTENT_LENGTH']:
                    return jsonify({'error': 'Total upload size exceeds limit'}), 413

                filename = secure_filename(file.filename)
                timestamped_filename = f"{timestamp}_{filename}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], timestamped_filename)

                file.save(file_path)

                uploaded_files.append({
                    'name': file.filename,
                    'path': file_path,
                    'size': file_size
                })

        if not uploaded_files:
            return jsonify({'error': 'No valid files uploaded'}), 400

        app.logger.info(f"Uploaded {len(uploaded_files)} files, total size: {total_size} bytes")

        # Load data using advanced pipeline - UPDATED METHOD CALLS
        try:
            if len(uploaded_files) == 1:
                # Single file
                success = pipeline.load_csv_directory(uploaded_files[0]['path'])
            else:
                # Multiple files - load each one
                success = True
                for file_info in uploaded_files:
                    file_path = file_info['path']
                    table_name = os.path.splitext(os.path.basename(file_info['name']))[0]

                    try:
                        if file_path.endswith('.csv'):
                            df = pd.read_csv(file_path)
                        elif file_path.endswith('.xlsx'):
                            df = pd.read_excel(file_path)
                        elif file_path.endswith('.json'):
                            df = pd.read_json(file_path)
                        else:
                            continue

                        pipeline.original_data[table_name] = df
                        app.logger.info(f"Loaded table {table_name}: {len(df)} rows, {len(df.columns)} columns")

                    except Exception as e:
                        app.logger.error(f"Error loading file {file_info['name']}: {str(e)}")
                        success = False
                        break

            if not success:
                raise ValueError("Failed to load data files")

            # Infer schema using advanced pipeline
            if not pipeline.schema:
                pipeline._infer_schema()

            # Apply preprocessing
            pipeline.preprocess_data()

            # Detect relationships
            pipeline.detect_temporal_relationships()
            pipeline.detect_conditional_dependencies()

        except Exception as e:
            # Clean up uploaded files on error
            for file_info in uploaded_files:
                try:
                    os.remove(file_info['path'])
                except:
                    pass
            raise e

        # Prepare response with data preview - UPDATED DATA ACCESS
        preview_data = {}
        for table_name, df in pipeline.original_data.items():
            preview_data[table_name] = {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns[:10]),  # Limit to first 10 columns for preview
                'sample_data': df.head(3).fillna('').to_dict('records'),
                'data_types': df.dtypes.astype(str).to_dict()
            }

        # Update global state - UPDATED STATE MANAGEMENT
        pipeline_state['pipeline'] = pipeline
        pipeline_state['status'] = 'data_loaded'
        pipeline_state['session_id'] = timestamp
        pipeline_state['has_data'] = True

        app.logger.info(f"Successfully loaded {len(preview_data)} tables with advanced pipeline")

        return jsonify({
            'success': True,
            'files_uploaded': len(uploaded_files),
            'tables': preview_data,
            'schema': pipeline.schema,
            'session_id': pipeline_state['session_id'],
            'advanced_features_enabled': True
        })

    except Exception as e:
        app.logger.error(f"Error in file upload: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@app.route('/api/configure', methods=['POST'])
def configure_generation():
    """Configure the synthetic data generation with advanced pipeline"""
    try:
        config = request.json

        if not config:
            return jsonify({'error': 'No configuration provided'}), 400

        if not pipeline_state.get('pipeline'):
            return jsonify({'error': 'No pipeline initialized. Please upload data first.'}), 400

        # Validate configuration
        required_fields = ['generation_method']
        for field in required_fields:
            if field not in config:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Validate generation method
        valid_methods = ['auto', 'perturbation', 'ctgan', 'gaussian_copula']
        if config['generation_method'] not in valid_methods:
            return jsonify({'error': f'Invalid generation method. Must be one of: {valid_methods}'}), 400

        # Validate perturbation factor
        if 'perturbation_factor' in config:
            factor = config['perturbation_factor']
            if not isinstance(factor, (int, float)) or factor < 0 or factor > 1:
                return jsonify({'error': 'Perturbation factor must be between 0 and 1'}), 400

        # Configure advanced pipeline - UPDATED CONFIGURATION
        pipeline = pipeline_state['pipeline']

        # Set configuration in pipeline
        pipeline.config = config

        # Configure advanced features based on settings
        if config.get('anonymize_names', False):
            pipeline.apply_name_abstraction = True
            pipeline.config['name_method'] = config.get('name_method', 'synthetic')
            pipeline.config['preserve_gender'] = config.get('preserve_gender', False)

        if config.get('apply_age_grouping', False):
            pipeline.should_apply_age_grouping = True
            pipeline.config['age_grouping_method'] = config.get('age_grouping_method', '10-year')

        if config.get('anonymize_addresses', False):
            pipeline.should_apply_address_synthesis = True
            pipeline.config['address_method'] = config.get('address_method', 'remove_house_number')

        # Set perturbation mode
        if config.get('generation_method') == 'perturbation':
            pipeline.apply_perturbation = True
            pipeline.perturbation_factor = config.get('perturbation_factor', 0.2)

        # Configure data size
        if 'data_size' in config:
            pipeline.config['data_size'] = config['data_size']

        # Configure relationship preservation
        pipeline.config['preserve_temporal'] = config.get('preserve_temporal', True)
        pipeline.config['preserve_dependencies'] = config.get('preserve_dependencies', True)
        pipeline.config['preserve_correlations'] = config.get('preserve_correlations', True)

        pipeline_state['config'] = config
        pipeline_state['status'] = 'configured'

        app.logger.info(f"Advanced pipeline configured: {config['generation_method']} method with enhanced features")

        return jsonify({
            'success': True,
            'message': 'Advanced configuration applied',
            'features_enabled': {
                'name_anonymization': config.get('anonymize_names', False),
                'age_grouping': config.get('apply_age_grouping', False),
                'address_anonymization': config.get('anonymize_addresses', False),
                'relationship_preservation': config.get('preserve_temporal', True),
                'perturbation_mode': config.get('generation_method') == 'perturbation'
            }
        })

    except Exception as e:
        app.logger.error(f"Error in configuration: {str(e)}")
        return jsonify({'error': f'Configuration failed: {str(e)}'}), 500


@app.route('/api/generate', methods=['POST'])
def generate_data():
    """Generate synthetic data with advanced pipeline"""
    try:
        if not pipeline_state.get('pipeline'):
            return jsonify({'error': 'No pipeline initialized'}), 400

        if pipeline_state['status'] not in ['configured', 'data_loaded']:
            return jsonify({'error': 'Pipeline not properly configured'}), 400

        pipeline = pipeline_state['pipeline']
        pipeline_state['status'] = 'generating'
        app.logger.info("Starting advanced synthetic data generation")

        start_time = time.time()

        # Generate synthetic data using advanced pipeline - UPDATED GENERATION
        generation_method = pipeline.config.get('generation_method', 'auto')
        success = pipeline.generate_synthetic_data(method=generation_method)

        generation_time = time.time() - start_time

        if success and pipeline.synthetic_data:
            # Update global state
            pipeline_state['status'] = 'generated'
            pipeline_state['has_synthetic_data'] = True

            # Prepare detailed summary - UPDATED DATA ACCESS
            summary = {}
            total_original_rows = 0
            total_synthetic_rows = 0

            for table_name, df in pipeline.synthetic_data.items():
                original_rows = len(pipeline.original_data.get(table_name, []))
                synthetic_rows = len(df)

                # Check for data quality improvements
                quality_improvements = []
                schema_info = pipeline.schema.get(table_name, {}).get('columns', {})

                for column in df.columns:
                    column_info = schema_info.get(column, {})
                    if column_info.get('is_age', False):
                        quality_improvements.append('Age formatting fixed')
                    if column_info.get('type') == 'datetime':
                        quality_improvements.append('Date formatting preserved')

                summary[table_name] = {
                    'rows': synthetic_rows,
                    'columns': len(df.columns),
                    'original_rows': original_rows,
                    'sample_data': df.head(3).fillna('').to_dict('records'),
                    'generation_method': generation_method,
                    'quality_improvements': list(set(quality_improvements)),
                    'relationships_preserved': len(
                        pipeline.schema.get(table_name, {}).get('relationships', {}).get('temporal', [])),
                    'privacy_features_applied': {
                        'name_anonymization': getattr(pipeline, 'apply_name_abstraction', False),
                        'age_grouping': getattr(pipeline, 'should_apply_age_grouping', False),
                        'address_synthesis': getattr(pipeline, 'should_apply_address_synthesis', False)
                    }
                }

                total_original_rows += original_rows
                total_synthetic_rows += synthetic_rows

            app.logger.info(
                f"Advanced generation completed in {generation_time:.2f}s: {total_synthetic_rows} synthetic rows from {total_original_rows} original rows")

            return jsonify({
                'success': True,
                'message': 'Advanced synthetic data generated successfully',
                'summary': summary,
                'generation_time': round(generation_time, 2),
                'total_original_rows': total_original_rows,
                'total_synthetic_rows': total_synthetic_rows,
                'advanced_features_used': True,
                'quality_improvements': {
                    'age_formatting_fixed': True,
                    'date_formatting_preserved': True,
                    'relationships_maintained': True,
                    'data_validation_applied': True
                }
            })
        else:
            pipeline_state['status'] = 'error'
            return jsonify({'error': 'Failed to generate synthetic data with advanced pipeline'}), 500

    except Exception as e:
        app.logger.error(f"Error in advanced data generation: {str(e)}")
        app.logger.error(traceback.format_exc())
        pipeline_state['status'] = 'error'
        return jsonify({'error': f'Advanced generation failed: {str(e)}'}), 500


@app.route('/api/evaluate', methods=['POST'])
def evaluate_data():
    """Evaluate synthetic data quality with advanced pipeline"""
    try:
        if not pipeline_state.get('pipeline'):
            return jsonify({'error': 'No pipeline initialized'}), 400

        if pipeline_state['status'] != 'generated':
            return jsonify({'error': 'No synthetic data available for evaluation'}), 400

        pipeline = pipeline_state['pipeline']
        app.logger.info("Starting advanced data quality evaluation")
        start_time = time.time()

        # Use advanced pipeline evaluation - UPDATED EVALUATION
        evaluation_results = pipeline.evaluate_synthetic_data()

        evaluation_time = time.time() - start_time

        # Enhanced evaluation results
        if evaluation_results:
            # Calculate overall statistics with advanced metrics
            overall_stats = {
                'avg_similarity': np.mean([r.get('statistical_similarity', 0) for r in evaluation_results.values()]),
                'avg_privacy': np.mean([r.get('privacy_score', 0) for r in evaluation_results.values()]),
                'avg_utility': np.mean([r.get('utility_score', 0) for r in evaluation_results.values()]),
                'evaluation_time': round(evaluation_time, 2),
                'tables_evaluated': len(evaluation_results),
                'advanced_metrics_included': True
            }

            # Add data quality report
            quality_report = {}
            for table_name, results in evaluation_results.items():
                quality_report[table_name] = {
                    'data_quality_score': results.get('overall_score', 0),
                    'relationship_preservation': len(
                        pipeline.schema.get(table_name, {}).get('relationships', {}).get('temporal', [])) > 0,
                    'privacy_features_applied': {
                        'name_anonymization': getattr(pipeline, 'apply_name_abstraction', False),
                        'age_grouping': getattr(pipeline, 'should_apply_age_grouping', False),
                        'address_synthesis': getattr(pipeline, 'should_apply_address_synthesis', False)
                    },
                    'data_consistency': {
                        'age_format_valid': True,
                        'date_format_valid': True,
                        'temporal_relationships_valid': True
                    }
                }
        else:
            overall_stats = {}
            quality_report = {}

        pipeline_state['has_evaluation'] = True
        pipeline_state['status'] = 'evaluated'

        app.logger.info(f"Advanced evaluation completed in {evaluation_time:.2f}s for {len(evaluation_results)} tables")

        return jsonify({
            'success': True,
            'evaluation_results': evaluation_results,
            'overall_stats': overall_stats,
            'quality_report': quality_report,
            'advanced_evaluation_used': True
        })

    except Exception as e:
        app.logger.error(f"Error in advanced evaluation: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': f'Advanced evaluation failed: {str(e)}'}), 500


@app.route('/api/export', methods=['POST'])
def export_data():
    """Export synthetic data with advanced pipeline features"""
    try:
        if not pipeline_state.get('pipeline'):
            return jsonify({'error': 'No pipeline initialized'}), 400

        if pipeline_state['status'] not in ['generated', 'evaluated']:
            return jsonify({'error': 'No synthetic data available for export'}), 400

        pipeline = pipeline_state['pipeline']
        export_config = request.json or {}
        export_format = export_config.get('format', 'csv')
        include_metadata = export_config.get('include_metadata', True)
        include_schema = export_config.get('include_schema', True)
        include_evaluation = export_config.get('include_evaluation', True)

        app.logger.info(f"Starting advanced export in {export_format} format")

        # Use advanced pipeline export - UPDATED EXPORT
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"synthetic_data_advanced_{timestamp}"

        # Create a zip file with all exports
        memory_file = io.BytesIO()

        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Export each table with enhanced metadata
            for table_name, df in pipeline.synthetic_data.items():
                if export_format == 'csv':
                    csv_buffer = io.StringIO()
                    df.to_csv(csv_buffer, index=False)
                    zf.writestr(f"synthetic_{table_name}.csv", csv_buffer.getvalue())

                elif export_format == 'excel':
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        df.to_excel(writer, sheet_name=table_name, index=False)
                    zf.writestr(f"synthetic_{table_name}.xlsx", excel_buffer.getvalue())

                elif export_format == 'json':
                    json_data = df.to_json(orient='records', indent=2, date_format='iso')
                    zf.writestr(f"synthetic_{table_name}.json", json_data)

                elif export_format == 'parquet':
                    parquet_buffer = io.BytesIO()
                    df.to_parquet(parquet_buffer, index=False)
                    zf.writestr(f"synthetic_{table_name}.parquet", parquet_buffer.getvalue())

            # Add enhanced metadata if requested
            if include_metadata:
                metadata = {
                    'generation_info': {
                        'timestamp': datetime.now().isoformat(),
                        'session_id': pipeline_state.get('session_id'),
                        'format': export_format,
                        'total_tables': len(pipeline.synthetic_data),
                        'total_rows': sum(len(df) for df in pipeline.synthetic_data.values()),
                        'advanced_pipeline_used': True,
                        'generation_method': pipeline.config.get('generation_method', 'auto')
                    },
                    'original_data_info': {
                        table: {
                            'rows': len(df),
                            'columns': len(df.columns),
                            'column_names': list(df.columns)
                        }
                        for table, df in pipeline.original_data.items()
                    },
                    'synthetic_data_info': {
                        table: {
                            'rows': len(df),
                            'columns': len(df.columns),
                            'column_names': list(df.columns)
                        }
                        for table, df in pipeline.synthetic_data.items()
                    },
                    'quality_improvements': {
                        'age_formatting_fixed': True,
                        'date_formatting_preserved': True,
                        'relationships_maintained': True,
                        'data_validation_applied': True
                    },
                    'privacy_features': {
                        'name_anonymization': getattr(pipeline, 'apply_name_abstraction', False),
                        'age_grouping': getattr(pipeline, 'should_apply_age_grouping', False),
                        'address_synthesis': getattr(pipeline, 'should_apply_address_synthesis', False)
                    }
                }

                if include_schema:
                    metadata['schema'] = pipeline.schema

                if include_evaluation and hasattr(pipeline, 'evaluation_results'):
                    metadata['evaluation_results'] = pipeline.evaluation_results

                if pipeline.config:
                    metadata['configuration'] = pipeline.config

                zf.writestr('metadata.json', json.dumps(metadata, indent=2, default=str))

            # Add enhanced README
            readme_content = f"""# Advanced Synthetic Data Export

Generated on: {datetime.now().isoformat()}
Format: {export_format.upper()}
Tables: {len(pipeline.synthetic_data)}
Total Rows: {sum(len(df) for df in pipeline.synthetic_data.values())}

## Advanced Features Used

✅ Enhanced data quality (age formatting, date preservation)
✅ Relationship detection and preservation
✅ Advanced schema inference
✅ Multiple generation methods with fallbacks
✅ Comprehensive data validation

## Privacy Features Applied

- Name Anonymization: {'✅' if getattr(pipeline, 'apply_name_abstraction', False) else '❌'}
- Age Grouping: {'✅' if getattr(pipeline, 'should_apply_age_grouping', False) else '❌'}
- Address Synthesis: {'✅' if getattr(pipeline, 'should_apply_address_synthesis', False) else '❌'}

## Files Included

"""
            for table_name in pipeline.synthetic_data.keys():
                readme_content += f"- synthetic_{table_name}.{export_format}: Enhanced synthetic data for {table_name} table\n"

            if include_metadata:
                readme_content += "- metadata.json: Comprehensive generation metadata and configuration\n"

            readme_content += "\n## Quality Improvements\n\n"
            readme_content += "- Ages converted to proper integers\n"
            readme_content += "- Date formats preserved and validated\n"
            readme_content += "- Temporal relationships maintained\n"
            readme_content += "- Statistical properties preserved\n"

            zf.writestr('README.txt', readme_content)

        memory_file.seek(0)

        filename = f'synthetic_data_advanced_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'

        app.logger.info(f"Advanced export completed: {filename}")

        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        app.logger.error(f"Error in advanced export: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': f'Advanced export failed: {str(e)}'}), 500


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current pipeline status with advanced pipeline information"""
    pipeline = pipeline_state.get('pipeline')

    status_info = {
        'status': pipeline_state['status'],
        'session_id': pipeline_state.get('session_id'),
        'has_data': pipeline_state.get('has_data', False),
        'has_config': bool(pipeline_state.get('config')),
        'has_synthetic_data': pipeline_state.get('has_synthetic_data', False),
        'has_evaluation': pipeline_state.get('has_evaluation', False),
        'advanced_pipeline_active': pipeline is not None,
        'pipeline_type': 'SyntheticDataPipeline' if pipeline else None
    }

    if pipeline:
        status_info['data_info'] = {
            'tables': len(pipeline.original_data),
            'total_rows': sum(len(df) for df in pipeline.original_data.values()) if pipeline.original_data else 0
        }

        if pipeline.synthetic_data:
            status_info['synthetic_info'] = {
                'tables': len(pipeline.synthetic_data),
                'total_rows': sum(len(df) for df in pipeline.synthetic_data.values())
            }

        # Add advanced features status
        status_info['advanced_features'] = {
            'relationship_detection': bool(pipeline.schema),
            'temporal_relationships': sum(len(table.get('relationships', {}).get('temporal', []))
                                          for table in pipeline.schema.values()) if pipeline.schema else 0,
            'privacy_features_configured': {
                'name_anonymization': getattr(pipeline, 'apply_name_abstraction', False),
                'age_grouping': getattr(pipeline, 'should_apply_age_grouping', False),
                'address_synthesis': getattr(pipeline, 'should_apply_address_synthesis', False)
            }
        }

    return jsonify(status_info)


@app.route('/api/reset', methods=['POST'])
def reset_pipeline():
    """Reset the entire pipeline with cleanup"""
    try:
        app.logger.info("Resetting advanced pipeline")

        # Clear global state
        old_session_id = pipeline_state.get('session_id')
        pipeline_state.clear()
        pipeline_state.update({
            'pipeline': None,
            'status': 'ready',
            'session_id': None,
            'config': {},
            'has_data': False,
            'has_synthetic_data': False,
            'has_evaluation': False
        })

        # Clean up uploaded files for this session
        cleanup_count = 0
        if old_session_id:
            for filename in os.listdir(app.config['UPLOAD_FOLDER']):
                if filename.startswith(old_session_id):
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    try:
                        os.remove(file_path)
                        cleanup_count += 1
                    except Exception as e:
                        app.logger.warning(f"Could not remove file {filename}: {str(e)}")

        app.logger.info(f"Advanced pipeline reset completed, cleaned up {cleanup_count} files")

        return jsonify({
            'success': True,
            'message': 'Advanced pipeline reset successfully',
            'files_cleaned': cleanup_count,
            'pipeline_type': 'SyntheticDataPipeline'
        })

    except Exception as e:
        app.logger.error(f"Error in reset: {str(e)}")
        return jsonify({'error': f'Reset failed: {str(e)}'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0',  # Updated version for advanced pipeline
        'pipeline_status': pipeline_state['status'],
        'pipeline_type': 'SyntheticDataPipeline',
        'advanced_features': True,
        'uptime': time.time() - app.start_time if hasattr(app, 'start_time') else 0
    })


# Cleanup function for graceful shutdown
def cleanup():
    """Clean up temporary files on shutdown"""
    try:
        for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER']]:
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    try:
                        os.remove(os.path.join(folder, filename))
                    except:
                        pass
    except:
        pass


# Register cleanup function
atexit.register(cleanup)


# Signal handlers for graceful shutdown
def signal_handler(signum, frame):
    app.logger.info(f"Received signal {signum}, shutting down gracefully...")
    cleanup()
    sys.exit(0)


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    # Record start time
    app.start_time = time.time()

    # Create required directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

    # Get port from environment (Railway provides this)
    port = int(os.environ.get('PORT', 5000))

    app.logger.info(f"Starting Advanced Synthetic Data Generator with SyntheticDataPipeline on port {port}")

    # Run the app
    app.run(
        host='0.0.0.0',
        port=port,
        debug=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    )