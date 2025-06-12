# app.py - Enhanced Flask Backend for Synthetic Data Generator
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

# Import the core generator after app creation to avoid circular imports
from synthetic_data_pipeline import SyntheticDataGenerator

# Global state for the pipeline
pipeline_state = {
    'data': {},
    'schema': {},
    'config': {},
    'synthetic_data': {},
    'evaluation_results': {},
    'status': 'ready',
    'session_id': None
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# Initialize the generator
generator = SyntheticDataGenerator()


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
    """Handle file uploads with enhanced error handling"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400

        files = request.files.getlist('files')
        if not files or all(file.filename == '' for file in files):
            return jsonify({'error': 'No files selected'}), 400

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
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{timestamp}_{filename}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                file.save(file_path)

                uploaded_files.append({
                    'name': file.filename,
                    'path': file_path,
                    'size': file_size
                })

        if not uploaded_files:
            return jsonify({'error': 'No valid files uploaded'}), 400

        app.logger.info(f"Uploaded {len(uploaded_files)} files, total size: {total_size} bytes")

        # Load data with timeout protection
        try:
            generator.load_data_from_files(uploaded_files)
        except Exception as e:
            # Clean up uploaded files on error
            for file_info in uploaded_files:
                try:
                    os.remove(file_info['path'])
                except:
                    pass
            raise e

        # Prepare response with data preview
        preview_data = {}
        for table_name, df in generator.original_data.items():
            preview_data[table_name] = {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns[:10]),  # Limit to first 10 columns for preview
                'sample_data': df.head(3).fillna('').to_dict('records'),
                'data_types': df.dtypes.astype(str).to_dict()
            }

        # Update global state
        pipeline_state['data'] = generator.original_data
        pipeline_state['schema'] = generator.schema
        pipeline_state['status'] = 'data_loaded'
        pipeline_state['session_id'] = timestamp

        app.logger.info(f"Successfully loaded {len(preview_data)} tables")

        return jsonify({
            'success': True,
            'files_uploaded': len(uploaded_files),
            'tables': preview_data,
            'schema': generator.schema,
            'session_id': pipeline_state['session_id']
        })

    except Exception as e:
        app.logger.error(f"Error in file upload: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@app.route('/api/configure', methods=['POST'])
def configure_generation():
    """Configure the synthetic data generation with validation"""
    try:
        config = request.json

        if not config:
            return jsonify({'error': 'No configuration provided'}), 400

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

        generator.configure_generation(config)

        pipeline_state['config'] = config
        pipeline_state['status'] = 'configured'

        app.logger.info(f"Configuration updated: {config['generation_method']} method")

        return jsonify({'success': True, 'message': 'Configuration saved'})

    except Exception as e:
        app.logger.error(f"Error in configuration: {str(e)}")
        return jsonify({'error': f'Configuration failed: {str(e)}'}), 500


@app.route('/api/generate', methods=['POST'])
def generate_data():
    """Generate synthetic data with progress tracking"""
    try:
        if pipeline_state['status'] not in ['configured', 'data_loaded']:
            return jsonify({'error': 'Pipeline not properly configured'}), 400

        pipeline_state['status'] = 'generating'
        app.logger.info("Starting synthetic data generation")

        start_time = time.time()

        # Generate synthetic data
        success = generator.generate_synthetic_data()

        generation_time = time.time() - start_time

        if success:
            # Update global state
            pipeline_state['synthetic_data'] = generator.synthetic_data
            pipeline_state['status'] = 'generated'

            # Prepare summary
            summary = {}
            total_original_rows = 0
            total_synthetic_rows = 0

            for table_name, df in generator.synthetic_data.items():
                original_rows = len(generator.original_data.get(table_name, []))
                synthetic_rows = len(df)

                summary[table_name] = {
                    'rows': synthetic_rows,
                    'columns': len(df.columns),
                    'original_rows': original_rows,
                    'sample_data': df.head(3).fillna('').to_dict('records')
                }

                total_original_rows += original_rows
                total_synthetic_rows += synthetic_rows

            app.logger.info(
                f"Generation completed in {generation_time:.2f}s: {total_synthetic_rows} synthetic rows from {total_original_rows} original rows")

            return jsonify({
                'success': True,
                'message': 'Synthetic data generated successfully',
                'summary': summary,
                'generation_time': round(generation_time, 2),
                'total_original_rows': total_original_rows,
                'total_synthetic_rows': total_synthetic_rows
            })
        else:
            pipeline_state['status'] = 'error'
            return jsonify({'error': 'Failed to generate synthetic data'}), 500

    except Exception as e:
        app.logger.error(f"Error in data generation: {str(e)}")
        app.logger.error(traceback.format_exc())
        pipeline_state['status'] = 'error'
        return jsonify({'error': f'Generation failed: {str(e)}'}), 500


@app.route('/api/evaluate', methods=['POST'])
def evaluate_data():
    """Evaluate synthetic data quality with comprehensive metrics"""
    try:
        if pipeline_state['status'] != 'generated':
            return jsonify({'error': 'No synthetic data available for evaluation'}), 400

        app.logger.info("Starting data quality evaluation")
        start_time = time.time()

        evaluation_results = generator.evaluate_synthetic_data()

        evaluation_time = time.time() - start_time

        # Add overall statistics
        if evaluation_results:
            overall_stats = {
                'avg_similarity': np.mean([r.get('statistical_similarity', 0) for r in evaluation_results.values()]),
                'avg_privacy': np.mean([r.get('privacy_score', 0) for r in evaluation_results.values()]),
                'avg_utility': np.mean([r.get('utility_score', 0) for r in evaluation_results.values()]),
                'evaluation_time': round(evaluation_time, 2),
                'tables_evaluated': len(evaluation_results)
            }
        else:
            overall_stats = {}

        pipeline_state['evaluation_results'] = evaluation_results
        pipeline_state['status'] = 'evaluated'

        app.logger.info(f"Evaluation completed in {evaluation_time:.2f}s for {len(evaluation_results)} tables")

        return jsonify({
            'success': True,
            'evaluation_results': evaluation_results,
            'overall_stats': overall_stats
        })

    except Exception as e:
        app.logger.error(f"Error in evaluation: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': f'Evaluation failed: {str(e)}'}), 500


@app.route('/api/export', methods=['POST'])
def export_data():
    """Export synthetic data with multiple format support"""
    try:
        if pipeline_state['status'] not in ['generated', 'evaluated']:
            return jsonify({'error': 'No synthetic data available for export'}), 400

        export_config = request.json or {}
        export_format = export_config.get('format', 'csv')
        include_metadata = export_config.get('include_metadata', True)
        include_schema = export_config.get('include_schema', True)
        include_evaluation = export_config.get('include_evaluation', True)

        app.logger.info(f"Starting export in {export_format} format")

        # Create a zip file with all exports
        memory_file = io.BytesIO()

        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Export each table
            for table_name, df in generator.synthetic_data.items():
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

            # Add metadata if requested
            if include_metadata:
                metadata = {
                    'generation_info': {
                        'timestamp': datetime.now().isoformat(),
                        'session_id': pipeline_state.get('session_id'),
                        'format': export_format,
                        'total_tables': len(generator.synthetic_data),
                        'total_rows': sum(len(df) for df in generator.synthetic_data.values())
                    },
                    'original_data_info': {
                        table: {
                            'rows': len(df),
                            'columns': len(df.columns),
                            'column_names': list(df.columns)
                        }
                        for table, df in generator.original_data.items()
                    },
                    'synthetic_data_info': {
                        table: {
                            'rows': len(df),
                            'columns': len(df.columns),
                            'column_names': list(df.columns)
                        }
                        for table, df in generator.synthetic_data.items()
                    }
                }

                if include_schema:
                    metadata['schema'] = generator.schema

                if include_evaluation and pipeline_state.get('evaluation_results'):
                    metadata['evaluation_results'] = pipeline_state['evaluation_results']

                if generator.config:
                    metadata['configuration'] = generator.config

                zf.writestr('metadata.json', json.dumps(metadata, indent=2, default=str))

            # Add README
            readme_content = f"""# Synthetic Data Export

Generated on: {datetime.now().isoformat()}
Format: {export_format.upper()}
Tables: {len(generator.synthetic_data)}
Total Rows: {sum(len(df) for df in generator.synthetic_data.values())}

## Files Included

"""
            for table_name in generator.synthetic_data.keys():
                readme_content += f"- synthetic_{table_name}.{export_format}: Synthetic data for {table_name} table\n"

            if include_metadata:
                readme_content += "- metadata.json: Generation metadata and configuration\n"

            zf.writestr('README.txt', readme_content)

        memory_file.seek(0)

        filename = f'synthetic_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'

        app.logger.info(f"Export completed: {filename}")

        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        app.logger.error(f"Error in export: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': f'Export failed: {str(e)}'}), 500


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current pipeline status with detailed information"""
    return jsonify({
        'status': pipeline_state['status'],
        'session_id': pipeline_state.get('session_id'),
        'has_data': bool(pipeline_state['data']),
        'has_config': bool(pipeline_state['config']),
        'has_synthetic_data': bool(pipeline_state['synthetic_data']),
        'has_evaluation': bool(pipeline_state['evaluation_results']),
        'data_info': {
            'tables': len(pipeline_state['data']),
            'total_rows': sum(len(df) for df in pipeline_state['data'].values()) if pipeline_state['data'] else 0
        } if pipeline_state['data'] else None,
        'synthetic_info': {
            'tables': len(pipeline_state['synthetic_data']),
            'total_rows': sum(len(df) for df in pipeline_state['synthetic_data'].values()) if pipeline_state[
                'synthetic_data'] else 0
        } if pipeline_state['synthetic_data'] else None
    })


@app.route('/api/reset', methods=['POST'])
def reset_pipeline():
    """Reset the entire pipeline with cleanup"""
    try:
        app.logger.info("Resetting pipeline")

        # Clear global state
        old_session_id = pipeline_state.get('session_id')
        pipeline_state.clear()
        pipeline_state.update({
            'data': {},
            'schema': {},
            'config': {},
            'synthetic_data': {},
            'evaluation_results': {},
            'status': 'ready',
            'session_id': None
        })

        # Reset generator
        generator.original_data = {}
        generator.synthetic_data = {}
        generator.schema = {}
        generator.config = {}

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

        app.logger.info(f"Pipeline reset completed, cleaned up {cleanup_count} files")

        return jsonify({
            'success': True,
            'message': 'Pipeline reset successfully',
            'files_cleaned': cleanup_count
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
        'version': '1.0.0',
        'pipeline_status': pipeline_state['status'],
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

    app.logger.info(f"Starting Synthetic Data Generator on port {port}")

    # Run the app
    app.run(
        host='0.0.0.0',
        port=port,
        debug=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    )