#!/usr/bin/env python3
"""
Simplified Flask app for Railway deployment
Fallback version with minimal dependencies
"""

import os
import sys
import logging
import traceback
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'railway-fallback-secret-key-2024'
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
    OUTPUT_FOLDER = os.environ.get('OUTPUT_FOLDER', 'outputs')
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 100 * 1024 * 1024))
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'json'}
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*')

# Create Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Initialize CORS
CORS(app, origins=app.config['CORS_ORIGINS'])

# Create directories
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['OUTPUT_FOLDER']).mkdir(exist_ok=True)

# Simple data storage
data_store = {
    'original_data': {},
    'synthetic_data': {},
    'status': 'ready'
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_simple_synthetic_data(original_df):
    """Simple synthetic data generation using basic techniques"""
    synthetic_df = original_df.copy()
    
    for column in synthetic_df.columns:
        if pd.api.types.is_numeric_dtype(synthetic_df[column]):
            # Add small random noise to numeric columns
            std_dev = synthetic_df[column].std()
            if std_dev > 0:
                noise = np.random.normal(0, std_dev * 0.1, len(synthetic_df))
                synthetic_df[column] = synthetic_df[column] + noise
                
                # Handle age columns specifically
                if 'age' in column.lower():
                    synthetic_df[column] = synthetic_df[column].clip(0, 120).round().astype(int)
        
        elif pd.api.types.is_datetime64_any_dtype(synthetic_df[column]):
            # Add random days to dates
            try:
                random_days = np.random.randint(-5, 6, len(synthetic_df))
                synthetic_df[column] = pd.to_datetime(synthetic_df[column]) + pd.Timedelta(days=1) * random_days
            except:
                pass
    
    return synthetic_df

# Routes
@app.route('/')
def index():
    return jsonify({
        'message': 'Synthetic Data Generator - Railway Simple Version',
        'status': 'running',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0-railway-simple',
        'data_status': data_store['status']
    })

@app.route('/api/health')
def api_health():
    return health_check()

@app.route('/api/upload', methods=['POST'])
def upload_files():
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        if not files or all(file.filename == '' for file in files):
            return jsonify({'error': 'No files selected'}), 400
        
        uploaded_files = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for file in files:
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                timestamped_filename = f"{timestamp}_{filename}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], timestamped_filename)
                
                file.save(file_path)
                
                # Load the data
                try:
                    if filename.endswith('.csv'):
                        df = pd.read_csv(file_path)
                    elif filename.endswith(('.xlsx', '.xls')):
                        df = pd.read_excel(file_path)
                    elif filename.endswith('.json'):
                        df = pd.read_json(file_path)
                    else:
                        continue
                    
                    table_name = filename.rsplit('.', 1)[0]
                    data_store['original_data'][table_name] = df
                    
                    uploaded_files.append({
                        'name': filename,
                        'table_name': table_name,
                        'rows': len(df),
                        'columns': len(df.columns)
                    })
                    
                    logger.info(f"Loaded {table_name}: {len(df)} rows, {len(df.columns)} columns")
                    
                except Exception as e:
                    logger.error(f"Error loading {filename}: {str(e)}")
                    continue
        
        if uploaded_files:
            data_store['status'] = 'data_loaded'
            
            # Create preview
            preview = {}
            for table_name, df in data_store['original_data'].items():
                preview[table_name] = {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'column_names': list(df.columns),
                    'sample_data': df.head(3).fillna('').to_dict('records')
                }
            
            return jsonify({
                'success': True,
                'files_uploaded': len(uploaded_files),
                'tables': preview,
                'message': 'Files uploaded successfully'
            })
        else:
            return jsonify({'error': 'No valid files processed'}), 400
            
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/generate', methods=['POST'])
def generate_data():
    try:
        if not data_store['original_data']:
            return jsonify({'error': 'No data loaded'}), 400
        
        logger.info("Starting simple synthetic data generation")
        data_store['status'] = 'generating'
        
        # Generate synthetic data for each table
        for table_name, original_df in data_store['original_data'].items():
            try:
                synthetic_df = generate_simple_synthetic_data(original_df)
                data_store['synthetic_data'][table_name] = synthetic_df
                logger.info(f"Generated synthetic data for {table_name}: {len(synthetic_df)} rows")
            except Exception as e:
                logger.error(f"Error generating data for {table_name}: {str(e)}")
                continue
        
        if data_store['synthetic_data']:
            data_store['status'] = 'generated'
            
            # Create summary
            summary = {}
            for table_name, df in data_store['synthetic_data'].items():
                summary[table_name] = {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'sample_data': df.head(2).fillna('').to_dict('records'),
                    'generation_method': 'simple_noise'
                }
            
            return jsonify({
                'success': True,
                'message': 'Synthetic data generated successfully',
                'summary': summary,
                'method_used': 'simple_synthetic_generation'
            })
        else:
            return jsonify({'error': 'Failed to generate synthetic data'}), 500
            
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        data_store['status'] = 'error'
        return jsonify({'error': f'Generation failed: {str(e)}'}), 500

@app.route('/api/export', methods=['POST'])
def export_data():
    try:
        if not data_store['synthetic_data']:
            return jsonify({'error': 'No synthetic data to export'}), 400
        
        export_format = request.json.get('format', 'csv') if request.json else 'csv'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        import io
        import zipfile
        
        memory_file = io.BytesIO()
        
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for table_name, df in data_store['synthetic_data'].items():
                if export_format == 'csv':
                    csv_buffer = io.StringIO()
                    df.to_csv(csv_buffer, index=False)
                    zf.writestr(f"synthetic_{table_name}.csv", csv_buffer.getvalue())
                elif export_format == 'json':
                    json_data = df.to_json(orient='records', indent=2)
                    zf.writestr(f"synthetic_{table_name}.json", json_data)
            
            # Add metadata
            metadata = {
                'generation_timestamp': datetime.now().isoformat(),
                'tables': list(data_store['synthetic_data'].keys()),
                'total_rows': sum(len(df) for df in data_store['synthetic_data'].values()),
                'method': 'simple_synthetic_generation',
                'version': '1.0.0-railway-simple'
            }
            zf.writestr('metadata.json', str(metadata))
        
        memory_file.seek(0)
        filename = f'synthetic_data_simple_{timestamp}.zip'
        
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        return jsonify({'error': f'Export failed: {str(e)}'}), 500

@app.route('/api/status')
def get_status():
    return jsonify({
        'status': data_store['status'],
        'has_data': bool(data_store['original_data']),
        'has_synthetic_data': bool(data_store['synthetic_data']),
        'tables': list(data_store['original_data'].keys()),
        'version': '1.0.0-railway-simple'
    })

@app.route('/api/reset', methods=['POST'])
def reset_data():
    data_store['original_data'] = {}
    data_store['synthetic_data'] = {}
    data_store['status'] = 'ready'
    
    return jsonify({
        'success': True,
        'message': 'Data reset successfully'
    })

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(500)
def internal_error(error):
    logger.error(f'Internal error: {str(error)}')
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Get port from environment (Railway provides this)
    port = int(os.environ.get('PORT', 5000))
    
    logger.info(f"Starting Simple Synthetic Data Generator on port {port}")
    
    # Run the app
    app.run(
        host='0.0.0.0',
        port=port,
        debug=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    )