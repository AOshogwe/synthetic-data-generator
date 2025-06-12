# Synthetic Data Generator API Documentation

## Overview

The Synthetic Data Generator API provides endpoints for uploading data, configuring generation parameters, generating synthetic data, and exporting results. This RESTful API is designed to be intuitive and comprehensive, supporting various data formats and advanced privacy-preserving techniques.

## Base URL

```
https://your-app.railway.app/api
```

## Authentication

Currently, the API does not require authentication. For production deployments, consider implementing API key authentication or OAuth2.

## Rate Limiting

- **Default**: 100 requests per minute
- **Burst**: 200 requests per minute
- Rate limits can be configured via environment variables

## Content Types

- **Request**: `application/json`, `multipart/form-data` (for file uploads)
- **Response**: `application/json`

---

## Endpoints

### 1. Health Check

Check the API health and status.

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "1.0.0",
  "pipeline_status": "ready",
  "uptime": 3600
}
```

---

### 2. Upload Files

Upload data files for synthetic data generation.

**Endpoint**: `POST /upload`

**Content-Type**: `multipart/form-data`

**Parameters**:
- `files[]` (required): One or more data files (CSV, Excel, JSON)

**Supported File Types**:
- CSV (`.csv`)
- Excel (`.xlsx`, `.xls`)
- JSON (`.json`)

**Request Example**:
```bash
curl -X POST \
  https://your-app.railway.app/api/upload \
  -F "files[]=@data.csv" \
  -F "files[]=@data2.xlsx"
```

**Response**:
```json
{
  "success": true,
  "files_uploaded": 2,
  "session_id": "20240101_120000",
  "tables": {
    "data": {
      "rows": 1000,
      "columns": 10,
      "column_names": ["id", "name", "age", "email", "address"],
      "sample_data": [
        {"id": 1, "name": "John Doe", "age": 30, "email": "john@example.com"},
        {"id": 2, "name": "Jane Smith", "age": 25, "email": "jane@example.com"}
      ],
      "data_types": {
        "id": "int64",
        "name": "object",
        "age": "int64",
        "email": "object"
      }
    }
  },
  "schema": {
    "data": {
      "columns": {
        "id": {
          "type": "numeric",
          "subtype": "id",
          "is_id": true,
          "null_count": 0,
          "unique_count": 1000
        },
        "name": {
          "type": "text",
          "subtype": "name",
          "is_name": true,
          "null_count": 0,
          "unique_count": 995
        }
      }
    }
  }
}
```

**Error Responses**:
```json
{
  "error": "No files provided"
}
```

---

### 3. Configure Generation

Configure synthetic data generation parameters.

**Endpoint**: `POST /configure`

**Content-Type**: `application/json`

**Request Body**:
```json
{
  "generation_method": "auto",
  "data_size": {
    "type": "percentage",
    "value": 100
  },
  "anonymize_names": true,
  "name_method": "synthetic",
  "preserve_gender": true,
  "apply_age_grouping": false,
  "age_grouping_method": "10-year",
  "anonymize_addresses": true,
  "address_method": "synthetic",
  "perturbation_factor": 0.2,
  "preserve_temporal": true,
  "preserve_dependencies": true,
  "preserve_correlations": true,
  "correlation_level": "moderate",
  "privacy_level": "balanced",
  "use_seed": false,
  "random_seed": 42
}
```

**Configuration Options**:

#### Generation Method
- `auto`: Automatically select best method based on data characteristics
- `perturbation`: Apply controlled modifications to original data
- `ctgan`: Use Conditional Tabular GAN (requires additional setup)
- `gaussian_copula`: Use Gaussian Copula method

#### Data Size
- `same`: Generate same number of rows as original
- `percentage`: Generate percentage of original rows
- `custom`: Generate specific number of rows

#### Privacy & Anonymization
- `anonymize_names`: Replace names with synthetic ones
- `name_method`: `synthetic`, `initials`, `random`
- `apply_age_grouping`: Convert ages to ranges
- `age_grouping_method`: `5-year`, `10-year`, `life-stages`, `custom`
- `anonymize_addresses`: Replace addresses with synthetic ones
- `address_method`: `remove-numbers`, `street-only`, `city-state`, `zip-only`, `synthetic`

#### Relationship Preservation
- `preserve_temporal`: Maintain temporal relationships between date columns
- `preserve_dependencies`: Maintain conditional dependencies
- `preserve_correlations`: Maintain statistical correlations
- `correlation_level`: `strict`, `moderate`, `loose`

**Response**:
```json
{
  "success": true,
  "message": "Configuration saved"
}
```

---

### 4. Generate Synthetic Data

Generate synthetic data based on uploaded files and configuration.

**Endpoint**: `POST /generate`

**Content-Type**: `application/json`

**Request Body**: Empty (uses previously uploaded data and configuration)

**Response**:
```json
{
  "success": true,
  "message": "Synthetic data generated successfully",
  "generation_time": 12.5,
  "total_original_rows": 1000,
  "total_synthetic_rows": 1000,
  "summary": {
    "data": {
      "rows": 1000,
      "columns": 10,
      "original_rows": 1000,
      "sample_data": [
        {"id": 1001, "name": "Alex Johnson", "age": 28, "email": "alex@example.com"}
      ]
    }
  }
}
```

**Error Responses**:
```json
{
  "error": "Pipeline not properly configured"
}
```

---

### 5. Evaluate Data Quality

Evaluate the quality of generated synthetic data.

**Endpoint**: `POST /evaluate`

**Content-Type**: `application/json`

**Request Body**: Empty

**Response**:
```json
{
  "success": true,
  "evaluation_results": {
    "data": {
      "statistical_similarity": 0.89,
      "privacy_score": 0.92,
      "utility_score": 0.85,
      "overall_score": 0.887,
      "record_count": {
        "original": 1000,
        "synthetic": 1000
      }
    }
  },
  "overall_stats": {
    "avg_similarity": 0.89,
    "avg_privacy": 0.92,
    "avg_utility": 0.85,
    "evaluation_time": 3.2,
    "tables_evaluated": 1
  }
}
```

**Metrics Explanation**:
- `statistical_similarity`: How well synthetic data matches original statistical properties (0-1)
- `privacy_score`: Level of privacy protection achieved (0-1, higher is better)
- `utility_score`: How useful the synthetic data is for analysis/ML (0-1)
- `overall_score`: Weighted average of all metrics

---

### 6. Export Synthetic Data

Export generated synthetic data in various formats.

**Endpoint**: `POST /export`

**Content-Type**: `application/json`

**Request Body**:
```json
{
  "format": "csv",
  "include_metadata": true,
  "include_schema": true,
  "include_evaluation": true,
  "compress_files": true
}
```

**Export Formats**:
- `csv`: Comma-separated values
- `excel`: Excel workbook
- `json`: JSON format
- `parquet`: Parquet format
- `sql`: SQL insert statements

**Response**: Binary ZIP file containing:
- Synthetic data files in requested format
- `metadata.json` (if requested)
- `README.txt` with export information

**Response Headers**:
```
Content-Type: application/zip
Content-Disposition: attachment; filename="synthetic_data_20240101_120000.zip"
```

---

### 7. Get Pipeline Status

Get current status of the data processing pipeline.

**Endpoint**: `GET /status`

**Response**:
```json
{
  "status": "generated",
  "session_id": "20240101_120000",
  "has_data": true,
  "has_config": true,
  "has_synthetic_data": true,
  "has_evaluation": true,
  "data_info": {
    "tables": 1,
    "total_rows": 1000
  },
  "synthetic_info": {
    "tables": 1,
    "total_rows": 1000
  }
}
```

**Status Values**:
- `ready`: System ready for data upload
- `data_loaded`: Data uploaded and processed
- `configured`: Generation configured
- `generating`: Currently generating synthetic data
- `generated`: Synthetic data generated successfully
- `evaluated`: Data quality evaluation completed
- `error`: Error occurred in pipeline

---

### 8. Reset Pipeline

Reset the pipeline and clear all data.

**Endpoint**: `POST /reset`

**Content-Type**: `application/json`

**Request Body**: Empty

**Response**:
```json
{
  "success": true,
  "message": "Pipeline reset successfully",
  "files_cleaned": 3
}
```

---

## Error Handling

### HTTP Status Codes

- `200 OK`: Request successful
- `400 Bad Request`: Invalid request parameters
- `413 Payload Too Large`: File too large
- `500 Internal Server Error`: Server error

### Error Response Format

```json
{
  "error": "Error description",
  "details": "Additional error details (optional)",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Common Errors

#### File Upload Errors
```json
{
  "error": "File too large. Maximum size is 100MB."
}
```

#### Configuration Errors
```json
{
  "error": "Invalid generation method. Must be one of: auto, perturbation, ctgan, gaussian_copula"
}
```

#### Processing Errors
```json
{
  "error": "No original data available for evaluation"
}
```

---

## Usage Examples

### Python Client Example

```python
import requests
import json

# Base URL
base_url = "https://your-app.railway.app/api"

# 1. Upload file
with open('data.csv', 'rb') as f:
    files = {'files': f}
    response = requests.post(f"{base_url}/upload", files=files)
    print(response.json())

# 2. Configure generation
config = {
    "generation_method": "perturbation",
    "anonymize_names": True,
    "perturbation_factor": 0.2
}
response = requests.post(f"{base_url}/configure", json=config)
print(response.json())

# 3. Generate synthetic data
response = requests.post(f"{base_url}/generate")
print(response.json())

# 4. Evaluate quality
response = requests.post(f"{base_url}/evaluate")
print(response.json())

# 5. Export data
export_config = {"format": "csv", "include_metadata": True}
response = requests.post(f"{base_url}/export", json=export_config)

# Save the zip file
with open('synthetic_data.zip', 'wb') as f:
    f.write(response.content)
```

### JavaScript Client Example

```javascript
// 1. Upload file
const formData = new FormData();
formData.append('files', fileInput.files[0]);

const uploadResponse = await fetch('/api/upload', {
    method: 'POST',
    body: formData
});

// 2. Configure generation
const config = {
    generation_method: 'auto',
    anonymize_names: true,
    perturbation_factor: 0.3
};

const configResponse = await fetch('/api/configure', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(config)
});

// 3. Generate synthetic data
const generateResponse = await fetch('/api/generate', {
    method: 'POST'
});

// 4. Export data
const exportConfig = {
    format: 'csv',
    include_metadata: true
};

const exportResponse = await fetch('/api/export', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(exportConfig)
});

// Download the file
const blob = await exportResponse.blob();
const url = window.URL.createObjectURL(blob);
const a = document.createElement('a');
a.href = url;
a.download = 'synthetic_data.zip';
a.click();
```

---

## Rate Limiting

The API implements rate limiting to prevent abuse:

- **Per IP**: 100 requests per minute
- **Burst allowance**: 200 requests in a short period
- **File upload**: Additional size-based limits

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1609459200
```

---

## Security Considerations

### File Upload Security
- File type validation based on extension and content
- Size limits enforced (default: 100MB)
- Secure filename handling
- Automatic cleanup of temporary files

### Data Privacy
- No data persistence between sessions
- Automatic cleanup on session end
- Memory-only processing (no disk storage)
- Privacy-preserving synthetic data generation

### CORS Configuration
- Configurable CORS origins
- Secure headers in production
- HTTPS enforcement recommended

---

## Performance Considerations

### File Size Limits
- Default maximum file size: 100MB per file
- Chunked processing for large files
- Memory management for datasets

### Processing Time
- Small datasets (< 1000 rows): < 10 seconds
- Medium datasets (1000-100K rows): 10-60 seconds
- Large datasets (> 100K rows): 1-10 minutes

### Optimization Tips
- Use CSV format for fastest processing
- Limit unnecessary columns before upload
- Consider data sampling for very large datasets

---

## Monitoring and Logging

### Health Monitoring
- `/health` endpoint for system status
- Automatic health checks every 30 seconds
- Performance metrics tracking

### Logging
- Request/response logging
- Error tracking with stack traces
- Performance metrics
- User activity monitoring (anonymized)

### Metrics Available
- Request count and timing
- Error rates
- File processing statistics
- Memory usage
- Generation success rates

---

## Support and Troubleshooting

### Common Issues

1. **File Upload Fails**
   - Check file size (max 100MB)
   - Verify file format (CSV, Excel, JSON)
   - Ensure file is not corrupted

2. **Generation Takes Too Long**
   - Large datasets require more time
   - Consider reducing data size
   - Check system resource availability

3. **Poor Quality Scores**
   - Try different generation methods
   - Adjust perturbation parameters
   - Ensure sufficient data quality in original

### Getting Help
- Check the error message details
- Review API documentation
- Use the `/status` endpoint to debug pipeline state
- Enable detailed logging for troubleshooting

### Reporting Issues
When reporting issues, include:
- API endpoint used
- Request payload (remove sensitive data)
- Error response received
- File characteristics (size, format, columns)
- Expected vs actual behavior