# Deployment Guide - Synthetic Data Generator

## 🚀 Production Deployment Checklist

### 1. Environment Setup

#### Required Environment Variables
```bash
# CRITICAL: Set a strong secret key
export SECRET_KEY="your-super-secure-random-key-at-least-32-characters-long"

# Flask Configuration
export FLASK_ENV="production"
export DEBUG="False"

# File Upload Limits
export MAX_CONTENT_LENGTH="100"  # MB
export UPLOAD_FOLDER="/app/uploads"
export OUTPUT_FOLDER="/app/outputs"

# CORS (restrict in production)
export CORS_ORIGINS="https://yourdomain.com"

# Logging
export LOG_LEVEL="INFO"
export LOG_FILE="/var/log/synthetic-data/app.log"

# Security
export RATE_LIMIT_ENABLED="True"
export MAX_REQUESTS_PER_MINUTE="60"
```

#### Copy Environment Template
```bash
cp .env.example .env
# Edit .env with your actual values
```

### 2. Security Hardening

#### ✅ Implemented Security Features:
- [x] File validation with MIME type checking
- [x] Rate limiting (100 requests per 5 minutes)
- [x] Input sanitization and injection prevention
- [x] Secure headers (X-Frame-Options, CSP, etc.)
- [x] File quarantine system for malicious uploads
- [x] Database connection pooling with prepared statements
- [x] Secret key validation
- [x] IP blocking for repeated violations

#### Additional Security Recommendations:
```bash
# Use a reverse proxy (nginx/Apache)
# Configure SSL/TLS certificates
# Set up firewall rules
# Enable fail2ban for brute force protection
```

### 3. Database Security

#### Connection String Examples:
```bash
# PostgreSQL (recommended)
export DATABASE_URL="postgresql://username:password@localhost:5432/synthetic_data?sslmode=require"

# MySQL
export MYSQL_URL="mysql://username:password@localhost:3306/synthetic_data?charset=utf8mb4&ssl=true"
```

#### Database Permissions:
- Create a dedicated database user with minimal permissions
- Grant only necessary privileges (SELECT, INSERT, UPDATE)
- Never use root/admin database accounts

### 4. File System Security

#### Directory Permissions:
```bash
# Create secure directories
mkdir -p /app/uploads /app/outputs /app/logs /app/quarantine
chmod 750 /app/uploads /app/outputs /app/quarantine
chmod 755 /app/logs

# Set ownership (assuming app runs as 'appuser')
chown -R appuser:appuser /app/
```

#### Storage Recommendations:
- Use separate mount points for uploads/outputs
- Implement disk quotas
- Regular cleanup of old files
- Consider cloud storage for production

### 5. Monitoring and Logging

#### Log Rotation:
```bash
# Add to /etc/logrotate.d/synthetic-data-app
/var/log/synthetic-data/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0644 appuser appuser
    postrotate
        systemctl reload synthetic-data-app
    endscript
}
```

#### Health Check Endpoint:
The app includes `/api/health` endpoint for monitoring.

### 6. Performance Optimization

#### Memory Management:
```bash
# Set appropriate limits
export WORKERS="4"
export TIMEOUT="300"
export MAX_MEMORY_PER_WORKER="512M"
```

#### Caching:
- Implement Redis for session storage
- Use CDN for static assets
- Enable gzip compression

### 7. Backup Strategy

#### Database Backups:
```bash
# Automated backup script
#!/bin/bash
pg_dump $DATABASE_URL > backup_$(date +%Y%m%d_%H%M%S).sql
# Upload to secure storage
```

#### File Backups:
- Regular backups of configuration files
- Encrypted backups of sensitive data
- Test restore procedures

### 8. Container Deployment (Docker)

#### Dockerfile:
```dockerfile
FROM python:3.9-slim

# Security: Create non-root user
RUN useradd -m -u 1000 appuser

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app

# Set ownership and permissions
RUN chown -R appuser:appuser /app
USER appuser

# Security hardening
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "app:app"]
```

### 9. Web Server Configuration

#### Nginx Configuration:
```nginx
server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    # SSL configuration
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";

    # File upload limits
    client_max_body_size 100M;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=60r/m;
    
    location /api/ {
        limit_req zone=api burst=20 nodelay;
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location / {
        try_files $uri $uri/ =404;
    }
}
```

### 10. Process Management

#### Systemd Service:
```ini
[Unit]
Description=Synthetic Data Generator
After=network.target

[Service]
Type=simple
User=appuser
WorkingDirectory=/app
Environment=PATH=/app/venv/bin
ExecStart=/app/venv/bin/gunicorn --bind 127.0.0.1:8000 --workers 4 app:app
Restart=always
RestartSec=10

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/app/uploads /app/outputs /var/log/synthetic-data

[Install]
WantedBy=multi-user.target
```

### 11. Testing Production Deployment

#### Pre-deployment Tests:
```bash
# Test file upload
curl -X POST -F "files=@test.csv" https://yourdomain.com/api/upload

# Test rate limiting
for i in {1..100}; do curl https://yourdomain.com/api/status & done

# Test security headers
curl -I https://yourdomain.com/

# Test error handling
curl -X POST https://yourdomain.com/api/upload
```

### 12. Disaster Recovery

#### Backup Verification:
- Automated backup testing
- Recovery time objectives (RTO < 4 hours)
- Recovery point objectives (RPO < 1 hour)

#### Incident Response:
- Monitoring alerts for failures
- Automatic failover procedures
- Communication plan for outages

---

## 🔧 Development vs Production

| Feature | Development | Production |
|---------|-------------|------------|
| SECRET_KEY | Auto-generated | Required env var |
| DEBUG | True | False |
| CORS_ORIGINS | * | Specific domains |
| File validation | Basic | Comprehensive + quarantine |
| Rate limiting | Disabled | Enabled |
| Logging | Console | File + rotation |
| Error details | Full stack traces | User-friendly messages |

---

## 📊 Monitoring Checklist

- [ ] Application logs are being collected
- [ ] Error rates are monitored
- [ ] Response times are tracked
- [ ] File upload/processing metrics
- [ ] Security alerts are configured
- [ ] Disk space monitoring
- [ ] Memory and CPU usage
- [ ] Database connection health

---

## 🚨 Security Incident Response

1. **Detect**: Monitor logs for suspicious activity
2. **Contain**: Block malicious IPs automatically
3. **Investigate**: Check quarantine directory for malicious files
4. **Recover**: Restore from clean backups if needed
5. **Learn**: Update security rules based on incidents