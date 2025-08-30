# Quick Fixes for Common Crashes

## 1. Import Errors
```bash
# Install missing dependencies
pip install -r requirements.txt

# If that fails, install one by one:
pip install flask flask-cors pandas numpy
```

## 2. Port Already in Use
```bash
# Kill processes on port 5000
pkill -f "python.*app.py"
# or
lsof -ti:5000 | xargs kill -9
```

## 3. Permission Errors
```bash
# Fix permissions
chmod +x app.py
mkdir -p uploads outputs logs
chmod 755 uploads outputs logs
```

## 4. Environment Variables
```bash
# Set required variables
export FLASK_ENV=development
export SECRET_KEY=dev-key-for-testing
```

## 5. Memory Issues
```bash
# Reduce memory usage
export FLASK_DEBUG=False
export WORKERS=1
```

## 6. Docker Issues
```bash
# Clean rebuild
docker system prune -f
docker build --no-cache -f Dockerfile.simple -t synthetic-data-app .
```