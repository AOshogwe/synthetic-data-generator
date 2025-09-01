#!/usr/bin/env python3
"""
Railway deployment script for Synthetic Data Generator
This script helps validate and deploy the application to Railway
"""

import os
import sys
import subprocess
import json
from datetime import datetime

def check_railway_cli():
    """Check if Railway CLI is installed"""
    try:
        result = subprocess.run(['railway', '--version'], capture_output=True, text=True)
        print(f"✅ Railway CLI found: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("❌ Railway CLI not found. Install it with:")
        print("   npm install -g @railway/cli")
        return False

def check_docker():
    """Check if Docker is available"""
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        print(f"✅ Docker found: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("⚠️  Docker not found. Railway will build the image for you.")
        return False

def validate_files():
    """Validate required files exist"""
    required_files = [
        'app.py',
        'requirements.txt',
        'Dockerfile.backend',
        'railway.toml'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files present")
    return True

def create_env_file():
    """Create .env file with Railway-specific settings"""
    env_content = f"""# Railway deployment environment variables
FLASK_ENV=production
FLASK_DEBUG=False
SECRET_KEY=railway-production-secret-key-{datetime.now().strftime('%Y%m%d%H%M%S')}
PORT=${{PORT}}
MAX_CONTENT_LENGTH=104857600
UPLOAD_FOLDER=uploads
OUTPUT_FOLDER=outputs
LOG_LEVEL=INFO
CORS_ORIGINS=*
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✅ Created .env file for Railway deployment")

def validate_deployment_files():
    """Validate all deployment files are ready"""
    deployment_files = [
        'app.py',
        'app-simple.py',
        'requirements-railway.txt', 
        'Dockerfile.backend',
        'railway.toml',
        'start.sh'
    ]
    
    missing = []
    for file in deployment_files:
        if not os.path.exists(file):
            missing.append(file)
    
    if missing:
        print(f"❌ Missing deployment files: {', '.join(missing)}")
        return False
    
    print("✅ All deployment files present")
    return True

def test_local_build():
    """Test Docker build locally if Docker is available"""
    if not check_docker():
        return True
    
    print("🔨 Testing local Docker build...")
    try:
        result = subprocess.run([
            'docker', 'build', 
            '-f', 'Dockerfile.backend',
            '-t', 'synthetic-data-generator:test',
            '.'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Local Docker build successful")
            return True
        else:
            print(f"❌ Local Docker build failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"⚠️  Docker build test failed: {e}")
        return False

def deploy_to_railway():
    """Deploy to Railway"""
    if not check_railway_cli():
        return False
    
    print("🚀 Deploying to Railway...")
    try:
        # Login check
        result = subprocess.run(['railway', 'whoami'], capture_output=True, text=True)
        if result.returncode != 0:
            print("Please login to Railway first:")
            print("   railway login")
            return False
        
        # Deploy
        result = subprocess.run(['railway', 'up'], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Deployment successful!")
            print(result.stdout)
            return True
        else:
            print(f"❌ Deployment failed:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Deployment error: {e}")
        return False

def main():
    """Main deployment function"""
    print("🚀 Railway Deployment Helper for Synthetic Data Generator")
    print("=" * 60)
    
    # Validate environment
    if not validate_files():
        return False
    
    # Validate deployment-specific files
    if not validate_deployment_files():
        return False
    
    # Create environment file
    create_env_file()
    
    # Test local build (optional)
    test_local_build()
    
    # Deploy
    deploy_successful = deploy_to_railway()
    
    if deploy_successful:
        print("\n" + "=" * 60)
        print("🎉 Deployment completed successfully!")
        print("📝 Next steps:")
        print("   1. Check Railway dashboard for deployment status")
        print("   2. Test your application at the provided URL")
        print("   3. Monitor logs: railway logs")
        print("   4. Check health endpoint: /health")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Deployment failed. Please check the errors above.")
        print("💡 Common fixes:")
        print("   1. Ensure Railway CLI is installed and you're logged in")
        print("   2. Check Dockerfile.backend syntax")
        print("   3. Verify all dependencies in requirements.txt")
        print("   4. Check Railway project settings")
        print("=" * 60)
    
    return deploy_successful

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)