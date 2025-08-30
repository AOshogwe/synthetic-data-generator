#!/usr/bin/env python3
"""Minimal test to find crash cause"""

print("🔍 Testing minimal app startup...")

try:
    print("1. Testing basic imports...")
    import os, sys, json, time
    from pathlib import Path
    print("✅ Basic imports OK")
    
    print("2. Testing Flask...")
    from flask import Flask, jsonify
    app = Flask(__name__)
    print("✅ Flask OK")
    
    print("3. Testing config...")
    import secrets
    app.config['SECRET_KEY'] = secrets.token_hex(16)
    print("✅ Config OK")
    
    print("4. Testing route...")
    @app.route('/test')
    def test():
        return jsonify({'status': 'ok'})
    print("✅ Route OK")
    
    print("5. Starting minimal server...")
    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=5000, debug=True)
        
except Exception as e:
    import traceback
    print(f"❌ Error: {e}")
    print(f"🔍 Full traceback:\n{traceback.format_exc()}")