#!/usr/bin/env python3
"""
Startup script for the Email Spam Classifier FastAPI server.
"""

import uvicorn
import os
import sys

def main():
    """Start the FastAPI server."""
    print("🚀 Starting Email Spam Classifier API Server")
    print("=" * 50)
    
    # Check if static directory exists
    if not os.path.exists("static"):
        print("❌ Static directory not found. Creating it...")
        os.makedirs("static", exist_ok=True)
    
    # Check if static/index.html exists
    if not os.path.exists("static/index.html"):
        print("❌ static/index.html not found!")
        print("Please ensure the HTML frontend is in place.")
        sys.exit(1)
    
    # Check if src directory and required files exist
    required_files = [
        "src/preprocess.py",
        "src/train.py", 
        "src/predict.py"
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Required file not found: {file_path}")
            sys.exit(1)
    
    print("✅ All required files found")
    print("✅ Starting server on http://localhost:8000")
    print("✅ API documentation available at http://localhost:8000/docs")
    print("✅ Web interface available at http://localhost:8000")
    print("=" * 50)
    
    # Start the server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main() 