#!/usr/bin/env python3
"""
Unified server startup script for Email Spam Classifier.
Starts both ML and LLM APIs simultaneously.
"""

import uvicorn
import os
import sys
import asyncio
from pathlib import Path

def check_requirements():
    """Check if all required files exist."""
    required_files = [
        "app.py",
        "llm_classifier/llm_api.py",
        "static/unified_interface.html",
        "static/index.html"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    return True

async def start_llm_server():
    """Start the LLM API server in a separate process."""
    try:
        print("🚀 Starting LLM API server on port 8001...")
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "uvicorn", 
            "llm_classifier.llm_api:app",
            "--host", "0.0.0.0",
            "--port", "8001",
            "--reload"
        )
        await process.wait()
    except Exception as e:
        print(f"❌ LLM server failed to start: {e}")
    except KeyboardInterrupt:
        print("🛑 LLM server stopped")

async def start_ml_server():
    """Start the ML API server in a separate process."""
    try:
        print("🚀 Starting ML API server on port 8000...")
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "uvicorn", 
            "app:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        )
        await process.wait()
    except Exception as e:
        print(f"❌ ML server failed to start: {e}")
    except KeyboardInterrupt:
        print("🛑 ML server stopped")

async def main():
    """Start both servers."""
    print("🤖 Email Spam Classifier - Unified Server")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("❌ Please ensure all required files are present.")
        sys.exit(1)
    
    print("✅ All required files found")
    print("✅ Starting unified server...")
    print("=" * 50)
    
    try:
        # Start both servers concurrently
        ml_task = asyncio.create_task(start_ml_server())
        llm_task = asyncio.create_task(start_llm_server())
        
        # Wait a moment for servers to start
        await asyncio.sleep(3)
        
        print("\n🎉 Servers are running!")
        print("=" * 50)
        print("📱 Main Interface: http://localhost:8000")
        print("🤖 ML API Docs: http://localhost:8000/docs")
        print("🧠 LLM API Docs: http://localhost:8001/docs")
        print("📊 Health Checks:")
        print("   - ML API: http://localhost:8000/health")
        print("   - LLM API: http://localhost:8001/health")
        print("=" * 50)
        print("💡 Use Ctrl+C to stop all servers")
        
        # Wait for both servers to complete
        await asyncio.gather(ml_task, llm_task)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down servers...")
        print("✅ All servers stopped")
    except Exception as e:
        print(f"❌ Server error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 