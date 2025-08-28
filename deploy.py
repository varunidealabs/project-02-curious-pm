#!/usr/bin/env python3
"""
Deployment script for Personal Calling Agent API
"""
import subprocess
import sys
import os

def install_requirements():
    """Install Python requirements"""
    print("📦 Installing Python requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✅ Requirements installed successfully")

def setup_environment():
    """Setup environment variables"""
    if not os.path.exists(".env"):
        print("❌ .env file not found!")
        print("Please create .env file with the following variables:")
        print("- PINECONE_API_KEY")
        print("- AZURE_OPENAI_ENDPOINT")
        print("- AZURE_OPENAI_API_KEY")
        print("- AZURE_OPENAI_DEPLOYMENT")
        print("- AZURE_OPENAI_API_VERSION")
        print("- API_SECRET_KEY")
        return False
    
    print("✅ Environment file found")
    return True

def run_server():
    """Run the FastAPI server"""
    print("🚀 Starting FastAPI server...")
    subprocess.call(["python", "-m", "uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"])

def main():
    print("🤖 Personal Memory Assistant API Deployment")
    print("=" * 50)
    
    if not setup_environment():
        sys.exit(1)
        
    install_requirements()
    
    print("\n🔧 Setup complete!")
    print("📱 API will be available at: http://localhost:8000")
    print("📚 API docs will be at: http://localhost:8000/docs")
    print("🔍 Health check: http://localhost:8000/api/health")
    
    try:
        run_server()
    except KeyboardInterrupt:
        print("\n👋 Server stopped")

if __name__ == "__main__":
    main()