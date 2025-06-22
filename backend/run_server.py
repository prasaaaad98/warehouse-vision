"""
Simple script to run the FastAPI server
"""
import uvicorn
from main import app

if __name__ == "__main__":
    print("Starting Warehouse Vision API Server...")
    print("Server will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )