#!/bin/bash

# Setup script for the warehouse backend

echo "Setting up Warehouse Vision Backend..."

# Create virtual environment
python -m venv warehouse_env

# Activate virtual environment
source warehouse_env/bin/activate  # On Windows: warehouse_env\Scripts\activate

# Install dependencies
python -m pip install -r requirements.txt

echo "Setup complete!"
echo "To run the server:"
echo "1. Activate the virtual environment: source warehouse_env/bin/activate"
echo "2. Run the server: python run_server.py"
echo "3. The API will be available at http://localhost:8000"