# Smart AI Farming Assistant - Makefile
# Cross-platform commands for development

.PHONY: help venv install train run-api run test clean

# Default target
help:
	@echo "Smart AI Farming Assistant - Available commands:"
	@echo ""
	@echo "  make venv      - Create Python virtual environment"
	@echo "  make install   - Install dependencies"
	@echo "  make train     - Train weather prediction models"
	@echo "  make run-api   - Run FastAPI backend only"
	@echo "  make run       - Run full application (API + Frontend)"
	@echo "  make test      - Run test suite"
	@echo "  make clean     - Clean generated files"
	@echo ""

# Create virtual environment
venv:
	python -m venv venv
	@echo "Virtual environment created. Activate with:"
	@echo "Windows: venv\\Scripts\\activate"
	@echo "Unix/Mac: source venv/bin/activate"

# Install dependencies
install:
	pip install -r requirements.txt

# Train weather models
train:
	cd backend && python train_weather_models.py

# Run API server only
run-api:
	cd backend && uvicorn app:app --reload --port 8000

# Run full application (API serves frontend)
run: train
	cd backend && uvicorn app:app --reload --port 8000 --host 0.0.0.0

# Run tests
test:
	python -m pytest tests/ -v

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	@if exist models\\weather_models.joblib del models\\weather_models.joblib
	@if exist __pycache__ rmdir /s /q __pycache__
	@if exist backend\\__pycache__ rmdir /s /q backend\\__pycache__
	@if exist tests\\__pycache__ rmdir /s /q tests\\__pycache__
	@if exist .pytest_cache rmdir /s /q .pytest_cache
	@echo "Clean complete."

# Development setup (Windows)
setup-windows: venv
	venv\\Scripts\\activate && pip install -r requirements.txt && cd backend && python train_weather_models.py
	@echo "Setup complete! Run 'make run' to start the application."

# Development setup (Unix/Mac)
setup-unix: venv
	. venv/bin/activate && pip install -r requirements.txt && cd backend && python train_weather_models.py
	@echo "Setup complete! Run 'make run' to start the application."