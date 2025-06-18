@echo off
python -m venv .venv
call .venv\Scripts\activate.bat
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
.venv\Scripts\deactivate.bat