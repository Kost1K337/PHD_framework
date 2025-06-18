@echo off
python -m venv .venv
call .venv\Scripts\activate.bat
pip install torch torchvision torchaudio
pip install -r requirements.txt
.venv\Scripts\deactivate.bat