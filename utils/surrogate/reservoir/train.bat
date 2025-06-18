@echo off
call .venv\Scripts\activate.bat
python train_model.py %1
.venv\Scripts\deactivate.bat