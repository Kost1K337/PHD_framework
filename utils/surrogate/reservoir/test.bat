@echo off
call .venv\Scripts\activate.bat
python validate_model.py %1
.venv\Scripts\deactivate.bat