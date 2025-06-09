# PHD_framework

PHD_framework/
│
├── data/                  # Все данные: входные параметры, результаты симуляций, обучающие выборки
│   ├── raw/               # Исходные симуляции или поля
│   ├── processed/         # Преобразованные и очищенные данные
│   └── surrogates/        # Наборы данных для обучения суррогатных моделей
│
├── utils/                 # Весь исходный код и алгоритмы
│   ├── multiscale.py      # Функции сопряжения моделей разной детальности
│   ├── surrogate.py       # Обучение и применение суррогатных моделей
│   ├── active_learning.py # Стратегии активного обучения
│   ├── simulation_io.py   # Парсинг и преобразование входных/выходных данных
│   ├── metrics.py         # Метрики качества аппроксимации
│   ├── visualization.py   # Построение графиков
│   └── config.py          # Настройки, пути, параметры по умолчанию
│
├── notebooks/             # Jupyter-ноутбуки для экспериментов и демонстраций
│   ├── 01_surrogate_vs_hifi.ipynb
│   ├── 02_active_learning_loop.ipynb
│   ├── 03_multiscale_modeling_case.ipynb
│   └── 04_real_case_analysis.ipynb
