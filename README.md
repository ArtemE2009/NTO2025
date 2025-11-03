# NTO AI 2025-2026: Baseline для рекомендательной системы

Baseline-решение для задачи предсказания оценок книг пользователями в рамках соревнования НТО 2025/2026 (Профиль «Искусственный интеллект»).

## Описание

Модель предсказывает оценку (rating) от 0 до 10, которую пользователь поставит книге. Решение основано на:
- **LightGBM** с 5-fold GroupKFold cross-validation
- **Feature Engineering**: агрегированные признаки (user/item biases), жанры
- **TF-IDF**: 500 фичей из текстовых описаний книг

## Быстрый старт

### Установка зависимостей

```bash
poetry install
```

### Подготовка данных

Поместите CSV-файлы в `data/raw/`:
- `stage1_public_train.csv`
- `stage1_public_test.csv`
- `stage1_public_users.csv`
- `stage1_public_books.csv`
- `stage1_public_book_genres.csv`
- `stage1_public_genres.csv`
- `stage1_public_book_descriptions.csv`

### Запуск пайплайна

```bash
# Обучение
poetry run python -m src.baseline.train

# Предсказание
poetry run python -m src.baseline.predict

# Валидация submission
poetry run python -m src.baseline.validate
```

Или через Makefile:
```bash
make train    # Обучение
make predict  # Предсказание
make validate # Валидация
make run      # Полный цикл
```

## Структура проекта

```
.
├── data/
│   └── raw/              # Исходные CSV-файлы
├── output/
│   ├── models/           # Обученные модели и TF-IDF векторайзер
│   └── submissions/      # Файлы submission
├── src/baseline/
│   ├── config.py         # Конфигурация и параметры
│   ├── constants.py      # Константы проекта
│   ├── data_processing.py # Загрузка и объединение данных
│   ├── features.py       # Feature engineering (агрегаты, TF-IDF)
│   ├── train.py          # Обучение модели
│   ├── predict.py        # Генерация предсказаний
│   └── validate.py       # Проверка submission
└── Makefile              # Удобные команды
```

## Особенности реализации

- **Предотвращение data leakage**: TF-IDF векторайзер обучается только на train данных
- **GroupKFold**: Разбиение по `user_id` для корректной валидации
- **Обработка пропусков**: Автоматическое заполнение для всех признаков
- **Типизация**: Полная типизация кода (type hints)
- **Code quality**: Ruff для линтинга и форматирования, pre-commit hooks

## Метрика

Score рассчитывается на основе RMSE и MAE:
```
Score = 1 - (0.5 * RMSE/10 + 0.5 * MAE/10)
```

## Зависимости

- Python >= 3.10
- pandas, scikit-learn, lightgbm, joblib
- ruff, pre-commit (dev)

## Лицензия

Проект создан для соревнования НТО 2025-2026.

