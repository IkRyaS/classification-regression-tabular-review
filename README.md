# classification-regression-tabular-review

**Обзор и сравнение методов классификации и регрессии табличных данных.**

## ТЗ

### Обзорная часть

Краткое описание семейств моделей с математической интуицией:

- Линейные модели: LogReg, Ridge, Lasso
- Деревья решений
- Бэггинг и Random Forest
- Градиентный бустинг: XGBoost, LightGBM, CatBoost
- kNN, SVM, наивный Байес
- Кратко: почему DL на табличке часто проигрывает бустингу

### Практическая часть

- **2–3 табличных датасета** из экономики/финансов.
- **Единый протокол:**
  - train / val / test,
  - кросс-валидация,
  - подбор гиперпараметров,
  - метрики: ROC-AUC, F1, RMSE, MAE, MAPE,
  - статистическая значимость различий:
    тест Фридмана + пост-хок Неменьи, критическая разница.
- **Вывод:**
  - таблица «метод × датасет × метрика»,
  - рекомендации: когда что работает.

### Что должен освоить

- `pandas`, `sklearn`
- базовая статистика
- оформление экспериментов
- воспроизводимость: seed, requirements

### Критерий оценки

**Корректный эксперимент**, а не число прочитанных статей.
**Не превращать в реферат.**

## Что внутри репозитория

- 3 датасета: Credit Risk (классификация), Telco Churn (классификация),
  Store Item Demand (регрессия, временной ряд).
- 9–10 моделей: LogReg, Ridge, Lasso, DecisionTree, RandomForest,
  XGBoost, LightGBM, CatBoost, kNN, SVM, NaiveBayes.
- Единый протокол: StratifiedKFold(5) для классификации,
  TimeSeriesSplit(5) для регрессии.
- Метрики: ROC-AUC, F1 (классификация); RMSE, MAE, MAPE (регрессия).
- Статистика: Friedman + posthoc Nemenyi, α = 0.05.
- Воспроизводимость: `RANDOM_STATE = 42` везде.

Теоретическая база, определения терминов и методология —
в отдельном файле [`Docs.md`](Docs.md).

## Требования

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) — менеджер зависимостей
- Git

## Установка

```bash
git clone https://github.com/<username>/classification-regression-tabular-review.git
cd classification-regression-tabular-review
uv sync
```

`uv sync` создаст виртуальное окружение `.venv/` и установит все
зависимости из `pyproject.toml`.

## Датасеты

Датасеты **не в репозитории** (большие файлы). Скачать с Kaggle:

| № | Название | Задача | Ссылка |
|---|---|---|---|
| 1 | Credit Risk Dataset | Классификация | https://www.kaggle.com/datasets/laotse/credit-risk-dataset |
| 2 | Telco Customer Churn | Классификация | https://www.kaggle.com/datasets/blastchar/telco-customer-churn |
| 3 | Store Item Demand Forecasting | Регрессия | https://www.kaggle.com/datasets/akshaymairal/store-item-demand-forecasting-challenge |

Положить CSV в `data/`:

```
data/
├── credit_risk_dataset.csv
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── train.csv
```

## Как запускать

### Jupyter

```bash
uv run jupyter lab
```

Открыть нужный ноутбук в `notebooks/` и выполнить **Restart & Run All**.

Порядок ноутбуков:

1. `01_eda.ipynb` — разведочный анализ
2. `02_credit.ipynb` — классификация credit
3. `03_churn.ipynb` — классификация churn
4. `04_demand.ipynb` — регрессия demand
5. `05_statistics.ipynb` — Friedman + Nemenyi

### Kaggle

Ноутбуки рассчитаны на запуск в Kaggle Notebooks. Пути к датасетам:

```python
PATH_CREDIT = '/kaggle/input/datasets/laotse/credit-risk-dataset/credit_risk_dataset.csv'
PATH_CHURN  = '/kaggle/input/datasets/blastchar/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv'
PATH_DEMAND = '/kaggle/input/datasets/akshaymairal/store-item-demand-forecasting-challenge/train.csv'
```

При локальном запуске заменить на пути в `data/`.
