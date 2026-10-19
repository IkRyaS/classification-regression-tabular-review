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
  Store Item Demand (регрессия, временной ряд, прогноз на 1 день вперёд).
- 9–10 моделей: LogReg, LinearReg, Ridge, Lasso, DecisionTree, RandomForest,
  XGBoost, LightGBM, CatBoost (+ CatBoost с нативными категориями на churn),
  kNN, SVM/SVR, NaiveBayes. Для demand - наивные бейзлайны (вчера / неделю назад).
- Единый протокол:
  - отложенный тест: 20% (credit, churn, стратифицированно), 2017 год (demand);
  - сравнение моделей: StratifiedKFold(5) / TimeSeriesSplit(5) по датам - только на train;
  - подбор гиперпараметров: RandomizedSearchCV на train, финальная оценка на тесте.
- Пропуски и невозможные значения: условное заполнение внутри `Pipeline`
  (статистики считаются только на train), флаги пропусков.
- Метрики: ROC-AUC, F1 (классификация); RMSE, MAE, MAPE (регрессия).
- Статистика: Friedman + Iman-Davenport + posthoc Nemenyi, CD-диаграммы, α = 0.05.
- Воспроизводимость: `RANDOM_STATE = 42` везде, версии в `uv.lock` / `requirements.txt`.

Теоретическая база, определения терминов и методология -
в отдельном файле [`Docs.md`](Docs.md).

Итоговые результаты экспериментов (таблицы, значимость, выводы) -
в [`REPORT.md`](REPORT.md).

### Дополнительные исследования (по заданию преподавателя)

- **06_imbalance** - сильный дисбаланс классов (Give Me Some Credit, 6.7% дефолтов):
  ROC-AUC vs PR-AUC / MCC / Balanced Accuracy / F1, бизнес-стоимость ошибок, подбор порога,
  `class_weight` / `scale_pos_weight` / `sample_weight`, SMOTE / ADASYN / SMOTEENN (только на train),
  калибровка (Platt, isotonic), двухстадийная модель.
- **07_skewed_target** - скошенный таргет (freMTPL2, страховые выплаты, 96% нулей):
  диагностика (skewness, kurtosis, Breusch-Pagan, White, QQ-plot, Cook's distance), log1p / Yeo-Johnson /
  винзоризация / smearing-поправка, Huber / Quantile / Poisson / Tweedie / асимметричная потеря, веса,
  двухстадийная модель, сегментный анализ ошибки (нули / bulk / tail), применимость Croston / SBA / TSB.

### Структура

```
config.py              пути к данным/результатам, RANDOM_STATE
scripts/prepare_extra_data.py  загрузка данных для 06 и 07
REPORT.md              итоговый отчёт по результатам прогона
docs/Курсовая_работа.docx  текст курсовой работы (оформление по образцу кафедры)
CHAT_BACKUP.md         бэкап переписки с ассистентом: что проверено и исправлено
requirements.txt       зафиксированные версии (экспорт из uv.lock)
notebooks/             01_eda ... 05_statistics
results/               графики и CSV, которые создают ноутбуки
  *_cv_folds.csv       метрики каждой модели на каждом фолде CV
  *_test.csv           метрики на тесте после подбора гиперпараметров + лучшие параметры
  final_table.csv      «метод × датасет × метрика» (CV и тест)
  cd_*.png             CD-диаграммы
```

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
зависимости ровно тех версий, что в `uv.lock`.

Без uv:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

## Датасеты

Датасеты **не в репозитории** (большие файлы). Скачать с Kaggle:

| № | Название | Задача | Ссылка |
|---|---|---|---|
| 1 | Credit Risk Dataset | Классификация | https://www.kaggle.com/datasets/laotse/credit-risk-dataset |
| 2 | Telco Customer Churn | Классификация | https://www.kaggle.com/datasets/blastchar/telco-customer-churn |
| 3 | Store Item Demand Forecasting | Регрессия | https://www.kaggle.com/datasets/akshaymairal/store-item-demand-forecasting-challenge |

Для дополнительных ноутбуков 06 и 07 данные скачиваются и конвертируются скриптом
(Give Me Some Credit с зеркала на GitHub, freMTPL2 из R-пакета CASdatasets):

```bash
uv run python scripts/prepare_extra_data.py
```

| № | Название | Задача | Источник |
|---|---|---|---|
| 4 | Give Me Some Credit | Классификация, сильный дисбаланс | https://www.kaggle.com/c/GiveMeSomeCredit |
| 5 | freMTPL2freq / freMTPL2sev | Регрессия, скошенный таргет | https://github.com/dutangc/CASdatasets |

Положить CSV основных датасетов в `data/`:

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
Ноутбуки 02–04 сохраняют результаты в `results/`, ноутбук 05 читает их -
поэтому 05 запускается последним.

Время полного прогона на 2 ядрах CPU: 02 и 04 - по 20-40 минут
(основное время - подбор гиперпараметров), остальные - минуты.

Порядок ноутбуков:

1. `01_eda.ipynb` — разведочный анализ
2. `02_credit.ipynb` — классификация credit
3. `03_churn.ipynb` — классификация churn
4. `04_demand.ipynb` — регрессия demand
5. `05_statistics.ipynb` — Friedman + Nemenyi
6. `06_imbalance.ipynb` — доп. задание 1, сильный дисбаланс (независим от 01–05)
7. `07_skewed_target.ipynb` — доп. задание 2, скошенный таргет (независим от 01–05)

### Kaggle

Ноутбуки рассчитаны на запуск в Kaggle Notebooks. Пути к датасетам:

```python
PATH_CREDIT = '/kaggle/input/datasets/laotse/credit-risk-dataset/credit_risk_dataset.csv'
PATH_CHURN  = '/kaggle/input/datasets/blastchar/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv'
PATH_DEMAND = '/kaggle/input/datasets/akshaymairal/store-item-demand-forecasting-challenge/train.csv'
```

Менять пути вручную не нужно: `config.py` сам определяет режим
(`kaggle`, если существует `/kaggle/input`, иначе `local` с путями в `data/`).
На Kaggle ноутбук 05 должен видеть CSV из `results/`, созданные ноутбуками 02–04
(запуск в одной сессии или подключение их output как датасета).
