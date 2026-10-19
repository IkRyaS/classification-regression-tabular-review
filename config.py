import os
from pathlib import Path


class Config:
    """
    Конфигурация проекта.

    Режим определяется автоматически:
        kaggle — если существует /kaggle/input;
        local  — иначе (пути data/ и results/ относительно корня проекта).

    Можно форсировать режим: Config(mode="local") или Config(mode="kaggle").
    """

    RANDOM_STATE = 42

    KAGGLE_INPUT = Path("/kaggle/input")
    KAGGLE_WORKING = Path("/kaggle/working")

    KAGGLE_PATHS = {
        "credit": "/kaggle/input/datasets/laotse/credit-risk-dataset/credit_risk_dataset.csv",
        "churn":  "/kaggle/input/datasets/blastchar/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "demand": "/kaggle/input/datasets/akshaymairal/store-item-demand-forecasting-challenge/train.csv",
    }

    def __init__(self, mode: str | None = None):
        """
        Инициализация конфигурации.

        Args:
            mode: "kaggle", "local" или None.
                  Если None — режим определяется автоматически.
        """
        self.mode = mode or self._detect_mode()

        if self.mode == "kaggle":
            self.PROJECT_ROOT = self.KAGGLE_WORKING
            self.DATA_DIR = self.KAGGLE_WORKING / "data"
            self.RESULTS_DIR = self.KAGGLE_WORKING / "results"
            self._paths = dict(self.KAGGLE_PATHS)
        else:
            self.PROJECT_ROOT = Path(__file__).resolve().parent
            self.DATA_DIR = self.PROJECT_ROOT / "data"
            self.RESULTS_DIR = self.PROJECT_ROOT / "results"
            self._paths = {
                "credit": str(self.DATA_DIR / "credit_risk_dataset.csv"),
                "churn":  str(self.DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"),
                "demand": str(self.DATA_DIR / "train.csv"),
            }

        # Доп. задания (06, 07): данные готовит scripts/prepare_extra_data.py
        self._paths.update({
            "imbalance":   str(self.DATA_DIR / "cs-training.csv"),
            "claims_freq": str(self.DATA_DIR / "freMTPL2freq.csv"),
            "claims_sev":  str(self.DATA_DIR / "freMTPL2sev.csv"),
        })

        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _detect_mode() -> str:
        """Определяет режим: kaggle, если есть /kaggle/input, иначе local."""
        if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
            return "kaggle"
        if Path("/kaggle/input").exists():
            return "kaggle"
        return "local"

    @property
    def path_credit(self) -> str:
        """Путь к датасету Credit Risk."""
        return self._paths["credit"]

    @property
    def path_churn(self) -> str:
        """Путь к датасету Telco Customer Churn."""
        return self._paths["churn"]

    @property
    def path_demand(self) -> str:
        """Путь к датасету Store Item Demand Forecasting."""
        return self._paths["demand"]

    @property
    def path_imbalance(self) -> str:
        """Путь к Give Me Some Credit (доп. задание 1, сильный дисбаланс)."""
        return self._paths["imbalance"]

    @property
    def path_claims_freq(self) -> str:
        """Путь к freMTPL2freq - полисы (доп. задание 2, скошенный таргет)."""
        return self._paths["claims_freq"]

    @property
    def path_claims_sev(self) -> str:
        """Путь к freMTPL2sev - суммы выплат (доп. задание 2)."""
        return self._paths["claims_sev"]

    @property
    def credit_plot(self) -> Path:
        """Путь к PNG-графику моделей на credit."""
        return self.RESULTS_DIR / "credit_models.png"

    @property
    def churn_plot(self) -> Path:
        """Путь к PNG-графику моделей на churn."""
        return self.RESULTS_DIR / "churn_models.png"

    @property
    def demand_plot(self) -> Path:
        """Путь к PNG-графику моделей на demand."""
        return self.RESULTS_DIR / "demand_models.png"

    @property
    def final_table(self) -> Path:
        """Путь к CSV-таблице «метод × датасет × метрика»."""
        return self.RESULTS_DIR / "final_table.csv"

    def cv_folds(self, dataset: str) -> Path:
        """Путь к CSV с метриками по фолдам CV (например, credit_cv_folds.csv)."""
        return self.RESULTS_DIR / f"{dataset}_cv_folds.csv"

    def test_results(self, dataset: str) -> Path:
        """Путь к CSV с метриками на отложенном тесте после подбора гиперпараметров."""
        return self.RESULTS_DIR / f"{dataset}_test.csv"

    def __repr__(self) -> str:
        """Строковое представление конфигурации для отладки."""
        return (
            f"Config(mode={self.mode!r}, "
            f"credit={self.path_credit!r}, "
            f"churn={self.path_churn!r}, "
            f"demand={self.path_demand!r}, "
            f"random_state={self.RANDOM_STATE})"
        )