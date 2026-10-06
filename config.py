from pathlib import Path


class Config:
    """
    Конфигурация проекта.

    Поддерживает два режима:
        kaggle — пути /kaggle/input/...
        local  — пути data/ относительно корня проекта.
    """

    MODE = "local"
    RANDOM_STATE = 42

    KAGGLE_PATHS = {
        "credit": "/kaggle/input/datasets/laotse/credit-risk-dataset/credit_risk_dataset.csv",
        "churn":  "/kaggle/input/datasets/blastchar/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "demand": "/kaggle/input/datasets/akshaymairal/store-item-demand-forecasting-challenge/train.csv",
    }

    PROJECT_ROOT = Path(__file__).resolve().parent
    DATA_DIR = PROJECT_ROOT / "data"
    RESULTS_DIR = PROJECT_ROOT / "results"

    LOCAL_PATHS = {
        "credit": str(DATA_DIR / "credit_risk_dataset.csv"),
        "churn":  str(DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"),
        "demand": str(DATA_DIR / "train.csv"),
    }

    def __init__(self, mode: str | None = None):
        """
        Инициализация конфигурации.

        Args:
            mode: "kaggle" или "local". Если None — Config.MODE.
        """
        self.mode = mode or self.MODE
        self._paths = self.KAGGLE_PATHS if self.mode == "kaggle" else self.LOCAL_PATHS
        self.RESULTS_DIR.mkdir(exist_ok=True)

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

    def __repr__(self) -> str:
        """Строковое представление конфигурации для отладки."""
        return (
            f"Config(mode={self.mode!r}, "
            f"credit={self.path_credit!r}, "
            f"churn={self.path_churn!r}, "
            f"demand={self.path_demand!r}, "
            f"random_state={self.RANDOM_STATE})"
        )