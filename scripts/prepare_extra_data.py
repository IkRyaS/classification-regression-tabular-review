"""
Загрузка и подготовка датасетов для дополнительных заданий (ноутбуки 06 и 07).

    python scripts/prepare_extra_data.py

1. Give Me Some Credit (Kaggle, 2011) - сильный дисбаланс классов (≈ 6.7% дефолтов).
   Зеркало на GitHub: jalajthanaki/credit-risk-modelling.
2. freMTPL2freq / freMTPL2sev (R-пакет CASdatasets) - страховые полисы и выплаты
   (≈ 96% полисов без выплат, тяжёлый хвост). Файлы .rda конвертируются в CSV.

Нужны пакеты: requests не нужен (urllib), rdata (pip install rdata) - для чтения .rda.
"""
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import Config  # noqa: E402

SOURCES = {
    "cs-training.csv": "https://raw.githubusercontent.com/jalajthanaki/credit-risk-modelling/master/data/cs-training.csv",
    "freMTPL2freq.rda": "https://raw.githubusercontent.com/dutangc/CASdatasets/master/data/freMTPL2freq.rda",
    "freMTPL2sev.rda": "https://raw.githubusercontent.com/dutangc/CASdatasets/master/data/freMTPL2sev.rda",
}


def download(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        target = data_dir / name
        if target.exists():
            print(f"есть: {target}")
            continue
        print(f"скачиваю {url}")
        urllib.request.urlretrieve(url, target)


def rda_to_csv(src: Path, dst: Path) -> None:
    import rdata

    if dst.exists():
        print(f"есть: {dst}")
        return
    parsed = rdata.parser.parse_file(src)
    df = list(rdata.conversion.convert(parsed).values())[0]
    df.to_csv(dst, index=False)
    print(f"{src.name} -> {dst.name}: {df.shape}")


if __name__ == "__main__":
    cfg = Config()
    data_dir = Path(cfg.path_imbalance).parent
    download(data_dir)
    rda_to_csv(data_dir / "freMTPL2freq.rda", Path(cfg.path_claims_freq))
    rda_to_csv(data_dir / "freMTPL2sev.rda", Path(cfg.path_claims_sev))
    print("готово")
