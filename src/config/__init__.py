"""設定事項."""

from pathlib import Path

DATABASE_CLASS = Path("25-database-class")

# プロジェクトルートからの相対パス
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"
