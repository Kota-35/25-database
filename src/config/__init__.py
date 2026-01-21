"""設定事項."""

from pathlib import Path

# プロジェクトルートからの相対パス
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATABASE_CLASS = PROJECT_ROOT / "25-database-class"
FIGURES_DIR = PROJECT_ROOT / "figures"
