"""図の出力パス管理ユーティリティ.

このモジュールでは以下の機能を提供する:
- figures ディレクトリのパス管理
- 図の保存パス生成
"""

from pathlib import Path

from config import FIGURES_DIR


def ensure_figures_dir() -> Path:
    """figures ディレクトリが存在することを確認し、パスを返す.

    Returns:
        figures ディレクトリのパス
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR


def get_figure_path(filename: str, subdir: str | None = None) -> Path:
    """図の保存パスを生成する.

    Args:
        filename: ファイル名（拡張子を含む）
        subdir: サブディレクトリ名（オプション）

    Returns:
        図の保存パス
    """
    base_dir = ensure_figures_dir()

    if subdir:
        target_dir = base_dir / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
    else:
        target_dir = base_dir

    return target_dir / filename
