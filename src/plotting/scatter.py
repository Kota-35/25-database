"""散布図関連のプロット関数."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def plot_scatter_by_category(
    df: pl.DataFrame,
    x_col: str,
    y_col: str,
    category_col: str,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    alpha: float = 0.4,
    figsize: tuple[float, float] = (8, 5),
    save_path: str | Path | None = None,
) -> None:
    """カテゴリ別の散布図を描画する.

    Args:
        df: データフレーム
        x_col: X軸の列名
        y_col: Y軸の列名
        category_col: カテゴリ列名
        title: グラフタイトル
        xlabel: X軸ラベル
        ylabel: Y軸ラベル
        alpha: 透明度
        figsize: 図のサイズ
        save_path: 図の保存パス（Noneの場合は保存しない）
    """
    categories = sorted(
        df.select(category_col).unique().to_series().drop_nulls().to_list(),
    )

    _, ax = plt.subplots(figsize=figsize)
    for cat in categories:
        sub = df.filter(pl.col(category_col) == cat)
        x_vals = sub.select(x_col).to_series().to_list()
        y_vals = sub.select(y_col).to_series().to_list()
        ax.scatter(x_vals, y_vals, alpha=alpha, label=cat)

    ax.set_xlabel(xlabel if xlabel else x_col)
    ax.set_ylabel(ylabel if ylabel else y_col)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved: {save_path}")

    plt.show()


def plot_jitter_scatter(
    df: pl.DataFrame,
    category_col: str,
    value_col: str,
    category_order: Sequence[str],
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    jitter_scale: float = 0.06,
    markersize: float = 3,
    alpha: float = 0.35,
    figsize: tuple[float, float] = (8, 5),
    seed: int = 42,
    save_path: str | Path | None = None,
) -> None:
    """カテゴリ別のジッター散布図を描画する.

    Args:
        df: データフレーム
        category_col: カテゴリ列名
        value_col: 値の列名
        category_order: カテゴリの表示順序
        title: グラフタイトル
        xlabel: X軸ラベル
        ylabel: Y軸ラベル
        jitter_scale: ジッターの広がり
        markersize: マーカーサイズ
        alpha: 透明度
        figsize: 図のサイズ
        seed: 乱数シード
        save_path: 図の保存パス（Noneの場合は保存しない）
    """
    rng = np.random.default_rng(seed)

    value_lists = []
    for cat in category_order:
        vals = (
            df.filter(pl.col(category_col) == cat)
            .select(value_col)
            .to_series()
            .to_list()
        )
        value_lists.append(vals)

    _, ax = plt.subplots(figsize=figsize)
    for i, vals in enumerate(value_lists, start=1):
        if len(vals) == 0:
            continue
        x = rng.normal(loc=i, scale=jitter_scale, size=len(vals))
        ax.plot(x, vals, "o", markersize=markersize, alpha=alpha)

    ax.set_xticks(range(1, len(category_order) + 1))
    ax.set_xticklabels(category_order)
    ax.set_xlabel(xlabel if xlabel else category_col)
    ax.set_ylabel(ylabel if ylabel else value_col)
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved: {save_path}")

    plt.show()
