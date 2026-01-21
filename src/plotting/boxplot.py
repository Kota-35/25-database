"""箱ひげ図・バイオリン図のプロット関数."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_boxplot_by_category(
    data_lists: Sequence[Sequence[float]],
    labels: Sequence[str],
    title: str,
    xlabel: str,
    ylabel: str,
    figsize: tuple[float, float] = (8, 5),
    *,
    showfliers: bool = True,
    save_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """カテゴリ別の箱ひげ図を描画する.

    Args:
        data_lists: カテゴリごとのデータリスト
        labels: カテゴリラベル
        title: グラフタイトル
        xlabel: X軸ラベル
        ylabel: Y軸ラベル
        figsize: 図のサイズ
        showfliers: 外れ値を表示するか
        save_path: 図の保存パス（Noneの場合は保存しない）

    Returns:
        Figure と Axes のタプル
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(data_lists, tick_labels=labels, vert=True, showfliers=showfliers)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved: {save_path}")

    return fig, ax


def plot_violin_with_boxplot(
    data_lists: Sequence[Sequence[float]],
    labels: Sequence[str],
    title: str,
    ylabel: str,
    figsize: tuple[float, float] = (8, 5),
    boxplot_width: float = 0.18,
    save_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """バイオリン図と箱ひげ図を重ねて描画する.

    Args:
        data_lists: カテゴリごとのデータリスト
        labels: カテゴリラベル
        title: グラフタイトル
        ylabel: Y軸ラベル
        figsize: 図のサイズ
        boxplot_width: 箱ひげ図の幅
        save_path: 図の保存パス（Noneの場合は保存しない）

    Returns:
        Figure と Axes のタプル
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.violinplot(
        data_lists,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    ax.boxplot(
        data_lists,
        widths=boxplot_width,
        vert=True,
    )

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved: {save_path}")

    return fig, ax


def plot_boxplot_by_user_type(
    df: pl.DataFrame,
    value_col: str,
    title: str,
    figsize: tuple[float, float] = (6, 4),
    user_types: Sequence[str] | None = None,
    save_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """ユーザータイプ別の箱ひげ図を描画する.

    Args:
        df: user_type 列を持つデータフレーム
        value_col: 値として使用する列名
        title: グラフタイトル
        figsize: 図のサイズ
        user_types: 表示するユーザータイプのリスト（デフォルト: ["staff", "student"]）
        save_path: 図の保存パス（Noneの場合は保存しない）

    Returns:
        Figure と Axes のタプル
    """
    if user_types is None:
        user_types = ["staff", "student"]

    data_lists = []
    for ut in user_types:
        vals = (
            df.filter(pl.col("user_type") == ut)
            .select(value_col)
            .to_series()
            .to_list()
        )
        data_lists.append(vals)

    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(
        data_lists,
        tick_labels=list(user_types),
        showfliers=True,
    )
    ax.set_title(title)
    ax.set_ylabel(value_col)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved: {save_path}")

    return fig, ax


def plot_jitter_scatter(
    data_lists: Sequence[Sequence[float]],
    labels: Sequence[str],
    title: str,
    xlabel: str,
    ylabel: str,
    figsize: tuple[float, float] = (8, 5),
    jitter_scale: float = 0.06,
    marker_size: float = 3,
    alpha: float = 0.35,
    seed: int = 42,
    save_path: str | Path | None = None,
) -> tuple[Figure, Axes]:
    """ジッター散布図を描画する.

    Args:
        data_lists: カテゴリごとのデータリスト
        labels: カテゴリラベル
        title: グラフタイトル
        xlabel: X軸ラベル
        ylabel: Y軸ラベル
        figsize: 図のサイズ
        jitter_scale: ジッターの標準偏差
        marker_size: マーカーサイズ
        alpha: 透明度
        seed: 乱数シード
        save_path: 図の保存パス（Noneの場合は保存しない）

    Returns:
        Figure と Axes のタプル
    """
    rng = np.random.default_rng(seed)

    fig, ax = plt.subplots(figsize=figsize)
    for i, data in enumerate(data_lists, start=1):
        if len(data) == 0:
            continue
        x = rng.normal(loc=i, scale=jitter_scale, size=len(data))
        ax.plot(x, data, "o", markersize=marker_size, alpha=alpha)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved: {save_path}")

    return fig, ax


def extract_values_by_category(
    df: pl.DataFrame,
    category_col: str,
    value_col: str,
    category_order: Sequence[str],
) -> list[list[Any]]:
    """カテゴリ別にデータを抽出する.

    Args:
        df: データフレーム
        category_col: カテゴリ列名
        value_col: 値列名
        category_order: カテゴリの順序

    Returns:
        カテゴリ順のデータリスト
    """
    return [
        df.filter(pl.col(category_col) == cat)
        .select(value_col)
        .to_series()
        .to_list()
        for cat in category_order
    ]
