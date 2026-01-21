"""ヒートマップ描画関数.

このモジュールでは以下の処理を提供する:
- 汎用ヒートマップ描画
- 曜日×時間帯ヒートマップ
- カテゴリ×カテゴリ ヒートマップ
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


import matplotlib.pyplot as plt
import matplotlib_fontja  # noqa: F401

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray


def plot_heatmap(
    mat: NDArray[np.float64],
    *,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
    xlabel: str,
    ylabel: str,
    cbar_label: str = "",
    figsize: tuple[float, float] = (8, 6),
    annotate: bool = False,
    fmt: str = ".2f",
    save_path: str | Path | None = None,
) -> None:
    """汎用ヒートマップを描画する.

    Args:
        mat: 2次元配列
        row_labels: 行ラベル
        col_labels: 列ラベル
        title: グラフのタイトル
        xlabel: X軸ラベル
        ylabel: Y軸ラベル
        cbar_label: カラーバーのラベル
        figsize: 図のサイズ
        annotate: セルに値を表示するか
        fmt: 値の表示フォーマット
        save_path: 図の保存パス（Noneの場合は保存しない）
    """
    _, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, aspect="auto", origin="upper")
    cbar = plt.colorbar(im, ax=ax)
    if cbar_label:
        cbar.set_label(cbar_label)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if annotate:
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                ax.text(
                    c,
                    r,
                    f"{mat[r, c]:{fmt}}",
                    ha="center",
                    va="center",
                )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved: {save_path}")

    plt.show()


def plot_dow_hour_heatmap(
    mat: NDArray[np.float64],
    title: str,
    value_label: str = "share_within_type",
    figsize: tuple[float, float] = (12, 3.5),
    save_path: str | Path | None = None,
) -> None:
    """曜日×時間帯のヒートマップを描画する.

    Args:
        mat: (7, 24) の2次元配列（行:曜日, 列:時間）
        title: グラフのタイトル
        value_label: カラーバーのラベル
        figsize: 図のサイズ
        save_path: 図の保存パス（Noneの場合は保存しない）
    """
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    hours = [str(h) for h in range(24)]

    _, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, aspect="auto")
    ax.set_title(title)
    ax.set_yticks(range(7))
    ax.set_yticklabels(dow_labels)
    ax.set_xticks(range(24))
    ax.set_xticklabels(hours)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Day of week")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(value_label)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved: {save_path}")

    plt.show()


def plot_category_heatmap(
    mat: NDArray[np.float64],
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
    xlabel: str,
    ylabel: str,
    figsize: tuple[float, float] = (8, 6),
    fmt: str = ".2f",
    save_path: str | Path | None = None,
) -> None:
    """カテゴリ×カテゴリのヒートマップを描画する（値注釈付き）.

    Args:
        mat: 2次元配列
        row_labels: 行ラベル
        col_labels: 列ラベル
        title: グラフのタイトル
        xlabel: X軸ラベル
        ylabel: Y軸ラベル
        figsize: 図のサイズ
        fmt: 値の表示フォーマット
        save_path: 図の保存パス（Noneの場合は保存しない）
    """
    plot_heatmap(
        mat,
        row_labels=row_labels,
        col_labels=col_labels,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        figsize=figsize,
        annotate=True,
        fmt=fmt,
        save_path=save_path,
    )
