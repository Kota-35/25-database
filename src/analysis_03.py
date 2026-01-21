"""分析3: 目的地数・乗客数の分布分析.

このモジュールでは以下の分析を行う:
- ユーザータイプ別の1乗車あたり目的地数
- 目的地数別の距離分布
- 乗客数×目的地数の割合ヒートマップ
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib_fontja  # noqa: F401
import numpy as np
import polars as pl

if TYPE_CHECKING:
    from numpy.typing import NDArray

from plotting.boxplot import (
    extract_values_by_category,
    plot_boxplot_by_category,
    plot_violin_with_boxplot,
)
from plotting.heatmap import plot_category_heatmap
from plotting.scatter import plot_jitter_scatter
from preprocessing.datetime import (
    add_duration_column,
    parse_datetime_columns_strptime,
)
from preprocessing.join import join_user_info
from preprocessing.transform import (
    PASSENGERS_BINS,
    SPOTS_BINS,
    add_bin_columns,
    cast_distance_to_float,
    count_spots_per_history,
)
from utils.laod import load_dataframes
from utils.paths import get_figure_path

# =============================================================================
# データ前処理
# =============================================================================


def build_ride_features(
    history: pl.DataFrame,
    trip: pl.DataFrame,
    user: pl.DataFrame,
) -> pl.DataFrame:
    """乗車レベルの特徴量を構築する.

    Args:
        history: 履歴データフレーム
        trip: tripデータフレーム
        user: ユーザーデータフレーム

    Returns:
        前処理済みデータフレーム
    """
    # trip: history_id ごとの目的地数
    trip_counts = count_spots_per_history(trip)

    df = (
        history.join(trip_counts, on="history_id", how="left")
        .pipe(join_user_info, user)
        .with_columns(
            # trip が無い場合は 0 に
            pl.col("spots_n").fill_null(0).cast(pl.Int64),
        )
        .pipe(cast_distance_to_float)
        .pipe(parse_datetime_columns_strptime)
        .pipe(
            add_duration_column,
            started_col="started_at",
            ended_col="ended_at",
        )
        .pipe(add_bin_columns)
        # 目的地数が 1以上、乗客数が 1以上の乗車だけに絞る
        .filter((pl.col("spots_n") >= 1) & (pl.col("passengers_count") >= 1))
        # 欠損を除外
        .drop_nulls(
            [
                "user_type",
                "distance",
                "passengers_count",
                "spots_bin",
                "passengers_bin",
            ],
        )
    )

    return df


# =============================================================================
# 集計処理
# =============================================================================


def aggregate_passengers_spots_counts(df: pl.DataFrame) -> pl.DataFrame:
    """乗客数ビン×目的地数ビンの件数を集計する.

    Args:
        df: 前処理済みデータフレーム

    Returns:
        集計結果データフレーム
    """
    return df.group_by(["passengers_bin", "spots_bin"]).agg(pl.len().alias("n"))


def create_heatmap_matrix(counts: pl.DataFrame) -> NDArray[np.float64]:
    """ヒートマップ用の割合行列を作成する.

    Args:
        counts: passengers_bin, spots_bin, n 列を持つ集計データフレーム

    Returns:
        (5, 5) の2次元配列（行:乗客数ビン, 列:目的地数ビン）
    """
    # pivot して行列化
    pivot = counts.pivot(
        values="n",
        index="passengers_bin",
        on="spots_bin",
        aggregate_function="first",
    ).fill_null(0)

    # 欠けてる列を追加
    for b in SPOTS_BINS:
        if b not in pivot.columns:
            pivot = pivot.with_columns(pl.lit(0).alias(b))

    # 欠けてる行を追加
    existing_rows = set(pivot["passengers_bin"].to_list())
    missing_rows = [b for b in PASSENGERS_BINS if b not in existing_rows]
    if missing_rows:
        add_rows = pl.DataFrame(
            {
                "passengers_bin": missing_rows,
                **{c: [0] * len(missing_rows) for c in SPOTS_BINS},
            },
        )
        pivot = pl.concat([pivot, add_rows], how="vertical")

    # 明示的な並べ替え（PASSENGERS_BINS順）
    pivot = (
        pivot.with_columns(
            pl.col("passengers_bin")
            .map_elements(
                lambda x: PASSENGERS_BINS.index(x),
                return_dtype=pl.Int64,
            )
            .alias("_row_order"),
        )
        .sort("_row_order")
        .drop("_row_order")
        .select(["passengers_bin", *SPOTS_BINS])
    )

    mat_counts = pivot.select(SPOTS_BINS).to_numpy()
    total = mat_counts.sum()
    mat_share: NDArray[np.float64] = (
        mat_counts / total if total > 0 else mat_counts.astype(np.float64)
    )

    return mat_share


# =============================================================================
# 分析実行関数
# =============================================================================


def analyze_spots_by_user_type(df: pl.DataFrame) -> None:
    """ユーザータイプ別の目的地数分析を実行する.

    Args:
        df: 前処理済みデータフレーム
    """
    print("--- ユーザータイプ別 目的地数 バイオリン＋箱ひげ図 ---")

    user_types = sorted(df.select("user_type").unique().to_series().to_list())
    data_lists = extract_values_by_category(
        df,
        category_col="user_type",
        value_col="spots_n",
        category_order=user_types,
    )

    _, _ = plot_violin_with_boxplot(
        data_lists,
        labels=user_types,
        title="user_type別：1乗車あたり目的地数",
        ylabel="目的地数（1乗車あたり） spots_n",
        save_path=get_figure_path("03_violin_spots_by_user_type.png"),
    )
    plt.show()


def analyze_distance_by_spots(df: pl.DataFrame) -> None:
    """目的地数別の距離分布分析を実行する.

    Args:
        df: 前処理済みデータフレーム
    """
    print("--- 目的地数×距離：ジッター散布図 ---")
    plot_jitter_scatter(
        df,
        category_col="spots_bin",
        value_col="distance",
        category_order=SPOTS_BINS,
        title="目的地数×距離：ジッター散布",
        xlabel="目的地数ビン（spots_n）",
        ylabel="移動距離（distance）",
        save_path=get_figure_path("03_jitter_distance_by_spots.png"),
    )

    print("--- 目的地数×距離：箱ひげ図 ---")
    data_lists = extract_values_by_category(
        df,
        category_col="spots_bin",
        value_col="distance",
        category_order=SPOTS_BINS,
    )

    _, _ = plot_boxplot_by_category(
        data_lists,
        labels=SPOTS_BINS,
        title="目的地数×距離",
        xlabel="目的地数ビン（spots_n）",
        ylabel="移動距離（distance）",
        save_path=get_figure_path("03_boxplot_distance_by_spots.png"),
    )
    plt.show()


def analyze_passengers_vs_spots(df: pl.DataFrame) -> None:
    """乗客数×目的地数の割合分析を実行する.

    Args:
        df: 前処理済みデータフレーム
    """
    print("--- 乗車人数×目的地数：割合ヒートマップ ---")
    counts = aggregate_passengers_spots_counts(df)
    mat_share = create_heatmap_matrix(counts)

    plot_category_heatmap(
        mat_share,
        row_labels=PASSENGERS_BINS,
        col_labels=SPOTS_BINS,
        title="乗車人数×目的地数：乗車割合ヒートマップ（全体比）",
        xlabel="目的地数ビン（spots_n）",
        ylabel="乗客人数ビン（passengers_count）",
        save_path=get_figure_path("03_heatmap_passengers_spots.png"),
    )


def main() -> None:
    """メイン関数."""
    dfs = load_dataframes()
    history = dfs["history"]
    trip = dfs["trip"]
    user = dfs["user"]

    print("=== データ前処理 ===")
    df = build_ride_features(history, trip, user)
    print(f"分析対象レコード数: {len(df)}")

    print("\n=== 分析1: ユーザータイプ別 目的地数 ===")
    analyze_spots_by_user_type(df)

    print("\n=== 分析2: 目的地数別 距離分布 ===")
    analyze_distance_by_spots(df)

    print("\n=== 分析3: 乗車人数×目的地数 割合 ===")
    analyze_passengers_vs_spots(df)


if __name__ == "__main__":
    main()
