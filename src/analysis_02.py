"""分析2: 距離・所要時間・乗客数の分布分析.

このモジュールでは以下の分析を行う:
- ユーザータイプ別の距離・所要時間の箱ひげ図
- 乗客数別の距離分布（散布図・箱ひげ図）
"""

import matplotlib.pyplot as plt
import matplotlib_fontja  # noqa: F401
import polars as pl

from plotting.boxplot import (
    extract_values_by_category,
    plot_boxplot_by_category,
    plot_boxplot_by_user_type,
)
from plotting.scatter import plot_scatter_by_category
from preprocessing.datetime import add_duration_column, parse_datetime_columns
from preprocessing.join import join_user_info
from preprocessing.transform import (
    PASSENGERS_BINS,
    add_passengers_bin,
    cast_distance_to_float,
)
from utils.laod import load_dataframes

# =============================================================================
# データ前処理
# =============================================================================


def preprocess_for_boxplot(
    history: pl.DataFrame,
    user: pl.DataFrame,
) -> pl.DataFrame:
    """箱ひげ図分析用のデータ前処理を行う.

    Args:
        history: 履歴データフレーム
        user: ユーザーデータフレーム

    Returns:
        前処理済みデータフレーム
    """
    df = (
        history.pipe(parse_datetime_columns)
        .pipe(cast_distance_to_float, target_col="distance_f")
        .pipe(join_user_info, user)
        .pipe(add_duration_column)
    )
    df = df.drop_nulls(["user_type", "distance_f", "duration_min"])
    # 異常値を除外
    df = df.filter((pl.col("duration_min") > 0) & (pl.col("distance_f") >= 0))
    return df


def preprocess_for_passengers_analysis(
    history: pl.DataFrame,
    user: pl.DataFrame,
) -> pl.DataFrame:
    """乗客数別分析用のデータ前処理を行う.

    Args:
        history: 履歴データフレーム
        user: ユーザーデータフレーム

    Returns:
        前処理済みデータフレーム
    """
    df = (
        join_user_info(history, user)
        .select(["user_type", "passengers_count", "distance"])
        .with_columns(
            pl.col("passengers_count").cast(pl.Int64),
            pl.col("distance").cast(pl.Float64, strict=False),
        )
        .filter(
            pl.col("user_type").is_not_null()
            & pl.col("passengers_count").is_not_null()
            & pl.col("distance").is_not_null()
            & (pl.col("passengers_count") >= 1)
            & (pl.col("distance") > 0),
        )
        .pipe(add_passengers_bin, target_col="passengers_cat")
    )
    return df


# =============================================================================
# 集計処理
# =============================================================================


def aggregate_distance_by_passengers(df: pl.DataFrame) -> pl.DataFrame:
    """乗客数カテゴリ別に距離の統計量を集計する.

    Args:
        df: 前処理済みデータフレーム

    Returns:
        集計結果データフレーム
    """
    return (
        df.group_by(["user_type", "passengers_cat"])
        .agg(
            pl.len().alias("n"),
            pl.col("distance").mean().alias("mean"),
            pl.col("distance").median().alias("median"),
            pl.col("distance").quantile(0.25).alias("q25"),
            pl.col("distance").quantile(0.75).alias("q75"),
            pl.col("distance").max().alias("max"),
        )
        .sort(["user_type", "passengers_cat"])
    )


# =============================================================================
# 分析実行関数
# =============================================================================


def analyze_boxplot_by_user_type(
    history: pl.DataFrame,
    user: pl.DataFrame,
) -> None:
    """ユーザータイプ別の距離・所要時間の箱ひげ図分析を実行する.

    Args:
        history: 履歴データフレーム
        user: ユーザーデータフレーム
    """
    df = preprocess_for_boxplot(history, user)

    print(f"分析対象レコード数: {len(df)}")

    # 距離の箱ひげ図
    _, _ = plot_boxplot_by_user_type(
        df,
        "distance_f",
        "user_type別 距離の分布",
    )
    plt.show()

    # 所要時間の箱ひげ図
    _, _ = plot_boxplot_by_user_type(
        df,
        "duration_min",
        "user_type別 所要時間(分)の分布",
    )
    plt.show()


def analyze_distance_by_passengers(
    history: pl.DataFrame,
    user: pl.DataFrame,
) -> None:
    """乗客数別の距離分布分析を実行する.

    Args:
        history: 履歴データフレーム
        user: ユーザーデータフレーム
    """
    df = preprocess_for_passengers_analysis(history, user)

    print(f"分析対象レコード数: {len(df)}")

    # 集計結果を表示
    summary = aggregate_distance_by_passengers(df)
    print("\n=== 乗客数カテゴリ別の距離統計量 ===")
    print(summary)

    # 散布図
    plot_scatter_by_category(
        df,
        x_col="passengers_count",
        y_col="distance",
        category_col="user_type",
        title="乗客数と距離の散布図 (user_type別)",
        xlabel="passengers_count",
        ylabel="distance",
    )

    # ユーザータイプ別の箱ひげ図
    user_types = sorted(df.select("user_type").unique().to_series().to_list())
    cat_order = PASSENGERS_BINS

    for ut in user_types:
        sub = df.filter(pl.col("user_type") == ut)
        data_lists = extract_values_by_category(
            sub,
            category_col="passengers_cat",
            value_col="distance",
            category_order=cat_order,
        )
        # 空でないデータのみ使用
        valid_data = []
        valid_labels = []
        for data, label in zip(data_lists, cat_order, strict=True):
            if len(data) > 0:
                valid_data.append(data)
                valid_labels.append(label)

        if len(valid_data) == 0:
            print(f"No data for user_type={ut}")
            continue

        _, _ = plot_boxplot_by_category(
            valid_data,
            valid_labels,
            title=f"乗客数カテゴリ別 距離の分布 ({ut})",
            xlabel="passengers_cat",
            ylabel="distance",
        )
        plt.show()


def main() -> None:
    """メイン関数."""
    dfs = load_dataframes()
    history, user = dfs["history"], dfs["user"]

    print("=== ユーザータイプ別 距離・所要時間 箱ひげ図分析 ===")
    analyze_boxplot_by_user_type(history, user)

    print("\n=== 乗客数別 距離分布分析 ===")
    analyze_distance_by_passengers(history, user)


if __name__ == "__main__":
    main()
