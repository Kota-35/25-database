"""分析1: ユーザータイプ別の利用パターン分析.

このモジュールでは以下の分析を行う:
- 曜日×時間帯別の利用割合ヒートマップ
"""

import numpy as np
import polars as pl
from numpy.typing import NDArray

from plotting.heatmap import plot_dow_hour_heatmap
from preprocessing.datetime import add_time_features, parse_datetime_columns
from preprocessing.join import join_user_info
from utils.laod import load_dataframes

# =============================================================================
# データ前処理
# =============================================================================


def preprocess_for_heatmap(
    history: pl.DataFrame,
    user: pl.DataFrame,
) -> pl.DataFrame:
    """ヒートマップ分析用のデータ前処理を行う.

    Args:
        history: 履歴データフレーム
        user: ユーザーデータフレーム

    Returns:
        前処理済みデータフレーム
    """
    df = parse_datetime_columns(history)
    df = join_user_info(df, user)
    df = add_time_features(df)
    df = df.drop_nulls(["user_type", "dow", "hour"])
    return df


# =============================================================================
# 集計処理
# =============================================================================


def aggregate_by_user_type_dow_hour(df: pl.DataFrame) -> pl.DataFrame:
    """user_type, dow, hour でグループ化して件数を集計する.

    Args:
        df: 前処理済みデータフレーム

    Returns:
        集計結果データフレーム
    """
    return (
        df.group_by(["user_type", "dow", "hour"])
        .len()
        .rename({"len": "ride_count"})
    )


def calculate_share_within_type(agg: pl.DataFrame) -> pl.DataFrame:
    """ユーザータイプ内での割合を計算する.

    Args:
        agg: 集計済みデータフレーム

    Returns:
        share_within_type 列が追加されたデータフレーム
    """
    total_by_type = agg.group_by("user_type").agg(
        pl.col("ride_count").sum().alias("total"),
    )

    return (
        agg.join(total_by_type, on="user_type", how="left")
        .with_columns(
            (pl.col("ride_count") / pl.col("total")).alias("share_within_type"),
        )
        .drop("total")
    )


def get_total_rides_by_type(agg: pl.DataFrame) -> dict[str, int]:
    """ユーザータイプ別の総利用回数を取得する.

    Args:
        agg: 集計済みデータフレーム

    Returns:
        ユーザータイプをキー、総利用回数を値とする辞書
    """
    total_by_type = agg.group_by("user_type").agg(
        pl.col("ride_count").sum().alias("total_ride_count"),
    )
    return dict(
        zip(
            total_by_type["user_type"].to_list(),
            total_by_type["total_ride_count"].to_list(),
            strict=True,
        ),
    )


# =============================================================================
# ヒートマップ用マトリックス作成
# =============================================================================


def create_dow_hour_grid() -> pl.DataFrame:
    """曜日(0-6) × 時間(0-23) の全組み合わせグリッドを作成する.

    Returns:
        全組み合わせのデータフレーム
    """
    return pl.DataFrame({"dow": list(range(7))}).join(
        pl.DataFrame({"hour": list(range(24))}),
        how="cross",
    )


def create_heatmap_matrix(
    agg_out: pl.DataFrame,
    user_type: str,
    value_col: str = "share_within_type",
) -> NDArray[np.float64]:
    """ヒートマップ用の2次元配列を作成する.

    Args:
        agg_out: share_within_type を含む集計データフレーム
        user_type: 対象のユーザータイプ ("staff" or "student")
        value_col: 値として使用する列名

    Returns:
        (7, 24) の2次元配列（行:曜日, 列:時間）
    """
    grid = create_dow_hour_grid()

    # user_typeでフィルタリング
    filtered = agg_out.filter(pl.col("user_type") == user_type).select(
        ["dow", "hour", value_col],
    )

    # 全グリッドに左結合して欠損を0埋め
    filled = (
        grid.join(filtered, on=["dow", "hour"], how="left")
        .with_columns(pl.col(value_col).fill_null(0.0))
        .sort(["dow", "hour"])
    )

    # pivot: rows=dow, cols=hour
    pivoted = filled.pivot(
        index="dow",
        on="hour",
        values=value_col,
        aggregate_function="first",
    ).sort("dow")

    # numpy配列に変換
    return pivoted.drop("dow").to_numpy()


# =============================================================================
# 分析実行関数
# =============================================================================


def analyze_heatmap(history: pl.DataFrame, user: pl.DataFrame) -> None:
    """曜日×時間帯別の利用割合ヒートマップ分析を実行する.

    Args:
        history: 履歴データフレーム
        user: ユーザーデータフレーム
    """
    # データ前処理
    df = preprocess_for_heatmap(history, user)

    # 集計
    agg = aggregate_by_user_type_dow_hour(df)
    agg_out = calculate_share_within_type(agg)

    # 総利用回数を取得
    total_map = get_total_rides_by_type(agg)
    staff_total = total_map.get("staff", 0)
    student_total = total_map.get("student", 0)

    # ヒートマップ用マトリックス作成
    mat_staff = create_heatmap_matrix(agg_out, "staff")
    mat_student = create_heatmap_matrix(agg_out, "student")

    print(
        f"Matrix shapes: staff={mat_staff.shape}, student={mat_student.shape}",
    )

    # プロット
    plot_dow_hour_heatmap(
        mat_staff,
        f"staff: share within type by dow x hour (total rides={staff_total})",
    )
    plot_dow_hour_heatmap(
        mat_student,
        f"student: share within type by dow x hour (total rides={student_total})",
    )


def main() -> None:
    """メイン関数."""
    dfs = load_dataframes()
    history, user = dfs["history"], dfs["user"]

    print("=== ヒートマップ分析 ===")
    analyze_heatmap(history, user)


if __name__ == "__main__":
    main()
