"""分析2: 距離・所要時間・乗客数の分布分析.

このモジュールでは以下の分析を行う:
- ユーザータイプ別の距離・所要時間の箱ひげ図
- 乗客数別の距離分布（散布図・箱ひげ図）
"""

import matplotlib.pyplot as plt
import polars as pl

from utils.laod import load_dataframes

# =============================================================================
# データ前処理
# =============================================================================


def parse_datetime_columns(df: pl.DataFrame) -> pl.DataFrame:
    """started_at, ended_at を datetime 型に変換する.

    Args:
        df: history データフレーム

    Returns:
        datetime 列が追加されたデータフレーム
    """
    return df.with_columns(
        pl.col("started_at")
        .str.to_datetime(strict=False)
        .alias("started_at_dt"),
        pl.col("ended_at").str.to_datetime(strict=False).alias("ended_at_dt"),
    )


def cast_distance_column(df: pl.DataFrame) -> pl.DataFrame:
    """distance 列を Float64 型に変換する.

    Args:
        df: history データフレーム

    Returns:
        distance_f 列が追加されたデータフレーム
    """
    return df.with_columns(
        pl.col("distance").cast(pl.Float64, strict=False).alias("distance_f"),
    )


def calculate_duration(df: pl.DataFrame) -> pl.DataFrame:
    """所要時間（分）を計算する.

    Args:
        df: started_at_dt, ended_at_dt 列を持つデータフレーム

    Returns:
        duration_min 列が追加されたデータフレーム
    """
    return df.with_columns(
        (pl.col("ended_at_dt") - pl.col("started_at_dt"))
        .dt.total_minutes()
        .alias("duration_min"),
    )


def join_user_info(
    history_df: pl.DataFrame,
    user_df: pl.DataFrame,
) -> pl.DataFrame:
    """history に user 情報を結合する.

    Args:
        history_df: 履歴データフレーム
        user_df: ユーザーデータフレーム

    Returns:
        結合されたデータフレーム
    """
    return history_df.join(user_df, on="user_id", how="left")


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
    df = parse_datetime_columns(history)
    df = cast_distance_column(df)
    df = join_user_info(df, user)
    df = calculate_duration(df)
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
            & (pl.col("distance") > 0)
        )
        .with_columns(
            pl.when(pl.col("passengers_count") >= 5)
            .then(pl.lit("5+"))
            .otherwise(pl.col("passengers_count").cast(pl.Utf8))
            .alias("passengers_cat"),
        )
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
# プロット関数
# =============================================================================


def plot_boxplot_by_user_type(
    df: pl.DataFrame,
    value_col: str,
    title: str,
) -> None:
    """ユーザータイプ別の箱ひげ図を描画する.

    Args:
        df: 前処理済みデータフレーム
        value_col: 値として使用する列名
        title: グラフのタイトル
    """
    staff_vals = (
        df.filter(pl.col("user_type") == "staff")
        .select(value_col)
        .to_numpy()
        .ravel()
    )
    student_vals = (
        df.filter(pl.col("user_type") == "student")
        .select(value_col)
        .to_numpy()
        .ravel()
    )

    _, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(
        [staff_vals, student_vals],
        tick_labels=["staff", "student"],
        showfliers=True,
    )
    ax.set_title(title)
    ax.set_ylabel(value_col)
    plt.tight_layout()
    plt.show()


def plot_distance_vs_passengers_scatter(df: pl.DataFrame) -> None:
    """乗客数と距離の散布図を描画する.

    Args:
        df: 前処理済みデータフレーム
    """
    user_types = sorted(
        df.select("user_type").unique().to_series().drop_nulls().to_list()
    )

    _, ax = plt.subplots(figsize=(8, 5))
    for ut in user_types:
        sub = df.filter(pl.col("user_type") == ut)
        passengers = sub.select("passengers_count").to_series().to_list()
        distance = sub.select("distance").to_series().to_list()
        ax.scatter(
            passengers,
            distance,
            alpha=0.4,
            label=ut,
        )

    ax.set_xlabel("passengers_count")
    ax.set_ylabel("distance")
    ax.set_title("Distance vs Passengers Count (by user_type)")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_distance_boxplot_by_passengers(
    df: pl.DataFrame,
    user_type: str,
) -> None:
    """乗客数カテゴリ別の距離箱ひげ図を描画する.

    Args:
        df: 前処理済みデータフレーム
        user_type: 対象のユーザータイプ
    """
    cat_order = ["1", "2", "3", "4", "5+"]

    sub = df.filter(pl.col("user_type") == user_type)

    data = []
    labels = []
    for c in cat_order:
        arr = (
            sub.filter(pl.col("passengers_cat") == c)
            .select("distance")
            .to_series()
            .to_list()
        )
        if len(arr) > 0:
            data.append(arr)
            labels.append(c)

    if len(data) == 0:
        print(f"No data for user_type={user_type}")
        return

    _, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, tick_labels=labels, showfliers=True)
    ax.set_xlabel("passengers_cat")
    ax.set_ylabel("distance")
    ax.set_title(f"Distance Distribution by Passengers Category ({user_type})")
    plt.tight_layout()
    plt.show()


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
    plot_boxplot_by_user_type(
        df,
        "distance_f",
        "Distance by user_type (boxplot)",
    )

    # 所要時間の箱ひげ図
    plot_boxplot_by_user_type(
        df,
        "duration_min",
        "Duration (min) by user_type (boxplot)",
    )


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
    plot_distance_vs_passengers_scatter(df)

    # ユーザータイプ別の箱ひげ図
    user_types = df.select("user_type").unique().to_series().to_list()
    for ut in sorted(user_types):
        plot_distance_boxplot_by_passengers(df, ut)


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
