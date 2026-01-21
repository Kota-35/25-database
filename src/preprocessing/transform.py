"""型変換・ビン化に関する前処理関数."""

import polars as pl

# =============================================================================
# 集計
# =============================================================================


def count_spots_per_history(trip: pl.DataFrame) -> pl.DataFrame:
    """history_id ごとの目的地数（trip行数）を集計する.

    Args:
        trip: trip データフレーム

    Returns:
        history_id と spots_n 列を持つデータフレーム
    """
    return trip.group_by("history_id").agg(pl.len().alias("spots_n"))


# =============================================================================
# 定数
# =============================================================================

SPOTS_BINS = ["1", "2", "3", "4", "5+"]
PASSENGERS_BINS = ["1", "2", "3", "4", "5+"]


# =============================================================================
# 型変換
# =============================================================================


def cast_distance_to_float(
    df: pl.DataFrame,
    source_col: str = "distance",
    target_col: str | None = None,
) -> pl.DataFrame:
    """distance 列を Float64 型に変換する.

    Args:
        df: データフレーム
        source_col: 変換元の列名
        target_col: 変換後の列名（Noneの場合は source_col を上書き）

    Returns:
        変換後のデータフレーム
    """
    target = target_col if target_col else source_col
    return df.with_columns(
        pl.col(source_col).cast(pl.Float64, strict=False).alias(target),
    )


def cast_passengers_count_to_int(
    df: pl.DataFrame,
    col: str = "passengers_count",
) -> pl.DataFrame:
    """passengers_count 列を Int64 型に変換する.

    Args:
        df: データフレーム
        col: 変換する列名

    Returns:
        変換後のデータフレーム
    """
    return df.with_columns(
        pl.col(col).cast(pl.Int64, strict=False),
    )


# =============================================================================
# ビン化
# =============================================================================


def add_passengers_bin(
    df: pl.DataFrame,
    source_col: str = "passengers_count",
    target_col: str = "passengers_bin",
    threshold: int = 5,
) -> pl.DataFrame:
    """乗客数ビン列を追加する（1,2,3,4,5+）.

    Args:
        df: データフレーム
        source_col: 元の列名
        target_col: 追加するビン列名
        threshold: この値以上を "5+" とする閾値

    Returns:
        ビン列が追加されたデータフレーム
    """
    return df.with_columns(
        pl.when(pl.col(source_col) >= threshold)
        .then(pl.lit(f"{threshold}+"))
        .otherwise(pl.col(source_col).cast(pl.Utf8))
        .alias(target_col),
    )


def add_spots_bin(
    df: pl.DataFrame,
    source_col: str = "spots_n",
    target_col: str = "spots_bin",
    threshold: int = 5,
) -> pl.DataFrame:
    """目的地数ビン列を追加する（1,2,3,4,5+）.

    Args:
        df: データフレーム
        source_col: 元の列名
        target_col: 追加するビン列名
        threshold: この値以上を "5+" とする閾値

    Returns:
        ビン列が追加されたデータフレーム
    """
    return df.with_columns(
        pl.when(pl.col(source_col) >= threshold)
        .then(pl.lit(f"{threshold}+"))
        .otherwise(pl.col(source_col).cast(pl.Utf8))
        .alias(target_col),
    )


def add_bin_columns(
    df: pl.DataFrame,
    spots_col: str = "spots_n",
    passengers_col: str = "passengers_count",
) -> pl.DataFrame:
    """目的地数ビンと乗客数ビンを両方追加する.

    Args:
        df: データフレーム
        spots_col: 目的地数の列名
        passengers_col: 乗客数の列名

    Returns:
        spots_bin, passengers_bin 列が追加されたデータフレーム
    """
    return df.pipe(add_spots_bin, source_col=spots_col).pipe(
        add_passengers_bin,
        source_col=passengers_col,
    )
