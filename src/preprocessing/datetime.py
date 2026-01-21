"""日時関連の前処理関数.

このモジュールでは以下の処理を提供する:
- 日時列のパース・変換
- 時間特徴量（曜日・時間）の追加
- 所要時間の計算
"""

import polars as pl


def parse_datetime_columns(
    df: pl.DataFrame,
    started_col: str = "started_at",
    ended_col: str = "ended_at",
    started_alias: str = "started_at_dt",
    ended_alias: str = "ended_at_dt",
) -> pl.DataFrame:
    """started_at, ended_at を datetime 型に変換する.

    Args:
        df: 対象データフレーム
        started_col: 開始日時の列名
        ended_col: 終了日時の列名
        started_alias: 変換後の開始日時列名
        ended_alias: 変換後の終了日時列名

    Returns:
        datetime 列が追加されたデータフレーム
    """
    return df.with_columns(
        pl.col(started_col).str.to_datetime(strict=False).alias(started_alias),
        pl.col(ended_col).str.to_datetime(strict=False).alias(ended_alias),
    )


def parse_datetime_columns_strptime(
    df: pl.DataFrame,
    started_col: str = "started_at",
    ended_col: str = "ended_at",
) -> pl.DataFrame:
    """started_at, ended_at を strptime で datetime 型に変換する.

    フォーマットが不明な場合に使用する。元の列を上書きする。

    Args:
        df: 対象データフレーム
        started_col: 開始日時の列名
        ended_col: 終了日時の列名

    Returns:
        datetime 列が変換されたデータフレーム
    """
    return df.with_columns(
        pl.col(started_col)
        .cast(pl.Utf8)
        .str.strptime(pl.Datetime, strict=False),
        pl.col(ended_col).cast(pl.Utf8).str.strptime(pl.Datetime, strict=False),
    )


def add_time_features(
    df: pl.DataFrame,
    datetime_col: str = "started_at_dt",
) -> pl.DataFrame:
    """曜日(dow)と時間(hour)の特徴量を追加する.

    Args:
        df: datetime 列を持つデータフレーム
        datetime_col: 日時列名

    Returns:
        dow, hour 列が追加されたデータフレーム
    """
    return df.with_columns(
        pl.col(datetime_col).dt.weekday().alias("dow"),
        pl.col(datetime_col).dt.hour().alias("hour"),
    )


def add_duration_column(
    df: pl.DataFrame,
    started_col: str = "started_at_dt",
    ended_col: str = "ended_at_dt",
    alias: str = "duration_min",
) -> pl.DataFrame:
    """所要時間（分）を計算する.

    Args:
        df: 開始・終了日時列を持つデータフレーム
        started_col: 開始日時の列名
        ended_col: 終了日時の列名
        alias: 出力列名

    Returns:
        所要時間列が追加されたデータフレーム
    """
    return df.with_columns(
        (pl.col(ended_col) - pl.col(started_col))
        .dt.total_minutes()
        .alias(alias),
    )
