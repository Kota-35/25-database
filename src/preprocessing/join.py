"""テーブル結合関連の前処理関数."""

import polars as pl


def join_user_info(
    history_df: pl.DataFrame,
    user_df: pl.DataFrame,
    select_columns: list[str] | None = None,
) -> pl.DataFrame:
    """history に user 情報を結合する.

    Args:
        history_df: 履歴データフレーム
        user_df: ユーザーデータフレーム
        select_columns: user_df から選択する列名のリスト（デフォルト: ["user_id", "user_type"]）

    Returns:
        結合されたデータフレーム
    """
    if select_columns is None:
        select_columns = ["user_id", "user_type"]

    return history_df.join(
        user_df.select(select_columns),
        on="user_id",
        how="left",
    )


def join_trip_counts(
    history_df: pl.DataFrame,
    trip_df: pl.DataFrame,
    count_col: str = "spots_n",
) -> pl.DataFrame:
    """history に trip の目的地数を結合する.

    Args:
        history_df: 履歴データフレーム
        trip_df: tripデータフレーム
        count_col: 目的地数の列名

    Returns:
        結合されたデータフレーム
    """
    trip_counts = trip_df.group_by("history_id").agg(
        pl.len().alias(count_col),
    )

    return history_df.join(trip_counts, on="history_id", how="left")
