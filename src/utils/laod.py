"""データをロードする."""

import polars as pl

from config import DATABASE_CLASS


def load_dataframes() -> dict[str, pl.DataFrame]:
    """全てのデータセットを読み込む."""
    history = pl.read_csv(DATABASE_CLASS / "history.csv")
    user = pl.read_csv(DATABASE_CLASS / "user.csv")
    spot = pl.read_csv(DATABASE_CLASS / "spot.csv")
    trip = pl.read_csv(DATABASE_CLASS / "trip.csv")

    return {"history": history, "user": user, "spot": spot, "trip": trip}
