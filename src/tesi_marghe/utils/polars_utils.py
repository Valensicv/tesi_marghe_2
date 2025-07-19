import polars as pl
import logging

logger = logging.getLogger(__name__)

def get_nulls(df):
    return (
        df.select(pl.all().null_count())
        .transpose(include_header=True)
        .rename({"column": "column_name", "column_0": "null_count"})
        .sort("null_count", descending=True)
        .filter(pl.col.null_count > 0)
        .with_columns((pl.col.null_count / df.height).alias("null_ratio"))
    )


def print_nulls(df):
    print(get_nulls(df))
    return df


def find_df_key(df):
    return (
        df.select(pl.all().n_unique())
        .transpose(include_header=True)
        .rename({"column": "column_name", "column_0": "n_unique_values"})
        .filter(pl.col.n_unique_values.eq(df.height))
    )


def get_cardinality(df):
    return (
        df.select(pl.all().n_unique())
        .transpose(include_header=True)
        .rename({"column": "column_name", "column_0": "n_unique_values"})
        .sort("n_unique_values", descending=True)
    )


def quantile_filtering(df, col, upper_quantile, lower_quantile=None):
    if lower_quantile is None:
        lower_quantile = 1 - upper_quantile
    return df.filter(
        pl.col(col).gt(pl.col(col).quantile(lower_quantile)),
        pl.col(col).lt(pl.col(col).quantile(upper_quantile)),
    )


def drop_columns_that_are_all_null(_df: pl.DataFrame) -> pl.DataFrame:
    return _df[[s.name for s in _df if not (s.null_count() == _df.height)]]


def print_shape(df, label=""):
    print(f"Shape {label}: {df.shape}")
    return df


def get_value_counts_per_columns(df):
    return (
        df.unpivot(on=df.columns)
        .group_by("variable")
        .agg(
            [
                pl.all()
                .value_counts(normalize=True, sort=True)
                .alias("proportion")
                .struct.rename_fields(["val_prop", "proportion"]),
                pl.all()
                .value_counts(normalize=False, sort=True)
                .alias("counts")
                .struct.field("count"),
            ],
        )
        .explode(["proportion", "counts"])
        .unnest("proportion")
        .with_columns(
            cum_prop=pl.col("proportion").cum_sum().over("variable"),
            cum_num=pl.col("counts").cum_sum().over("variable"),
        )
    )


def _value_count_both(series):
    print(series.name)
    return series.value_counts(normalize=True).join(
        series.value_counts(), on=series.name
    )


def print_estimated_size(df, label="", unit="mb"):
    print(f"Estimated size {label}: {df.estimated_size(unit=unit)} {unit}")
    return df
