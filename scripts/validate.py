import pandera as pa
from pandera import Column, Check
import pandas as pd

def validate_common(df: pd.DataFrame):
    schema = pa.DataFrameSchema({
        "GEO": Column(str, checks=Check.str_length(min_value=1)),
        "YEAR": Column(int, checks=Check.ge(1900)),
    }, coerce=True)
    schema.validate(df, lazy=True)

if __name__ == "__main__":
    pass
