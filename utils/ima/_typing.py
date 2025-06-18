"""Common types used in solver."""
from pathlib import Path
from typing import Union
import pandas
import datetime

OptionalPath = Union[str, Path, None]
StrOrTimeStamp = Union[str, pandas.Timestamp, datetime.datetime]
