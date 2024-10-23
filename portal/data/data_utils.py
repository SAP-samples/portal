import datetime
from math import log, nan
from typing import List, Optional, Union
import holidays
import warnings

import numpy as np
import pandas as pd
import pytz
import torch
from tqdm import tqdm

warnings.filterwarnings('ignore', message='Diwali and Holi holidays available from 2001 to 2030 only', category=Warning)

# For scientific notation m * base**n, we take n to be between -127 and 127 (like for Float32)
# To avoid bugs, keep the following number odd.
EXPONENT_BITS = 255
assert EXPONENT_BITS % 2, 'EXPONENT_BITS must be odd'
HALF_EXPONENT_BITS = EXPONENT_BITS // 2
# and for m we use 1000 bins (plus one more bit for sign), unless we use L2 / Huber loss
FRACTION_BINS = 1000
# For now, base is always 2; a lower value, like 1.1, might give better results. Must always be > 1.0
BASE = 2.0

date_cache = {}



def _str_to_num(x: str):
    try:
        return float(x)
    except ValueError:
        pass

    if ',' in x:
        try:
            return float(x.replace(',', '.'))
        except ValueError:
            pass

    if '.' in x and ',' in x:
        # If there are both . and , it could be both 1,234.56 (US) or 1.234,56 (EU)
        # Try just guessing it from which comes first
        if x.index('.') < x.index(','):
            number = x.replace('.', '').replace(',', '.')
        else:
            number = x.replace(',', '')
        try:
            return float(number)
        except ValueError:
            pass

    # TODO: ideally we'd like to also be able to parse other numbers, such as
    # '1 234' --> 1234 (e.g. in France they like putting these spaces)
    # '~234' or '234€' --> 234 (of course it's not a correct replacement, but the string embedding will also be there to support)

    return nan


def to_numeric(row: pd.Series,
               date_row: Optional['pd.Series[datetime.date]'] = None,
               time_row: Optional['pd.Series[datetime.time]'] = None) -> pd.Series:
    """
    Converts a row of a dataframe to a row of floats.
    For each value, if it's already a float, or convertible with pd.to_numeric (independently of the Serie type), then okay.
    Otherwise, if a date (with or without a time) is provided, it's converted to timestamp (seconds from 01/01/1970).
    If instead a time is provided but not a date, it's converted to float hours in [0.0, 24.0).
    If all of the above fails, also tries to convert strings like 0,0 for non-English locales.
    """
    if date_row is None:
        date_row = pd.Series([None] * len(row), index=row.index)
    if time_row is None:
        time_row = pd.Series([None] * len(row), index=row.index)

    string_row: 'pd.Series[str]' = row.apply(str)

    numeric_series = pd.to_numeric(string_row, errors='coerce')
    null_indices = numeric_series.index[numeric_series.isnull()]

    for idx in null_indices:
        if not pd.isnull(date_row[idx]):
            # We have a date; value is timestamp
            if pd.isnull(time_row[idx]):
                time = datetime.time(0)
            else:
                time = time_row[idx]
            dt = datetime.datetime.combine(date_row[idx], time)
            numeric_series[idx] = dt.timestamp()
        elif not pd.isnull(time_row[idx]):
            # We have a time, but not a date; value is hours
            numeric_series[idx] = time_row[idx].hour + time_row[idx].minute / 60 + time_row[idx].second / 3600
        else:
            # We have a string; try to convert it to a number
            numeric_series[idx] = _str_to_num(string_row[idx])

    return numeric_series.astype(float)


class DateFeatures:
    # 120 countries supported by the holidays package
    # Currently we ignore local holidays and only consider country-wide ones.
    country_codes = [
        'AL', 'AS', 'AD', 'AO', 'AR', 'AM', 'AW', 'AU', 'AT', 'AZ', 'BH', 'BD', 'BY', 'BE', 'BO', 'BA', 'BW', 'BR',
        'BG', 'BI', 'CA', 'CL', 'CN', 'CO', 'CR', 'HR', 'CU', 'CW', 'CY', 'CZ', 'DK', 'DJ', 'DO', 'EC', 'EG', 'EE',
        'SZ', 'ET', 'FI', 'FR', 'GE', 'DE', 'GR', 'GU', 'HN', 'HK', 'HU', 'IS', 'IN', 'ID', 'IE', 'IM', 'IL', 'IT',
        'JM', 'JP', 'KZ', 'KE', 'KG', 'LV', 'LS', 'LI', 'LT', 'LU', 'MG', 'MW', 'MY', 'MT', 'MX', 'MD', 'MC', 'ME',
        'MA', 'MZ', 'NA', 'NL', 'NZ', 'NI', 'NG', 'MP', 'MK', 'NO', 'PK', 'PA', 'PY', 'PE', 'PH', 'PL', 'PT', 'PR',
        'RO', 'RU', 'SM', 'SA', 'RS', 'SG', 'SK', 'SI', 'ZA', 'KR', 'ES', 'SE', 'CH', 'TW', 'TH', 'TN', 'TR', 'UA',
        'AE', 'GB', 'UM', 'US', 'UY', 'UZ', 'VA', 'VE', 'VN', 'VI', 'ZM', 'ZW'
    ]
    dictionaries = [holidays.country_holidays(x) for x in country_codes]

    def __init__(self, date):
        not_a_date = (
            date is None or \
            (not isinstance(date, datetime.datetime) and not isinstance(date, datetime.date)) \
            or pd.isnull(date)
        )
        if not_a_date:
            self.year = self.month = self.day = self.weekday = 0
            self.holidays = [0] * len(self.country_codes)
        else:
            assert isinstance(date, datetime.date) or isinstance(date, datetime.datetime)
            # Days and months are numbered from 1 to 31 and from 1 to 12
            self.month = date.month
            self.day = date.day
            # Year is just a number, we agree to use it as-is
            self.year = date.year
            # date.weekday() is from 0 to 6; we already have NaT encoded above,
            # no need to specify it again here
            self.weekday = date.weekday()
            try:
                self.holidays = [int(date in dictionary) for dictionary in self.dictionaries]
            except:
                # Most countries are only implemented between e.g. 1949 and 2099
                # Outside this range, just set them as non-holiday.
                # Also sometimes this is trying to sum None to datetime.timedelta?
                # And sometimes tries to access an attribute of None?
                # Quite a buggy package...
                self.holidays = [0] * len(self.country_codes)

    @classmethod
    def from_year_month_day(cls, year, month, day):
        try:
            date = datetime.date(year, month, day)
        except (TypeError, ValueError):
            date = None

        return cls(date)

    def to_tuple(self):
        # Year is 0 for NaT, between 1 and 50 otherwise.
        # (clipped to 1 until 2001, and to 50 after 2050)
        if self.year:
            year = max(min(self.year, 2050), 2001) - 2000
        else:
            year = 0

        # For day, month, and weekday we don't change the original value, which uses
        # 0 for NaT and 1-12, 1-31, or 1-7 for the rest.

        # For holidays we leave the boolean vector of length 120 as is.
        return year, self.month, self.day, self.weekday, self.holidays


# Used only in encode_numbers here below
def encode_numbers_no_binning(number_row: Union[pd.Series, np.ndarray, List[float]]):
    """
    Input: pandas series or numpy array or list of length n
    Output:
        - is_positive: numpy array of shape (n,) and dtype bool
        - exponent: numpy array of shape (n,) and dtype int, values in [0, 255] (255 being reserved for NaN)
        - fraction: numpy array of shape (n,) and dtype float, values in [1, 2) range
    """
    # Number embedding:
    # Create encodings based on scientific notation y = ±m * 2**n:
    # - 1 bit for sign ±
    # - integer exponent n between -127 and 127; plus an additional one (128) for NaN
    #   (we approximate 0 = 2**-127 and inf = 2**127)
    #   Actual values are shifted in the 0-255 range.
    # - the fraction m in [1, 2] is left unchanged
    # number_row is a pandas Series with all entries of type float64, but we only keep range of float32
    # (the rest will be approximated as ±inf)
    # We switch from pandas to numpy array for speed
    row_values = np.asarray(number_row, dtype=np.float64)
    is_positive = row_values >= 0  # bools
    row_values = np.clip(np.abs(row_values), np.exp2(-HALF_EXPONENT_BITS), np.exp2(HALF_EXPONENT_BITS))
    exponent = np.floor(np.log(row_values) / log(BASE))
    approx = BASE**exponent
    exponent = np.where(np.isnan(exponent), HALF_EXPONENT_BITS + 1, exponent).astype(int) + HALF_EXPONENT_BITS
    # exponent are ints between 0 and 255: 0 stands for 0, 254 for ±inf, 255 for NaN.
    fraction = np.clip(row_values / approx, 1, BASE)  # floats between 1 and 2, or NaN
    return is_positive, exponent, fraction


# Used for input fields inside row_tokenizer
def encode_numbers(number_row: Union[pd.Series, np.ndarray, List[float]]):
    """
    Input: pandas series or numpy array or list of length n
    Output:
        - is_positive: numpy array of shape (n,) and dtype bool
        - exponent: numpy array of shape (n,) and dtype int, values in [0, 255] (255 being reserved for NaN)
        - fraction_bin: numpy array of shape (n,) and dtype int, values [0, 999] range
        - delta: numpy array of shape (n,) and dtype float, values in [0, 1) range
    """
    # Number embedding:
    # Create encodings based on scientific notation y = ±m * 2**n:
    # - 1 bit for sign ±
    # - integer exponent n between -127 and 127; plus an additional one (128) for NaN
    #   (we approximate 0 = 2**-127 and inf = 2**127)
    #   Actual values are shifted in the 0-255 range.
    # - 1000 bins (maybe fewer will be enough?) for the fraction m in [1, 2], i.e. an integer in the 0-999 range.
    #   E.g. if the fraction is 1.53124 , then the bin index will be 531.
    # - one float in the [0, 1] range, with the missing part to the next bin.
    #   In the 1.53124 example above, this would be 0.24.
    # number_row is a pandas Series with all entries of type float64, but we only keep range of float32
    # (the rest will be approximated as ±inf)
    # We switch from pandas to numpy array for speed
    is_positive, exponent, fraction = encode_numbers_no_binning(number_row)

    fraction -= 1.0  # floats between 0 and 1, or NaN

    fraction_bin = np.where(np.isnan(fraction), 0, np.floor(fraction * FRACTION_BINS)).astype(int)
    fraction_bin = np.clip(fraction_bin, 0, FRACTION_BINS - 1)
    # fraction_bin is int between 0 and 999; the clip should do nothing. For NaN, this is 0.
    delta = fraction * FRACTION_BINS - fraction_bin
    delta = np.where(np.isnan(delta), 0.0, delta)
    # Hence the number is encoded as four values:
    # 1) is_positive (bool)
    # 2) exponent (int between 0 and 255)
    # 3) fraction_bin (int between 0 and 999)
    # 4) delta (float between 0 and 1)
    # To get back the original number, one should do:
    # value = (-1)**is_positive * 2**(exponent - 127) * (1 + (fraction_bin + delta) / 1000)
    # NaN is encoded as exponent = 255, everything else doesn't matter.
    # 0 is encoded as exponent = 0, the rest is also 0 but doesn't matter
    # ±inf is encoded as exponent = 254, the rest is 0 but doesn't matter
    return is_positive, exponent, fraction_bin, delta.astype(np.float32)


# Used both in encode_numbers_torch here below and to compute mixed loss
def encode_numbers_torch_no_binning(number_row: torch.Tensor):
    """
    This is currently logically different from encode_numbers_no_binning: it changes the fraction
    to go from the natural [1, 2) range to the artificial [0, 1], by:
        - subtracting 1 if exponent is even
        - mapping x -> 2 - x if exponent is odd
    Which makes changes from one exponent to the next smooth for fraction.
    Input: torch tensor of shape (n,)
    Output:
        - is_positive: torch tensor of shape (n,) and dtype bool
        - exponent: torch tensor of shape (n,) and dtype int, values in [0, 255] (255 being reserved for NaN)
        - fraction01: torch tensor of shape (n,) and dtype float, values in [0, 1] range
    """
    # Copy of the above method, but using torch only. Should run on GPU.
    row_values = number_row.double()
    is_positive = row_values >= 0  # bools
    row_values = torch.abs(row_values)
    exponent_unbounded = torch.floor(torch.log(row_values) / log(BASE))
    exponent = torch.clip(exponent_unbounded, -HALF_EXPONENT_BITS, HALF_EXPONENT_BITS)
    approx = BASE**exponent
    exponent = torch.where(torch.isnan(exponent), HALF_EXPONENT_BITS + 1, exponent).long() + HALF_EXPONENT_BITS
    # exponent are ints between 0 and 255: 0 stands for 0, 254 for ±inf, 255 for NaN.
    fraction = torch.clip(row_values / approx, 1, BASE)  # floats between 1 and 2, or NaN

    # Fraction above is of shape (n,) in the [1, 2] range, it is the "real" fraction
    # fraction_logits is of shape (n, 1) with values in R
    # Need to squeeze the latter and transform the former to [0, 1] (depending on the exponent),
    # then apply cross entropy loss
    # For transforming [1, 2] to [0, 1], we do:
    # x -> x - 1 where exponent is even
    # x -> 2 - x where exponent is odd.
    # This is summarized and vectorized to x -> (-1)^exponent * (x - 1.5) + 0.5
    fraction01 = (-1)**exponent * (fraction - 1.5) + 0.5

    return is_positive, exponent, fraction01


# Used to compute cross entropy loss
def encode_numbers_torch(number_row: torch.Tensor):
    """
    Inheriting from encode_numbers_torch_no_binning, this is different from encode_numbers:
    - for even exponent, fraction_bin = 0 and delta = 0 corresponds to fraction = 1.0,
        while fraction_bin = 999 and delta = 1 to fraction = 2.0
    - for odd exponent, fraction_bin = 0 and delta = 0 corresponds to fraction = 2.0,
        while fraction_bin = 999 and delta = 1 to fraction = 1.0
    Input: torch tensor of shape (n,)
    Output:
        - is_positive: torch tensor of shape (n,) and dtype bool
        - exponent: torch tensor of shape (n,) and dtype int, values in [0, 255] (255 being reserved for NaN)
        - fraction_bin: torch tensor of shape (n,) and dtype int, values [0, 999] range
        - delta: torch tensor of shape (n,) and dtype float, values in [0, 1) range
    """
    # Copy of the above method, but using torch only. Should run on GPU.

    is_positive, exponent, fraction01 = encode_numbers_torch_no_binning(number_row)

    fraction_bin = torch.where(torch.isnan(fraction01), 0, torch.floor(fraction01 * FRACTION_BINS)).long()
    fraction_bin = torch.clip(fraction_bin, 0, FRACTION_BINS - 1)
    # fraction_bin is int between 0 and 999; the clip should do nothing. For NaN, this is 0.
    delta = fraction01 * FRACTION_BINS - fraction_bin
    delta = torch.where(torch.isnan(delta), 0.0, delta)
    # Hence the number is encoded as four values:
    # 1) is_positive (bool)
    # 2) exponent (int between 0 and 255)
    # 3) fraction_bin (int between 0 and 999)
    # 4) delta (float between 0 and 1)
    # To get back the original number, one should do:
    # value = (-1)**is_positive * 2**(exponent - 127) * (1 + (fraction_bin + delta) / 1000)
    # NaN is encoded as exponent = 255, everything else doesn't matter.
    # 0 is encoded as exponent = 0, the rest is also 0 but doesn't matter
    # ±inf is encoded as exponent = 254, the rest is 0 but doesn't matter
    return is_positive, exponent, fraction_bin, delta.float()


def decode_numbers_no_binning(is_positive, exponent, mantissa):
    return (-1)**(1 - is_positive) * BASE**(exponent - HALF_EXPONENT_BITS) * mantissa


def decode_numbers(is_positive, exponent, fraction_bin, delta=0.5):
    """
    Applies the inverse transformation as that in encode_numbers
    TODO: if we change encode_numbers to make it smooth for the fraction part, we also need to change this.
    is_positive: bool, treated as 0 or 1 here
    exponent: int between 0 and 255, then remapped to [-127, 127]
    fraction_bin: int between 0 and 999 (or more generally 0 and FRACTION_BINS - 1)
        then remapped to [0, 1) and finally to [1, 2) (or more generally [1, BASE))
    """
    part_of_mantissa = (fraction_bin + delta) / FRACTION_BINS
    if BASE == 2:
        # Typical case, faster, let's avoid always multiplying by 1...
        mantissa = 1 + part_of_mantissa
    else:
        mantissa = 1 + part_of_mantissa * (BASE - 1)
    return decode_numbers_no_binning(is_positive, exponent, mantissa)


def decode_numbers_torch_no_binning(is_positive, exponent, fraction01):
    """
    Applies the inverse transformation as that in encode_numbers_torch_no_binning
    To be used:
    - here below
    - in MultiHeadedModel extract_predictions for mixed regression
    Input:
        - is_positive: bool tensor of shape (n,)
        - exponent: int tensor of shape (n,), values in [0, 255] (255 being reserved for NaN)
        - fraction01: float tensor of shape (n,), values in [0, 1] range
    The fraction is first remapped to [1, 2] with a formula that depends on whether exponent is even or odd,
    similarly to how it is done in encode_numbers_torch_no_binning
    """
    # Linear transformation:
    # - x + 1 if exponent is even
    # - 2 - x if exponent is odd
    # This is summarized and vectorized as 1.5 + (-1)^exponent * (x - 0.5)
    fraction = 1.5 + (-1)**exponent * (fraction01 - 0.5)

    return (-1)**(1 - is_positive.float()) * BASE**(exponent - HALF_EXPONENT_BITS) * fraction


def decode_numbers_torch(is_positive, exponent, fraction_bin, delta=0.5):
    """
    Applies the inverse transformation as that in encode_numbers_torch
    To be used:
    - in MultiHeadedModel extract_predictions for cross entropy regression
    """
    assert BASE == 2, 'Only base 2 is supported for now'
    fraction01 = (fraction_bin + delta) / FRACTION_BINS
    return decode_numbers_torch_no_binning(is_positive, exponent, fraction01)


def encode_dates(raw_row: pd.Series):
    # Dates embedding:
    date_year = []
    date_month = []
    date_day = []
    date_weekday = []
    date_holidays = []
    for column in raw_row.index:
        date = raw_row[column]

        if (isinstance(date, datetime.datetime)
                or isinstance(date, datetime.date)) and (date.year, date.month, date.day) in date_cache:
            year, month, day, weekday, holidays = date_cache[(date.year, date.month, date.day)]
        else:
            year, month, day, weekday, holidays = DateFeatures(date).to_tuple()

        date_year.append(year)
        date_month.append(month)
        date_day.append(day)
        date_weekday.append(weekday)
        date_holidays.append(holidays)
    # dates_row should now be an array-like of shape (self.df.shape[1], 220);
    # 220 comes from:
    # - 50 bits for the year ("1-hot" - setting all previous one to 1 - between 2001 and 2050)
    # - 12 bits for one-hot encoded month + NaT special value
    # - 31 bits for one-hot encoded day of month
    # - 7 bits for one-hot encoded weekday
    # - 120 bits for is-a-holiday in each country supported by holidays package
    return date_year, date_month, date_day, date_weekday, date_holidays
