import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os


# ─────────────────────────────────────────────
# API endpoints
# ─────────────────────────────────────────────
LIVE_URL    = "https://www.deribit.com/api/v2/public"
HISTORY_URL = "https://history.deribit.com/api/v2/public"


def _get(base_url, endpoint, params, retries=3, wait=2):
    """Robust GET with retry logic."""
    url = f"{base_url}/{endpoint}"
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            result = r.json()
            if "error" in result and result["error"]:
                raise ValueError(f"API error: {result['error']}")
            return result["result"]
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(wait)
            else:
                raise e


# ─────────────────────────────────────────────
# Parse instrument name
# ─────────────────────────────────────────────

def parse_instrument_name(name):
    """
    'BTC-27DEC24-50000-C' -> (currency, expiry, strike, option_type)
    """
    parts       = name.split("-")
    currency    = parts[0]
    expiry_str  = parts[1]
    strike      = float(parts[2])
    option_type = parts[3]
    expiry_date = datetime.strptime(expiry_str, "%d%b%y")
    return currency, expiry_date, strike, option_type


# ─────────────────────────────────────────────
# Core historical trade fetcher
# ─────────────────────────────────────────────

def fetch_trades_for_day(date, currency="BTC", count_per_page=1000):
    """
    Fetch all BTC option trades for a single day from history.deribit.com.
    Paginates through all pages to get every trade of the day.

    Parameters
    ----------
    date     : datetime   The day to fetch
    currency : str        "BTC" or "ETH"

    Returns
    -------
    df : pd.DataFrame   All trades with iv, instrument, strike, expiry etc.
    """
    start_ts = int(date.replace(hour=0,  minute=0,  second=0).timestamp() * 1000)
    end_ts   = int(date.replace(hour=23, minute=59, second=59).timestamp() * 1000)

    all_trades = []
    current_start = start_ts

    while True:
        try:
            result = _get(HISTORY_URL,
                          "get_last_trades_by_currency_and_time", {
                "currency":        currency,
                "kind":            "option",
                "start_timestamp": current_start,
                "end_timestamp":   end_ts,
                "count":           count_per_page,
                "sorting":         "asc",
            })

            trades = result.get("trades", [])
            if not trades:
                break

            all_trades.extend(trades)

            # If fewer than max returned, we have all trades
            if len(trades) < count_per_page:
                break

            # Move start to just after last trade timestamp
            current_start = trades[-1]["timestamp"] + 1

        except Exception as e:
            print(f"    Error fetching {date.strftime('%Y-%m-%d')}: {e}")
            break

    if not all_trades:
        return pd.DataFrame()

    df = pd.DataFrame(all_trades)
    df["date"] = date.strftime("%Y-%m-%d")

    # Parse instrument name
    parsed = df["instrument_name"].apply(
        lambda x: pd.Series(parse_instrument_name(x),
                             index=["currency", "expiry", "strike", "option_type"])
    )
    df = pd.concat([df, parsed], axis=1)

    # Time to maturity
    df["T"] = (df["expiry"] - pd.to_datetime(df["date"])) \
              .dt.total_seconds() / (365.25 * 24 * 3600)

    # Moneyness
    df["moneyness"] = df["strike"] / df["index_price"]

    # IV to decimal
    df["iv"] = df["iv"] / 100.0

    return df


# ─────────────────────────────────────────────
# Build full historical dataset
# ─────────────────────────────────────────────

def build_historical_dataset(
        currency="BTC",
        start_date="2024-01-01",
        end_date="2025-03-01",
        option_type="C",
        min_moneyness=0.7,
        max_moneyness=1.3,
        min_T_days=7,
        max_T_days=180,
        min_iv=0.01,
        max_iv=5.0,
        save_path=None,
        day_sleep=0.5):
    """
    Build a historical BTC/ETH options dataset by fetching all trades
    day by day from history.deribit.com.

    Each row is one actual market trade with:
        date, instrument_name, currency, expiry, strike, option_type,
        iv (implied vol), price, index_price, mark_price,
        amount, T (time to maturity), moneyness

    Parameters
    ----------
    currency      : str   "BTC" or "ETH"
    start_date    : str   "YYYY-MM-DD"
    end_date      : str   "YYYY-MM-DD"
    option_type   : str   "C", "P", or "both"
    min_moneyness : float
    max_moneyness : float
    min_T_days    : int
    max_T_days    : int
    min_iv        : float minimum IV in decimal (filter bad quotes)
    max_iv        : float maximum IV in decimal (filter bad quotes)
    save_path     : str   CSV output path
    day_sleep     : float seconds to wait between days

    Returns
    -------
    df : pd.DataFrame
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt   = datetime.strptime(end_date,   "%Y-%m-%d")

    print(f"\nBuilding {currency} options dataset")
    print(f"Period: {start_date} to {end_date}")
    print(f"Filters: {option_type} options, "
          f"moneyness [{min_moneyness}, {max_moneyness}], "
          f"T [{min_T_days}, {max_T_days}] days\n")

    all_days = []
    current  = start_dt
    n_days   = (end_dt - start_dt).days

    while current <= end_dt:
        date_str = current.strftime("%Y-%m-%d")

        df_day = fetch_trades_for_day(current, currency)

        if not df_day.empty:
            # Apply filters
            mask = pd.Series([True] * len(df_day))

            if option_type in ["C", "P"]:
                mask &= df_day["option_type"] == option_type

            mask &= (df_day["moneyness"] >= min_moneyness) & \
                    (df_day["moneyness"] <= max_moneyness)
            mask &= (df_day["T"] >= min_T_days / 365.25) & \
                    (df_day["T"] <= max_T_days / 365.25)
            mask &= (df_day["iv"] >= min_iv) & \
                    (df_day["iv"] <= max_iv)
            mask &= df_day["iv"].notna()

            df_filtered = df_day[mask]

            if not df_filtered.empty:
                all_days.append(df_filtered)
                days_done = (current - start_dt).days + 1
                print(f"  {date_str}: {len(df_day):5d} raw trades "
                      f"-> {len(df_filtered):4d} after filter "
                      f"  [{days_done}/{n_days} days]")
            else:
                print(f"  {date_str}: {len(df_day):5d} raw trades "
                      f"->    0 after filter")
        else:
            print(f"  {date_str}: no trades")

        current  += timedelta(days=1)
        time.sleep(day_sleep)

    if not all_days:
        print("No data retrieved.")
        return pd.DataFrame()

    df_all = pd.concat(all_days, ignore_index=True)

    # Keep useful columns
    cols = ["date", "instrument_name", "currency", "expiry", "strike",
            "option_type", "T", "iv", "price", "mark_price",
            "index_price", "amount", "moneyness"]
    df_all = df_all[[c for c in cols if c in df_all.columns]]
    df_all = df_all.sort_values(["date", "expiry", "strike"]) \
                   .reset_index(drop=True)

    print(f"\n{'='*50}")
    print(f"Final dataset: {len(df_all):,} trade observations")
    print(f"Date range:    {df_all['date'].min()} to {df_all['date'].max()}")
    print(f"Unique dates:  {df_all['date'].nunique()}")
    print(f"Unique expiries: {df_all['expiry'].nunique()}")
    print(f"Strike range:  {df_all['strike'].min():,.0f} "
          f"to {df_all['strike'].max():,.0f}")
    print(f"IV range:      {df_all['iv'].min():.2%} "
          f"to {df_all['iv'].max():.2%}")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        df_all.to_csv(save_path, index=False)
        print(f"\nSaved to {save_path}")

    return df_all


# ─────────────────────────────────────────────
# Aggregate trades to daily smile
# (one IV per strike per day, using median)
# ─────────────────────────────────────────────

def aggregate_to_daily_smile(df, min_trades=2):
    """
    Aggregate trade-level data to one IV observation per
    (date, expiry, strike) using the volume-weighted median IV.
    This is what we use for calibration.

    Parameters
    ----------
    df         : pd.DataFrame   Output of build_historical_dataset
    min_trades : int            Minimum trades required per (date, expiry, strike)

    Returns
    -------
    df_smile : pd.DataFrame
    """
    grouped = df.groupby(["date", "expiry", "strike", "option_type"])

    df_smile = grouped.agg(
        iv_median    = ("iv",          "median"),
        iv_mean      = ("iv",          "mean"),
        iv_vwap      = ("iv",          lambda x:
                         np.average(x, weights=df.loc[x.index, "amount"])),
        n_trades     = ("iv",          "count"),
        total_volume = ("amount",      "sum"),
        index_price  = ("index_price", "last"),
        T            = ("T",           "last"),
        moneyness    = ("moneyness",   "last"),
    ).reset_index()

    # Filter minimum trades
    df_smile = df_smile[df_smile["n_trades"] >= min_trades]

    # Use VWAP as primary IV estimate
    df_smile = df_smile.rename(columns={"iv_vwap": "mark_iv"})
    df_smile["currency"] = df_smile["strike"].apply(
        lambda x: "BTC"
    )

    df_smile = df_smile.sort_values(["date", "expiry", "strike"]) \
                       .reset_index(drop=True)

    print(f"Daily smile dataset: {len(df_smile):,} observations")
    print(f"Unique (date, expiry) pairs: "
          f"{df_smile.groupby(['date','expiry']).ngroups}")

    return df_smile


# ─────────────────────────────────────────────
# Live snapshot (for current surface)
# ─────────────────────────────────────────────

def fetch_live_surface(currency="BTC", min_open_interest=0,
                       min_moneyness=0.7, max_moneyness=1.3,
                       min_T_days=7, max_T_days=180):
    """
    Fetch the current full IV surface using get_book_summary_by_currency.
    Returns one row per instrument with mark_iv already included.
    Much faster than fetching ticker by ticker.
    """
    print(f"Fetching live {currency} option surface...")
    result = _get(LIVE_URL, "get_book_summary_by_currency", {
        "currency": currency,
        "kind":     "option",
    })

    now     = datetime.utcnow()
    records = []

    for ins in result:
        try:
            name = ins["instrument_name"]
            _, expiry, strike, opt_type = parse_instrument_name(name)
            T = max((expiry - now).total_seconds() /
                    (365.25 * 24 * 3600), 1e-6)
            underlying = ins.get("underlying_price", np.nan)
            moneyness  = strike / underlying if underlying else np.nan

            records.append({
                "date":             now.strftime("%Y-%m-%d"),
                "instrument_name":  name,
                "currency":         currency,
                "expiry":           expiry,
                "strike":           strike,
                "option_type":      opt_type,
                "T":                T,
                "mark_iv":          ins.get("mark_iv", np.nan) / 100.0,
                "mark_price":       ins.get("mark_price", np.nan),
                "index_price":      underlying,
                "open_interest":    ins.get("open_interest", 0),
                "volume":           ins.get("volume", 0),
                "bid_price":        ins.get("bid_price", np.nan),
                "ask_price":        ins.get("ask_price", np.nan),
                "moneyness":        moneyness,
            })
        except Exception:
            continue

    df = pd.DataFrame(records)

    # Filters
    df = df[df["option_type"] == "C"]
    df = df[(df["moneyness"] >= min_moneyness) &
            (df["moneyness"] <= max_moneyness)]
    df = df[(df["T"] >= min_T_days / 365.25) &
            (df["T"] <= max_T_days / 365.25)]
    df = df[df["mark_iv"].notna() & (df["mark_iv"] > 0)]
    df = df[df["open_interest"] >= min_open_interest]

    print(f"Live surface: {len(df)} options after filtering")
    return df.sort_values(["expiry", "strike"]).reset_index(drop=True)


# ─────────────────────────────────────────────
# Calibration helpers
# ─────────────────────────────────────────────

def get_smile_slice(df, date, expiry, currency="BTC",
                    iv_col="mark_iv"):
    """
    Extract a single (date, maturity) smile for calibration.

    Returns
    -------
    strikes     : np.array
    market_vols : np.array
    F           : float   Forward / index price
    T           : float   Time to maturity
    """
    if "date" in df.columns:
        mask = ((df["date"] == date) &
                (pd.to_datetime(df["expiry"]) == pd.to_datetime(expiry)))
    else:
        mask = (pd.to_datetime(df["expiry"]) == pd.to_datetime(expiry))

    slice_df = df[mask].sort_values("strike")

    if len(slice_df) < 3:
        raise ValueError(
            f"Not enough options for date={date}, expiry={expiry} "
            f"(found {len(slice_df)})"
        )

    strikes     = slice_df["strike"].values.astype(float)
    market_vols = slice_df[iv_col].values.astype(float)
    F           = slice_df["index_price"].iloc[0]
    T           = slice_df["T"].iloc[0]

    return strikes, market_vols, F, T


def list_available_smiles(df, min_strikes=5):
    """
    List all (date, expiry) pairs available for calibration,
    with enough strikes.
    """
    grouped = df.groupby(["date", "expiry"])["strike"].count()
    available = grouped[grouped >= min_strikes].reset_index()
    available.columns = ["date", "expiry", "n_strikes"]
    return available.sort_values(["date", "expiry"])