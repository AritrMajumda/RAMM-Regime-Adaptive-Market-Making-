# ===============================================================================
# RAMM PIPELINE — GOOGLE COLAB OOM-SAFE PREPROCESSING (USD-M FUTURES)
# Downloads Binance Vision L1 tick archives & exports [N, 7] PyTorch Tensors
# ===============================================================================

import os
import gc
import json
import urllib.request
from urllib.error import HTTPError
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------------------------------------
# 0. COLAB HARDWARE SETUP & GOOGLE DRIVE MOUNT
# -------------------------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")

# Mount Google Drive (Required for Colab)
try:
    from google.colab import drive
    drive.mount('/content/gdrive', force_remount=True)
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    print("⚠️ Not running in Colab. Skipping Drive mount.")

# Define Paths
if IN_COLAB:
    GDRIVE_OUTPUT = '/content/gdrive/My Drive/data for hft model 2023 tick/2023_processed_data_sol'
    LOCAL_TEMP = '/tmp/ramm_processing'
else:
    GDRIVE_OUTPUT = '/workspace/data' # Fallback for local/RunPod
    LOCAL_TEMP = '/tmp/ramm_processing'

os.makedirs(GDRIVE_OUTPUT, exist_ok=True)
os.makedirs(LOCAL_TEMP, exist_ok=True)

print(f"✅ Permanent Storage (Drive) : {GDRIVE_OUTPUT}")
print(f"✅ Ephemeral Storage (Local) : {LOCAL_TEMP}")

# -------------------------------------------------------------------------------
# 1. BINANCE VISION ARCHIVE DOWNLOAD & ASYNCHRONOUS MERGE
# -------------------------------------------------------------------------------

def download_binance_archive(symbol: str, date_str: str, data_type: str) -> str:
    """Downloads raw zip archive directly from Binance Vision FUTURES archives."""
    base_url = "https://data.binance.vision/data/futures/um/daily"
    url = f"{base_url}/{data_type}/{symbol}/{symbol}-{data_type}-{date_str}.zip"
    temp_zip = os.path.join(LOCAL_TEMP, f"{symbol}_{data_type}_{date_str}.zip")

    try:
        urllib.request.urlretrieve(url, temp_zip)
        return temp_zip
    except HTTPError as e:
        if e.code == 404:
            return None  # Data unavailable/not published yet
        print(f"      ⚠️ Archive download error ({url}): {e}")
        return None

def load_and_merge_tick_data(date_str: str, symbol: str = 'ETHUSDT') -> pd.DataFrame:
    """
    Downloads aggTrades and bookTicker for the given day, handles unpredictable
    Binance CSV headers, and performs an asynchronous backward merge.
    """
    agg_zip = download_binance_archive(symbol, date_str, 'aggTrades')
    book_zip = download_binance_archive(symbol, date_str, 'bookTicker')

    if not agg_zip or not book_zip:
        return None

    try:
        # 1. Load Trades (Handle unpredictable Binance headers natively)
        trades = pd.read_csv(
            agg_zip, header=None, usecols=[1, 2, 5, 6],
            names=['price', 'qty', 'transact_time', 'is_buyer_maker']
        )
        # Force numeric, pushing any string headers to NaN, then drop them
        trades['transact_time'] = pd.to_numeric(trades['transact_time'], errors='coerce')
        trades = trades.dropna(subset=['transact_time']).copy()
        trades['price'] = pd.to_numeric(trades['price'])
        trades['qty'] = pd.to_numeric(trades['qty'])
        trades['is_buyer_maker'] = trades['is_buyer_maker'].astype(str).str.lower().map(
            {'true': True, 'false': False, '1': True, '0': False}
        )

        # 2. Load BookTicker
        book = pd.read_csv(
            book_zip, header=None, usecols=[1, 3, 5],
            names=['best_bid', 'best_ask', 'transaction_time']
        )
        book['transaction_time'] = pd.to_numeric(book['transaction_time'], errors='coerce')
        book = book.dropna(subset=['transaction_time']).copy()
        book['best_bid'] = pd.to_numeric(book['best_bid'])
        book['best_ask'] = pd.to_numeric(book['best_ask'])

        # 3. Sort & Merge
        trades = trades.sort_values('transact_time').reset_index(drop=True)
        book = book.sort_values('transaction_time').reset_index(drop=True)

        # Asynchronous backward merge: Pair trade execution with exact prior L1 quotes
        df = pd.merge_asof(
            trades, book,
            left_on='transact_time', right_on='transaction_time',
            direction='backward'
        )

        return df.dropna().reset_index(drop=True)

    finally:
        # COLAB DISK CLEANUP: Clean raw zips immediately to preserve 78GB disk limit
        if agg_zip and os.path.exists(agg_zip): os.remove(agg_zip)
        if book_zip and os.path.exists(book_zip): os.remove(book_zip)

# -------------------------------------------------------------------------------
# 2. OOM-SAFE GPU ROLLING VECTORIZATION
# -------------------------------------------------------------------------------

def _rolling_sum_gpu(tensor: torch.Tensor, window: int) -> torch.Tensor:
    """OOM-Safe rolling sum using Cumulative Sums."""
    cumsum = torch.cumsum(tensor, dim=0)
    res = torch.zeros_like(tensor)
    res[window:] = cumsum[window:] - cumsum[:-window]
    res[:window] = cumsum[:window]
    return res

def _rolling_std_gpu(tensor: torch.Tensor, window: int) -> torch.Tensor:
    """OOM-Safe rolling standard deviation: Var(X) = E[X^2] - (E[X])^2."""
    window = min(window, len(tensor))

    cumsum_x = torch.cumsum(tensor, dim=0)
    sum_x = torch.zeros_like(tensor)
    sum_x[window:] = cumsum_x[window:] - cumsum_x[:-window]
    sum_x[:window] = cumsum_x[:window]

    counts = torch.arange(1, len(tensor) + 1, device=tensor.device, dtype=torch.float32)
    counts[window:] = float(window)
    mean_x = sum_x / counts

    cumsum_x2 = torch.cumsum(tensor ** 2, dim=0)
    sum_x2 = torch.zeros_like(tensor)
    sum_x2[window:] = cumsum_x2[window:] - cumsum_x2[:-window]
    sum_x2[:window] = cumsum_x2[:window]
    mean_x2 = sum_x2 / counts

    variance = (mean_x2 - mean_x ** 2).clamp(min=1e-8)
    return torch.sqrt(variance)

def compute_7col_features(df: pd.DataFrame, ofi_window=1000, vol_window=100):
    """Computes the exact 7 features matching notebooks 1, 2, and 3."""
    direction_np = np.where(df['is_buyer_maker'].values, -1.0, 1.0)

    price_t     = torch.tensor(df['price'].values, dtype=torch.float32, device=device)
    volume_t    = torch.tensor(df['qty'].values, dtype=torch.float32, device=device)
    direction_t = torch.tensor(direction_np, dtype=torch.float32, device=device)
    best_bid_t  = torch.tensor(df['best_bid'].values, dtype=torch.float32, device=device)
    best_ask_t  = torch.tensor(df['best_ask'].values, dtype=torch.float32, device=device)

    log_price_t = torch.log(price_t.clamp(min=1e-6))
    log_price_mean = log_price_t.mean().item()
    log_price_std = log_price_t.std().item()
    price_norm = (log_price_t - log_price_mean) / (log_price_std + 1e-8)

    volume_norm = volume_t / (volume_t.mean() + 1e-8)

    midpoint_t = (best_ask_t + best_bid_t) / 2.0
    spread_bps = ((best_ask_t - best_bid_t) / midpoint_t * 10000.0).clamp(0, 100)

    signed_volume = volume_t * direction_t
    ofi_t = _rolling_sum_gpu(signed_volume, window=ofi_window)
    ofi_norm = ofi_t / (ofi_t.std() + 1e-8)

    direction_feature = direction_t

    shifted_price = F.pad(price_t[:-1], (1, 0), value=price_t[0].item())
    price_change_t = torch.log((price_t / shifted_price).clamp(min=1e-6)).nan_to_num(0)
    volatility_t = _rolling_std_gpu(price_change_t, window=vol_window)

    features_gpu = torch.stack([
        price_norm, volume_norm, spread_bps, ofi_norm,
        direction_feature, volatility_t, price_change_t
    ], dim=1)

    norm_params = {
        "log_price_mean": float(log_price_mean),
        "log_price_std": float(log_price_std)
    }

    return features_gpu.cpu().numpy().astype(np.float32), norm_params

# -------------------------------------------------------------------------------
# 3. COLAB CHUNKING EXECUTION ENGINE
# -------------------------------------------------------------------------------

def run_colab_chunking_pipeline(symbol='SOLUSDT', start_date_str='2023-05-01', end_date_str='2023-12-31'):
    start_dt = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_dt   = datetime.strptime(end_date_str, '%Y-%m-%d')
    curr_dt  = start_dt

    asset_name = symbol.replace('USDT', '')
    master_norm_params = {}
    total_processed_ticks = 0
    all_means, all_stds = [], []

    print("=" * 80)
    print(f"🚀 STARTING COLAB PIPELINE FOR {asset_name} ({start_date_str} to {end_date_str})")
    print("=" * 80)

    pbar = tqdm(total=(end_dt - start_dt).days + 1, desc="Processing Days")

    while curr_dt <= end_dt:
        date_str = curr_dt.strftime('%Y-%m-%d')
        tensor_filename = f"tensor_OFI_Enhanced_{asset_name}_{date_str}.pt"
        gdrive_filepath = os.path.join(GDRIVE_OUTPUT, tensor_filename)

        # Resume Check: If Colab timed out, this skips days we already successfully uploaded
        if os.path.exists(gdrive_filepath):
            curr_dt += timedelta(days=1)
            pbar.update(1)
            continue

        try:
            df = load_and_merge_tick_data(date_str, symbol=symbol)
            if df is not None and len(df) > 0:
                features_np, norm_params = compute_7col_features(df)

                local_filepath = os.path.join(LOCAL_TEMP, tensor_filename)

                # 1. Save tensor locally
                torch.save(torch.tensor(features_np, dtype=torch.float32), local_filepath)

                # 2. Copy tensor to Google Drive
                os.system(f'cp "{local_filepath}" "{gdrive_filepath}"')

                # 3. COLAB DISK CLEANUP: Delete local tensor immediately
                if os.path.exists(local_filepath):
                    os.remove(local_filepath)

                all_means.append(norm_params["log_price_mean"])
                all_stds.append(norm_params["log_price_std"])
                total_processed_ticks += len(features_np)

                # COLAB RAM/VRAM CLEANUP
                del df, features_np
                gc.collect()
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n❌ Error processing {date_str}: {e}")

        curr_dt += timedelta(days=1)
        pbar.update(1)

    pbar.close()

    # Save the master normalization parameters needed by backtesting.ipynb
    overall_mean = float(np.mean(all_means)) if all_means else 11.30
    overall_std  = float(np.mean(all_stds)) if all_stds else 0.10

    master_norm_params[asset_name] = {
        "log_price_mean": overall_mean,
        "log_price_std": overall_std
    }

    json_path_out = os.path.join(GDRIVE_OUTPUT, "normalization_params1.json")
    with open(json_path_out, 'w') as f:
        json.dump(master_norm_params, f, indent=2)

    print("\n" + "=" * 80)
    print("✅ COLAB PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Total Ticks Processed : {total_processed_ticks:,}")
    print(f"Saved Tensors Path    : {GDRIVE_OUTPUT}/tensor_OFI_Enhanced_{asset_name}_*.pt")
    print(f"Saved Norm Params     : {json_path_out}")

if __name__ == '__main__':
    # Process 2024 BTCUSDT
    run_colab_chunking_pipeline(symbol='SOLUSDT', start_date_str='2023-05-01', end_date_str='2023-12-31')