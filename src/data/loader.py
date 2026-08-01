#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据加载器 — tushare中转站 + 本地缓存"""
from __future__ import annotations

import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"
_TS_PRO = None


def _init_tushare():
    global _TS_PRO
    if _TS_PRO is not None:
        return _TS_PRO
    try:
        import tushare as ts
        token = os.environ.get("TUSHARE_TOKEN", "")
        if not token:
            return None
        ts.set_token(token)
        _TS_PRO = ts.pro_api()
        server = os.environ.get("TUSHARE_API_SERVER", "https://fastapic.stockai888.top")
        _TS_PRO._DataApi__http_url = server
    except Exception:
        _TS_PRO = None
    return _TS_PRO


def _to_ts_code(code: str) -> str:
    c = str(code)
    if c.startswith(("60", "68", "90")):
        return c + ".SH"
    return c + ".SZ"


def load_daily(code: str, start_date: str = "", end_date: str = "", use_cache: bool = True) -> pd.DataFrame:
    """
    加载个股日K数据
    
    返回: trade_date, open, close, high, low, pct_chg, vol, amount
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _CACHE_DIR / f"daily_{code}.csv"
    
    # 默认拉2年
    if not end_date:
        end_date = datetime.now().strftime("%Y%m%d")
    if not start_date:
        start_date = (datetime.now() - timedelta(days=730)).strftime("%Y%m%d")
    
    # 尝试缓存
    if use_cache and cache_file.exists():
        df = pd.read_csv(cache_file, dtype={'trade_date': str})
        df = df[(df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)]
        if len(df) > 30:
            return df.sort_values('trade_date').reset_index(drop=True)
    
    # tushare拉取
    pro = _init_tushare()
    if pro is None:
        raise RuntimeError("TUSHARE_TOKEN 未配置")
    
    ts_code = _to_ts_code(code)
    raw = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    
    if raw is None or len(raw) == 0:
        raise RuntimeError(f"拉取 {code} 日K失败")
    
    df = raw.rename(columns={
        'trade_date': 'trade_date', 'open': 'open', 'close': 'close',
        'high': 'high', 'low': 'low', 'pct_chg': 'pct_chg',
        'vol': 'vol', 'amount': 'amount',
    })
    df['trade_date'] = df['trade_date'].astype(str)
    df = df.sort_values('trade_date').reset_index(drop=True)
    
    # 写缓存
    df.to_csv(cache_file, index=False)
    
    return df


def load_index(code: str = "000300", days: int = 365) -> pd.DataFrame:
    """加载指数日K（如沪深300）"""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _CACHE_DIR / f"index_{code}.csv"
    
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    
    if cache_file.exists():
        df = pd.read_csv(cache_file, dtype={'trade_date': str})
        df = df[(df['trade_date'] >= start_date)]
        if len(df) > 30:
            return df.sort_values('trade_date').reset_index(drop=True)
    
    pro = _init_tushare()
    if pro is None:
        return pd.DataFrame()
    
    ts_code = code + ".SH" if code.startswith("000") else code + ".SZ"
    raw = pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    
    df = raw.rename(columns={
        'trade_date': 'trade_date', 'pct_chg': 'pct_chg',
        'close': 'close', 'open': 'open',
    })
    df['trade_date'] = df['trade_date'].astype(str)
    df = df.sort_values('trade_date').reset_index(drop=True)
    df.to_csv(cache_file, index=False)
    
    return df
