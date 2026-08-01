#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""连板概率统计器

功能：
- 用tushare拉全市场daily数据（按交易日批量），识别涨停股（pct_chg >= 9.8%）
- 计算连板晋级率：1板→2板、2板→3板的概率（近5日滚动）
- 计算涨停股平均次日收益
- 输出LimitUpStats数据类：日期、涨停数、连板晋级率、平均次日收益
- 数据缓存到 data/cache/limit_up.csv
"""
from __future__ import annotations

import os
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.loader import _init_tushare

_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"
LIMIT_UP_THRESHOLD = 9.8  # 涨停判定阈值 pct_chg >= 9.8% （兼容10%/20%涨跌幅）


@dataclass
class LimitUpStats:
    """连板统计结果（单日）"""
    trade_date: str
    limit_up_count: int = 0           # 当日涨停股数量
    first_board_count: int = 0        # 首板数量
    second_board_count: int = 0       # 2板数量
    third_board_count: int = 0        # 3板+数量
    promotion_1to2: float = 0.0       # 1板→2板晋级率（近5日滚动）
    promotion_2to3: float = 0.0       # 2板→3板晋级率（近5日滚动）
    avg_next_day_return: float = 0.0  # 涨停股平均次日收益（%）
    total_stocks: int = 0             # 全市场股票数


def _get_trade_dates(pro, start_date: str, end_date: str) -> List[str]:
    """获取交易日列表"""
    cal = pro.trade_cal(exchange="SSE", start_date=start_date, end_date=end_date, is_open="1")
    if cal is None or len(cal) == 0:
        # fallback: 用工作日
        dates = pd.bdate_range(start_date, end_date)
        return [d.strftime("%Y%m%d") for d in dates]
    return sorted(cal["cal_date"].tolist())


def _fetch_daily_by_date(pro, trade_date: str, retry: int = 3) -> pd.DataFrame:
    """按交易日拉全市场daily数据"""
    for attempt in range(retry):
        try:
            df = pro.daily(trade_date=trade_date)
            if df is not None and len(df) > 0:
                return df
        except Exception as e:
            if attempt == retry - 1:
                print(f"  [WARN] 拉取 {trade_date} 失败(第{attempt+1}次): {e}")
            import time
            time.sleep(1)
    return pd.DataFrame()


def _calc_consecutive_boards(grouped: dict, trade_date: str) -> pd.DataFrame:
    """
    给定某日的涨停股，结合前序交易日数据计算连板数。

    grouped: {trade_date: DataFrame(ts_code, pct_chg, close, ...)} 按日期排序
    trade_date: 当前交易日
    """
    # 收集近10个交易日数据，用于连板判定
    sorted_dates = sorted(grouped.keys())
    if trade_date not in sorted_dates:
        return pd.DataFrame()
    idx = sorted_dates.index(trade_date)
    lookback_dates = sorted_dates[max(0, idx - 10):idx + 1]

    # 构建全量 daily 拼接表
    frames = []
    for d in lookback_dates:
        f = grouped[d][["ts_code", "pct_chg", "close", "trade_date"]].copy()
        frames.append(f)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    # 计算每只股票的连板计数
    result_rows = []
    today_df = grouped[trade_date]
    today_limit = today_df[today_df["pct_chg"] >= LIMIT_UP_THRESHOLD].copy()

    for _, row in today_limit.iterrows():
        ts_code = row["ts_code"]
        hist = combined[combined["ts_code"] == ts_code].sort_values("trade_date")
        if len(hist) < 1:
            continue

        # 从最后一天往前数连续涨停天数
        consec = 0
        for i in range(len(hist) - 1, -1, -1):
            if hist.iloc[i]["pct_chg"] >= LIMIT_UP_THRESHOLD:
                consec += 1
            else:
                break
        result_rows.append({
            "ts_code": ts_code,
            "trade_date": trade_date,
            "pct_chg": row["pct_chg"],
            "consecutive_boards": consec,
        })

    return pd.DataFrame(result_rows)


def _calc_next_day_return(grouped: dict, trade_date: str) -> float:
    """计算涨停股的次日平均收益"""
    sorted_dates = sorted(grouped.keys())
    idx = sorted_dates.index(trade_date)
    if idx + 1 >= len(sorted_dates):
        return 0.0
    next_date = sorted_dates[idx + 1]

    today_limit = grouped[trade_date][grouped[trade_date]["pct_chg"] >= LIMIT_UP_THRESHOLD]
    next_df = grouped[next_date]

    merged = today_limit[["ts_code"]].merge(
        next_df[["ts_code", "pct_chg"]], on="ts_code", how="inner"
    )
    if len(merged) == 0:
        return 0.0
    return round(merged["pct_chg"].mean(), 2)


def compute_limit_up_stats(
    start_date: str = "",
    end_date: str = "",
    lookback_days: int = 30,
    use_cache: bool = True,
    cache_file: Optional[str] = None,
) -> List[LimitUpStats]:
    """
    计算全市场连板统计。

    Args:
        start_date: 起始日期 YYYYMMDD
        end_date:   截止日期 YYYYMMDD
        lookback_days: 向前回看天数（用于拉取足够数据计算连板）
        use_cache:  是否使用缓存
        cache_file: 缓存文件路径

    Returns:
        List[LimitUpStats]
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if cache_file is None:
        cache_file = str(_CACHE_DIR / "limit_up.csv")

    if not end_date:
        end_date = datetime.now().strftime("%Y%m%d")
    if not start_date:
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y%m%d")

    # 拉取额外10个交易日用于连板判定
    fetch_start = (datetime.strptime(start_date, "%Y%m%d") - timedelta(days=20)).strftime("%Y%m%d")

    pro = _init_tushare()
    if pro is None:
        raise RuntimeError("TUSHARE_TOKEN 未配置，无法拉取全市场数据")

    # 获取交易日
    trade_dates = _get_trade_dates(pro, fetch_start, end_date)
    if not trade_dates:
        raise RuntimeError("无法获取交易日历")

    print(f"[limit_up_stats] 拉取 {len(trade_dates)} 个交易日全市场数据...")

    # 按交易日拉取daily
    grouped = {}
    for td in trade_dates:
        df = _fetch_daily_by_date(pro, td)
        if len(df) > 0:
            grouped[td] = df
            print(f"  {td}: {len(df)} 只股票")

    if not grouped:
        raise RuntimeError("未拉取到任何daily数据")

    sorted_dates = sorted(grouped.keys())
    # 只统计目标范围内的日期
    target_dates = [d for d in sorted_dates if start_date <= d <= end_date]

    # 逐日计算连板统计
    daily_records = []
    board_data_by_date = {}  # {date: DataFrame(consecutive_boards)}

    for td in target_dates:
        boards_df = _calc_consecutive_boards(grouped, td)
        board_data_by_date[td] = boards_df

        today_df = grouped[td]
        total = len(today_df)
        limit_count = len(boards_df)

        first_b = len(boards_df[boards_df["consecutive_boards"] == 1]) if len(boards_df) > 0 else 0
        second_b = len(boards_df[boards_df["consecutive_boards"] == 2]) if len(boards_df) > 0 else 0
        third_b = len(boards_df[boards_df["consecutive_boards"] >= 3]) if len(boards_df) > 0 else 0

        avg_ret = _calc_next_day_return(grouped, td)

        daily_records.append(LimitUpStats(
            trade_date=td,
            limit_up_count=limit_count,
            first_board_count=first_b,
            second_board_count=second_b,
            third_board_count=third_b,
            avg_next_day_return=avg_ret,
            total_stocks=total,
        ))

    # 计算近5日滚动晋级率
    for i, rec in enumerate(daily_records):
        window = daily_records[max(0, i - 4):i + 1]
        total_first = sum(r.first_board_count + r.second_board_count for r in window)  # 需要1板→2板的基数
        # 1→2: 今天2板数 / 昨天1板数（近5日平均）
        # 更简单：近5日 2板数 / 近5日(1板数+2板数) 作为晋级率近似
        sum_1b = sum(r.first_board_count for r in window)
        sum_2b = sum(r.second_board_count for r in window)
        sum_3b = sum(r.third_board_count for r in window)

        # 1→2 晋级率 = 近5日2板数 / 近5日(1板+2板)
        denom_12 = sum_1b + sum_2b
        rec.promotion_1to2 = round(sum_2b / denom_12 * 100, 1) if denom_12 > 0 else 0.0

        # 2→3 晋级率 = 近5日3板+数 / 近5日(2板+3板+)
        denom_23 = sum_2b + sum_3b
        rec.promotion_2to3 = round(sum_3b / denom_23 * 100, 1) if denom_23 > 0 else 0.0

    # 缓存
    cache_df = pd.DataFrame([
        {
            "trade_date": r.trade_date,
            "limit_up_count": r.limit_up_count,
            "first_board_count": r.first_board_count,
            "second_board_count": r.second_board_count,
            "third_board_count": r.third_board_count,
            "promotion_1to2": r.promotion_1to2,
            "promotion_2to3": r.promotion_2to3,
            "avg_next_day_return": r.avg_next_day_return,
            "total_stocks": r.total_stocks,
        }
        for r in daily_records
    ])
    cache_df.to_csv(cache_file, index=False)
    print(f"[limit_up_stats] 缓存写入 {cache_file} ({len(cache_df)} 行)")

    return daily_records


def load_cached_stats(cache_file: Optional[str] = None) -> pd.DataFrame:
    """读取缓存的连板统计"""
    if cache_file is None:
        cache_file = str(_CACHE_DIR / "limit_up.csv")
    if not os.path.exists(cache_file):
        return pd.DataFrame()
    return pd.read_csv(cache_file, dtype={"trade_date": str})


if __name__ == "__main__":
    stats = compute_limit_up_stats(lookback_days=10)
    for s in stats[-5:]:
        print(f"  {s.trade_date}: 涨停{s.limit_up_count}只 "
              f"1→2:{s.promotion_1to2}% 2→3:{s.promotion_2to3}% "
              f"次日均收:{s.avg_next_day_return}%")
