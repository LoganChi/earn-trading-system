#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""全市场趋势筛选器（批量优化版）

用tushare批量接口一次拉全市场日K，不逐只拉。
速度：全市场5500只 < 10分钟
"""
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional

from src.data.loader import _init_tushare
from src.signals.macd_area import calc_macd, find_green_peaks, calc_price_position


@dataclass
class ScreenedStock:
    code: str
    name: str
    industry: str
    close: float
    pct_chg: float
    green_peak_area: float
    green_severity: str
    macd_bar: float
    dif: float
    price_position: float
    composite_score: float
    price_tier: str = ""  # 仙股(<3) / 低价(3-10) / 中价(10-50) / 高价(>50)
    description: str = ""


def batch_load_daily(trade_date: str) -> pd.DataFrame:
    """批量拉全市场某天日K"""
    cache = Path(__file__).resolve().parents[1] / "data" / "cache" / f"market_{trade_date}.csv"
    if cache.exists():
        return pd.read_csv(cache, dtype={'ts_code': str})
    
    pro = _init_tushare()
    if not pro:
        raise RuntimeError("TUSHARE_TOKEN 未配置")
    
    df = pro.daily(trade_date=trade_date)
    df.to_csv(cache, index=False)
    return df


def batch_load_history(ts_codes: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """批量拉多只股票的历史日K"""
    cache = Path(__file__).resolve().parents[1] / "data" / "cache" / f"history_{start_date}_{end_date}.csv"
    if cache.exists():
        return pd.read_csv(cache, dtype={'ts_code': str})
    
    pro = _init_tushare()
    if not pro:
        raise RuntimeError("TUSHARE_TOKEN 未配置")
    
    all_data = []
    # 按日期批量拉（每个交易日一次API调用）
    # 找出日期范围内的交易日
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    
    current = start
    while current <= end:
        date_str = current.strftime("%Y%m%d")
        try:
            df = pro.daily(trade_date=date_str)
            if df is not None and len(df) > 0:
                # 过滤目标股票
                df = df[df['ts_code'].isin(ts_codes)]
                if len(df) > 0:
                    all_data.append(df)
        except:
            pass
        current += timedelta(days=1)
    
    if not all_data:
        return pd.DataFrame()
    
    raw = pd.concat(all_data, ignore_index=True)
    raw.to_csv(cache, index=False)
    return raw


def screen_market_fast(end_date: str = "20260731", 
                       lookback_days: int = 120,
                       min_score: float = 0.3,
                       verbose: bool = True) -> List[ScreenedStock]:
    """
    全市场快速筛选
    
    策略：
    1. 批量拉全市场近N个交易日的日K（按交易日批量）
    2. 按股票分组，计算MACD+绿峰面积+价格位置
    3. 筛选符合条件的
    """
    pro = _init_tushare()
    if not pro:
        raise RuntimeError("TUSHARE_TOKEN 未配置")
    
    # 1. 获取股票列表
    cache_stocks = Path(__file__).resolve().parents[1] / "data" / "cache" / "all_stocks.csv"
    if cache_stocks.exists():
        stocks = pd.read_csv(cache_stocks, dtype={'symbol': str})
    else:
        stocks = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,industry')
        stocks.to_csv(cache_stocks, index=False)
    
    # 过滤ST和北交所
    stocks = stocks[~stocks['name'].str.contains('ST|退', na=False)]
    stocks = stocks[~stocks['ts_code'].str.endswith('.BJ')]
    
    if verbose:
        print(f"股票池: {len(stocks)}只（已排除ST+北交所）")
    
    # 2. 批量拉历史日K
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=lookback_days * 2)).strftime("%Y%m%d")
    
    # 获取交易日历
    cal = pro.trade_cal(exchange='SSE', start_date=start_date, end_date=end_date)
    trade_dates = sorted(cal[cal['is_open'] == 1]['cal_date'].tolist())[-lookback_days:]
    
    if verbose:
        print(f"拉取 {len(trade_dates)} 个交易日数据...")
    
    # 批量拉（按交易日）
    cache_history = Path(__file__).resolve().parents[1] / "data" / "cache" / f"market_history_{trade_dates[0]}_{trade_dates[-1]}.csv"
    
    if cache_history.exists():
        if verbose:
            print("使用缓存...")
        all_daily = pd.read_csv(cache_history, dtype={'ts_code': str})
    else:
        all_data = []
        valid_codes = set(stocks['ts_code'])
        
        for i, date_str in enumerate(trade_dates):
            try:
                df = pro.daily(trade_date=date_str)
                if df is not None and len(df) > 0:
                    df = df[df['ts_code'].isin(valid_codes)]
                    all_data.append(df)
                if verbose and (i + 1) % 20 == 0:
                    print(f"  [{i+1}/{len(trade_dates)}] 累计{sum(len(d) for d in all_data):,}条")
            except:
                pass
        
        all_daily = pd.concat(all_data, ignore_index=True)
        all_daily.to_csv(cache_history, index=False)
    
    if verbose:
        print(f"总计: {len(all_daily):,}条日K, {all_daily['ts_code'].nunique()}只股票")
    
    # 3. 按股票分组筛选
    if verbose:
        print(f"开始筛选...")
    
    results = []
    stock_map = {row['ts_code']: row for _, row in stocks.iterrows()}
    
    for i, (ts_code, group) in enumerate(all_daily.groupby('ts_code')):
        if verbose and (i + 1) % 500 == 0:
            print(f"  [{i+1}] 已筛选出 {len(results)} 只")
        
        group = group.sort_values('trade_date').reset_index(drop=True)
        if len(group) < 35:
            continue
        
        close = group['close'].values
        pct_chg = group['pct_chg'].values
        
        # 排除涨停状态
        if pct_chg[-1] > 9.5:
            continue
        
        # 排除价格过高（但保留仙股，低价股弹性大）
        if close[-1] > 500:
            continue
        
        # 计算MACD
        dif, dea, macd_bar = calc_macd(close)
        
        # 找绿峰
        dates = group['trade_date'].values
        green_peaks = find_green_peaks(macd_bar, dif, dates, close)
        
        last_gp = None
        for gp in reversed(green_peaks):
            if gp.end_idx <= len(close) - 1:
                last_gp = gp
                break
        
        if not last_gp or last_gp.area < 5:
            continue
        
        # MACD状态：翻红或接近翻红
        if macd_bar[-1] < -0.1:
            continue
        
        # 价格位置
        price_pos = calc_price_position(close, len(close) - 1, lookback=60)
        if price_pos > 0.6:
            continue
        
        # 综合评分
        score = 0.0
        if last_gp.severity == "deep":
            score += 0.3
        elif last_gp.severity == "moderate":
            score += 0.15
        
        if macd_bar[-1] > 0:
            score += 0.2
            if dif[-1] > dif[-2]:
                score += 0.1
        
        if price_pos < 0.2:
            score += 0.2
        elif price_pos < 0.4:
            score += 0.1
        
        score = min(1.0, score)
        
        if score < min_score:
            continue
        
        stock_info = stock_map.get(ts_code, {})
        
        # 描述
        desc = (f"绿峰{last_gp.area:.1f}({last_gp.severity}) "
                f"MACD{'红' if macd_bar[-1] > 0 else '近翻红'} "
                f"位置{price_pos:.0%} 评分{score:.0%}")
        
        # 价格分档
        p = close[-1]
        if p < 3:
            tier = "仙股"
        elif p < 10:
            tier = "低价"
        elif p < 50:
            tier = "中价"
        else:
            tier = "高价"
        
        results.append(ScreenedStock(
            code=ts_code.split('.')[0],
            name=stock_info.get('name', ''),
            industry=stock_info.get('industry', ''),
            close=round(close[-1], 2),
            pct_chg=round(pct_chg[-1], 2),
            green_peak_area=round(last_gp.area, 1),
            green_severity=last_gp.severity,
            macd_bar=round(macd_bar[-1], 4),
            dif=round(dif[-1], 4),
            price_position=round(price_pos, 2),
            composite_score=round(score, 2),
            price_tier=tier,
            description=desc,
        ))
    
    if verbose:
        print(f"  完成！筛选出 {len(results)} 只")
    
    results.sort(key=lambda x: -x.composite_score)
    return results


if __name__ == "__main__":
    results = screen_market_fast(
        end_date="20260731",
        lookback_days=120,
        min_score=0.3,
        verbose=True,
    )
    
    print(f"\n{'='*100}")
    print(f"  全市场趋势筛选结果")
    print(f"{'='*100}")
    print(f"\n{'代码':8s} {'名称':8s} {'行业':10s} {'现价':>7s} {'涨跌':>6s} {'绿峰':>6s} {'级别':8s} {'MACD柱':>7s} {'位置':>5s} {'评分':>5s}")
    print(f"{'-'*100}")
    
    for r in results[:50]:
        print(f"{r.code:8s} {r.name:8s} {r.industry:10s} {r.close:7.2f} {r.pct_chg:+5.1f}% "
              f"{r.green_peak_area:6.1f} {r.green_severity:8s} {r.macd_bar:+7.4f} {r.price_position:4.0%} {r.composite_score:4.0%}")
    
    if len(results) > 50:
        print(f"\n  ... 还有 {len(results) - 50} 只")
