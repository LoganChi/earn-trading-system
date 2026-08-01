#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""全市场趋势筛选器

按用户的MACD面积战法标准，从全市场筛选符合条件的股票：
1. 日线MACD绿柱面积充分消耗（深跌后筹码出清）
2. 红柱开始积累（大资金开始推动）
3. 价格在底部区域（不追高）
4. 成交量分布配合（VP边缘信号）

不限于用户持仓，全市场扫描。
"""
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.data.loader import _init_tushare, load_daily
from src.signals.macd_area import calc_macd, generate_signals, find_green_peaks, calc_price_position
from src.signals.volume_profile import calc_volume_profile, generate_vp_signals


@dataclass
class ScreenedStock:
    """筛选出的股票"""
    code: str
    name: str
    industry: str
    close: float
    pct_chg: float
    
    # MACD面积信号
    signal_type: str
    signal_strength: float
    green_peak_area: float
    green_severity: str
    red_expanding: bool
    dif_turning_up: bool
    price_position: float
    
    # VP信号
    vp_signal: str
    poc_price: float
    at_edge: bool
    edge_breakout: bool
    
    # 综合评分
    composite_score: float
    description: str


def get_all_stocks() -> pd.DataFrame:
    """获取全市场A股列表"""
    cache = Path(__file__).resolve().parents[1] / "data" / "cache" / "all_stocks.csv"
    
    if cache.exists():
        return pd.read_csv(cache, dtype={'symbol': str})
    
    pro = _init_tushare()
    if not pro:
        raise RuntimeError("TUSHARE_TOKEN 未配置")
    
    df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,industry')
    df.to_csv(cache, index=False)
    return df


def screen_stock(code: str, name: str = "", industry: str = "",
                 days: int = 120, end_date: str = "") -> Optional[ScreenedStock]:
    """
    筛选单只股票是否符合战法标准
    
    标准：
    1. 近期有绿峰面积消耗（深跌后）
    2. MACD翻红（红柱开始积累）
    3. 价格在底部区域（price_position < 0.5）
    4. VP信号配合
    
    排除：
    - ST/退市
    - 涨停状态（买不进）
    - 新股（数据不足）
    """
    try:
        df = load_daily(code, start_date="", end_date=end_date, use_cache=True)
        if len(df) < 60:
            return None
        
        df = df.sort_values('trade_date').reset_index(drop=True)
        recent = df.tail(days).reset_index(drop=True)
        
        # 计算MACD
        close = recent['close'].values
        dates = recent['trade_date'].values
        dif, dea, macd_bar = calc_macd(close)
        
        # 找绿峰
        green_peaks = find_green_peaks(macd_bar, dif, dates, close)
        red_peaks_signals = generate_signals(recent)
        
        # 取最新信号
        if not red_peaks_signals:
            return None
        
        latest_sig = red_peaks_signals[-1]
        current_price = close[-1]
        current_pct = recent.iloc[-1].get('pct_chg', 0)
        
        # VP信号
        vp_sigs = generate_vp_signals(recent, lookback=20, bins=30)
        latest_vp = vp_sigs[-1] if vp_sigs else None
        
        # 找最近的绿峰
        last_gp = None
        for gp in reversed(green_peaks):
            if gp.end_idx <= len(recent) - 1:
                last_gp = gp
                break
        
        # === 筛选标准 ===
        
        # 标准1：有显著的绿峰面积消耗（area > 5 或 severity=deep/moderate）
        if not last_gp or last_gp.area < 5:
            return None
        
        # 标准2：MACD翻红或在零轴附近拐头
        if macd_bar[-1] < 0:  # 还在绿柱
            # 但如果绿柱在缩短且接近0轴，也算候选
            if macd_bar[-1] > -0.1 and macd_bar[-1] > macd_bar[-2]:
                pass  # 接近翻红
            else:
                return None
        
        # 标准3：价格位置（不在半山腰以上）
        price_pos = calc_price_position(close, len(close) - 1, lookback=60)
        if price_pos > 0.6:
            return None  # 太高了
        
        # 标准4：排除涨停
        if current_pct > 9.5:
            return None
        
        # 计算综合评分
        score = 0.0
        
        # 绿峰面积
        if last_gp.severity == "deep":
            score += 0.3
        elif last_gp.severity == "moderate":
            score += 0.15
        
        # MACD状态
        if macd_bar[-1] > 0:  # 已翻红
            score += 0.2
            if dif[-1] > dif[-2]:  # DIF拐头
                score += 0.1
        
        # 价格位置
        if price_pos < 0.2:
            score += 0.2
        elif price_pos < 0.4:
            score += 0.1
        
        # VP配合
        vp_desc = ""
        poc = 0
        at_edge = False
        edge_breakout = False
        vp_type = "neutral"
        
        if latest_vp:
            vp_type = latest_vp.signal_type
            poc = latest_vp.poc_price if hasattr(latest_vp, 'poc_price') else 0
            at_edge = latest_vp.at_edge
            edge_breakout = latest_vp.edge_breakout
            
            if edge_breakout:
                score += 0.2  # VP边缘放量
            elif latest_vp.at_poc:
                score += 0.1  # 在POC附近
            
            vp_desc = latest_vp.description
        
        score = min(1.0, score)
        
        # 描述
        desc_parts = []
        desc_parts.append(f"绿峰{last_gp.area:.1f}({last_gp.severity})")
        desc_parts.append(f"MACD{'红' if macd_bar[-1] > 0 else '接近翻红'}")
        desc_parts.append(f"价格位置{price_pos:.0%}")
        if latest_vp:
            desc_parts.append(f"VP:{vp_type}")
        desc_parts.append(f"评分{score:.0%}")
        
        return ScreenedStock(
            code=code, name=name, industry=industry,
            close=round(current_price, 2),
            pct_chg=round(current_pct, 2),
            signal_type=latest_sig.signal_type,
            signal_strength=round(latest_sig.signal_strength, 2),
            green_peak_area=round(last_gp.area, 1),
            green_severity=last_gp.severity,
            red_expanding=(macd_bar[-1] > 0 and macd_bar[-1] > macd_bar[-2]),
            dif_turning_up=(dif[-1] > dif[-2] if len(dif) > 1 else False),
            price_position=round(price_pos, 2),
            vp_signal=vp_type,
            poc_price=round(poc, 2) if poc else 0,
            at_edge=at_edge,
            edge_breakout=edge_breakout,
            composite_score=round(score, 2),
            description=" | ".join(desc_parts),
        )
        
    except Exception as e:
        return None


def screen_market(end_date: str = "", max_stocks: int = 500, 
                  min_score: float = 0.3, verbose: bool = True) -> List[ScreenedStock]:
    """
    全市场扫描，筛选符合战法标准的股票
    
    策略：
    1. 先用日K快速筛选（不拉分钟数据）
    2. 按绿峰面积+MACD状态+价格位置打分
    3. 按综合评分排序
    
    参数：
    - end_date: 截止日期
    - max_stocks: 最多扫描多少只（全市场5500+只太慢）
    - min_score: 最低评分阈值
    """
    all_stocks = get_all_stocks()
    
    # 过滤ST
    all_stocks = all_stocks[~all_stocks['name'].str.contains('ST|退', na=False)]
    
    # 按行业抽样（保证覆盖各行业）
    if len(all_stocks) > max_stocks:
        # 每行业取前N只
        n_per_industry = max_stocks // all_stocks['industry'].nunique()
        all_stocks = all_stocks.groupby('industry').head(max(1, n_per_industry)).reset_index(drop=True)
    
    if verbose:
        print(f"扫描 {len(all_stocks)} 只股票...")
    
    results = []
    errors = 0
    
    for i, (_, row) in enumerate(all_stocks.iterrows()):
        code = str(row['symbol'])
        name = row.get('name', '')
        industry = row.get('industry', '')
        
        if verbose and (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(all_stocks)}] 已筛选出 {len(results)} 只")
        
        screened = screen_stock(code, name, industry, days=120, end_date=end_date)
        
        if screened and screened.composite_score >= min_score:
            results.append(screened)
    
    if verbose:
        print(f"  完成！筛选出 {len(results)} 只（评分≥{min_score:.0%}）")
    
    # 按评分排序
    results.sort(key=lambda x: -x.composite_score)
    
    return results


def print_screening_results(results: List[ScreenedStock]):
    """打印筛选结果"""
    if not results:
        print("无符合条件的股票")
        return
    
    print(f"\n{'='*90}")
    print(f"  全市场趋势筛选结果（MACD面积战法标准）")
    print(f"  共 {len(results)} 只符合条件")
    print(f"{'='*90}")
    print(f"\n{'代码':8s} {'名称':8s} {'行业':8s} {'现价':>7s} {'涨跌':>6s} {'绿峰':>6s} {'级别':6s} {'位置':>5s} {'VP':10s} {'评分':>5s} 描述")
    print(f"{'-'*120}")
    
    for r in results[:50]:  # 最多显示50只
        print(f"{r.code:8s} {r.name:8s} {r.industry:8s} {r.close:7.2f} {r.pct_chg:+5.1f}% "
              f"{r.green_peak_area:6.1f} {r.green_severity:6s} {r.price_position:4.0%} "
              f"{r.vp_signal:10s} {r.composite_score:4.0%} {r.description}")
    
    if len(results) > 50:
        print(f"\n  ... 还有 {len(results) - 50} 只未显示")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="全市场趋势筛选器")
    parser.add_argument("--end", type=str, default="", help="截止日期 YYYYMMDD")
    parser.add_argument("--max", type=int, default=500, help="最多扫描股票数")
    parser.add_argument("--min-score", type=float, default=0.3, help="最低评分")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    
    results = screen_market(end_date=args.end, max_stocks=args.max, 
                           min_score=args.min_score, verbose=True)
    print_screening_results(results)
