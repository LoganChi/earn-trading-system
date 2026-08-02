#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""价格区间分析器

核心概念（来自用户实战经验）：
- 价格在某个区间内反复震荡=筹码充分交换=蓄势
- 区间上沿=阻力位（多次碰到回落）
- 区间下沿=支撑位（多次碰到反弹）
- 当价格在区间下沿+MACD翻红=双重确认入场

与成交量分布（VP）的逻辑相通：
- POC=区间内成交量最大的价位=市场共识
- 价格在POC下方=偏弱，在POC上方=偏强
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class PriceZone:
    """价格区间"""
    upper: float          # 区间上沿（阻力位）
    lower: float          # 区间下沿（支撑位）
    poc: float            # 成交量最大价位（POC）
    touches_upper: int    # 触碰上沿次数
    touches_lower: int    # 触碰下沿次数
    width_pct: float      # 区间宽度占价格的比例
    consolidation_days: int  # 整理天数
    current_position: str  # "upper" / "middle" / "lower" / "below" / "above"
    breakout_direction: str  # "none" / "up" / "down"
    strength: float       # 区间强度 0-1（触碰次数越多越强）


def analyze_price_zone(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                       vol: np.ndarray = None, lookback: int = 60) -> Optional[PriceZone]:
    """分析价格区间
    
    逻辑：
    1. 取最近lookback天的价格
    2. 找出多次触碰的高点和低点（至少各2次）
    3. 确定区间上沿和下沿
    4. 判断当前价格在区间中的位置
    """
    if len(close) < lookback:
        lookback = len(close)
    
    recent_close = close[-lookback:]
    recent_high = high[-lookback:]
    recent_low = low[-lookback:]
    recent_vol = vol[-lookback:] if vol is not None else None
    
    current_price = recent_close[-1]
    
    # 方法：用价格聚类找区间
    # 把价格分成20个bin，找出成交量最大/最密集的价格带
    n_bins = 20
    price_min = recent_low.min()
    price_max = recent_high.max()
    if price_max <= price_min:
        return None
    
    bin_width = (price_max - price_min) / n_bins
    
    # 计算每个bin的"成交量"（如果没有vol就用天数）
    bin_weights = np.zeros(n_bins)
    for i in range(len(recent_close)):
        bin_idx = int((recent_close[i] - price_min) / bin_width)
        bin_idx = min(bin_idx, n_bins - 1)
        if recent_vol is not None:
            bin_weights[bin_idx] += recent_vol[i]
        else:
            bin_weights[bin_idx] += 1
    
    # POC = 成交量最大的bin
    poc_bin = np.argmax(bin_weights)
    poc_price = price_min + (poc_bin + 0.5) * bin_width
    
    # 找区间上沿和下沿
    # 上沿 = 价格分布的85%分位（偏高但不是最高）
    # 下沿 = 价格分布的15%分位
    sorted_prices = np.sort(recent_close)
    upper_raw = sorted_prices[int(len(sorted_prices) * 0.85)]
    lower_raw = sorted_prices[int(len(sorted_prices) * 0.15)]
    
    # 精确化：找实际触碰的高点/低点
    # 上沿：近期高点中接近upper_raw的
    touch_threshold = bin_width * 1.5  # 触碰容忍范围
    
    touches_upper = 0
    touches_lower = 0
    
    for i in range(len(recent_high)):
        if abs(recent_high[i] - upper_raw) < touch_threshold:
            touches_upper += 1
        if abs(recent_low[i] - lower_raw) < touch_threshold:
            touches_lower += 1
    
    # 区间宽度
    zone_width = upper_raw - lower_raw
    width_pct = zone_width / current_price if current_price > 0 else 0
    
    # 整理天数：价格在区间内的天数比例
    in_zone = sum(1 for c in recent_close if lower_raw <= c <= upper_raw)
    consolidation_ratio = in_zone / len(recent_close)
    consolidation_days = int(consolidation_ratio * lookback)
    
    # 当前位置
    if current_price > upper_raw + touch_threshold:
        current_pos = "above"
        breakout = "up"
    elif current_price < lower_raw - touch_threshold:
        current_pos = "below"
        breakout = "down"
    elif current_price < poc_price:
        current_pos = "lower"
        breakout = "none"
    elif current_price > poc_price:
        current_pos = "upper"
        breakout = "none"
    else:
        current_pos = "middle"
        breakout = "none"
    
    # 区间强度：触碰次数+整理天数
    touch_score = min((touches_upper + touches_lower) / 8, 1.0)
    consol_score = min(consolidation_days / 30, 1.0)
    strength = touch_score * 0.4 + consol_score * 0.6
    
    return PriceZone(
        upper=round(upper_raw, 2),
        lower=round(lower_raw, 2),
        poc=round(poc_price, 2),
        touches_upper=touches_upper,
        touches_lower=touches_lower,
        width_pct=round(width_pct, 3),
        consolidation_days=consolidation_days,
        current_position=current_pos,
        breakout_direction=breakout,
        strength=round(strength, 2),
    )
