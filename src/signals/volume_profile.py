#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""成交量分布（Volume Profile）信号

核心逻辑（来自用户素材"成交量分布定涨跌"）：
- POC（Point of Control）= 成交量最大的价格 = 市场共识价值中心
- 正态分布边缘只占7%的时间和7%的成交量
- 价格到边缘无量 → 碰一下回去 → 继续围绕POC震荡
- 价格到边缘放量 → 边缘被否定 → 寻找新价值 → 趋势开始

一句话：量比线重要。同一个价格，上次无量=边缘有效，这次放量=边缘失效。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class VolumeProfile:
    """成交量分布"""
    poc_price: float           # 成交量最大价位（价值中心）
    poc_volume: float          # POC处成交量
    value_area_high: float     # 价值区域上沿（70%成交量上边界）
    value_area_low: float      # 价值区域下沿（70%成交量下边界）
    total_volume: float        # 总成交量
    bins: int                  # 价格分桶数
    histogram: dict            # {price_bucket: volume}


@dataclass
class VPSignal:
    """成交量分布信号"""
    date: str
    price: float
    signal_type: str          # "edge_test" / "breakout" / "value_test" / "neutral"
    signal_strength: float    # 0-1
    
    at_edge: bool = False     # 是否在VP边缘
    edge_volume_ratio: float = 0.0  # 边缘成交量占比（<7%=正常边缘）
    edge_breakout: bool = False     # 边缘放量=突破
    
    at_poc: bool = False      # 是否在POC附近
    description: str = ""


def calc_volume_profile(df: pd.DataFrame, lookback: int = 20, bins: int = 50) -> VolumeProfile:
    """
    计算成交量分布（Volume Profile）
    
    用近N日的日K（开盘/收盘/最高/最低/成交量）估算价格分布。
    简化方法：用每日的成交均价（(open+close)/2）和成交量构建直方图。
    
    更精确的方法需要分钟数据（tick分布），这里用日线近似。
    """
    recent = df.tail(lookback).copy()
    
    # 用每日价格范围构建分布
    # 简化：把每日成交量分配到当日的价格区间
    prices = []
    volumes = []
    
    for _, row in recent.iterrows():
        lo = row['low']
        hi = row['high']
        vol = row.get('vol', row.get('volume', 0))
        if hi == lo:
            prices.append((lo + hi) / 2)
            volumes.append(vol)
        else:
            # 把成交量均匀分配到价格区间（近似）
            # 实际中应该用分钟数据做更精确的分配
            prices.append((lo + hi) / 2)
            volumes.append(vol)
    
    prices = np.array(prices)
    volumes = np.array(volumes, dtype=float)
    
    if len(prices) == 0 or volumes.sum() == 0:
        return VolumeProfile(0, 0, 0, 0, 0, 0, {})
    
    # 构建价格直方图
    price_min = prices.min()
    price_max = prices.max()
    if price_max == price_min:
        return VolumeProfile(price_min, volumes[0], price_max, price_min, volumes.sum(), 1, {price_min: volumes[0]})
    
    bin_edges = np.linspace(price_min, price_max, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # 把成交量分配到价格桶
    hist = np.zeros(bins)
    for i, (p, v) in enumerate(zip(prices, volumes)):
        # 找到最接近的价格桶
        idx = np.clip(int((p - price_min) / (price_max - price_min) * bins), 0, bins - 1)
        hist[idx] += v
    
    # POC = 成交量最大的桶
    poc_idx = np.argmax(hist)
    poc_price = bin_centers[poc_idx]
    poc_volume = hist[poc_idx]
    total_volume = hist.sum()
    
    # 价值区域（70%成交量）
    # 从POC向两边扩展，累计70%的成交量
    target = total_volume * 0.7
    accumulated = hist[poc_idx]
    va_high_idx = poc_idx
    va_low_idx = poc_idx
    
    while accumulated < target and (va_low_idx > 0 or va_high_idx < bins - 1):
        up_vol = hist[va_high_idx + 1] if va_high_idx + 1 < bins else 0
        down_vol = hist[va_low_idx - 1] if va_low_idx - 1 >= 0 else 0
        
        if up_vol >= down_vol and va_high_idx + 1 < bins:
            va_high_idx += 1
            accumulated += hist[va_high_idx]
        elif va_low_idx - 1 >= 0:
            va_low_idx -= 1
            accumulated += hist[va_low_idx]
        else:
            break
    
    histogram = {round(bin_centers[i], 2): hist[i] for i in range(bins) if hist[i] > 0}
    
    return VolumeProfile(
        poc_price=round(poc_price, 2),
        poc_volume=round(poc_volume, 0),
        value_area_high=round(bin_centers[va_high_idx], 2),
        value_area_low=round(bin_centers[va_low_idx], 2),
        total_volume=round(total_volume, 0),
        bins=bins,
        histogram=histogram,
    )


def generate_vp_signals(df: pd.DataFrame, lookback: int = 20, bins: int = 50) -> List[VPSignal]:
    """
    为每个交易日生成VP信号
    
    使用滚动窗口：每天的VP用过去N天的数据计算
    """
    signals = []
    n = len(df)
    
    for i in range(lookback, n):
        # 用过去N天计算VP
        window = df.iloc[i - lookback:i + 1]
        vp = calc_volume_profile(window, lookback=lookback, bins=bins)
        
        current_price = df.iloc[i]['close']
        current_date = str(df.iloc[i]['trade_date'])
        current_vol = df.iloc[i].get('vol', df.iloc[i].get('volume', 0))
        
        # 判断当前价格位置
        at_edge_high = current_price >= vp.value_area_high
        at_edge_low = current_price <= vp.value_area_low
        at_edge = at_edge_high or at_edge_low
        
        # 计算边缘成交量占比
        edge_vol_ratio = 0.0
        if at_edge:
            # 简化：如果当日价格在边缘区间，用当日成交量/近N日平均
            avg_vol = vp.total_volume / lookback
            if avg_vol > 0:
                edge_vol_ratio = current_vol / avg_vol
        
        # 判断信号
        signal = VPSignal(
            date=current_date,
            price=round(current_price, 2),
            signal_type="neutral",
            signal_strength=0.0,
        )
        
        if at_edge:
            signal.at_edge = True
            signal.edge_volume_ratio = round(edge_vol_ratio, 2)
            
            if edge_vol_ratio > 1.5:
                # 边缘放量 → 突破信号
                signal.signal_type = "breakout"
                signal.edge_breakout = True
                signal.signal_strength = min(1.0, (edge_vol_ratio - 1.0) / 2.0)
                signal.description = (
                    f"VP边缘放量({edge_vol_ratio:.1f}x均值) "
                    f"{'上沿突破' if at_edge_high else '下沿突破'} "
                    f"POC={vp.poc_price}"
                )
            else:
                # 边缘无量 → 测试信号（大概率回去）
                signal.signal_type = "edge_test"
                signal.signal_strength = 0.3
                signal.description = (
                    f"VP边缘无量({edge_vol_ratio:.1f}x均值) "
                    f"{'上沿' if at_edge_high else '下沿'} "
                    f"大概率回归POC={vp.poc_price}"
                )
        elif abs(current_price - vp.poc_price) / vp.poc_price < 0.02:
            # 在POC附近
            signal.at_poc = True
            signal.signal_type = "value_test"
            signal.signal_strength = 0.2
            signal.description = f"在POC附近，价值中心={vp.poc_price}"
        
        signals.append(signal)
    
    return signals


def check_resonance(macd_signal, vp_signal) -> float:
    """
    MACD面积信号与VP信号的共振检测
    
    返回共振强度（0-1）
    """
    if not macd_signal or not vp_signal:
        return 0.0
    
    strength = 0.0
    
    # MACD入场信号 + VP突破
    if macd_signal.signal_type == "entry_candidate":
        if vp_signal.edge_breakout:
            # MACD翻红 + VP边缘放量 = 强共振
            strength = min(1.0, macd_signal.signal_strength + vp_signal.signal_strength)
        elif vp_signal.signal_type == "edge_test":
            # MACD翻红 + VP边缘无量 = 中等共振（可能假突破）
            strength = macd_signal.signal_strength * 0.5
        elif vp_signal.at_poc:
            # MACD翻红 + 在POC附近 = 良好入场位
            strength = macd_signal.signal_strength * 0.8
    
    # MACD出场信号 + VP边缘
    if macd_signal.signal_type == "exit_warning":
        if vp_signal.at_edge and not vp_signal.edge_breakout:
            # MACD动能衰竭 + VP边缘无量 = 出场确认
            strength = 0.7
    
    return round(strength, 2)
