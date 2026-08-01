#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MACD 绿峰面积分析器

核心逻辑（来自用户实战经验）：
- 不是看MACD金叉/死叉（机械信号，已被证伪）
- 看的是红绿柱的"面积"——大资金体量和动能持续性
- 绿柱面积充分消耗 = 筹码出清 = 底部反转前提
- 红柱面积开始积累 = 大资金开始推动

关键概念：
- 绿峰：连续绿柱（MACD<0）的完整区间
- 面积：绿柱绝对值累加（代表卖压总量）
- 深度：DIF最低点（代表下跌力度）
- 多峰结构：两段独立大跌（筹码出清更充分）

用户规则：
- 单绿峰面积大 → 可能反转但博弈空间看位置
- 红柱连续放大 + DIF拐头向上 → 入场信号
- 底部价格位置关键（同样的面积，底部 vs 半山腰结果不同）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GreenPeak:
    """一个完整的绿峰（连续绿柱区间）"""
    start_idx: int
    end_idx: int          # 绿柱结束（翻红）的位置
    start_date: str
    end_date: str
    area: float           # 绿柱绝对值累加（卖压总量）
    duration: int         # 持续天数
    min_dif: float        # DIF最低点（下跌深度）
    min_macd: float       # MACD柱最低点
    start_price: float
    end_price: float
    price_drop_pct: float # 区间跌幅

    @property
    def severity(self) -> str:
        """下跌严重度分级"""
        if self.area > 30 or self.min_dif < -1.5:
            return "deep"       # 深跌
        elif self.area > 10 or self.min_dif < -0.5:
            return "moderate"   # 中等
        else:
            return "shallow"    # 浅跌


@dataclass
class RedPeak:
    """一个完整的红峰（连续红柱区间）"""
    start_idx: int
    end_idx: int
    start_date: str
    end_date: str
    area: float
    duration: int
    max_dif: float
    max_macd: float
    is_expanding: bool    # 红柱是否还在放大（动能还在）


@dataclass
class MACDAreaSignal:
    """MACD面积信号"""
    date: str
    price: float
    signal_type: str      # "entry_candidate" / "exit_warning" / "neutral"
    signal_strength: float # 0-1
    
    # 入场信号细节
    green_peak_area: float = 0.0      # 前序绿峰面积
    green_peak_severity: str = ""     # deep/moderate/shallow
    red_peak_expanding: bool = False  # 红柱是否在放大
    dif_turning_up: bool = False      # DIF是否拐头向上
    price_position: float = 0.0       # 价格位置（0=最低 1=最高，近N日）
    
    # 出场信号细节
    red_peak_shrinking: bool = False  # 红柱开始缩短
    high_open_reject: bool = False    # 高开回落
    
    description: str = ""


def calc_macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    """计算MACD（DIF/DEA/MACD柱）"""
    ema_fast = np.zeros(len(close))
    ema_slow = np.zeros(len(close))
    ema_fast[0] = ema_slow[0] = close[0]
    
    alpha_fast = 2 / (fast + 1)
    alpha_slow = 2 / (slow + 1)
    
    for i in range(1, len(close)):
        ema_fast[i] = close[i] * alpha_fast + ema_fast[i-1] * (1 - alpha_fast)
        ema_slow[i] = close[i] * alpha_slow + ema_slow[i-1] * (1 - alpha_slow)
    
    dif = ema_fast - ema_slow
    dea = np.zeros(len(close))
    dea[0] = dif[0]
    alpha_signal = 2 / (signal + 1)
    for i in range(1, len(close)):
        dea[i] = dif[i] * alpha_signal + dea[i-1] * (1 - alpha_signal)
    
    macd_bar = (dif - dea) * 2
    return dif, dea, macd_bar


def find_green_peaks(macd_bar: np.ndarray, dif: np.ndarray, dates: np.ndarray, 
                     close: np.ndarray) -> List[GreenPeak]:
    """识别所有绿峰（连续绿柱区间）"""
    peaks = []
    i = 0
    n = len(macd_bar)
    
    while i < n:
        if macd_bar[i] < 0:
            start = i
            area = 0.0
            min_dif = dif[i]
            min_macd = macd_bar[i]
            
            while i < n and macd_bar[i] < 0:
                area += abs(macd_bar[i])
                min_dif = min(min_dif, dif[i])
                min_macd = min(min_macd, macd_bar[i])
                i += 1
            
            end = i
            duration = end - start
            start_price = close[start - 1] if start > 0 else close[0]
            end_price = close[end - 1] if end > 0 else close[0]
            price_drop = (end_price / start_price - 1) * 100 if start_price > 0 else 0
            
            # 只记录有意义的绿峰（面积>0.5 或持续>3天）
            if area > 0.5 or duration > 3:
                peaks.append(GreenPeak(
                    start_idx=start, end_idx=end,
                    start_date=str(dates[start]), end_date=str(dates[min(end, n-1)]),
                    area=round(area, 2), duration=duration,
                    min_dif=round(min_dif, 3), min_macd=round(min_macd, 3),
                    start_price=round(start_price, 2), end_price=round(end_price, 2),
                    price_drop_pct=round(price_drop, 1),
                ))
        else:
            i += 1
    
    return peaks


def find_red_peaks(macd_bar: np.ndarray, dif: np.ndarray, dates: np.ndarray,
                   close: np.ndarray) -> List[RedPeak]:
    """识别所有红峰（连续红柱区间）"""
    peaks = []
    i = 0
    n = len(macd_bar)
    
    while i < n:
        if macd_bar[i] > 0:
            start = i
            area = 0.0
            max_dif = dif[i]
            max_macd = macd_bar[i]
            
            while i < n and macd_bar[i] > 0:
                area += macd_bar[i]
                max_dif = max(max_dif, dif[i])
                max_macd = max(max_macd, macd_bar[i])
                i += 1
            
            end = i
            # 红柱是否还在放大（最后3根趋势）
            recent = macd_bar[max(start, end-3):end]
            is_expanding = len(recent) >= 2 and recent[-1] >= recent[0]
            
            if area > 0.5 or (end - start) > 2:
                peaks.append(RedPeak(
                    start_idx=start, end_idx=end,
                    start_date=str(dates[start]), end_date=str(dates[min(end, n-1)]),
                    area=round(area, 2), duration=end-start,
                    max_dif=round(max_dif, 3), max_macd=round(max_macd, 3),
                    is_expanding=is_expanding,
                ))
        else:
            i += 1
    
    return peaks


def calc_price_position(close: np.ndarray, idx: int, lookback: int = 60) -> float:
    """计算当前价格在近N日的高低位置（0=最低 1=最高）"""
    start = max(0, idx - lookback)
    window = close[start:idx + 1]
    if len(window) < 2:
        return 0.5
    lo, hi = min(window), max(window)
    if hi == lo:
        return 0.5
    return (close[idx] - lo) / (hi - lo)


def generate_signals(df: pd.DataFrame, lookback: int = 60) -> List[MACDAreaSignal]:
    """
    主函数：从日K数据生成MACD面积信号序列
    
    df 需要：trade_date, close 列
    
    返回信号列表，每个交易日一个信号
    """
    close = df['close'].values
    dates = df['trade_date'].values
    n = len(close)
    
    if n < 35:
        return []
    
    dif, dea, macd_bar = calc_macd(close)
    green_peaks = find_green_peaks(macd_bar, dif, dates, close)
    red_peaks = find_red_peaks(macd_bar, dif, dates, close)
    
    signals = []
    
    for i in range(30, n):
        sig = MACDAreaSignal(
            date=str(dates[i]),
            price=round(close[i], 2),
            signal_type="neutral",
            signal_strength=0.0,
        )
        
        # 找最近的绿峰（已结束的）
        last_gp = None
        for gp in reversed(green_peaks):
            if gp.end_idx <= i:
                last_gp = gp
                break
        
        # 找当前红峰（如果有的话）
        current_rp = None
        for rp in red_peaks:
            if rp.start_idx <= i < rp.end_idx + 1:
                current_rp = rp
                break
        
        # 入场信号判断
        if last_gp and last_gp.end_idx <= i and (i - last_gp.end_idx) <= 5:
            # 绿峰刚结束（5天内），红柱开始出现
            if macd_bar[i] > 0 and dif[i] < 0:  # 零轴下方翻红
                sig.signal_type = "entry_candidate"
                sig.green_peak_area = last_gp.area
                sig.green_peak_severity = last_gp.severity
                sig.red_peak_expanding = current_rp.is_expanding if current_rp else False
                sig.dif_turning_up = dif[i] > dif[i-1] if i > 0 else False
                sig.price_position = calc_price_position(close, i, lookback)
                
                # 信号强度计算
                strength = 0.0
                if last_gp.severity == "deep":
                    strength += 0.4
                elif last_gp.severity == "moderate":
                    strength += 0.2
                
                if sig.red_peak_expanding:
                    strength += 0.3
                if sig.dif_turning_up:
                    strength += 0.2
                
                # 价格位置加成：越接近底部越强
                if sig.price_position < 0.3:
                    strength += 0.1
                
                sig.signal_strength = min(1.0, strength)
                
                sig.description = (
                    f"绿峰面积{last_gp.area:.1f}({last_gp.severity}) "
                    f"DIF={dif[i]:.3f} 红柱{'放大' if sig.red_peak_expanding else '未放大'} "
                    f"价格位置{sig.price_position:.0%} "
                    f"强度{sig.signal_strength:.0%}"
                )
        
        # 出场预警
        elif current_rp and i > current_rp.start_idx + 2:
            # 红柱持续中，检查是否开始缩短
            if macd_bar[i] < macd_bar[i-1] and macd_bar[i-1] < macd_bar[i-2]:
                sig.signal_type = "exit_warning"
                sig.red_peak_shrinking = True
                sig.signal_strength = 0.5
                sig.description = f"红柱连续缩短（动能衰竭），考虑减仓"
        
        signals.append(sig)
    
    return signals
