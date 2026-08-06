#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""红峰二波段识别器

核心逻辑：
MACD翻红后涨一波→回调（红柱缩短但不转绿）→红柱再次放大+价格突破=第二波入场

与绿峰反转器的区别：
- 绿峰反转：看卖压消耗（深跌后底部翻红）
- 红峰二波段：看买盘延续（上涨中继回调后再涨）

识别条件（同时满足）：
1. 当前处于红柱区间（MACD柱>0）
2. 前面有一个完整的红峰（面积≥3）
3. 红峰中间有过回调（红柱连续缩短≥3天，但未转绿）
4. 红柱再次放大（当前柱>前一根）
5. 价格突破回调前高（或者接近）
6. DIF在零轴上方（趋势向上确认）

评分维度：
- 第一波涨幅（越大=资金越强势）
- 回调深度（越浅=控盘越强）
- 回调天数（适中最好3-7天）
- 红柱再放大力度（加速度）
- 量能配合（回调缩量+再放大放量）
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class RedPeak:
    """红峰信息"""
    start_idx: int
    end_idx: int
    area: float          # 红柱累积面积
    duration: int        # 持续天数
    max_bar: float       # 最大红柱值
    price_gain: float    # 期间涨幅%
    has_pullback: bool   # 是否包含回调
    pullback_start: int  # 回调起始位置
    pullback_days: int   # 回调天数
    pullback_depth: float  # 回调深度%（价格从高点回落%）
    pullback_bar_shrink: float  # 红柱从峰值缩小的比例
    is_second_wave: bool  # 是否当前处于二波段


@dataclass
class SecondWaveSignal:
    """二波段信号"""
    is_signal: bool = False
    score: float = 0       # 0-100
    first_wave_gain: float = 0  # 第一波涨幅
    pullback_depth: float = 0   # 回调深度
    pullback_days: int = 0      # 回调天数
    red_bar_accel: float = 0    # 红柱加速度
    volume_shrink: float = 0    # 量能收缩比
    dif_pos: bool = False       # DIF零轴上方
    description: str = ""
    
    # 买卖建议
    entry_price: float = 0      # 建议入场价（回调前高突破价）
    stop_loss: float = 0        # 止损价（回调最低点下方）
    target: float = 0           # 目标价（第一波涨幅等比）
    risk_reward: float = 0      # 风险回报比


def detect_second_wave(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    macd_bar: np.ndarray,
    dif: np.ndarray,
    lookback: int = 40
) -> SecondWaveSignal:
    """检测红峰二波段信号
    
    Args:
        close: 收盘价
        high: 最高价
        low: 最低价
        volume: 成交量
        macd_bar: MACD柱
        dif: DIF线
        lookback: 回看天数（默认40天）
    
    Returns:
        SecondWaveSignal
    """
    signal = SecondWaveSignal()
    n = len(close)
    if n < lookback + 10:
        return signal
    
    # 只看最近lookback天
    start = n - lookback
    bars = macd_bar[start:]
    prices = close[start:]
    highs = high[start:]
    lows = low[start:]
    vols = volume[start:]
    difs = dif[start:]
    
    # 条件1：当前必须是红柱
    if macd_bar[-1] <= 0:
        signal.description = "当前绿柱，不满足二波段"
        return signal
    
    # 条件2：前面有一个完整红峰（从翻红开始）
    # 找当前红柱区间的起点
    red_start = len(bars) - 1
    while red_start > 0 and bars[red_start - 1] > 0:
        red_start -= 1
    
    if red_start < 10:  # 红柱区间太短
        signal.description = f"红柱区间仅{len(bars)-red_start}天，太短"
        return signal
    
    # 分析红柱区间内的结构
    red_bars = bars[red_start:]
    red_prices = prices[red_start:]
    red_highs = highs[red_start:]
    red_lows = lows[red_start:]
    red_vols = vols[red_start:]
    
    # 找红柱峰值（最大红柱）
    peak_idx = np.argmax(red_bars)
    peak_bar = red_bars[peak_idx]
    peak_price = red_highs[peak_idx]  # 峰值时的高点
    
    # 第一波涨幅（从红柱起点到峰值）
    first_wave_gain = (red_highs[peak_idx] / red_prices[0] - 1) * 100 if red_prices[0] > 0 else 0
    
    # 条件3：峰值后必须有回调（红柱连续缩短≥3天）
    if peak_idx >= len(red_bars) - 2:
        signal.description = "红柱还在加速放大，没有回调"
        return signal
    
    # 检测回调
    after_peak = red_bars[peak_idx:]  # 峰值后的红柱
    shrink_days = 0
    for i in range(1, len(after_peak)):
        if after_peak[i] < after_peak[i - 1]:
            shrink_days += 1
        else:
            break  # 缩短结束
    
    if shrink_days < 2:
        signal.description = f"回调仅{shrink_days}天，太短"
        return signal
    
    # 回调最低点（价格）
    pullback_lows = red_lows[peak_idx:peak_idx + shrink_days + 1]
    pullback_low_price = min(pullback_lows) if len(pullback_lows) > 0 else red_prices[peak_idx]
    
    # 回调深度
    pullback_depth = (peak_price / pullback_low_price - 1) * 100 if pullback_low_price > 0 else 0
    
    # 红柱缩小比例
    min_bar_after_peak = min(after_peak[:shrink_days + 1]) if shrink_days > 0 else peak_bar
    bar_shrink_ratio = (1 - min_bar_after_peak / peak_bar) * 100 if peak_bar > 0 else 0
    
    # 条件4：回调结束后红柱再次放大
    # 找回调结束后的位置
    rebound_start = peak_idx + shrink_days
    if rebound_start >= len(red_bars):
        signal.description = "回调刚结束，尚未再放大"
        return signal
    
    rebound_bars = red_bars[rebound_start:]
    if len(rebound_bars) < 2:
        signal.description = "再放大天数不足"
        return signal
    
    # 红柱是否在放大
    is_accelerating = rebound_bars[-1] > rebound_bars[-2] if len(rebound_bars) >= 2 else False
    if not is_accelerating:
        signal.description = "红柱尚未再次放大"
        return signal
    
    # 红柱加速度
    red_bar_accel = rebound_bars[-1] - rebound_bars[-2]
    
    # 条件5：DIF在零轴上方
    dif_above_zero = difs[-1] > 0
    
    # 条件6：量能配合（回调缩量 + 再放大放量）
    if len(red_vols) > rebound_start + 1 and rebound_start > peak_idx:
        pullback_vols = red_vols[peak_idx:rebound_start]
        rebound_vols = red_vols[rebound_start:]
        avg_pullback_vol = np.mean(pullback_vols) if len(pullback_vols) > 0 else 1
        avg_rebound_vol = np.mean(rebound_vols) if len(rebound_vols) > 0 else 1
        avg_peak_vol = np.mean(red_vols[max(0,peak_idx-3):peak_idx+1]) if peak_idx > 0 else 1
        
        # 量能收缩比：回调均量/峰值均量（越小=缩量越好）
        volume_shrink = avg_pullback_vol / avg_peak_vol if avg_peak_vol > 0 else 1
    else:
        volume_shrink = 1.0
    
    # ===== 评分 =====
    
    # 1. 第一波涨幅（越大=资金越强势）
    if first_wave_gain >= 20:
        gain_score = 30
    elif first_wave_gain >= 10:
        gain_score = 25
    elif first_wave_gain >= 5:
        gain_score = 20
    else:
        gain_score = 10
    
    # 2. 回调深度（越浅=控盘越强）
    if pullback_depth <= 3:
        depth_score = 25  # 极浅回调=强控盘
    elif pullback_depth <= 5:
        depth_score = 22
    elif pullback_depth <= 8:
        depth_score = 18
    elif pullback_depth <= 12:
        depth_score = 12
    else:
        depth_score = 5   # 回调太深
    
    # 3. 回调天数（3-7天最佳）
    if 3 <= shrink_days <= 7:
        days_score = 15
    elif shrink_days <= 10:
        days_score = 10
    else:
        days_score = 5
    
    # 4. 红柱再放大力度
    if red_bar_accel > 0.05:
        accel_score = 15
    elif red_bar_accel > 0.02:
        accel_score = 12
    elif red_bar_accel > 0:
        accel_score = 8
    else:
        accel_score = 0
    
    # 5. 量能配合
    if volume_shrink < 0.6:
        vol_score = 15  # 大幅缩量=洗盘充分
    elif volume_shrink < 0.8:
        vol_score = 10
    elif volume_shrink < 1.0:
        vol_score = 5
    else:
        vol_score = 0   # 没缩量
    
    # DIF加成
    dif_bonus = 5 if dif_above_zero else 0
    
    total = gain_score + depth_score + days_score + accel_score + vol_score + dif_bonus
    
    # 是否为信号
    is_signal = (
        macd_bar[-1] > 0 and           # 红柱
        is_accelerating and             # 红柱再放大
        shrink_days >= 2 and            # 有过回调
        pullback_depth <= 15 and        # 回调不能太深
        first_wave_gain >= 3 and        # 第一波有涨幅
        total >= 50                     # 评分≥50
    )
    
    signal.is_signal = is_signal
    signal.score = total
    signal.first_wave_gain = round(first_wave_gain, 1)
    signal.pullback_depth = round(pullback_depth, 1)
    signal.pullback_days = shrink_days
    signal.red_bar_accel = round(red_bar_accel, 4)
    signal.volume_shrink = round(volume_shrink, 2)
    signal.dif_pos = dif_above_zero
    
    # 买卖建议
    current_price = close[-1]
    signal.entry_price = round(peak_price, 2)  # 突破回调前高入场
    signal.stop_loss = round(pullback_low_price * 0.97, 2)  # 回调低点下方3%
    # 目标：第一波涨幅等比延伸
    signal.target = round(current_price * (1 + first_wave_gain / 100), 2)
    risk = (current_price - signal.stop_loss) / current_price * 100 if current_price > 0 else 100
    reward = (signal.target - current_price) / current_price * 100 if current_price > 0 else 0
    signal.risk_reward = round(reward / risk, 1) if risk > 0 else 0
    
    # 描述
    tags = []
    if first_wave_gain >= 15: tags.append('第一波强势')
    if pullback_depth <= 5: tags.append('浅回调')
    if volume_shrink < 0.7: tags.append('缩量洗盘')
    if red_bar_accel > 0.05: tags.append('红柱加速')
    if dif_above_zero: tags.append('DIF零轴上方')
    signal.description = ' | '.join(tags) if tags else '二波段信号'
    
    return signal
