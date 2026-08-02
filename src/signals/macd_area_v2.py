#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MACD面积分析器 v2 — 周K趋势 + 多绿峰累积

核心改进（vs v1）：
1. 周K趋势判断（大方向：下跌/横盘/上涨）
2. 日K多绿峰累积面积（不只看最后一个，看卖压逐步衰竭过程）
3. 绿峰递减模式（后一个比前一个小=衰竭信号）
4. 入场信号综合考虑：周K方向 + 绿峰累积消耗 + 红柱刚开始积累
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional


def calc_macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    """计算MACD"""
    if len(close) < slow + signal:
        return np.zeros(len(close)), np.zeros(len(close)), np.zeros(len(close))
    
    ema_fast = np.zeros(len(close))
    ema_slow = np.zeros(len(close))
    ema_fast[:fast] = close[0]
    ema_slow[:slow] = close[0]
    
    k_fast = 2 / (fast + 1)
    k_slow = 2 / (slow + 1)
    
    for i in range(1, len(close)):
        ema_fast[i] = ema_fast[i-1] * (1 - k_fast) + close[i] * k_fast
        ema_slow[i] = ema_slow[i-1] * (1 - k_slow) + close[i] * k_slow
    
    dif = ema_fast - ema_slow
    dea = np.zeros(len(close))
    k_signal = 2 / (signal + 1)
    dea[0] = dif[0]
    for i in range(1, len(close)):
        dea[i] = dea[i-1] * (1 - k_signal) + dif[i] * k_signal
    
    macd_bar = 2 * (dif - dea)
    return dif, dea, macd_bar


@dataclass
class GreenPeak:
    """日K绿峰"""
    start_idx: int
    end_idx: int
    area: float
    min_dif: float
    start_price: float
    end_price: float
    price_drop_pct: float
    severity: str  # shallow/moderate/deep


@dataclass 
class WeeklyTrend:
    """周K趋势"""
    direction: str  # "down" / "flat" / "up"
    strength: float  # 0-1
    dif_weekly: float
    macd_bar_weekly: float
    consecutive_green: int  # 周K绿柱连续根数
    green_area_weekly: float  # 周K绿峰面积


@dataclass
class PriorTrade:
    """历史交易记录（用于失败后重建的权重提升）"""
    entry_date: str
    exit_date: str
    pnl_pct: float
    was_loss: bool


@dataclass
class MultiPeakAnalysis:
    """多绿峰累积分析"""
    peaks: List[GreenPeak] = field(default_factory=list)
    total_area: float = 0.0  # 累积面积
    is_decreasing: bool = False  # 绿峰递减（衰竭信号）
    decrease_ratio: float = 0.0  # 递减比例
    peak_count: int = 0
    last_peak: Optional[GreenPeak] = None
    
    # 入场判断
    exhaustion_level: float = 0.0  # 卖压耗尽程度 0-1
    
    # 历史交易记录
    prior_losses: List[PriorTrade] = field(default_factory=list)
    had_prior_loss: bool = False  # 是否有过亏损交易
    full_cycle_after_loss: bool = False  # 亏损后是否经历了完整绿峰


@dataclass
class SignalV2:
    """v2信号"""
    date: str
    price: float
    signal_type: str  # "entry" / "wait" / "avoid"
    signal_strength: float  # 0-1
    
    # 周K
    weekly_direction: str
    weekly_strength: float
    
    # 多绿峰
    green_peak_count: int
    total_green_area: float
    is_decreasing: bool
    exhaustion_level: float
    
    # 红柱
    red_bar: float
    red_accumulation: float  # 当前红柱累积
    
    # 价格位置
    price_position: float
    
    description: str = ""


def find_all_green_peaks(macd_bar: np.ndarray, dif: np.ndarray, 
                         dates: np.ndarray, close: np.ndarray) -> List[GreenPeak]:
    """找到所有绿峰（不只是最后一个）"""
    peaks = []
    in_green = False
    start_idx = 0
    
    for i in range(len(macd_bar)):
        if macd_bar[i] < 0 and not in_green:
            in_green = True
            start_idx = i
        elif macd_bar[i] >= 0 and in_green:
            in_green = False
            if i - start_idx >= 3:  # 至少3根才算绿峰
                area = abs(sum(macd_bar[start_idx:i]))
                min_dif = min(dif[start_idx:i])
                start_price = close[start_idx]
                end_price = close[i-1]
                price_drop = (end_price / start_price - 1) * 100 if start_price > 0 else 0
                
                if area >= 1.0:
                    if area >= 15:
                        severity = "deep"
                    elif area >= 5:
                        severity = "moderate"
                    else:
                        severity = "shallow"
                    
                    peaks.append(GreenPeak(
                        start_idx=start_idx, end_idx=i-1,
                        area=round(area, 2),
                        min_dif=round(min_dif, 4),
                        start_price=round(start_price, 2),
                        end_price=round(end_price, 2),
                        price_drop_pct=round(price_drop, 1),
                        severity=severity,
                    ))
    
    return peaks


def analyze_multi_peak(peaks: List[GreenPeak]) -> MultiPeakAnalysis:
    """分析多绿峰累积模式"""
    analysis = MultiPeakAnalysis()
    analysis.peaks = peaks[-4:]  # 最近4个绿峰
    analysis.peak_count = len(peaks)
    analysis.last_peak = peaks[-1] if peaks else None
    analysis.total_area = sum(p.area for p in peaks[-4:])  # 最近4个累积
    
    if len(peaks) >= 2:
        recent = peaks[-4:] if len(peaks) >= 4 else peaks
        # 判断递减：后面的绿峰比前面的小
        areas = [p.area for p in recent]
        if len(areas) >= 2:
            later_half = areas[len(areas)//2:]
            earlier_half = areas[:len(areas)//2]
            avg_later = np.mean(later_half) if later_half else 0
            avg_earlier = np.mean(earlier_half) if earlier_half else 1
            
            analysis.is_decreasing = avg_later < avg_earlier
            analysis.decrease_ratio = avg_later / avg_earlier if avg_earlier > 0 else 1
    
    # 卖压耗尽程度（动态判断）
    # 因素1：绿峰累积面积（最重要）
    # 单个深绿峰(≥10) = 充分消耗
    # 多个中等绿峰累积≥15 = 充分消耗
    area_factor = min(analysis.total_area / 15, 1.0)
    
    # 因素2：绿峰数量
    count_factor = min(len(peaks[-4:]) / 2, 1.0) if peaks else 0
    
    # 因素3：递减模式（加分项不是必须）
    decrease_bonus = (1.0 - analysis.decrease_ratio) * 0.5 if analysis.is_decreasing else 0
    
    # 综合：面积为主，数量为辅，递减加分
    analysis.exhaustion_level = min(1.0, area_factor * 0.6 + count_factor * 0.2 + decrease_bonus * 0.2)
    
    return analysis


def calc_weekly_trend(daily_df: pd.DataFrame) -> WeeklyTrend:
    """从日线数据计算周K趋势"""
    df = daily_df.copy()
    df['trade_date'] = df['trade_date'].astype(str)
    df['week'] = df['trade_date'].str[:4] + df['trade_date'].str[4:6]
    
    # 按周聚合
    weekly = df.groupby('week').agg(
        open=('open', 'first'),
        close=('close', 'last'),
        high=('high', 'max'),
        low=('low', 'min'),
    )
    
    if len(weekly) < 10:
        return WeeklyTrend("flat", 0.3, 0, 0, 0, 0)
    
    wclose = weekly['close'].values
    wdif, wdea, wmacd = calc_macd(wclose)
    
    last_dif = wdif[-1]
    last_bar = wmacd[-1]
    
    # 连续绿柱根数
    consecutive_green = 0
    for b in reversed(wmacd):
        if b < 0:
            consecutive_green += 1
        else:
            break
    
    # 周K绿峰面积
    green_area = 0
    for b in reversed(wmacd):
        if b < 0:
            green_area += abs(b)
        else:
            break
    
    # 方向判断
    if last_bar < 0 and consecutive_green >= 3:
        direction = "down"
        strength = min(consecutive_green / 8, 1.0)
    elif last_bar < 0 and consecutive_green < 3:
        direction = "down" if last_dif < 0 else "flat"
        strength = 0.4
    elif last_bar > 0 and last_dif < 0:
        direction = "flat"  # 红柱但DIF在零轴下方=筑底
        strength = 0.5
    elif last_bar > 0 and last_dif > 0:
        direction = "up"
        strength = 0.7
    else:
        direction = "flat"
        strength = 0.3
    
    return WeeklyTrend(
        direction=direction,
        strength=round(strength, 2),
        dif_weekly=round(last_dif, 4),
        macd_bar_weekly=round(last_bar, 4),
        consecutive_green=consecutive_green,
        green_area_weekly=round(green_area, 2),
    )


def calc_price_position(close: np.ndarray, idx: int, lookback: int = 120) -> float:
    """价格位置（0=最低，1=最高）"""
    start = max(0, idx - lookback)
    if idx <= start:
        return 0.5
    window = close[start:idx+1]
    if len(window) < 2:
        return 0.5
    high = np.max(window)
    low = np.min(window)
    if high == low:
        return 0.5
    return float((close[idx] - low) / (high - low))


def generate_signals_v2(daily_df: pd.DataFrame, prior_trades: list = None) -> List[SignalV2]:
    """生成v2信号（周K+多绿峰+失败重建）
    
    Args:
        prior_trades: 历史交易记录，格式 [{'entry_date': str, 'exit_date': str, 'pnl_pct': float}]
    """
    df = daily_df.sort_values('trade_date').reset_index(drop=True)
    close = df['close'].values
    dates = df['trade_date'].values
    
    if len(close) < 60:
        return []
    
    dif, dea, macd_bar = calc_macd(close)
    
    # 周K趋势
    weekly = calc_weekly_trend(df)
    
    # 所有绿峰
    all_peaks = find_all_green_peaks(macd_bar, dif, dates, close)
    
    # 多绿峰分析
    multi_peak = analyze_multi_peak(all_peaks)
    
    # 分析历史交易：是否有过亏损，亏损后是否经历了完整绿峰
    if prior_trades:
        for pt in prior_trades:
            if pt.get('pnl_pct', 0) < 0:  # 亏损交易
                exit_date = pt.get('exit_date', '')
                # 找亏损出场后的绿峰
                try:
                    exit_idx = list(dates).index(exit_date)
                    # 找出场后形成的绿峰（面积≥5）
                    for p in all_peaks:
                        if p.start_idx >= exit_idx and p.area >= 5:
                            multi_peak.had_prior_loss = True
                            multi_peak.full_cycle_after_loss = True
                            multi_peak.prior_losses.append(PriorTrade(
                                entry_date=pt.get('entry_date', ''),
                                exit_date=exit_date,
                                pnl_pct=pt.get('pnl_pct', 0),
                                was_loss=True,
                            ))
                            break
                except ValueError:
                    pass
    
    signals = []
    
    # 在最近60天生成信号（足够覆盖一个月的回测窗口）
    start_i = max(60, len(close) - 60)
    
    for i in range(start_i, len(close)):
        # 找截至i为止的最后一个绿峰
        peaks_up_to_i = [p for p in all_peaks if p.end_idx < i]
        if not peaks_up_to_i:
            continue
        
        last_peak = peaks_up_to_i[-1]
        days_after_peak = i - last_peak.end_idx
        
        # 红柱累积（只算最近5根，不是从翻红第一天）
        red_accum = 0
        for j in range(i, max(i-5, 0), -1):
            if macd_bar[j] > 0:
                red_accum += macd_bar[j]
            else:
                break
        
        # 当前单根红柱的大小（判断是否在加速）
        current_red_bar = macd_bar[i] if macd_bar[i] > 0 else 0
        prev_red_bar = macd_bar[i-1] if i > 0 and macd_bar[i-1] > 0 else 0
        red_accelerating = current_red_bar > prev_red_bar
        red_ratio = red_accum / max(abs(last_peak.area), 0.01)
        
        price_pos = calc_price_position(close, i)
        
        # ===== 入场判断（评分制，非硬阈值）=====
        # 不设硬门槛，而是综合打分，分数>0.5入场
        score = 0.0
        reasons = []
        
        # 1. 周K趋势（0-0.25分）
        if weekly.direction == "up":
            score += 0.25
            reasons.append("周K上涨")
        elif weekly.direction == "flat":
            score += 0.15
            reasons.append("周K横盘")
        elif weekly.direction == "down" and multi_peak.exhaustion_level > 0.7:
            score += 0.05  # 极度超跌时周K下跌也能小仓位
            reasons.append("周K下跌但极度超跌")
        else:
            reasons.append(f"周K{weekly.direction}")
        
        # 2. 卖压耗尽度（0-0.30分）
        score += multi_peak.exhaustion_level * 0.30
        
        # 2.5 失败后重建加分（0-0.15分）
        # 如果之前亏损过，然后经历了一个完整绿峰（面积≥5）后再次翻红
        # 说明卖压又充分释放了一轮，这次翻红的可信度更高
        if multi_peak.full_cycle_after_loss:
            score += 0.15
            reasons.append("失败后完整绿峰重建✅")
        
        # 3. 入场时机（0-0.20分）
        # 绿峰结束后越早越好
        if days_after_peak <= 2:
            score += 0.20
        elif days_after_peak <= 5:
            score += 0.10
        elif days_after_peak <= 10:
            score += 0.05
        
        # 4. 红柱刚开始（0-0.15分）
        if red_ratio < 0.10:
            score += 0.15
        elif red_ratio < 0.20:
            score += 0.10
        elif red_ratio < 0.30:
            score += 0.05
        
        # 5. DIF在零轴下方（0.10分）
        if dif[i] < 0:
            score += 0.10
        
        # 6. MACD翻红（必须）
        if macd_bar[i] > 0:
            score += 0.0  # 不加分但是必须条件
        else:
            score = 0  # 没翻红直接清零
        
        # 判断
        if score >= 0.50:
            signal_type = "entry"
        else:
            signal_type = "wait"
        
        strength = min(1.0, score)
        
        desc = (
            f"周K:{weekly.direction}({weekly.strength:.0%}) "
            f"绿峰{multi_peak.peak_count}个累积{multi_peak.total_area:.1f} "
            f"{'递减✅' if multi_peak.is_decreasing else '未递减'} "
            f"耗尽{multi_peak.exhaustion_level:.0%} "
            f"红柱{red_accum:.2f}({red_ratio:.0%}) "
            f"位置{price_pos:.0%} "
            f"→{signal_type}({strength:.0%})"
        )
        
        signals.append(SignalV2(
            date=str(dates[i]),
            price=round(close[i], 2),
            signal_type=signal_type,
            signal_strength=round(strength, 2),
            weekly_direction=weekly.direction,
            weekly_strength=weekly.strength,
            green_peak_count=multi_peak.peak_count,
            total_green_area=round(multi_peak.total_area, 1),
            is_decreasing=multi_peak.is_decreasing,
            exhaustion_level=round(multi_peak.exhaustion_level, 2),
            red_bar=round(macd_bar[i], 4),
            red_accumulation=round(red_accum, 2),
            price_position=round(price_pos, 2),
            description=desc,
        ))
    
    return signals
