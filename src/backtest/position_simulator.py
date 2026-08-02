#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""持仓周期模拟器

核心逻辑（还原用户真实操作）：
1. 一只票建仓一次（不重复买入）
2. 每天收盘后检查：该不该出场？
   - 止损：跌破入场价-8%→走
   - 止盈：涨到+15%~20%→看连板概率/板块决定
   - 动能衰竭：日K MACD红柱连续缩短3天→减仓信号
   - 高开回落：当天高开后收在最低附近→走
3. 没有固定持仓天数
4. 出场后这个周期结束，不会重新买入
5. 分时级别：用分钟数据判断盘中红柱拐头的精确出场点
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class Position:
    """一个持仓周期"""
    code: str
    name: str
    entry_date: str
    entry_price: float
    allocation: float  # 资金比例
    
    # 状态
    is_open: bool = True
    exit_date: str = ""
    exit_price: float = 0.0
    exit_reason: str = ""
    exit_time: str = ""  # 精确到分钟
    
    # 持仓追踪
    peak_price: float = 0.0  # 持仓期间最高价
    peak_profit: float = 0.0  # 最大浮盈
    holding_days: int = 0
    holding_minutes: int = 0
    
    # 减仓记录（支持分批减仓）
    reductions: List[Dict] = field(default_factory=list)
    
    # 最终结果
    pnl_pct: float = 0.0  # 最终收益率
    
    # 信号
    signal_strength: float = 0.0
    tier: str = ""  # 低价/中价/高价


def simulate_position(
    code: str,
    name: str,
    entry_date: str,
    entry_price: float,
    daily_df: pd.DataFrame,
    minute_df: pd.DataFrame,
    allocation: float = 0.08,
    signal_strength: float = 0.5,
    market_df: pd.DataFrame = None,
    stop_loss_pct: float = -8.0,
    take_profit_pct: float = 20.0,
) -> Optional[Position]:
    """模拟一个持仓周期
    
    从entry_date开始，每天检查是否该出场。
    出场条件（每天收盘检查+盘中分钟检查）：
    1. 止损：收盘价跌破入场价-8%
    2. 止盈：浮盈达到+20%
    3. 动能衰竭：日K MACD红柱连续缩短3天（需要先有红柱）
    4. 分时红柱拐头：盘中分时MACD红柱缩短→精确出场点
    
    没有固定持仓天数限制。
    """
    df = daily_df.sort_values('trade_date').reset_index(drop=True)
    
    # 找入场日在数据中的位置
    entry_rows = df[df['trade_date'].astype(str) == entry_date]
    if len(entry_rows) == 0:
        return None
    entry_idx = entry_rows.index[0]
    
    pos = Position(
        code=code, name=name,
        entry_date=entry_date, entry_price=entry_price,
        allocation=allocation, signal_strength=signal_strength,
        peak_price=entry_price,
        tier='低价' if entry_price < 10 else ('中价' if entry_price < 30 else '高价'),
    )
    
    # 计算日K MACD（用入场前60天+持仓期）
    close_all = df['close'].values
    high_all = df['high'].values
    low_all = df['low'].values
    
    # 简化MACD计算
    dif, dea, macd_bar = _calc_macd(close_all)
    
    # 追踪日K红柱缩短
    red_shrink_days = 0  # 连续缩短天数
    had_red_peak = False  # 是否出现过红柱峰值
    
    # 从入场日次日开始逐日检查
    for i in range(entry_idx + 1, len(df)):
        row = df.iloc[i]
        current_date = str(row['trade_date'])
        current_close = row['close']
        current_high = row['high']
        current_low = row['low']
        
        pos.holding_days += 1
        pos.peak_price = max(pos.peak_price, current_high)
        pos.peak_profit = max(pos.peak_profit, (pos.peak_price / entry_price - 1) * 100)
        
        profit = (current_close / entry_price - 1) * 100
        
        # === 每日收盘检查 ===
        
        # 1. 止损
        if profit <= stop_loss_pct:
            pos.exit_date = current_date
            pos.exit_price = current_close
            pos.exit_reason = "止损"
            pos.pnl_pct = profit
            pos.is_open = False
            
            # 尝试用分钟数据找精确出场时间
            exit_time = _find_exit_time_minute(code, current_date, entry_price, stop_loss_pct, minute_df)
            pos.exit_time = exit_time or f"{current_date} 15:00"
            pos.holding_minutes = pos.holding_days * 240
            return pos
        
        # 2. 止盈
        if profit >= take_profit_pct:
            pos.exit_date = current_date
            pos.exit_price = current_close
            pos.exit_reason = "目标止盈"
            pos.pnl_pct = profit
            pos.is_open = False
            exit_time = _find_exit_time_minute(code, current_date, entry_price, take_profit_pct, minute_df, is_profit=True)
            pos.exit_time = exit_time or f"{current_date} 15:00"
            pos.holding_minutes = pos.holding_days * 240
            return pos
        
        # 3. 日K MACD动能衰竭
        if macd_bar[i] > 0:
            had_red_peak = True
            if i > 0 and macd_bar[i] < macd_bar[i-1]:
                red_shrink_days += 1
            else:
                red_shrink_days = 0
        else:
            # 红柱消失（转绿）
            if had_red_peak and red_shrink_days >= 2:
                # 红柱曾经存在，缩短了2天以上，现在转绿→出场
                pos.exit_date = current_date
                pos.exit_price = current_close
                pos.exit_reason = "日K动能衰竭"
                pos.pnl_pct = profit
                pos.is_open = False
                pos.exit_time = f"{current_date} 15:00"
                pos.holding_minutes = pos.holding_days * 240
                return pos
            red_shrink_days = 0
        
        # 4. 分时红柱拐头（只在有浮盈时触发）
        if profit >= 5.0 and had_red_peak:
            exit_time = _check_intraday_macd_turn(
                code, current_date, entry_price, minute_df
            )
            if exit_time:
                # 分时级别找到出场点
                pos.exit_date = current_date
                pos.exit_price = _get_price_at_time(minute_df, current_date, exit_time) or current_close
                pos.exit_reason = "分时红柱拐头"
                pos.pnl_pct = (pos.exit_price / entry_price - 1) * 100
                pos.is_open = False
                pos.exit_time = exit_time
                pos.holding_minutes = pos.holding_days * 240 + _time_to_minutes(exit_time)
                return pos
        
        # 5. 高开回落（当天高开后收在最低附近）
        if i > 0:
            prev_close = df.iloc[i-1]['close']
            open_pct = (row['open'] / prev_close - 1) * 100
            close_vs_high = (current_close - current_high) / (current_high - current_low) * 100 if current_high > current_low else 50
            if open_pct > 3 and close_vs_high < -50 and profit > 0:
                # 高开3%+但收在最低附近+还有浮盈→锁定利润
                pos.exit_date = current_date
                pos.exit_price = current_close
                pos.exit_reason = "高开回落"
                pos.pnl_pct = profit
                pos.is_open = False
                pos.exit_time = f"{current_date} 15:00"
                pos.holding_minutes = pos.holding_days * 240
                return pos
    
    # 如果到数据末尾还没出场→期末平仓
    last_row = df.iloc[-1]
    pos.exit_date = str(last_row['trade_date'])
    pos.exit_price = last_row['close']
    pos.exit_reason = "期末平仓"
    pos.pnl_pct = (pos.exit_price / entry_price - 1) * 100
    pos.is_open = False
    pos.exit_time = f"{pos.exit_date} 15:00"
    pos.holding_minutes = pos.holding_days * 240
    
    return pos


def _calc_macd(close: np.ndarray, fast=12, slow=26, signal=9):
    """计算MACD"""
    if len(close) < slow + signal:
        return np.zeros(len(close)), np.zeros(len(close)), np.zeros(len(close))
    
    ema_f = np.zeros(len(close))
    ema_s = np.zeros(len(close))
    ema_f[:fast] = close[0]
    ema_s[:slow] = close[0]
    
    kf = 2 / (fast + 1)
    ks = 2 / (slow + 1)
    
    for i in range(1, len(close)):
        ema_f[i] = ema_f[i-1] * (1-kf) + close[i] * kf
        ema_s[i] = ema_s[i-1] * (1-ks) + close[i] * ks
    
    dif = ema_f - ema_s
    dea = np.zeros(len(close))
    kd = 2 / (signal + 1)
    dea[0] = dif[0]
    for i in range(1, len(close)):
        dea[i] = dea[i-1] * (1-kd) + dif[i] * kd
    
    macd_bar = 2 * (dif - dea)
    return dif, dea, macd_bar


def _find_exit_time_minute(code, date_str, entry_price, threshold_pct, minute_df, is_profit=False):
    """在分钟数据中找精确的止损/止盈时间"""
    if minute_df is None or minute_df.empty:
        return None
    
    # 筛选当天的分钟数据
    day_data = _filter_minute_by_date(minute_df, date_str)
    if day_data is None or len(day_data) == 0:
        return None
    
    for _, row in day_data.iterrows():
        price = row.get('close', 0)
        if price <= 0:
            continue
        profit = (price / entry_price - 1) * 100
        if is_profit and profit >= threshold_pct:
            return f"{date_str} {row.get('trade_time', '15:00')[-5:]}"
        if not is_profit and profit <= threshold_pct:
            return f"{date_str} {row.get('trade_time', '15:00')[-5:]}"
    
    return None


def _check_intraday_macd_turn(code, date_str, entry_price, minute_df):
    """检查当天分时MACD是否红柱拐头
    
    逻辑：计算分时MACD，如果红柱连续缩短2根+浮盈≥5%→返回出场时间
    """
    if minute_df is None or minute_df.empty:
        return None
    
    day_data = _filter_minute_by_date(minute_df, date_str)
    if day_data is None or len(day_data) < 30:
        return None
    
    prices = day_data['close'].values
    
    # 计算分时MACD（用5/13/4参数，更快响应）
    dif_m, dea_m, macd_m = _calc_macd(prices, fast=5, slow=13, signal=4)
    
    peak_profit = 0
    peak_idx = 0
    
    for i in range(5, len(prices)):
        profit = (prices[i] / entry_price - 1) * 100
        if profit > peak_profit:
            peak_profit = profit
            peak_idx = i
        
        # 红柱拐头检测
        if macd_m[i] > 0 and i >= 2:
            if macd_m[i] < macd_m[i-1] and macd_m[i-1] < macd_m[i-2]:
                # 红柱连续缩短2根
                if profit >= 5.0 and peak_profit >= profit + 1:
                    # 有5%以上浮盈+从峰值回落
                    time_str = day_data.iloc[i].get('trade_time', '15:00')
                    if isinstance(time_str, str) and len(time_str) >= 5:
                        return f"{date_str} {time_str[-5:]}"
                    return f"{date_str} 14:30"
    
    return None


def _filter_minute_by_date(minute_df, date_str):
    """从分钟数据中筛选某一天"""
    if 'trade_time' not in minute_df.columns:
        return None
    mask = minute_df['trade_time'].astype(str).str.startswith(date_str)
    result = minute_df[mask]
    return result if len(result) > 0 else None


def _get_price_at_time(minute_df, date_str, time_str):
    """获取某时间点的价格"""
    day_data = _filter_minute_by_date(minute_df, date_str)
    if day_data is None:
        return None
    # 找最接近time_str的记录
    time_part = time_str.split(' ')[-1] if ' ' in time_str else time_str[-5:]
    for _, row in day_data.iterrows():
        t = str(row.get('trade_time', ''))
        if time_part in t:
            return row.get('close', None)
    return day_data.iloc[-1].get('close', None) if len(day_data) > 0 else None


def _time_to_minutes(time_str):
    """时间字符串转分钟数"""
    try:
        parts = time_str.split(' ')
        if len(parts) >= 2:
            h, m = parts[-1].split(':')[:2]
            return int(h) * 60 + int(m)
    except:
        pass
    return 0
