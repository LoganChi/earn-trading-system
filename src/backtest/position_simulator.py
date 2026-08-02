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
    daily_df: pd.DataFrame,
    minute_df: pd.DataFrame,
    allocation: float = 0.08,
    signal_strength: float = 0.5,
    market_df: pd.DataFrame = None,
    stop_loss_pct: float = -8.0,
    take_profit_pct: float = 20.0,
) -> Optional[Position]:
    """模拟一个持仓周期（分时级别入场+出场）
    
    流程：
    1. entry_date是日K信号日（当天收盘后确认信号）
    2. 第二天盘中找买入时机（开盘后等回调，不追高）
    3. 买入后逐日检查出场条件（分时级别精确到分钟）
    4. 没有固定持仓天数
    """
    df = daily_df.sort_values('trade_date').reset_index(drop=True)
    
    # 找信号日在数据中的位置
    entry_rows = df[df['trade_date'].astype(str) == entry_date]
    if len(entry_rows) == 0:
        return None
    signal_idx = entry_rows.index[0]
    
    # 第二天才是实际买入日
    if signal_idx + 1 >= len(df):
        return None
    buy_idx = signal_idx + 1
    buy_date = str(df.iloc[buy_idx]['trade_date'])
    
    # 分时级别找买入时机：MACD与价格协同确认
    entry_price, entry_time = _find_entry_time_minute(code, buy_date, minute_df, df.iloc[signal_idx]['close'])
    if entry_price is None:
        # 全天没有MACD+价格协同信号 → 不入场（放弃这个信号）
        return None
    
    pos = Position(
        code=code, name=name,
        entry_date=buy_date, entry_price=entry_price,
        allocation=allocation, signal_strength=signal_strength,
        peak_price=entry_price,
        tier='低价' if entry_price < 10 else ('中价' if entry_price < 30 else '高价'),
    )
    
    close_all = df['close'].values
    dif, dea, macd_bar = _calc_macd(close_all)
    
    red_shrink_days = 0
    had_red_peak = False
    
    # 从买入日开始逐日检查
    for i in range(buy_idx, len(df)):
        row = df.iloc[i]
        current_date = str(row['trade_date'])
        current_close = row['close']
        current_high = row['high']
        current_low = row['low']
        
        if i > buy_idx:
            pos.holding_days += 1
        
        pos.peak_price = max(pos.peak_price, current_high)
        pos.peak_profit = max(pos.peak_profit, (pos.peak_price / entry_price - 1) * 100)
        
        # === 分时级别出场检查 ===
        # 每天用分钟数据找精确出场点
        exit_result = _check_intraday_exit(
            code, current_date, entry_price, minute_df,
            stop_loss_pct, take_profit_pct,
            pos.peak_profit if pos.holding_days > 0 else 0,
        )
        
        if exit_result:
            exit_time, exit_price, exit_reason = exit_result
            pos.exit_date = current_date
            pos.exit_price = exit_price
            pos.exit_reason = exit_reason
            pos.pnl_pct = (exit_price / entry_price - 1) * 100
            pos.is_open = False
            pos.exit_time = exit_time
            pos.holding_minutes = pos.holding_days * 240 + _time_to_minutes(exit_time)
            return pos
        
        # === 日K级别补充检查（盘后确认）===
        profit = (current_close / entry_price - 1) * 100
        
        # 日K动能衰竭
        if macd_bar[i] > 0:
            had_red_peak = True
            if i > 0 and macd_bar[i] < macd_bar[i-1]:
                red_shrink_days += 1
            else:
                red_shrink_days = 0
        else:
            if had_red_peak and red_shrink_days >= 3:
                pos.exit_date = current_date
                pos.exit_price = current_close
                pos.exit_reason = "日K动能衰竭"
                pos.pnl_pct = profit
                pos.is_open = False
                pos.exit_time = f"{current_date} 15:00"
                pos.holding_minutes = pos.holding_days * 240
                return pos
            red_shrink_days = 0
        
        # 高开回落
        if i > buy_idx:
            prev_close = df.iloc[i-1]['close']
            open_pct = (row['open'] / prev_close - 1) * 100
            range_pct = (current_high - current_low) / current_close * 100
            close_position = (current_close - current_low) / (current_high - current_low) if current_high > current_low else 0.5
            if open_pct > 3 and close_position < 0.2 and profit > 0:
                pos.exit_date = current_date
                pos.exit_price = current_close
                pos.exit_reason = "高开回落"
                pos.pnl_pct = profit
                pos.is_open = False
                pos.exit_time = f"{current_date} 15:00"
                pos.holding_minutes = pos.holding_days * 240
                return pos
    
    # 期末平仓
    last_row = df.iloc[-1]
    pos.exit_date = str(last_row['trade_date'])
    pos.exit_price = last_row['close']
    pos.exit_reason = "期末平仓"
    pos.pnl_pct = (pos.exit_price / entry_price - 1) * 100
    pos.is_open = False
    pos.exit_time = f"{pos.exit_date} 15:00"
    pos.holding_minutes = pos.holding_days * 240
    
    return pos


def _find_entry_time_minute(code, date_str, minute_df, signal_close):
    """分时级别找买入时机：MACD与价格协同确认
    
    核心逻辑：
    不是"看到回调就买"，而是等MACD红柱放大+价格同步突破的时刻
    
    判断条件（同时满足）：
    1. 分时MACD柱 > 0（红柱区间）
    2. 分时MACD柱在放大（当前 > 前一根）
    3. 价格突破近期高点（前面N根的最高价）
    4. 如果MACD红柱放大但价格没突破→跳过（假信号）
    
    如果全天没有协同信号→不入场（返回None）
    """
    if minute_df is None or minute_df.empty:
        return None, None
    
    day_data = _filter_minute_by_date(minute_df, date_str)
    if day_data is None or len(day_data) < 30:
        return None, None
    
    prices = day_data['close'].values
    times = day_data['trade_time'].values if 'trade_time' in day_data.columns else None
    
    # 计算分时MACD
    dif_m, dea_m, macd_m = _calc_macd(prices, fast=5, slow=13, signal=4)
    
    # 追踪窗口高点（用前20根作为参照）
    window = 20
    
    for i in range(max(window, 5), len(prices)):
        # 条件1: MACD柱为正
        if macd_m[i] <= 0:
            continue
        
        # 条件2: MACD柱在放大
        if macd_m[i] <= macd_m[i-1]:
            continue
        
        # 条件3: 价格突破窗口高点
        window_high = np.max(prices[i-window:i])
        if prices[i] <= window_high:
            continue
        
        # 条件4: 价格突破幅度≥0.3%（不是一分钱的波动）
        breakout_pct = (prices[i] / window_high - 1) * 100
        if breakout_pct < 0.3:
            continue
        
        # 协同确认：MACD红柱放大 + 价格突破 = 买入
        time_str = str(times[i]) if times is not None else "10:00"
        # 从"2026-01-06 09:30:00"提取"09:30"
        if ' ' in time_str and ':' in time_str:
            t = time_str.split(' ')[-1][:5]  # 取"09:30:00"的前5位"09:30"
        elif len(time_str) >= 5:
            t = time_str[-5:]
        else:
            t = "10:00"
        
        return prices[i], f"{date_str} {t}"


def _check_intraday_exit(code, date_str, entry_price, minute_df, stop_loss, take_profit, peak_profit):
    """分时级别出场检查
    
    返回 (exit_time, exit_price, exit_reason) 或 None
    每天的分钟数据逐根检查
    """
    if minute_df is None or minute_df.empty:
        return None
    
    day_data = _filter_minute_by_date(minute_df, date_str)
    if day_data is None or len(day_data) < 20:
        return None
    
    prices = day_data['close'].values
    times = day_data['trade_time'].values if 'trade_time' in day_data.columns else None
    
    # 计算分时MACD
    dif_m, dea_m, macd_m = _calc_macd(prices, fast=5, slow=13, signal=4)
    
    day_peak = 0  # 当天峰值
    
    for i in range(5, len(prices)):
        price = prices[i]
        profit = (price / entry_price - 1) * 100
        day_peak = max(day_peak, profit)
        
        time_str = str(times[i]) if times is not None else "15:00"
        t = time_str[-5:] if len(str(time_str)) >= 5 else "15:00"
        
        # 1. 止损
        if profit <= stop_loss:
            return f"{date_str} {t}", price, "止损"
        
        # 2. 止盈
        if profit >= take_profit:
            return f"{date_str} {t}", price, "目标止盈"
        
        # 3. 分时红柱拐头（核心alpha）
        if i >= 2 and macd_m[i] > 0:
            if macd_m[i] < macd_m[i-1] and macd_m[i-1] < macd_m[i-2]:
                # 红柱连续缩短2根 + 有浮盈
                if profit >= 5.0 and day_peak >= profit + 1:
                    return f"{date_str} {t}", price, "分时红柱拐头"
        
        # 4. 盘中冲高回落（利润回吐超过3%）
        if day_peak >= 8.0 and profit < day_peak - 3.0:
            return f"{date_str} {t}", price, "盘中冲高回落"
    
    return None


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
    """从分钟数据中筛选某一天（兼容多种日期格式）"""
    if 'trade_time' not in minute_df.columns:
        return None
    # date_str可能是20260106或2026-01-06，统一处理
    clean_date = date_str.replace('-', '')
    # 把trade_time转成和clean_date一样的格式
    times = minute_df['trade_time'].astype(str)
    # trade_time可能是2026-01-06 09:30:00或20260106 09:30:00
    mask = times.str.replace('-', '').str.startswith(clean_date)
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
