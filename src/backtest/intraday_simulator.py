#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分时级别战法回测

用分钟K线模拟真实盘中操作：
- 日线MACD面积信号确定入场日
- 入场后切到分钟级别，模拟盘中分时MACD出场
- 分时红柱拐头 = 实时减仓信号（你的真实操作逻辑）

核心：日K定方向，分钟定出场点。
"""
from __future__ import annotations

import sys
import json
import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.signals.macd_area import calc_macd, generate_signals, MACDAreaSignal
from src.data.loader import _init_tushare, _to_ts_code


@dataclass
class IntradayTrade:
    """分时级别交易记录"""
    code: str
    name: str
    entry_date: str         # 日线信号日
    entry_price: float      # 入场价（信号日收盘）
    
    # 分时出场细节
    exit_time: str = ""     # 精确到分钟的出场时间
    exit_price: float = 0.0
    exit_reason: str = ""
    
    # 盘中峰值
    intraday_max_price: float = 0.0
    intraday_max_profit: float = 0.0
    
    # 最终结果
    pnl_pct: float = 0.0
    holding_days: int = 0
    holding_minutes: int = 0  # 持有多少个交易分钟
    signal_strength: float = 0.0
    
    # 分时出场触发记录
    exit_triggers: List[Dict] = field(default_factory=list)
    
    @property
    def profit_giveback(self) -> float:
        """利润回吐比例 = (峰值收益 - 最终收益) / 峰值收益"""
        if self.intraday_max_profit <= 0:
            return 0
        return (self.intraday_max_profit - self.pnl_pct) / self.intraday_max_profit * 100


def calc_intraday_macd(minute_close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    """计算分时MACD（用分钟数据）"""
    if len(minute_close) < slow + signal:
        return None, None, None
    return calc_macd(minute_close, fast, slow, signal)


def load_minute_data(code: str, start_date: str = "20260101", end_date: str = "") -> pd.DataFrame:
    """加载分钟K线数据"""
    if not end_date:
        end_date = datetime.now().strftime("%Y%m%d")
    
    cache_dir = Path(__file__).resolve().parents[1] / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"min_{code}_{start_date}_{end_date}.csv"
    
    if cache_file.exists():
        return pd.read_csv(cache_file)
    
    pro = _init_tushare()
    if not pro:
        raise RuntimeError("TUSHARE_TOKEN 未配置")
    
    ts_code = _to_ts_code(code)
    
    # 分段拉取（避免单次太多）
    all_data = []
    # 按月分段
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    
    current = start
    while current < end:
        next_month = datetime(current.year, current.month + 1, 1) if current.month < 12 else datetime(current.year + 1, 1, 1)
        seg_end = min(next_month - pd.Timedelta(days=1), end)
        
        try:
            df = pro.stk_mins(
                ts_code=ts_code,
                start_date=f"{current.strftime('%Y-%m-%d')} 09:00:00",
                end_date=f"{seg_end.strftime('%Y-%m-%d')} 15:00:00",
                freq="1min"
            )
            if df is not None and len(df) > 0:
                all_data.append(df)
                print(f"  {current.strftime('%Y-%m')} : {len(df)}条")
        except Exception as e:
            print(f"  {current.strftime('%Y-%m')} : 失败 {str(e)[:50]}")
        
        current = next_month
    
    if not all_data:
        return pd.DataFrame()
    
    raw = pd.concat(all_data, ignore_index=True)
    # 排序（tushare返回的是倒序）
    raw = raw.sort_values('trade_time').reset_index(drop=True)
    
    # 缓存
    raw.to_csv(cache_file, index=False)
    
    return raw


def run_intraday_backtest(
    daily_df: pd.DataFrame,
    minute_df: pd.DataFrame,
    code: str,
    name: str = "",
    stop_loss_pct: float = -8.0,
    take_profit_pct: float = 20.0,
    take_profit_full: float = 30.0,
    # 分时出场参数
    intraday_red_shrink_bars: int = 3,    # 分时MACD红柱连续缩短N根=拐头
    intraday_min_profit_for_exit: float = 8.0,  # 盘中浮盈>8%才开始看分时出场
    intraday_pullback_from_peak: float = 5.0,   # 从盘中高点回落>5%=减仓
    max_holding_days: int = 20,          # 最多持有天数
) -> List[IntradayTrade]:
    """
    分时级别战法回测
    
    日线MACD面积信号确定入场日 → 切到分钟级别模拟盘中出场
    """
    # 1. 日线信号
    daily_signals = generate_signals(daily_df)
    entry_signals = [s for s in daily_signals if s.signal_type == "entry_candidate" and s.signal_strength >= 0.4]
    
    if not entry_signals:
        return []
    
    # 2. 准备分钟数据查找表
    if minute_df.empty:
        return []
    
    # 按交易日分组分钟数据
    minute_df['trade_date'] = minute_df['trade_time'].str[:10].str.replace('-', '')
    minute_by_date = {date: group.sort_values('trade_time').reset_index(drop=True) 
                      for date, group in minute_df.groupby('trade_date')}
    
    trades = []
    last_entry_date = ""
    
    for sig in entry_signals:
        entry_date = sig.date
        
        # 避免重复入场
        if last_entry_date and entry_date <= last_entry_date:
            continue
        
        entry_price = sig.price
        
        # 找入场后的分钟数据
        all_dates = sorted(daily_df['trade_date'].astype(str).tolist())
        try:
            entry_idx = all_dates.index(entry_date)
        except ValueError:
            continue
        
        trade = IntradayTrade(
            code=code, name=name,
            entry_date=entry_date, entry_price=entry_price,
            signal_strength=sig.signal_strength,
        )
        
        # 3. 逐日遍历分钟数据
        position_open = True
        shares = 1000
        daily_count = 0
        
        for d_idx in range(entry_idx + 1, min(entry_idx + max_holding_days + 1, len(all_dates))):
            if not position_open:
                break
            
            current_date = all_dates[d_idx]
            day_minutes = minute_by_date.get(current_date)
            
            if day_minutes is None or len(day_minutes) < 30:
                # 没有分钟数据，用日K近似
                daily_row = daily_df[daily_df['trade_date'].astype(str) == current_date]
                if len(daily_row) == 0:
                    continue
                daily_close = daily_row.iloc[0]['close']
                profit = (daily_close / entry_price - 1) * 100
                
                trade.intraday_max_price = max(trade.intraday_max_price, daily_close)
                trade.intraday_max_profit = max(trade.intraday_max_profit, profit)
                
                if profit <= stop_loss_pct:
                    trade.exit_time = f"{current_date} 15:00"
                    trade.exit_price = daily_close
                    trade.exit_reason = "止损"
                    trade.pnl_pct = profit
                    position_open = False
                elif profit >= take_profit_full:
                    trade.exit_time = f"{current_date} 15:00"
                    trade.exit_price = daily_close
                    trade.exit_reason = "强制止盈"
                    trade.pnl_pct = profit
                    position_open = False
                elif profit >= take_profit_pct and daily_count > 2:
                    trade.exit_time = f"{current_date} 15:00"
                    trade.exit_price = daily_close
                    trade.exit_reason = "目标止盈"
                    trade.pnl_pct = profit
                    position_open = False
                elif daily_count >= max_holding_days - 1:
                    trade.exit_time = f"{current_date} 15:00"
                    trade.exit_price = daily_close
                    trade.exit_reason = "超时平仓"
                    trade.pnl_pct = profit
                    position_open = False
                
                daily_count += 1
                trade.holding_days = daily_count
                continue
            
            # 有分钟数据 → 分时级别模拟
            closes = day_minutes['close'].values
            times = day_minutes['trade_time'].values
            
            # 计算分时MACD（用当日全部分钟收盘价）
            # 注意：分时MACD应该是从开盘开始算的，不是叠加历史的
            if len(closes) >= 35:  # 至少35根分钟K线
                i_dif, i_dea, i_macd = calc_intraday_macd(closes)
                if i_macd is None:
                    i_macd = np.zeros(len(closes))
            else:
                i_macd = np.zeros(len(closes))
            
            day_high = max(closes)
            day_minute_count = len(closes)
            
            for m_idx in range(1, len(closes)):
                price = closes[m_idx]
                profit = (price / entry_price - 1) * 100
                time_str = str(times[m_idx])
                
                # 更新峰值
                trade.intraday_max_price = max(trade.intraday_max_price, price)
                trade.intraday_max_profit = max(trade.intraday_max_profit, profit)
                
                # === 分时出场条件检查 ===
                
                # 止损
                if profit <= stop_loss_pct:
                    trade.exit_time = time_str
                    trade.exit_price = price
                    trade.exit_reason = "止损"
                    trade.pnl_pct = profit
                    trade.exit_triggers.append({"time": time_str, "trigger": f"止损 {profit:.1f}%"})
                    position_open = False
                    break
                
                # 强制止盈
                if profit >= take_profit_full:
                    trade.exit_time = time_str
                    trade.exit_price = price
                    trade.exit_reason = "强制止盈"
                    trade.pnl_pct = profit
                    trade.exit_triggers.append({"time": time_str, "trigger": f"强制止盈 {profit:.1f}%"})
                    position_open = False
                    break
                
                # 分时MACD红柱拐头（你的真实操作逻辑）
                if profit >= intraday_min_profit_for_exit and m_idx >= intraday_red_shrink_bars:
                    # 检查红柱是否连续缩短
                    recent_bars = i_macd[max(0, m_idx - intraday_red_shrink_bars):m_idx + 1]
                    if len(recent_bars) >= intraday_red_shrink_bars + 1:
                        all_shrinking = all(
                            recent_bars[i] < recent_bars[i - 1] 
                            for i in range(1, len(recent_bars))
                        ) and recent_bars[-1] > 0  # 还在红柱区但缩短
                        
                        if all_shrinking:
                            trade.exit_time = time_str
                            trade.exit_price = price
                            trade.exit_reason = "分时红柱拐头"
                            trade.pnl_pct = profit
                            trade.exit_triggers.append({
                                "time": time_str, 
                                "trigger": f"分时MACD红柱连续{intraday_red_shrink_bars}根缩短，浮盈{profit:.1f}%"
                            })
                            position_open = False
                            break
                
                # 从盘中高点回落
                if profit >= intraday_min_profit_for_exit and trade.intraday_max_profit > profit:
                    pullback = trade.intraday_max_profit - profit
                    if pullback >= intraday_pullback_from_peak:
                        trade.exit_time = time_str
                        trade.exit_price = price
                        trade.exit_reason = "盘中回落"
                        trade.pnl_pct = profit
                        trade.exit_triggers.append({
                            "time": time_str,
                            "trigger": f"从峰值{trade.intraday_max_profit:.1f}%回落{pullback:.1f}%"
                        })
                        position_open = False
                        break
            
            daily_count += 1
            trade.holding_days = daily_count
            trade.holding_minutes += day_minute_count
        
        # 如果还持有，按最后可用价格平仓
        if position_open:
            last_date = all_dates[min(entry_idx + max_holding_days, len(all_dates) - 1)]
            last_day = minute_by_date.get(last_date)
            if last_day is not None and len(last_day) > 0:
                last_price = last_day.iloc[-1]['close']
            else:
                daily_row = daily_df[daily_df['trade_date'].astype(str) == last_date]
                last_price = daily_row.iloc[0]['close'] if len(daily_row) > 0 else entry_price
            
            trade.exit_time = f"{last_date} 15:00"
            trade.exit_price = last_price
            trade.exit_reason = "超时平仓"
            trade.pnl_pct = (last_price / entry_price - 1) * 100
        
        last_entry_date = trade.exit_time[:8] if trade.exit_time else entry_date
        trades.append(trade)
    
    return trades


def print_intraday_results(trades: List[IntradayTrade], code: str = ""):
    """打印分时回测结果"""
    if not trades:
        print("无交易记录")
        return
    
    completed = [t for t in trades if t.exit_reason]
    wins = [t for t in completed if t.pnl_pct > 0]
    losses = [t for t in completed if t.pnl_pct <= 0]
    
    total_pnl = sum(t.pnl_pct for t in completed)
    avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0
    win_rate = len(wins) / len(completed) * 100 if completed else 0
    avg_giveback = np.mean([t.profit_giveback for t in completed if t.profit_giveback > 0]) if completed else 0
    
    # 出场原因统计
    reason_count = {}
    for t in completed:
        reason_count[t.exit_reason] = reason_count.get(t.exit_reason, 0) + 1
    
    print(f"\n{'='*60}")
    print(f"  分时级别战法回测 {f'· {code}' if code else ''}")
    print(f"{'='*60}")
    print(f"  总交易数: {len(completed)}")
    print(f"  胜率: {win_rate:.1f}% ({len(wins)}胜 / {len(losses)}负)")
    print(f"  平均收益: {total_pnl/len(completed):+.2f}%")
    print(f"  平均盈利: +{avg_win:.2f}% / 平均亏损: {avg_loss:.2f}%")
    print(f"  盈亏比: {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "")
    print(f"  累计收益: {total_pnl:+.2f}%")
    print(f"  平均持有: {np.mean([t.holding_days for t in completed]):.1f}天")
    print(f"  平均利润回吐: {avg_giveback:.1f}%")
    print(f"\n  出场原因分布:")
    for reason, count in sorted(reason_count.items(), key=lambda x: -x[1]):
        pct = count / len(completed) * 100
        print(f"    {reason}: {count}次 ({pct:.0f}%)")
    
    print(f"\n  交易明细:")
    print(f"  {'入场日':12s} {'入场价':>7s} {'出场时间':16s} {'出场价':>7s} {'收益':>8s} {'峰值':>8s} {'回吐':>6s} {'原因':10s}")
    print(f"  {'-'*85}")
    for t in completed:
        giveback = f"{t.profit_giveback:.0f}%" if t.profit_giveback > 0 else "-"
        print(f"  {t.entry_date:12s} {t.entry_price:7.2f} {t.exit_time:16s} {t.exit_price:7.2f} "
              f"{t.pnl_pct:+7.2f}% {t.intraday_max_profit:+7.2f}% {giveback:>6s} {t.exit_reason}")
    print(f"{'='*60}")
