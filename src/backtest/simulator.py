#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""动态持仓回测模拟器

核心区别与传统回测：
- 传统：信号 → 固定持有N天 → 算收益
- 本系统：信号 → 每天检查动态出场条件 → 条件驱动离场

用户的真实交易逻辑：
1. MACD面积信号入场
2. 每天检查：止损/止盈/动能/板块/大盘 → 动态调整仓位
3. 出场不是时间驱动，是条件驱动

回测流程：
  for each trading_day:
    if not holding:
      check_entry_signal() → 可能入场
    else:
      update_position()
      check_exit_conditions() → 可能减仓/清仓
    record_daily_state()
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum

import sys
sys.path.insert(0, ".")
from src.signals.macd_area import calc_macd, generate_signals, MACDAreaSignal
from src.risk.dynamic_exit import Position, ExitConfig, ExitDecision, check_exit, ExitReason


@dataclass
class Trade:
    """一笔完整交易"""
    code: str
    name: str
    entry_date: str
    entry_price: float
    exit_date: str = ""
    exit_price: float = 0.0
    shares: int = 0
    initial_shares: int = 0
    exit_reason: str = ""
    holding_days: int = 0
    pnl_pct: float = 0.0
    max_profit_pct: float = 0.0
    signal_strength: float = 0.0
    partial_exits: List[Dict] = field(default_factory=list)
    
    def close(self, date: str, price: float, reason: str):
        self.exit_date = date
        self.exit_price = price
        self.exit_reason = reason
        if self.entry_price > 0:
            self.pnl_pct = (price / self.entry_price - 1) * 100
    
    def partial_close(self, date: str, price: float, ratio: float, reason: str):
        reduce_shares = int(self.shares * ratio)
        pnl = (price / self.entry_price - 1) * 100
        self.shares -= reduce_shares
        self.partial_exits.append({
            "date": date, "price": price, "shares": reduce_shares,
            "pnl_pct": round(pnl, 2), "reason": reason
        })


@dataclass
class BacktestResult:
    """回测结果"""
    trades: List[Trade] = field(default_factory=list)
    total_pnl_pct: float = 0.0
    win_rate: float = 0.0
    avg_holding_days: float = 0.0
    avg_profit_pct: float = 0.0
    max_single_profit: float = 0.0
    max_single_loss: float = 0.0
    total_trades: int = 0
    win_trades: int = 0
    loss_trades: int = 0
    
    def summarize(self):
        self.total_trades = len(self.trades)
        completed = [t for t in self.trades if t.exit_date]
        self.win_trades = sum(1 for t in completed if t.pnl_pct > 0)
        self.loss_trades = sum(1 for t in completed if t.pnl_pct <= 0)
        
        if completed:
            self.win_rate = self.win_trades / len(completed) * 100
            self.avg_holding_days = sum(t.holding_days for t in completed) / len(completed)
            self.avg_profit_pct = sum(t.pnl_pct for t in completed) / len(completed)
            self.max_single_profit = max(t.pnl_pct for t in completed)
            self.max_single_loss = min(t.pnl_pct for t in completed)
            self.total_pnl_pct = sum(t.pnl_pct for t in completed)
        
        return self


def run_backtest(
    df: pd.DataFrame,
    code: str,
    name: str = "",
    exit_config: ExitConfig = None,
    signal_min_strength: float = 0.4,
    position_size: int = 1000,
    # 市场环境参数（可传入外部数据）
    market_df: pd.DataFrame = None,    # 大盘日K（trade_date, pct_chg）
    sector_df: pd.DataFrame = None,    # 板块日K（trade_date, pct_chg）
    limit_up_prob: float = 0.5,        # 默认连板概率
    verbose: bool = False,
) -> BacktestResult:
    """
    运行动态持仓回测
    
    参数：
    - df: 个股日K（trade_date, close, open, high, low, pct_chg）
    - code/name: 股票代码/名称
    - exit_config: 出场参数
    - signal_min_strength: 最低信号强度阈值
    - position_size: 每次入场股数
    - market_df: 大盘数据（可选）
    - sector_df: 板块数据（可选）
    - limit_up_prob: 市场连板概率
    """
    if exit_config is None:
        exit_config = ExitConfig()
    
    df = df.sort_values('trade_date').reset_index(drop=True)
    signals = generate_signals(df)
    
    # 构建信号查找表
    sig_map = {s.date: s for s in signals}
    
    # 构建大盘数据查找表
    mkt_map = {}
    if market_df is not None:
        for _, row in market_df.iterrows():
            mkt_map[str(row['trade_date'])] = row.get('pct_chg', 0)
    
    result = BacktestResult()
    current_trade: Optional[Trade] = None
    current_pos: Optional[Position] = None
    
    # 大盘连续下跌计数
    mkt_consecutive_down = 0
    
    for idx in range(len(df)):
        row = df.iloc[idx]
        date = str(row['trade_date'])
        close = row['close']
        open_price = row.get('open', close)
        high = row.get('high', close)
        
        # 更新大盘状态
        mkt_chg = mkt_map.get(date, 0)
        if isinstance(mkt_chg, str):
            try:
                mkt_chg = float(mkt_chg)
            except:
                mkt_chg = 0
        if mkt_chg < 0:
            mkt_consecutive_down += 1
        else:
            mkt_consecutive_down = 0
        
        if current_trade is None:
            # === 检查入场信号 ===
            sig = sig_map.get(date)
            if sig and sig.signal_type == "entry_candidate" and sig.signal_strength >= signal_min_strength:
                current_trade = Trade(
                    code=code, name=name,
                    entry_date=date, entry_price=close,
                    shares=position_size, initial_shares=position_size,
                    signal_strength=sig.signal_strength,
                )
                current_pos = Position(
                    code=code, name=name,
                    entry_date=date, entry_price=close,
                    shares=position_size, cost_basis=close * position_size,
                )
                current_pos.update(date, close)
                if verbose:
                    print(f"[入场] {date} {code} @{close:.2f} 强度{sig.signal_strength:.0%} {sig.description}")
        
        else:
            # === 更新持仓 + 检查出场 ===
            current_pos.update(date, close)
            current_trade.holding_days = current_pos.holding_days
            current_trade.max_profit_pct = current_pos.max_profit_pct
            
            # 高开回落检测
            high_open_rejected = False
            if open_price > current_pos.entry_price * 1.05:  # 高开5%以上
                if (high - close) / high * 100 > 3:  # 从高点回落3%
                    high_open_rejected = True
            
            # MACD红柱缩短检测
            sig = sig_map.get(date)
            macd_shrinking = sig is not None and sig.signal_type == "exit_warning"
            
            # 检查出场
            decision = check_exit(
                position=current_pos,
                config=exit_config,
                macd_red_shrinking=macd_shrinking,
                high_open_rejected=high_open_rejected,
                sector_vs_market=0,  # TODO: 接入板块数据
                market_change=mkt_chg,
                market_consecutive_down=mkt_consecutive_down,
                limit_up_probability=limit_up_prob,
                sector_strength="neutral",
            )
            
            if decision.action == "close":
                current_trade.close(date, close, decision.reason.value)
                result.trades.append(current_trade)
                if verbose:
                    print(f"[清仓] {date} {code} @{close:.2f} {decision.description} 收益{current_trade.pnl_pct:+.1f}%")
                current_trade = None
                current_pos = None
            
            elif decision.action == "reduce" and current_trade.shares > 100:
                current_trade.partial_close(date, close, decision.reduce_ratio, decision.reason.value)
                current_pos.shares = current_trade.shares
                if verbose:
                    print(f"[减仓] {date} {code} @{close:.2f} {decision.description}")
    
    # 如果回测结束时还持有，按最后收盘价平仓
    if current_trade is not None:
        last_close = df.iloc[-1]['close']
        last_date = str(df.iloc[-1]['trade_date'])
        current_trade.close(last_date, last_close, "回测结束平仓")
        result.trades.append(current_trade)
        if verbose:
            print(f"[期末平仓] {last_date} @{last_close:.2f} 收益{current_trade.pnl_pct:+.1f}%")
    
    return result.summarize()


def print_result(result: BacktestResult, code: str = ""):
    """打印回测结果"""
    print(f"\n{'='*60}")
    print(f"  动态持仓回测结果 {f'· {code}' if code else ''}")
    print(f"{'='*60}")
    print(f"  总交易次数: {result.total_trades}")
    print(f"  胜率: {result.win_rate:.1f}% ({result.win_trades}胜 / {result.loss_trades}负)")
    print(f"  平均持有天数: {result.avg_holding_days:.1f}天")
    print(f"  平均收益: {result.avg_profit_pct:+.2f}%")
    print(f"  最大单笔盈利: {result.max_single_profit:+.2f}%")
    print(f"  最大单笔亏损: {result.max_single_loss:+.2f}%")
    print(f"  累计收益: {result.total_pnl_pct:+.2f}%")
    print(f"{'='*60}")
    
    print(f"\n  交易明细:")
    print(f"  {'入场日':12s} {'入场价':>7s} {'出场日':12s} {'出场价':>7s} {'天数':>4s} {'收益':>8s} {'原因':8s}")
    print(f"  {'-'*70}")
    for t in result.trades:
        print(f"  {t.entry_date:12s} {t.entry_price:7.2f} {t.exit_date:12s} {t.exit_price:7.2f} "
              f"{t.holding_days:4d} {t.pnl_pct:+7.2f}% {t.exit_reason}")
