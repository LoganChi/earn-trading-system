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
    intraday_red_shrink_bars: int = 2,    # P1优化: 3→2 更灵敏
    intraday_min_profit_for_exit: float = 5.0,  # P1优化: 8→5 更早锁利润
    intraday_pullback_from_peak: float = 3.0,   # P1优化: 5→3 更快锁利润
    max_holding_days: int = 20,          # 最多持有天数
    # P0改进：风控过滤器
    market_consecutive_down_limit: int = 3,  # 大盘连跌N天=强制空仓
    market_df: pd.DataFrame = None,          # 大盘日K(trade_date, pct_chg)
    cooldown_days: int = 30,                 # 连续止损后冷却天数
    consecutive_stop_limit: int = 2,         # 连续N次止损触发冷却
    # P1改进：入场确认
    require_next_day_confirm: bool = True,   # 次日开盘站稳才入场
    next_day_confirm_pct: float = -2.0,      # 次日跌幅不超过2%=站稳
    # P2改进：自适应regime参数
    regime_params: dict = None,              # 从regime_params.py获取的自适应参数
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
    
    # P0: 大盘连跌查找表
    mkt_down_map = {}  # date -> consecutive_down_days
    if market_df is not None:
        mkt_sorted = market_df.sort_values('trade_date').reset_index(drop=True)
        consec = 0
        for _, row in mkt_sorted.iterrows():
            d = str(row['trade_date'])
            chg = row.get('pct_chg', 0)
            try:
                chg = float(chg)
            except:
                chg = 0
            if chg < 0:
                consec += 1
            else:
                consec = 0
            mkt_down_map[d] = consec
    
    # 按交易日分组分钟数据
    minute_df['trade_date'] = minute_df['trade_time'].str[:10].str.replace('-', '')
    minute_by_date = {date: group.sort_values('trade_time').reset_index(drop=True) 
                      for date, group in minute_df.groupby('trade_date')}
    
    trades = []
    last_entry_date = ""
    
    # P0: 连续止损冷却追踪
    consecutive_stops = 0
    cooldown_until = ""  # 冷却期截止日期
    
    all_dates = sorted(daily_df['trade_date'].astype(str).tolist())
    
    for sig in entry_signals:
        entry_date = sig.date
        
        # 避免重复入场（出场后至少冷却3个交易日，不管盈亏）
        if last_entry_date:
            # 找last_entry_date在all_dates中的位置
            try:
                last_idx = all_dates.index(last_entry_date)
                if last_idx + 3 < len(all_dates) and entry_date < all_dates[last_idx + 3]:
                    continue  # 出场后3个交易日内不再入场
            except ValueError:
                pass
        
        # P0: 大盘连跌过滤
        if market_df is not None:
            mkt_down = mkt_down_map.get(entry_date, 0)
            if mkt_down >= market_consecutive_down_limit:
                continue  # 大盘连跌≥3天，跳过
        
        # P0: 冷却期检查
        if cooldown_until and entry_date < cooldown_until:
            continue
        
        # P2: 自适应regime参数 —— 根据当前市场状态调整止损/止盈
        current_stop = stop_loss_pct
        current_take_profit = take_profit_pct
        current_intraday_threshold = intraday_min_profit_for_exit
        skip_entry = False
        
        if regime_params and market_df is not None:
            from src.risk.regime_params import classify_market_trend, classify_market_vol, classify_sector_strength, get_regime_optimal_params
            mkt_trend = classify_market_trend(market_df, entry_date)
            mkt_vol = classify_market_vol(market_df, entry_date)
            try:
                stock_row = daily_df[daily_df['trade_date'].astype(str) == entry_date]
                stock_pct = float(stock_row.iloc[0].get('pct_chg', 0)) if len(stock_row) > 0 else 0
                mkt_row = market_df[market_df['trade_date'].astype(str) == entry_date]
                mkt_pct = float(mkt_row.iloc[0].get('pct_chg', 0)) if len(mkt_row) > 0 else 0
            except:
                stock_pct, mkt_pct = 0, 0
            sector_str = classify_sector_strength(stock_pct, mkt_pct)
            
            rp = get_regime_optimal_params(regime_params, mkt_trend, mkt_vol, sector_str)
            current_stop = rp['stop_loss']
            current_take_profit = rp['take_profit']
            current_intraday_threshold = rp.get('intraday_threshold', intraday_min_profit_for_exit)
            
            # P2: 胜率极低的状态直接跳过
            if rp.get('expected_win_rate', 50) < 15 and rp.get('sample_count', 0) >= 3:
                skip_entry = True
        
        if skip_entry:
            continue
        
        # P1: 入场确认——次日不跌破-2%才真正入场
        if require_next_day_confirm:
            try:
                sig_idx = all_dates.index(entry_date)
                if sig_idx + 1 < len(all_dates):
                    next_date = all_dates[sig_idx + 1]
                    next_row = daily_df[daily_df['trade_date'].astype(str) == next_date]
                    if len(next_row) > 0:
                        next_open = next_row.iloc[0].get('open', 0)
                        next_close = next_row.iloc[0]['close']
                        # 次日收盘相对信号日收盘的涨跌
                        next_chg = (next_close / sig.price - 1) * 100
                        if next_chg < next_day_confirm_pct:
                            continue  # 次日跌破-2%，信号取消
                        # 用次日开盘价作为实际入场价（更真实）
                        if next_open > 0:
                            entry_price_actual = next_open
                        else:
                            entry_price_actual = next_close
                        actual_entry_date = next_date
                    else:
                        continue
                else:
                    continue
            except (ValueError, KeyError):
                continue
        else:
            entry_price_actual = sig.price
            actual_entry_date = entry_date
        
        entry_price = entry_price_actual
        
        # 找入场后的分钟数据
        try:
            entry_idx = all_dates.index(actual_entry_date)
        except ValueError:
            continue
        
        trade = IntradayTrade(
            code=code, name=name,
            entry_date=actual_entry_date, entry_price=entry_price,
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
                
                if profit <= current_stop:
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
                elif profit >= current_take_profit and daily_count > 2:
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
                if profit <= current_stop:
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
                if profit >= current_intraday_threshold and m_idx >= intraday_red_shrink_bars:
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
                if profit >= current_intraday_threshold and trade.intraday_max_profit > profit:
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
        
        last_entry_date = trade.exit_time[:8] if trade.exit_time else actual_entry_date
        
        # P0: 连续止损冷却
        if "止损" in trade.exit_reason:
            consecutive_stops += 1
            if consecutive_stops >= consecutive_stop_limit:
                # 触发冷却
                try:
                    exit_date_idx = all_dates.index(last_entry_date)
                    if exit_date_idx + cooldown_days < len(all_dates):
                        cooldown_until = all_dates[exit_date_idx + cooldown_days]
                    else:
                        cooldown_until = all_dates[-1]
                except ValueError:
                    pass
                consecutive_stops = 0  # 重置计数器
        else:
            consecutive_stops = 0  # 非止损出场，重置
        
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
