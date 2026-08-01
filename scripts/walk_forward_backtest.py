#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""滚动窗口分时回测

解决前视偏差：用前N个月的数据估计regime参数，应用到第N+1个月。
参数估计和数据应用完全不重叠。

流程：
  1月: 固定参数（无历史数据可估计）
  2月: 用1月交易估计regime → 应用到2月
  3月: 用1-2月交易估计regime → 应用到3月
  ...
  7月: 用1-6月交易估计regime → 应用到7月
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_daily, load_index
from src.backtest.intraday_simulator import load_minute_data, run_intraday_backtest
from src.risk.regime_params import estimate_conditional_params, get_regime_optimal_params


def run_walk_forward_backtest(
    codes: List[str],
    start_date: str = "20260101",
    end_date: str = "20260731",
    verbose: bool = True,
) -> dict:
    """
    滚动窗口分时回测（无前视偏差）
    
    每个月用之前所有月份的交易数据估计regime参数
    第一个月用固定参数（无历史可参考）
    """
    # 加载大盘
    mkt = load_index('000300', days=400)
    
    # 加载所有股票数据
    all_daily = {}
    all_minute = {}
    all_pct = []
    
    if verbose:
        print("加载数据...")
    
    for code in codes:
        try:
            daily = load_daily(code, start_date=start_date, end_date=end_date, use_cache=True)
            minute = load_minute_data(code, start_date, end_date)
            if len(daily) < 35 or minute.empty:
                continue
            all_daily[code] = daily
            all_minute[code] = minute
            for _, row in daily.iterrows():
                all_pct.append({
                    'trade_date': str(row['trade_date']),
                    'pct_chg': row.get('pct_chg', 0),
                    'code': code,
                })
        except:
            pass
    
    if verbose:
        print(f"加载完成: {len(all_daily)}只股票")
    
    stock_pct_df = pd.DataFrame(all_pct)
    
    # 获取所有交易月份
    all_dates = sorted(set(d for code in all_daily.values() for d in code['trade_date'].astype(str).tolist()))
    months = sorted(set(d[:6] for d in all_dates))
    
    if verbose:
        print(f"交易月份: {months}")
    
    # 按月分割交易
    monthly_trades = defaultdict(list)  # month -> [trades]
    regime_params = None
    all_results_trades = []
    
    for mi, month in enumerate(months):
        if verbose:
            print(f"\n--- {month} ---")
        
        # 确定本月的参数
        if mi == 0 or not monthly_trades:
            # 第一个月：用固定参数
            current_regime = None
            if verbose:
                print(f"  参数: 固定（无历史数据）")
        else:
            # 用之前所有月份的交易估计regime参数
            history_trades = []
            for prev_month in months[:mi]:
                history_trades.extend(monthly_trades[prev_month])
            
            if len(history_trades) >= 10:  # 至少10笔交易才能估计
                current_regime = estimate_conditional_params(history_trades, mkt, stock_pct_df)
                if verbose:
                    print(f"  参数: 基于{len(history_trades)}笔历史交易估计的regime（{len(current_regime)}个状态）")
            else:
                current_regime = None
                if verbose:
                    print(f"  参数: 固定（历史交易不足{len(history_trades)}笔）")
        
        # 本月回测
        month_start = f"{month}01"
        month_end = f"{month}31"
        
        month_trades = []
        for code, daily in all_daily.items():
            # 筛选本月数据
            month_daily = daily[(daily['trade_date'].astype(str) >= month_start) & 
                                (daily['trade_date'].astype(str) <= month_end)].reset_index(drop=True)
            
            if len(month_daily) < 5:
                continue
            
            # 需要完整的日K历史来计算MACD信号（不只本月）
            # run_intraday_backtest内部会调用generate_signals，需要完整历史
            # 所以传入完整daily_df，但只记录本月入场的交易
            
            minute = all_minute.get(code)
            if minute is None or minute.empty:
                continue
            
            try:
                trades = run_intraday_backtest(
                    daily, minute, code, '',
                    market_df=mkt,
                    regime_params=current_regime,
                    intraday_red_shrink_bars=2,
                    intraday_min_profit_for_exit=5.0,
                    intraday_pullback_from_peak=3.0,
                )
                
                # 只保留本月入场的交易
                for t in trades:
                    if t.entry_date[:6] == month:
                        month_trades.append(t)
            except:
                pass
        
        monthly_trades[month] = month_trades
        all_results_trades.extend(month_trades)
        
        if verbose:
            completed = [t for t in month_trades if t.exit_reason]
            wins = [t for t in completed if t.pnl_pct > 0]
            pnl = sum(t.pnl_pct for t in completed)
            wr = len(wins) / len(completed) * 100 if completed else 0
            print(f"  交易: {len(completed)}笔 胜率: {wr:.0f}% 收益: {pnl:+.2f}%")
    
    # 汇总
    completed = [t for t in all_results_trades if t.exit_reason]
    wins = [t for t in completed if t.pnl_pct > 0]
    losses = [t for t in completed if t.pnl_pct <= 0]
    stops = [t for t in completed if '止损' in t.exit_reason]
    
    total_pnl = sum(t.pnl_pct for t in completed)
    win_rate = len(wins) / len(completed) * 100 if completed else 0
    avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0
    
    print(f"\n{'='*70}")
    print(f"  滚动窗口分时回测结果（无前视偏差）")
    print(f"{'='*70}")
    print(f"  总交易: {len(completed)}")
    print(f"  胜率: {win_rate:.1f}%")
    print(f"  平均盈利: +{avg_win:.2f}% | 平均亏损: {avg_loss:.2f}%")
    print(f"  盈亏比: {abs(avg_win/avg_loss):.2f}" if avg_loss else "")
    print(f"  累计收益: {total_pnl:+.2f}%")
    print(f"  止损率: {len(stops)/len(completed)*100:.0f}%" if completed else "")
    
    # 分月统计
    print(f"\n  分月表现:")
    print(f"  {'月份':8s} {'交易':>4s} {'胜率':>5s} {'收益':>8s} {'参数来源':15s}")
    print(f"  {'-'*50}")
    for mi, month in enumerate(months):
        mt = monthly_trades.get(month, [])
        mc = [t for t in mt if t.exit_reason]
        mw = [t for t in mc if t.pnl_pct > 0]
        mp = sum(t.pnl_pct for t in mc)
        wr = len(mw) / len(mc) * 100 if mc else 0
        param_src = "固定" if mi == 0 else f"前{mi}个月"
        print(f"  {month:8s} {len(mc):4d} {wr:4.0f}% {mp:+7.2f}% {param_src:15s}")
    
    # 出场原因
    reason_count = Counter(t.exit_reason for t in completed)
    print(f"\n  出场原因:")
    for reason, cnt in reason_count.most_common():
        wr = sum(1 for t in completed if t.exit_reason == reason and t.pnl_pct > 0) / cnt * 100
        avg = np.mean([t.pnl_pct for t in completed if t.exit_reason == reason])
        print(f"    {reason}: {cnt}次 ({cnt/len(completed)*100:.0f}%) 胜率{wr:.0f}% 均{avg:+.1f}%")
    
    print(f"\n{'='*70}")
    print(f"  四版对比")
    print(f"{'='*70}")
    print(f"  原始版:        194笔 胜率51.0% 收益+63.46%")
    print(f"  P0+P1版:       150笔 胜率62.0% 收益+77.78%")
    print(f"  P2（有前视）:   173笔 胜率63.0% 收益+123.79%")
    print(f"  P2（滚动无偏）: {len(completed)}笔 胜率{win_rate:.1f}% 收益{total_pnl:+.2f}%")
    
    return {
        'trades': all_results_trades,
        'total_pnl': total_pnl,
        'win_rate': win_rate,
        'total_trades': len(completed),
    }


if __name__ == "__main__":
    codes = '000519,000628,000688,000733,000762,000766,000792,000831,000880,000969,001211,001212,001215,001220,001221,001230,001238,001268,001269,001280,001301,001306,001325,001332,001335,001360,001373,002028,002048'.split(',')
    
    result = run_walk_forward_backtest(codes, verbose=True)
