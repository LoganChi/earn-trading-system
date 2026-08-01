#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""条件概率参数估计器

核心问题：个股、板块、大盘不是独立的。
同一套出场参数，在不同市场环境下最优值不同。

层次影响模型：
  大盘（系统Beta）> 板块（行业Alpha）> 个股（信号）

条件概率矩阵：
  P(盈利 | 大盘涨 + 板块强 + 信号强) = ?
  P(盈利 | 大盘跌 + 板块强 + 信号强) = ?
  P(盈利 | 大盘跌 + 板块弱 + 信号强) = ?

用户要求：参数估计不能假设独立，必须考虑交互效应。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


@dataclass
class MarketRegime:
    """市场状态分类"""
    market_trend: str   # "up" / "flat" / "down"（大盘趋势）
    market_vol: str     # "high" / "normal" / "low"（大盘波动率）
    sector_strength: str  # "strong" / "neutral" / "weak"（板块相对大盘）
    
    @property
    def regime_id(self) -> str:
        return f"{self.market_trend}_{self.market_vol}_{self.sector_strength}"


@dataclass
class RegimeStats:
    """某个市场状态下的统计"""
    regime_id: str
    sample_count: int = 0
    win_count: int = 0
    win_rate: float = 0.0
    avg_profit: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    optimal_stop_loss: float = -8.0
    optimal_take_profit: float = 20.0
    optimal_intraday_threshold: float = 5.0
    
    # 交互效应
    market_marginal_effect: float = 0.0  # 大盘单独影响
    sector_marginal_effect: float = 0.0  # 板块单独影响
    interaction_effect: float = 0.0       # 交互项（大盘×板块）


def classify_market_trend(market_df: pd.DataFrame, date: str, lookback: int = 10) -> str:
    """分类大盘趋势"""
    market_sorted = market_df.sort_values('trade_date').reset_index(drop=True)
    try:
        idx = market_sorted[market_sorted['trade_date'].astype(str) == date].index[0]
    except (IndexError, KeyError):
        return "flat"
    
    start = max(0, idx - lookback)
    recent = market_sorted.iloc[start:idx + 1]
    if len(recent) < 3:
        return "flat"
    
    # 用均线斜率判断趋势
    closes = recent['close'].values
    ma5 = np.mean(closes[-5:]) if len(closes) >= 5 else np.mean(closes)
    ma10 = np.mean(closes[-10:]) if len(closes) >= 10 else np.mean(closes)
    current = closes[-1]
    
    if current > ma5 > ma10:
        return "up"
    elif current < ma5 < ma10:
        return "down"
    else:
        return "flat"


def classify_market_vol(market_df: pd.DataFrame, date: str, lookback: int = 20) -> str:
    """分类大盘波动率"""
    market_sorted = market_df.sort_values('trade_date').reset_index(drop=True)
    try:
        idx = market_sorted[market_sorted['trade_date'].astype(str) == date].index[0]
    except (IndexError, KeyError):
        return "normal"
    
    start = max(0, idx - lookback)
    recent = market_sorted.iloc[start:idx + 1]
    if len(recent) < 5:
        return "normal"
    
    pct_chgs = recent['pct_chg'].astype(float).values
    vol = np.std(pct_chgs)
    
    # 分位数判断
    if vol > 1.5:  # 日均波动>1.5%
        return "high"
    elif vol < 0.7:
        return "low"
    else:
        return "normal"


def classify_sector_strength(stock_pct: float, market_pct: float) -> str:
    """分类板块/个股相对大盘强度"""
    diff = stock_pct - market_pct
    if diff > 2.0:
        return "strong"
    elif diff < -2.0:
        return "weak"
    else:
        return "neutral"


def estimate_conditional_params(
    trades: List,  # IntradayTrade列表
    market_df: pd.DataFrame,
    stock_daily: pd.DataFrame,  # 个股日K（含pct_chg）
) -> Dict[str, RegimeStats]:
    """
    估计不同市场状态下的条件参数
    
    返回 {regime_id: RegimeStats} 字典
    """
    # 为每笔交易标注市场状态
    regime_trades = defaultdict(list)
    
    for trade in trades:
        entry_date = trade.entry_date
        
        # 分类市场状态
        mkt_trend = classify_market_trend(market_df, entry_date)
        mkt_vol = classify_market_vol(market_df, entry_date)
        
        # 个股相对大盘强度
        try:
            stock_row = stock_daily[stock_daily['trade_date'].astype(str) == entry_date]
            stock_pct = float(stock_row.iloc[0]['pct_chg']) if len(stock_row) > 0 else 0
        except:
            stock_pct = 0
        
        try:
            mkt_row = market_df[market_df['trade_date'].astype(str) == entry_date]
            mkt_pct = float(mkt_row.iloc[0]['pct_chg']) if len(mkt_row) > 0 else 0
        except:
            mkt_pct = 0
        
        sector_str = classify_sector_strength(stock_pct, mkt_pct)
        
        regime = MarketRegime(
            market_trend=mkt_trend,
            market_vol=mkt_vol,
            sector_strength=sector_str,
        )
        
        trade_dict = {
            'pnl_pct': trade.pnl_pct,
            'exit_reason': trade.exit_reason,
            'signal_strength': trade.signal_strength,
            'intraday_max_profit': trade.intraday_max_profit,
            'holding_days': trade.holding_days,
            'mkt_trend': mkt_trend,
            'mkt_vol': mkt_vol,
            'sector_str': sector_str,
        }
        regime_trades[regime.regime_id].append(trade_dict)
    
    # 计算每个regime的统计
    results = {}
    for regime_id, trade_list in regime_trades.items():
        pnls = [t['pnl_pct'] for t in trade_list]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        stats = RegimeStats(
            regime_id=regime_id,
            sample_count=len(trade_list),
            win_count=len(wins),
            win_rate=len(wins) / len(trade_list) * 100 if trade_list else 0,
            avg_profit=np.mean(pnls) if pnls else 0,
            avg_win=np.mean(wins) if wins else 0,
            avg_loss=np.mean(losses) if losses else 0,
        )
        
        # 估计最优参数（在该regime下）
        # 止损：找使总收益最大化的止损位
        best_stop = -8.0
        best_stop_pnl = float('-inf')
        for test_stop in [-5, -6, -7, -8, -10, -12]:
            sim_pnls = [max(p, test_stop) for p in pnls]  # 简化模拟
            total = sum(sim_pnls)
            if total > best_stop_pnl:
                best_stop_pnl = total
                best_stop = test_stop
        stats.optimal_stop_loss = best_stop
        
        # 止盈
        best_tp = 20.0
        best_tp_pnl = float('-inf')
        for test_tp in [10, 15, 20, 25, 30]:
            sim_pnls = [min(p, test_tp) if p > 0 else p for p in pnls]
            total = sum(sim_pnls)
            if total > best_tp_pnl:
                best_tp_pnl = total
                best_tp = test_tp
        stats.optimal_take_profit = best_tp
        
        # 分时出场阈值
        profits = [t['intraday_max_profit'] for t in trade_list]
        if profits:
            percentiles = np.percentile(profits, [25, 50, 75])
            stats.optimal_intraday_threshold = percentiles[0]  # 25%分位
        
        results[regime_id] = stats
    
    # 计算边际效应和交互效应
    _compute_interaction_effects(regime_trades, results)
    
    return results


def _compute_interaction_effects(regime_trades: dict, results: Dict[str, RegimeStats]):
    """
    计算大盘、板块的边际效应和交互效应
    
    用2×3方差分解的思想：
    - 大盘主效应：大盘涨 vs 大盘跌 的收益差异
    - 板块主效应：板块强 vs 板块弱 的收益差异
    - 交互效应：(大盘涨+板块强) 和 (大盘跌+板块弱) 的额外效应
    """
    
    # 按大盘趋势分组
    up_pnls = []
    down_pnls = []
    flat_pnls = []
    
    for regime_id, trades in regime_trades.items():
        for t in trades:
            if t['mkt_trend'] == 'up':
                up_pnls.append(t['pnl_pct'])
            elif t['mkt_trend'] == 'down':
                down_pnls.append(t['pnl_pct'])
            else:
                flat_pnls.append(t['pnl_pct'])
    
    market_up_avg = np.mean(up_pnls) if up_pnls else 0
    market_down_avg = np.mean(down_pnls) if down_pnls else 0
    market_effect = market_up_avg - market_down_avg
    
    # 按板块强度分组
    strong_pnls = []
    weak_pnls = []
    neutral_pnls = []
    
    for regime_id, trades in regime_trades.items():
        for t in trades:
            if t['sector_str'] == 'strong':
                strong_pnls.append(t['pnl_pct'])
            elif t['sector_str'] == 'weak':
                weak_pnls.append(t['pnl_pct'])
            else:
                neutral_pnls.append(t['pnl_pct'])
    
    sector_strong_avg = np.mean(strong_pnls) if strong_pnls else 0
    sector_weak_avg = np.mean(weak_pnls) if weak_pnls else 0
    sector_effect = sector_strong_avg - sector_weak_avg
    
    # 交互效应：观察"大盘涨+板块强"和"大盘跌+板块弱"的实际表现
    # vs 各自主效应之和
    up_strong_pnls = []
    down_weak_pnls = []
    
    for regime_id, trades in regime_trades.items():
        for t in trades:
            if t['mkt_trend'] == 'up' and t['sector_str'] == 'strong':
                up_strong_pnls.append(t['pnl_pct'])
            elif t['mkt_trend'] == 'down' and t['sector_str'] == 'weak':
                down_weak_pnls.append(t['pnl_pct'])
    
    if up_strong_pnls and down_weak_pnls:
        actual_spread = np.mean(up_strong_pnls) - np.mean(down_weak_pnls)
        expected_spread = market_effect + sector_effect
        interaction = actual_spread - expected_spread
    else:
        interaction = 0
    
    # 写入每个regime的统计
    for regime_id, stats in results.items():
        parts = regime_id.split('_')
        mkt_trend = parts[0] if parts else 'flat'
        sector_str = parts[2] if len(parts) > 2 else 'neutral'
        
        stats.market_marginal_effect = round(market_effect, 2)
        stats.sector_marginal_effect = round(sector_effect, 2)
        stats.interaction_effect = round(interaction, 2)


def print_regime_analysis(results: Dict[str, RegimeStats]):
    """打印条件概率分析"""
    print(f"\n{'='*80}")
    print(f"  条件概率参数估计（个股×板块×大盘交互效应）")
    print(f"{'='*80}")
    
    # 按胜率排序
    sorted_regimes = sorted(results.items(), key=lambda x: -x[1].win_rate)
    
    print(f"\n{'状态':25s} {'样本':>4s} {'胜率':>5s} {'均收益':>7s} {'均盈':>6s} {'均亏':>6s} {'最优止损':>7s} {'最优止盈':>7s}")
    print(f"{'-'*85}")
    
    for regime_id, s in sorted_regimes:
        parts = regime_id.split('_')
        mkt_trend = {'up': '大盘↑', 'down': '大盘↓', 'flat': '大盘→'}.get(parts[0], parts[0])
        mkt_vol = {'high': '高波', 'normal': '常波', 'low': '低波'}.get(parts[1] if len(parts) > 1 else '', '')
        sector = {'strong': '板块强', 'neutral': '板块平', 'weak': '板块弱'}.get(parts[2] if len(parts) > 2 else '', '')
        
        regime_label = f"{mkt_trend} {mkt_vol} {sector}"
        
        print(f"{regime_label:25s} {s.sample_count:4d} {s.win_rate:4.0f}% {s.avg_profit:+6.2f}% "
              f"{s.avg_win:+5.1f}% {s.avg_loss:+5.1f}% {s.optimal_stop_loss:+6.0f}% {s.optimal_take_profit:+6.0f}%")
    
    # 交互效应
    print(f"\n{'='*80}")
    print(f"  交互效应分解")
    print(f"{'='*80}")
    
    any_stats = list(results.values())[0] if results else None
    if any_stats:
        print(f"  大盘主效应（涨-跌）: {any_stats.market_marginal_effect:+.2f}%")
        print(f"  板块主效应（强-弱）: {any_stats.sector_marginal_effect:+.2f}%")
        print(f"  交互效应（非独立项）: {any_stats.interaction_effect:+.2f}%")
        print(f"  合计: {any_stats.market_marginal_effect + any_stats.sector_marginal_effect + any_stats.interaction_effect:+.2f}%")
        print(f"\n  解读:")
        if abs(any_stats.interaction_effect) > abs(any_stats.market_marginal_effect) * 0.3:
            print(f"  ⚠️ 交互效应显著（{any_stats.interaction_effect:+.2f}%），大盘和板块不是独立的")
            print(f"     大盘涨+板块强的收益 > 大盘效应 + 板块效应之和")
            print(f"     → 好的时候比预期更好，差的时候比预期更差")
        else:
            print(f"  交互效应较小，大盘和板块近似独立")


def get_regime_optimal_params(results: Dict[str, RegimeStats], 
                               mkt_trend: str, mkt_vol: str, 
                               sector_str: str) -> dict:
    """获取特定市场状态下的最优参数"""
    regime_id = f"{mkt_trend}_{mkt_vol}_{sector_str}"
    
    if regime_id in results:
        s = results[regime_id]
        return {
            'stop_loss': s.optimal_stop_loss,
            'take_profit': s.optimal_take_profit,
            'intraday_threshold': s.optimal_intraday_threshold,
            'expected_win_rate': s.win_rate,
            'expected_return': s.avg_profit,
            'sample_size': s.sample_count,
        }
    
    # 降级：找最接近的regime
    # 先去掉mkt_vol维度
    for vol in ['normal', 'high', 'low']:
        alt_id = f"{mkt_trend}_{vol}_{sector_str}"
        if alt_id in results:
            s = results[alt_id]
            return {
                'stop_loss': s.optimal_stop_loss,
                'take_profit': s.optimal_take_profit,
                'intraday_threshold': s.optimal_intraday_threshold,
                'expected_win_rate': s.win_rate,
                'expected_return': s.avg_profit,
                'sample_size': s.sample_count,
                'note': f'降级匹配 {alt_id}',
            }
    
    # 默认参数
    return {
        'stop_loss': -8.0,
        'take_profit': 20.0,
        'intraday_threshold': 5.0,
        'expected_win_rate': 50.0,
        'expected_return': 0.0,
        'sample_size': 0,
        'note': '默认参数（无匹配regime）',
    }
