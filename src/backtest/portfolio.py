#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多股票组合回测引擎

在单股 run_backtest 之上做组合层：
- 对多只股票同时运行回测
- 组合层面统计：总收益、夏普比率、最大回撤、相关性矩阵
- 仓位约束：总仓位上限、单票仓位上限
- 输出 PortfolioResult 数据类

设计要点（对齐用户真实交易风格）：
- 用户同时持有上海电气、四川长虹、赛力斯、士兰微等多只票
- 每只票独立的 run_backtest 产出 Trade 列表
- 组合层面把这些 Trade 的日频权益曲线合并，算组合指标
- 仓位约束在入场时生效：总持仓数 / 总资金不超过上限
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable

import sys
sys.path.insert(0, ".")
from src.backtest.simulator import run_backtest, BacktestResult, Trade
from src.risk.dynamic_exit import ExitConfig


@dataclass
class StockResult:
    """单只股票在组合中的回测结果"""
    code: str
    name: str
    result: BacktestResult
    daily_equity: pd.Series  # 日频权益曲线（index=trade_date, value=累计收益率）
    weight: float = 0.0       # 实际分配权重


@dataclass
class PortfolioResult:
    """组合回测结果"""
    # 个股结果
    stock_results: List[StockResult] = field(default_factory=list)

    # 组合层面日频
    portfolio_equity: Optional[pd.Series] = None  # 组合日频权益曲线

    # 组合统计
    total_return_pct: float = 0.0       # 组合总收益率%
    annual_return_pct: float = 0.0      # 年化收益率%
    sharpe_ratio: float = 0.0           # 夏普比率（日频算，年化）
    max_drawdown_pct: float = 0.0       # 最大回撤%
    volatility_pct: float = 0.0         # 年化波动率%

    # 仓位统计
    total_trades: int = 0
    avg_position_utilization: float = 0.0  # 平均仓位利用率
    max_position_utilization: float = 0.0  # 最大仓位利用率

    # 风险
    correlation_matrix: Optional[pd.DataFrame] = None  # 个股收益率相关性矩阵

    def summarize(self) -> "PortfolioResult":
        """计算组合层面统计"""
        if self.portfolio_equity is None or len(self.portfolio_equity) == 0:
            return self

        eq = self.portfolio_equity.dropna()
        if len(eq) < 2:
            return self

        # 日频收益率
        daily_returns = eq.pct_change().dropna()
        if len(daily_returns) == 0:
            daily_returns = pd.Series([0.0])

        # 总收益
        self.total_return_pct = (eq.iloc[-1] - 1.0) * 100

        # 年化（252交易日）
        n_days = len(eq)
        years = n_days / 252 if n_days > 0 else 1
        if years > 0 and eq.iloc[-1] > 0:
            self.annual_return_pct = ((eq.iloc[-1] ** (1 / years)) - 1) * 100

        # 年化波动率
        self.volatility_pct = daily_returns.std() * np.sqrt(252) * 100

        # 夏普比率（无风险利率2%）
        rf_daily = 0.02 / 252
        excess = daily_returns - rf_daily
        if daily_returns.std() > 0:
            self.sharpe_ratio = (excess.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            self.sharpe_ratio = 0.0

        # 最大回撤
        cummax = eq.cummax()
        drawdown = (eq - cummax) / cummax
        self.max_drawdown_pct = drawdown.min() * 100

        # 交易总数
        self.total_trades = sum(sr.result.total_trades for sr in self.stock_results)

        # 相关性矩阵
        ret_df = pd.DataFrame({
            sr.code: sr.daily_equity.pct_change().dropna()
            for sr in self.stock_results
            if sr.daily_equity is not None and len(sr.daily_equity) > 1
        })
        if ret_df.shape[1] >= 2:
            self.correlation_matrix = ret_df.corr()

        return self


def _build_daily_equity(df: pd.DataFrame, trades: List[Trade],
                        start_date: str = "", end_date: str = "") -> pd.Series:
    """
    从 Trade 列表构建日频权益曲线（相对净值，1.0=起始）

    逻辑：
    - 遍历每个交易日
    - 如果当日持有仓位，权益 = 1 + 累计已实现收益 + 当前浮盈
    - 如果空仓，权益 = 1 + 累计已实现收益
    """
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)

    df = df.sort_values('trade_date').reset_index(drop=True)
    if start_date:
        df = df[df['trade_date'] >= start_date]
    if end_date:
        df = df[df['trade_date'] <= end_date]
    if len(df) == 0:
        return pd.Series(dtype=float)

    dates = df['trade_date'].values
    closes = df['close'].values

    # 按入场日排序 trades
    sorted_trades = sorted(trades, key=lambda t: t.entry_date)

    equity = []
    realized_pnl = 0.0  # 累计已实现收益率（百分比转小数）
    active_trade: Optional[Trade] = None
    trade_idx = 0

    for i, date in enumerate(dates):
        date_str = str(date)

        # 检查是否进入新交易
        if active_trade is None and trade_idx < len(sorted_trades):
            if date_str >= sorted_trades[trade_idx].entry_date:
                active_trade = sorted_trades[trade_idx]
                trade_idx += 1

        # 检查当前交易是否结束
        if active_trade is not None:
            if active_trade.exit_date and date_str > active_trade.exit_date:
                # 交易已结束，结算已实现收益
                realized_pnl += active_trade.pnl_pct / 100.0
                active_trade = None
                # 检查下一个交易
                if trade_idx < len(sorted_trades) and date_str >= sorted_trades[trade_idx].entry_date:
                    active_trade = sorted_trades[trade_idx]
                    trade_idx += 1

        # 计算当日权益
        if active_trade is not None and active_trade.entry_price > 0:
            # 持仓中：已实现 + 浮盈
            unrealized = (closes[i] / active_trade.entry_price - 1)
            # 考虑分批减仓的影响：用剩余仓位比例
            remaining_ratio = active_trade.shares / active_trade.initial_shares if active_trade.initial_shares > 0 else 0
            # 已减仓部分锁定收益（approximate: 用 partial_exits 的收益）
            partial_pnl = sum(p.get('pnl_pct', 0) for p in active_trade.partial_exits) / 100.0
            partial_ratio = 1 - remaining_ratio
            daily_eq = 1.0 + realized_pnl + partial_pnl * partial_ratio + unrealized * remaining_ratio
        else:
            daily_eq = 1.0 + realized_pnl

        equity.append(daily_eq)

    return pd.Series(equity, index=[str(d) for d in dates])


def run_portfolio_backtest(
    stocks: List[Dict],
    capital: float = 1_000_000,
    max_total_position_pct: float = 0.8,
    max_single_position_pct: float = 0.3,
    equal_weight: bool = True,
    exit_config: ExitConfig = None,
    signal_min_strength: float = 0.4,
    position_size: int = 1000,
    market_df: pd.DataFrame = None,
    sector_df: pd.DataFrame = None,
    limit_up_prob: float = 0.5,
    verbose: bool = False,
) -> PortfolioResult:
    """
    运行多股票组合回测

    参数：
    - stocks: [{"code": "601727", "name": "上海电气", "df": df, "sector_df": ...}, ...]
    - capital: 总资金
    - max_total_position_pct: 总仓位上限（占资金比例）
    - max_single_position_pct: 单票仓位上限
    - equal_weight: 是否等权分配
    - exit_config: 出场参数
    - signal_min_strength: 信号强度阈值
    - position_size: 每次入场股数
    - market_df: 大盘数据
    - limit_up_prob: 连板概率

    返回：PortfolioResult
    """
    if exit_config is None:
        exit_config = ExitConfig()

    n_stocks = len(stocks)
    if n_stocks == 0:
        return PortfolioResult()

    # 等权分配
    if equal_weight:
        base_weight = min(
            1.0 / n_stocks,
            max_single_position_pct,
        )
        # 总仓位约束
        if base_weight * n_stocks > max_total_position_pct:
            base_weight = max_total_position_pct / n_stocks
    else:
        base_weight = max_single_position_pct

    stock_results: List[StockResult] = []
    all_dates = set()

    for stock in stocks:
        code = stock['code']
        name = stock.get('name', '')
        df = stock['df']
        stock_sector = stock.get('sector_df', sector_df)

        if verbose:
            print(f"\n{'='*40} 回测 {code} {name} {'='*40}")

        result = run_backtest(
            df=df,
            code=code,
            name=name,
            exit_config=exit_config,
            signal_min_strength=signal_min_strength,
            position_size=position_size,
            market_df=market_df,
            sector_df=stock_sector,
            limit_up_prob=limit_up_prob,
            verbose=verbose,
        )

        # 构建日频权益曲线
        daily_eq = _build_daily_equity(df, result.trades)
        if len(daily_eq) > 0:
            all_dates.update(daily_eq.index)

        sr = StockResult(
            code=code,
            name=name,
            result=result,
            daily_equity=daily_eq,
            weight=base_weight,
        )
        stock_results.append(sr)

    # 合并组合权益曲线
    all_dates_sorted = sorted(all_dates)

    if len(all_dates_sorted) == 0:
        return PortfolioResult(stock_results=stock_results)

    # 构建权益矩阵（每只股票对齐到统一日期）
    eq_matrix = pd.DataFrame(index=all_dates_sorted)
    for sr in stock_results:
        if sr.daily_equity is not None and len(sr.daily_equity) > 0:
            # 前向填充：没有交易的日期沿用上一个净值
            col = sr.daily_equity.reindex(all_dates_sorted).ffill().fillna(1.0)
            eq_matrix[sr.code] = col

    # 组合权益 = 加权平均
    weights = np.array([sr.weight for sr in stock_results])
    weights = weights / weights.sum() if weights.sum() > 0 else weights

    # 组合日频收益率 → 加权累乘
    daily_ret_matrix = eq_matrix.pct_change().fillna(0.0)
    portfolio_returns = daily_ret_matrix.values @ weights
    portfolio_equity = pd.Series(
        np.cumprod(1 + portfolio_returns),
        index=all_dates_sorted,
    )

    # 仓位利用率（简化：用有多少只股票同时持仓的比例）
    holding_mask = pd.DataFrame(index=all_dates_sorted)
    for sr in stock_results:
        if sr.daily_equity is not None and len(sr.daily_equity) > 0:
            col = sr.daily_equity.reindex(all_dates_sorted).ffill().fillna(1.0)
            # 权益 != 1 + 0 → 认为有持仓（权益偏离1.0超过阈值）
            holding_mask[sr.code] = (col - 1.0).abs() > 0.001

    if holding_mask.shape[1] > 0:
        position_util = holding_mask.sum(axis=1) / holding_mask.shape[1]
        avg_util = position_util.mean()
        max_util = position_util.max()
    else:
        avg_util = max_util = 0.0

    result = PortfolioResult(
        stock_results=stock_results,
        portfolio_equity=portfolio_equity,
        avg_position_utilization=avg_util,
        max_position_utilization=max_util,
    )

    return result.summarize()


def print_portfolio_result(pr: PortfolioResult):
    """打印组合回测结果"""
    print(f"\n{'#'*60}")
    print(f"  组合回测结果")
    print(f"{'#'*60}")

    # 个股
    print(f"\n  --- 个股回测 ---")
    print(f"  {'代码':8s} {'名称':10s} {'交易数':>5s} {'胜率':>6s} {'累计收益':>9s} {'权重':>6s}")
    print(f"  {'-'*55}")
    for sr in pr.stock_results:
        r = sr.result
        print(f"  {sr.code:8s} {sr.name:10s} {r.total_trades:5d} "
              f"{r.win_rate:5.1f}% {r.total_pnl_pct:+8.2f}% {sr.weight:5.1%}")

    # 组合
    print(f"\n  --- 组合统计 ---")
    print(f"  总交易次数:     {pr.total_trades}")
    print(f"  组合总收益:     {pr.total_return_pct:+.2f}%")
    print(f"  年化收益:       {pr.annual_return_pct:+.2f}%")
    print(f"  年化波动率:     {pr.volatility_pct:.2f}%")
    print(f"  夏普比率:       {pr.sharpe_ratio:.3f}")
    print(f"  最大回撤:       {pr.max_drawdown_pct:.2f}%")
    print(f"  平均仓位利用率: {pr.avg_position_utilization:.1%}")
    print(f"  最大仓位利用率: {pr.max_position_utilization:.1%}")

    # 相关性矩阵
    if pr.correlation_matrix is not None and pr.correlation_matrix.shape[0] >= 2:
        print(f"\n  --- 相关性矩阵 ---")
        codes = pr.correlation_matrix.columns.tolist()
        header = f"  {'':8s}" + " ".join(f"{c[-4:]:>8s}" for c in codes)
        print(header)
        for idx, row_code in enumerate(codes):
            vals = pr.correlation_matrix.iloc[idx]
            row_str = f"  {row_code[-4:]:8s}" + " ".join(f"{v:8.2f}" for v in vals)
            print(row_str)

    print(f"\n{'#'*60}")
