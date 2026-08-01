#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""组合级仓位管理回测引擎

模拟用户真实的组合操作：
- 初始资金100万，总仓位由五因子投票决定（满仓/半仓/空仓）
- 恶魔股阶段调整：低匹配度时总仓位×0.5
- 单票仓位上限：不超过总仓位的30%
- 多板块分散：同时最多持有5只
- 入场时分配资金，出场时回收

核心模块：
  1. PortfolioManager  — 组合仓位管理器（资金分配、约束检查）
  2. run_portfolio_intraday_backtest  — 多股组合分时回测
  3. PortfolioAnalysis — 组合统计分析（年化、夏普、回撤、换手率等）

与 src/backtest/portfolio.py 的区别：
  - portfolio.py 是 日线级别 + 事后合并权益曲线
  - 本模块是 分钟级别 + 前向组合约束（仓位上限实时生效）
"""
from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from enum import Enum

import numpy as np
import pandas as pd

# ── 项目内模块 ──────────────────────────────────────────────
_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from src.signals.macd_area import generate_signals, calc_macd, MACDAreaSignal
from src.risk.dynamic_exit import Position, ExitConfig, check_exit, ExitDecision, ExitReason
from src.position.five_factor import (
    FiveFactorResult, Direction, FactorScore,
    score_valuation, score_capital_flow, score_technical,
    score_sentiment, score_fundamental,
)
from src.position.stage_detector import detect_stage, MarketStage, StageResult
from src.data.loader import load_daily, load_index, _init_tushare, _to_ts_code


# ===========================================================================
# 常量 / 枚举
# ===========================================================================
INITIAL_CAPITAL = 1_000_000.0   # 初始资金 100万
MAX_HOLDINGS = 5                 # 同时最多持有5只
MAX_SINGLE_RATIO = 0.30          # 单票不超过总仓位的30%


class PositionLevel(str, Enum):
    """五因子投票仓位级别"""
    FULL = "满仓"       # 100%
    HALF = "半仓"       # 50%
    EMPTY = "空仓"      # 0%


# ===========================================================================
# 数据类
# ===========================================================================
@dataclass
class PortfolioState:
    """组合在某交易日的快照状态"""
    cash: float                          # 可用现金
    total_value: float                   # 总市值 (cash + positions_market_value)
    positions: List[Position]            # 当前持仓列表（引用）
    total_position_ratio: float          # 当前已用仓位比例
    max_position_ratio: float            # 当日最大允许仓位
    date: str                            # 交易日期 YYYYMMDD


@dataclass
class PortfolioTrade:
    """组合层面的一笔完整交易（含资金流）"""
    code: str
    name: str
    entry_date: str
    entry_price: float
    exit_date: str = ""
    exit_price: float = 0.0
    shares: int = 0
    allocated_capital: float = 0.0       # 入场时分配的资金
    pnl_amount: float = 0.0              # 绝对盈亏金额（含部分减仓累计）
    pnl_pct: float = 0.0                 # 收益率（按最终完整出场算）
    exit_reason: str = ""
    holding_days: int = 0
    signal_strength: float = 0.0
    realized_pnl: float = 0.0            # 部分减仓已实现盈亏（累加）

    def close(self, exit_date: str, exit_price: float, reason: str):
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.exit_reason = reason
        if self.entry_price > 0:
            self.pnl_pct = (exit_price / self.entry_price - 1) * 100


@dataclass
class PortfolioAnalysis:
    """组合统计分析结果"""
    # ── 核心指标 ──
    total_return_pct: float = 0.0        # 组合总收益率%
    annual_return_pct: float = 0.0       # 年化收益率%
    max_drawdown_pct: float = 0.0        # 最大回撤%
    sharpe_ratio: float = 0.0            # 夏普比率
    volatility_pct: float = 0.0          # 年化波动率%

    # ── 净值曲线 ──
    daily_nav: Optional[pd.Series] = None       # 每日净值（1.0起始）
    daily_returns: Optional[pd.Series] = None    # 每日收益率

    # ── 月度收益 ──
    monthly_returns: Optional[pd.DataFrame] = None  # 分月收益分布

    # ── 仓位统计 ──
    avg_position_utilization: float = 0.0   # 平均仓位利用率%
    max_position_utilization: float = 0.0   # 最大仓位利用率%

    # ── 交易统计 ──
    total_trades: int = 0
    win_rate: float = 0.0
    turnover_rate: float = 0.0              # 换手率（年化）

    # ── 持仓贡献 ──
    best_stock: Optional[Dict] = None       # 最优持仓 {code, name, pnl_amount}
    worst_stock: Optional[Dict] = None      # 最差持仓

    # ── 明细 ──
    trades: List[PortfolioTrade] = field(default_factory=list)
    position_history: List[PortfolioState] = field(default_factory=list)


# ===========================================================================
# 组合仓位管理器
# ===========================================================================
class PortfolioManager:
    """
    组合仓位管理器

    职责：
    1. 维护现金 / 持仓 / 总市值
    2. 每日根据五因子投票 + 恶魔股阶段更新最大仓位上限
    3. 检查入场约束（总仓位空间、单票上限、持仓数上限）
    4. 入场分配资金、出场回收资金
    """

    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        max_holdings: int = MAX_HOLDINGS,
        max_single_ratio: float = MAX_SINGLE_RATIO,
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.max_holdings = max_holdings
        self.max_single_ratio = max_single_ratio

        # 当前持仓（code -> PortfolioTrade + Position）
        self.holdings: Dict[str, PortfolioTrade] = {}
        self._positions: Dict[str, Position] = {}

        # 当日最大允许仓位（由五因子+恶魔股决定）
        self.max_position_ratio: float = 0.8  # 初始默认80%
        self.current_date: str = ""

        # 历史净值记录
        self.nav_history: List[Tuple[str, float]] = []
        self.position_ratio_history: List[Tuple[str, float]] = []

        # 已完成交易
        self.completed_trades: List[PortfolioTrade] = []

    # ── 仓位上限更新 ────────────────────────────────────────

    def update_position_limit(
        self,
        date: str,
        five_factor_ratio: float = 0.5,
        stage_adjustment: float = 1.0,
    ):
        """
        每日开盘前更新总仓位上限。

        参数：
        - five_factor_ratio: 五因子投票建议仓位（0-1）
        - stage_adjustment: 恶魔股阶段调整系数（0.5/1.0/1.5）
        """
        self.current_date = date
        # 总仓位 = 五因子建议 × 阶段调整
        adjusted = five_factor_ratio * stage_adjustment
        # 但不能超过1.0
        self.max_position_ratio = min(1.0, max(0.0, adjusted))

    # ── 入场检查 ────────────────────────────────────────────

    def can_enter(self, code: str) -> Tuple[bool, str]:
        """检查是否可以开新仓"""
        # 已持有该股？
        if code in self.holdings:
            return False, "已持有该股"

        # 持仓数上限
        if len(self.holdings) >= self.max_holdings:
            return False, f"持仓数已达上限{self.max_holdings}"

        # 总仓位空间
        current_ratio = self.current_position_ratio
        if current_ratio >= self.max_position_ratio - 0.001:
            return False, (
                f"总仓位已满 {current_ratio:.0%} >= {self.max_position_ratio:.0%}"
            )

        return True, "OK"

    def compute_entry_size(self, signal_strength: float = 0.5) -> Tuple[float, float]:
        """
        计算单次入场可分配的资金量。

        规则：
        1. 单票不超过 max_single_ratio × total_value
        2. 不超过剩余仓位空间 (max_position_ratio - current_ratio) × total_value
        3. 不超过可用现金
        4. 信号强度影响仓位：基础 = max_single_ratio × (0.5 + 0.5 × strength)
        """
        total_val = self.total_value
        if total_val <= 0:
            return 0, 0.0

        # 信号强度加权的基础仓位
        base_ratio = self.max_single_ratio * (0.5 + 0.5 * max(0.0, min(1.0, signal_strength)))

        # 约束1：不超过单票上限
        cap1 = self.max_single_ratio * total_val

        # 约束2：不超过剩余仓位空间
        remaining_ratio = self.max_position_ratio - self.current_position_ratio
        cap2 = max(0, remaining_ratio) * total_val

        # 约束3：不超过可用现金
        cap3 = self.cash

        # 取最小
        target_capital = min(base_ratio * total_val, cap1, cap2, cap3)

        if target_capital < 1000:
            return 0, 0.0

        return target_capital, target_capital

    # ── 入场 / 出场 ────────────────────────────────────────

    def enter_position(
        self,
        code: str,
        name: str,
        entry_date: str,
        entry_price: float,
        signal_strength: float = 0.5,
    ) -> Optional[PortfolioTrade]:
        """开新仓，返回 PortfolioTrade 或 None（不满足条件）"""
        ok, reason = self.can_enter(code)
        if not ok:
            return None

        _, target_capital = self.compute_entry_size(signal_strength)
        if target_capital < 1000:
            return None

        # 按整手（100股）计算
        shares = int(target_capital / entry_price / 100) * 100
        if shares <= 0:
            return None

        actual_cost = shares * entry_price
        if actual_cost > self.cash:
            shares = int(self.cash / entry_price / 100) * 100
            if shares <= 0:
                return None
            actual_cost = shares * entry_price

        # 扣款
        self.cash -= actual_cost

        # 创建持仓
        trade = PortfolioTrade(
            code=code,
            name=name,
            entry_date=entry_date,
            entry_price=entry_price,
            shares=shares,
            allocated_capital=actual_cost,
            signal_strength=signal_strength,
        )
        self.holdings[code] = trade

        pos = Position(
            code=code,
            name=name,
            entry_date=entry_date,
            entry_price=entry_price,
            shares=shares,
            cost_basis=actual_cost,
            current_price=entry_price,
            current_date=entry_date,
        )
        self._positions[code] = pos

        return trade

    def exit_position(
        self,
        code: str,
        exit_date: str,
        exit_price: float,
        reason: str = "",
        ratio: float = 1.0,   # 1.0=全部出场，0.5=减半
    ) -> Optional[PortfolioTrade]:
        """
        出场（支持部分出场）。
        ratio=1.0 全部平仓，ratio<1.0 部分减仓。
        """
        if code not in self.holdings:
            return None

        trade = self.holdings[code]
        pos = self._positions[code]

        exit_shares = int(trade.shares * ratio / 100) * 100
        if exit_shares <= 0:
            exit_shares = trade.shares if ratio >= 0.9 else 0
        if exit_shares <= 0:
            return None

        proceeds = exit_shares * exit_price
        self.cash += proceeds

        # 计算本次出场的已实现盈亏
        cost_per_share = trade.entry_price
        partial_pnl = (exit_price - cost_per_share) * exit_shares

        # 更新持仓shares
        trade.shares -= exit_shares
        pos.shares -= exit_shares

        if trade.shares <= 0 or ratio >= 0.99:
            # 完全平仓
            trade.realized_pnl += partial_pnl
            trade.close(exit_date, exit_price, reason)
            trade.pnl_amount = trade.realized_pnl
            trade.holding_days = pos.holding_days
            self.completed_trades.append(trade)
            del self.holdings[code]
            del self._positions[code]
            return trade
        else:
            # 部分出场
            trade.realized_pnl += partial_pnl
            trade.pnl_amount = trade.realized_pnl
            # 按比例减少allocated_capital
            exit_capital = trade.allocated_capital * (exit_shares / (exit_shares + trade.shares))
            trade.allocated_capital -= exit_capital
            return trade

    # ── 估值 / 状态 ────────────────────────────────────────

    def update_prices(self, date: str, price_map: Dict[str, float]):
        """用当日收盘价更新所有持仓的估值"""
        self.current_date = date
        for code, price in price_map.items():
            if code in self._positions:
                self._positions[code].update(date, price)

    @property
    def positions_market_value(self) -> float:
        return sum(
            self._positions[c].current_price * self._positions[c].shares
            for c in self._positions
        )

    @property
    def total_value(self) -> float:
        return self.cash + self.positions_market_value

    @property
    def current_position_ratio(self) -> float:
        """当前已用仓位比例 = 持仓市值 / 总市值"""
        tv = self.total_value
        if tv <= 0:
            return 0.0
        return self.positions_market_value / tv

    @property
    def positions(self) -> List[Position]:
        return list(self._positions.values())

    def get_state(self) -> PortfolioState:
        """获取当前组合快照"""
        return PortfolioState(
            cash=self.cash,
            total_value=self.total_value,
            positions=list(self._positions.values()),
            total_position_ratio=self.current_position_ratio,
            max_position_ratio=self.max_position_ratio,
            date=self.current_date,
        )

    def record_daily(self, date: str):
        """记录当日净值和仓位"""
        self.nav_history.append((date, self.total_value))
        self.position_ratio_history.append((date, self.current_position_ratio))


# ===========================================================================
# 五因子仓位简化评估（回测内联版）
# ===========================================================================
def _quick_five_factor_ratio(
    daily_df: pd.DataFrame,
    index_pct: float = 0.0,
) -> float:
    """
    快速五因子评估，返回建议仓位比例（0-1）。

    在回测中逐日调用，用截至当日的数据进行评分。
    不拉取额外数据（PE、涨停列表），只用日K和MACD信号。
    """
    if len(daily_df) < 35:
        return 0.5  # 数据不足，默认半仓

    # 因子1: 资金（成交量趋势）
    f_capital = score_capital_flow(daily_df)

    # 因子2: 技术（MACD面积）
    f_technical = score_technical(daily_df)

    # 因子3: 基本面（个股 vs 大盘，简化）
    stock_pct = float(daily_df["pct_chg"].iloc[-1]) if "pct_chg" in daily_df.columns else 0.0
    f_fundamental = score_fundamental(stock_pct, index_pct)

    # 投票（3因子简化版，凑齐5个的占位）
    factors = [f_capital, f_technical, f_fundamental]
    long_votes = sum(1 for f in factors if f.direction == Direction.LONG)
    short_votes = sum(1 for f in factors if f.direction == Direction.SHORT)

    if long_votes >= 2:
        return min(1.0, 0.8 + (long_votes - 2) * 0.1)
    elif long_votes == 1 and short_votes <= 1:
        return 0.5
    else:
        return max(0.0, 0.2 - short_votes * 0.05)


def _quick_stage_adjustment(
    daily_df: pd.DataFrame,
    recent_days: int = 20,
    signal_hold_days: int = 5,
) -> float:
    """
    快速恶魔股阶段检测，返回仓位调整系数。

    在回测中用截至当日的数据评估信号胜率。
    """
    if len(daily_df) < 60:
        return 1.0

    signals = generate_signals(daily_df)
    if not signals:
        return 1.0

    latest_date = str(daily_df["trade_date"].iloc[-1])
    cutoff_dt = pd.to_datetime(latest_date, format="%Y%m%d") - pd.Timedelta(days=recent_days)
    cutoff = cutoff_dt.strftime("%Y%m%d")

    entry_sigs = [s for s in signals if s.signal_type == "entry_candidate" and s.date >= cutoff]
    if not entry_sigs:
        return 1.0

    date_to_idx = {str(d): i for i, d in enumerate(daily_df["trade_date"].values)}
    closes = daily_df["close"].values
    n = len(closes)

    wins = 0
    evaluated = 0
    for sig in entry_sigs:
        idx = date_to_idx.get(sig.date)
        if idx is None:
            continue
        end_idx = min(idx + signal_hold_days, n - 1)
        if end_idx <= idx:
            continue
        ret = (closes[end_idx] / closes[idx] - 1) * 100
        evaluated += 1
        if ret > 0:
            wins += 1

    if evaluated == 0:
        return 1.0

    win_rate = wins / evaluated
    if win_rate < 0.30:
        return 0.5  # 恶魔股阶段
    elif win_rate <= 0.50:
        return 1.0  # 中性
    else:
        return 1.5  # 匹配阶段


# ===========================================================================
# 组合分时回测
# ===========================================================================
def run_portfolio_intraday_backtest(
    stocks: List[Dict],
    start_date: str = "20260101",
    end_date: str = "20260731",
    initial_capital: float = INITIAL_CAPITAL,
    max_holdings: int = MAX_HOLDINGS,
    max_single_ratio: float = MAX_SINGLE_RATIO,
    signal_min_strength: float = 0.4,
    exit_config: Optional[ExitConfig] = None,
    verbose: bool = False,
) -> PortfolioAnalysis:
    """
    对多只股票同时运行分时级别组合回测，加入组合约束。

    参数：
    - stocks: [{"code": "002580", "name": "圣阳股份", "daily_df": df, "minute_df": mdf}, ...]
    - start_date / end_date: 回测日期范围 YYYYMMDD
    - initial_capital: 初始资金
    - signal_min_strength: MACD信号强度阈值
    - exit_config: 出场参数

    返回: PortfolioAnalysis
    """
    if exit_config is None:
        exit_config = ExitConfig()

    # ── 准备数据 ──────────────────────────────────────────
    stock_data: Dict[str, Dict] = {}
    for s in stocks:
        code = s["code"]
        name = s.get("name", code)
        daily_df = s.get("daily_df")
        minute_df = s.get("minute_df")

        # 如果没传数据，自动拉取
        if daily_df is None:
            daily_df = load_daily(code, start_date=start_date, end_date=end_date)
        daily_df = daily_df.copy()
        daily_df["trade_date"] = daily_df["trade_date"].astype(str)
        daily_df = daily_df.sort_values("trade_date").reset_index(drop=True)

        # 信号
        all_signals = generate_signals(daily_df)
        entry_signals = {
            sig.date: sig
            for sig in all_signals
            if sig.signal_type == "entry_candidate"
            and sig.signal_strength >= signal_min_strength
        }

        # 日线收盘价查找表
        daily_close = dict(zip(daily_df["trade_date"].values, daily_df["close"].values))

        stock_data[code] = {
            "name": name,
            "daily_df": daily_df,
            "minute_df": minute_df,
            "entry_signals": entry_signals,
            "daily_close": daily_close,
            "all_signals": {sig.date: sig for sig in all_signals},
        }

    # ── 统一交易日历 ────────────────────────────────────
    all_dates_set = set()
    for sd in stock_data.values():
        df = sd["daily_df"]
        mask = (df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)
        all_dates_set.update(df.loc[mask, "trade_date"].tolist())
    all_dates = sorted(all_dates_set)

    if not all_dates:
        print("[WARN] 无可用交易日")
        return PortfolioAnalysis()

    if verbose:
        print(f"[portfolio] 回测区间 {all_dates[0]} ~ {all_dates[-1]}，共 {len(all_dates)} 个交易日")
        print(f"[portfolio] 股票池: {list(stock_data.keys())}")

    # ── 大盘指数数据（用于基本面因子）──────────────────
    try:
        index_df = load_index("000300", days=400)
        index_pct_map = dict(zip(
            index_df["trade_date"].astype(str).values,
            index_df["pct_chg"].astype(float).values,
        ))
    except Exception:
        index_pct_map = {}

    # ── 初始化组合管理器 ────────────────────────────────
    pm = PortfolioManager(
        initial_capital=initial_capital,
        max_holdings=max_holdings,
        max_single_ratio=max_single_ratio,
    )

    # ── 逐日回测 ────────────────────────────────────────
    for date in all_dates:
        # 1) 出场检查（在入场之前，释放仓位空间）
        codes_to_exit = []
        for code, trade in list(pm.holdings.items()):
            sd = stock_data[code]
            daily_df = sd["daily_df"]
            daily_close = sd["daily_close"]

            if date not in daily_close:
                continue

            current_price = daily_close[date]
            pos = pm._positions[code]

            # 检查出场条件
            profit = pos.current_profit_pct if pos.current_price > 0 else 0
            # 更新价格
            pos.update(date, current_price)
            profit = (current_price / trade.entry_price - 1) * 100

            # MACD红柱缩短检查（用日线信号）
            sig_today = sd["all_signals"].get(date)
            macd_red_shrinking = (
                sig_today is not None
                and sig_today.signal_type == "exit_warning"
                and sig_today.red_peak_shrinking
            )

            # 大盘信息
            mkt_pct = index_pct_map.get(date, 0.0)
            mkt_consec_down = _calc_consecutive_down(index_pct_map, date)

            decision = check_exit(
                position=pos,
                config=exit_config,
                macd_red_shrinking=macd_red_shrinking,
                high_open_rejected=False,
                sector_vs_market=profit - mkt_pct,
                market_change=mkt_pct,
                market_consecutive_down=mkt_consec_down,
                limit_up_probability=0.5,
                sector_strength="neutral",
            )

            if decision.action == "close":
                pm.exit_position(code, date, current_price, decision.description, ratio=1.0)
                if verbose:
                    print(f"  [{date}] EXIT {code} @ {current_price:.2f} ({profit:+.1f}%) "
                          f"| {decision.description}")
            elif decision.action == "reduce" and decision.reduce_ratio > 0:
                pm.exit_position(code, date, current_price, decision.description,
                                 ratio=decision.reduce_ratio)
                if verbose:
                    print(f"  [{date}] REDUCE {code} {decision.reduce_ratio:.0%} "
                          f"@ {current_price:.2f} | {decision.description}")

        # 2) 更新仓位上限（五因子 + 恶魔股）
        #    用第一只持仓股或第一只池内股的日K作为代理
        ref_code = list(pm.holdings.keys())[0] if pm.holdings else list(stock_data.keys())[0]
        ref_df = stock_data[ref_code]["daily_df"]
        ref_asof = ref_df[ref_df["trade_date"] <= date]
        if len(ref_asof) > 0:
            ff_ratio = _quick_five_factor_ratio(
                ref_asof,
                index_pct_map.get(date, 0.0),
            )
            stage_adj = _quick_stage_adjustment(ref_asof)
            pm.update_position_limit(date, ff_ratio, stage_adj)

        # 3) 入场检查（遍历股票池，先到先得）
        for code, sd in stock_data.items():
            if code in pm.holdings:
                continue  # 已持有

            sig = sd["entry_signals"].get(date)
            if sig is None:
                continue

            daily_close = sd["daily_close"]
            if date not in daily_close:
                continue

            entry_price = daily_close[date]

            ok, reason = pm.can_enter(code)
            if not ok:
                if verbose:
                    print(f"  [{date}] SKIP {code}: {reason}")
                continue

            trade = pm.enter_position(
                code=code,
                name=sd["name"],
                entry_date=date,
                entry_price=entry_price,
                signal_strength=sig.signal_strength,
            )
            if trade and verbose:
                print(f"  [{date}] ENTER {code}({sd['name']}) @ {entry_price:.2f} "
                      f"x{trade.shares}股 资金{trade.allocated_capital:.0f} "
                      f"强度{sig.signal_strength:.0%}")

        # 4) 更新所有持仓当日估值
        price_map = {}
        for code in list(pm.holdings.keys()):
            sd = stock_data[code]
            dc = sd["daily_close"]
            if date in dc:
                price_map[code] = dc[date]
        pm.update_prices(date, price_map)

        # 5) 记录每日净值
        pm.record_daily(date)

    # ── 期末平仓所有持仓 ──────────────────────────────
    for code in list(pm.holdings.keys()):
        sd = stock_data[code]
        last_date = all_dates[-1]
        last_price = sd["daily_close"].get(last_date, pm._positions[code].current_price)
        pm.exit_position(code, last_date, last_price, "期末平仓", ratio=1.0)

    # ── 分析统计 ──────────────────────────────────────
    analysis = _analyze_portfolio(pm, initial_capital, all_dates)

    if verbose:
        _print_analysis(analysis)

    return analysis


# ===========================================================================
# 辅助函数
# ===========================================================================
def _calc_consecutive_down(index_pct_map: Dict[str, float], date: str) -> int:
    """计算截至date的大盘连续下跌天数"""
    dates = sorted(index_pct_map.keys())
    if date not in index_pct_map:
        return 0
    idx = dates.index(date)
    count = 0
    for i in range(idx, -1, -1):
        if index_pct_map[dates[i]] < 0:
            count += 1
        else:
            break
    return count


# ===========================================================================
# 组合统计分析
# ===========================================================================
def _analyze_portfolio(
    pm: PortfolioManager,
    initial_capital: float,
    all_dates: List[str],
) -> PortfolioAnalysis:
    """从 PortfolioManager 的历史记录计算统计分析"""

    # ── 净值曲线 ──
    if not pm.nav_history:
        return PortfolioAnalysis(trades=pm.completed_trades)

    nav_dates = [d for d, _ in pm.nav_history]
    nav_values = [v for _, v in pm.nav_history]
    nav_series = pd.Series(nav_values, index=nav_dates, name="nav")
    nav_series = nav_series / initial_capital  # 归一化为1.0起始

    # 日收益率
    daily_ret = nav_series.pct_change().dropna()

    # ── 核心统计 ──
    total_return = (nav_series.iloc[-1] - 1.0) * 100

    n_days = len(nav_series)
    years = max(n_days / 252, 1 / 252)
    if nav_series.iloc[-1] > 0:
        annual_return = ((nav_series.iloc[-1] ** (1 / years)) - 1) * 100
    else:
        annual_return = 0.0

    vol = daily_ret.std() * np.sqrt(252) * 100 if len(daily_ret) > 1 else 0.0

    rf_daily = 0.02 / 252
    excess = daily_ret - rf_daily
    if len(daily_ret) > 1 and daily_ret.std() > 0:
        sharpe = (excess.mean() / daily_ret.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    cummax = nav_series.cummax()
    drawdown = (nav_series - cummax) / cummax
    max_dd = drawdown.min() * 100

    # ── 仓位利用率 ──
    pos_ratios = [r for _, r in pm.position_ratio_history]
    avg_util = np.mean(pos_ratios) * 100 if pos_ratios else 0.0
    max_util = max(pos_ratios) * 100 if pos_ratios else 0.0

    # ── 交易统计 ──
    trades = pm.completed_trades
    total_trades = len(trades)
    wins = [t for t in trades if t.pnl_pct > 0]
    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0.0

    # 换手率（年化）= 总交易额 / 平均总市值 × (252 / 天数)
    total_turnover = sum(t.allocated_capital for t in trades)
    avg_nav = np.mean(nav_values) if nav_values else initial_capital
    turnover_annual = (total_turnover / avg_nav) * (252 / max(n_days, 1)) if avg_nav > 0 else 0.0

    # ── 持仓贡献 ──
    stock_pnl: Dict[str, float] = {}
    stock_names: Dict[str, str] = {}
    for t in trades:
        stock_pnl[t.code] = stock_pnl.get(t.code, 0) + t.pnl_amount
        stock_names[t.code] = t.name

    best_stock = None
    worst_stock = None
    if stock_pnl:
        best_code = max(stock_pnl, key=stock_pnl.get)
        worst_code = min(stock_pnl, key=stock_pnl.get)
        best_stock = {"code": best_code, "name": stock_names.get(best_code, ""),
                      "pnl_amount": stock_pnl[best_code]}
        worst_stock = {"code": worst_code, "name": stock_names.get(worst_code, ""),
                       "pnl_amount": stock_pnl[worst_code]}

    # ── 月度收益 ──
    monthly_df = _calc_monthly_returns(nav_series)

    return PortfolioAnalysis(
        total_return_pct=round(total_return, 2),
        annual_return_pct=round(annual_return, 2),
        max_drawdown_pct=round(max_dd, 2),
        sharpe_ratio=round(sharpe, 3),
        volatility_pct=round(vol, 2),
        daily_nav=nav_series,
        daily_returns=daily_ret,
        monthly_returns=monthly_df,
        avg_position_utilization=round(avg_util, 1),
        max_position_utilization=round(max_util, 1),
        total_trades=total_trades,
        win_rate=round(win_rate, 1),
        turnover_rate=round(turnover_annual, 2),
        best_stock=best_stock,
        worst_stock=worst_stock,
        trades=trades,
    )


def _calc_monthly_returns(nav_series: pd.Series) -> pd.DataFrame:
    """计算分月收益分布"""
    if nav_series is None or len(nav_series) == 0:
        return pd.DataFrame()

    # 将index转为datetime
    idx = pd.to_datetime(nav_series.index, format="%Y%m%d")
    s = pd.Series(nav_series.values, index=idx)

    # 月末净值
    monthly_nav = s.resample("ME").last()
    # 月初净值（=上月末净值），首月用起始净值1.0
    monthly_nav_prev = monthly_nav.shift(1).fillna(1.0)

    monthly_ret = (monthly_nav / monthly_nav_prev - 1) * 100

    df = pd.DataFrame({
        "month": monthly_ret.index.strftime("%Y-%m"),
        "return_pct": monthly_ret.values.round(2),
        "nav": monthly_nav.values.round(4),
    })
    return df


# ===========================================================================
# 打印
# ===========================================================================
def _print_analysis(a: PortfolioAnalysis):
    """打印组合分析结果"""
    print(f"\n{'='*65}")
    print(f"  组合级分时回测分析报告")
    print(f"{'='*65}")

    print(f"\n  ── 核心指标 ──")
    print(f"  总收益率:     {a.total_return_pct:+.2f}%")
    print(f"  年化收益:     {a.annual_return_pct:+.2f}%")
    print(f"  最大回撤:     {a.max_drawdown_pct:.2f}%")
    print(f"  夏普比率:     {a.sharpe_ratio:.3f}")
    print(f"  年化波动率:   {a.volatility_pct:.2f}%")

    print(f"\n  ── 仓位管理 ──")
    print(f"  平均仓位利用率: {a.avg_position_utilization:.1f}%")
    print(f"  最大仓位利用率: {a.max_position_utilization:.1f}%")
    print(f"  年化换手率:     {a.turnover_rate:.2f}")

    print(f"\n  ── 交易统计 ──")
    print(f"  总交易数:  {a.total_trades}")
    print(f"  胜率:      {a.win_rate:.1f}%")

    if a.best_stock:
        print(f"\n  ── 持仓贡献 ──")
        print(f"  最佳: {a.best_stock['code']}({a.best_stock['name']}) "
              f"盈亏 {a.best_stock['pnl_amount']:+,.0f}")
        print(f"  最差: {a.worst_stock['code']}({a.worst_stock['name']}) "
              f"盈亏 {a.worst_stock['pnl_amount']:+,.0f}")

    # 月度收益
    if a.monthly_returns is not None and len(a.monthly_returns) > 0:
        print(f"\n  ── 分月收益 ──")
        print(f"  {'月份':10s} {'收益%':>8s} {'月末净值':>10s}")
        print(f"  {'-'*32}")
        for _, row in a.monthly_returns.iterrows():
            print(f"  {row['month']:10s} {row['return_pct']:+7.2f}% {row['nav']:10.4f}")

    # 交易明细
    if a.trades:
        print(f"\n  ── 交易明细 ──")
        print(f"  {'代码':8s} {'入场日':10s} {'入场价':>8s} {'出场日':10s} "
              f"{'出场价':>8s} {'收益%':>7s} {'盈亏额':>10s} {'原因'}")
        print(f"  {'-'*85}")
        for t in a.trades:
            print(f"  {t.code:8s} {t.entry_date:10s} {t.entry_price:8.2f} "
                  f"{t.exit_date:10s} {t.exit_price:8.2f} "
                  f"{t.pnl_pct:+6.2f}% {t.pnl_amount:+10,.0f} {t.exit_reason[:20]}")

    print(f"\n{'='*65}")


# ===========================================================================
# CLI / 验证
# ===========================================================================
def _load_test_stocks(codes: List[str], start_date: str, end_date: str) -> List[Dict]:
    """加载测试股票数据"""
    from src.backtest.intraday_simulator import load_minute_data

    stocks = []
    for code in codes:
        print(f"\n  加载 {code} ...")
        try:
            daily_df = load_daily(code, start_date=start_date, end_date=end_date)
        except Exception as e:
            print(f"  [WARN] {code} 日线数据加载失败: {e}")
            continue

        # 尝试加载分钟数据，失败则跳过
        try:
            minute_df = load_minute_data(code, start_date=start_date, end_date=end_date)
        except Exception as e:
            print(f"  [INFO] {code} 分钟数据加载失败: {e}，使用日线模式")
            minute_df = pd.DataFrame()

        stocks.append({
            "code": code,
            "name": code,  # 简化，实际可从tushare拿名称
            "daily_df": daily_df,
            "minute_df": minute_df,
        })
        print(f"  {code}: 日线{len(daily_df)}行, 分钟{len(minute_df)}行")

    return stocks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="组合级仓位管理分时回测")
    parser.add_argument("--codes", nargs="+", default=["002580", "601727", "600839", "601127", "600460"],
                        help="股票代码列表")
    parser.add_argument("--start", default="20260101", help="起始日期")
    parser.add_argument("--end", default="20260731", help="结束日期")
    parser.add_argument("--capital", type=float, default=INITIAL_CAPITAL, help="初始资金")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    args = parser.parse_args()

    print(f"\n{'#'*65}")
    print(f"  组合级分时回测验证")
    print(f"  股票: {args.codes}")
    print(f"  区间: {args.start} ~ {args.end}")
    print(f"  资金: {args.capital:,.0f}")
    print(f"{'#'*65}")

    stocks = _load_test_stocks(args.codes, args.start, args.end)
    if not stocks:
        print("无可用数据，退出")
        sys.exit(1)

    analysis = run_portfolio_intraday_backtest(
        stocks=stocks,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        verbose=args.verbose,
    )

    _print_analysis(analysis)
