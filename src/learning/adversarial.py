#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对抗验证器（Adversarial Validator）

灵感来源：
- LOOP文章：对抗辩论——让AI主动找反例，而不是只看支持信号
- 用户要求："给定一个MACD面积信号，让程序自动寻找在什么情况下这个信号会亏钱"
- 桥水PAT：信号失效场景的持续监控

核心逻辑：
  1. 扫描历史数据中所有 entry_candidate 信号
  2. 对每个信号，计算入场后N天的实际收益
  3. 将亏损的交易按特征维度分组（大盘连续下跌/PE分位/换手率/价格位置等）
  4. 输出信号失效场景Top-N，每个场景带样本数和平均亏损

输出示例：
  信号失效场景Top3：
  1. 大盘连续下跌≥3天时入场：5次样本，平均-8.2%
  2. PE历史分位>80%时入场：3次样本，平均-11.5%
  3. 换手率>20%时入场：4次样本，平均-6.7%
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd

# 项目内模块
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from signals.macd_area import generate_signals, calc_macd, MACDAreaSignal
from data.loader import load_daily, load_index, _init_tushare, _to_ts_code, _CACHE_DIR


# ===========================================================================
# 数据结构
# ===========================================================================
@dataclass
class SignalOutcome:
    """一个entry_candidate信号的实际结果"""
    date: str
    price: float
    signal_strength: float
    price_position: float           # 价格位置 0-1
    green_peak_area: float          # 前序绿峰面积
    green_peak_severity: str        # deep/moderate/shallow

    # 入场后N天收益
    forward_returns: Dict[int, float] = field(default_factory=dict)  # {5: 2.3, 10: -1.5, 20: ...}

    # 特征维度（用于对抗分析）
    market_consecutive_down: int = 0       # 入场时大盘连续下跌天数
    market_5d_chg: float = 0.0             # 大盘近5日涨跌幅
    turnover_ratio: float = 1.0            # 换手率比率（近5日均量 / 近20日均量）
    pe_percentile: Optional[float] = None  # PE历史分位
    vol_relative: float = 1.0              # 当日成交量 / 近20日均量
    macd_bar_value: float = 0.0            # 当日MACD柱值
    price_above_ma20: bool = False         # 价格是否在20日均线上方
    consecutive_up_days: int = 0           # 入场前个股连涨天数

    # 结果标记
    is_loss: bool = False                  # 是否亏损（基于forward_returns中的关键周期）
    worst_return: float = 0.0              # 最大亏损幅度


@dataclass
class FailureScenario:
    """信号失效场景"""
    name: str                      # 场景描述
    condition: str                 # 条件描述
    total_samples: int             # 总信号样本数
    failure_samples: int           # 失效样本数
    failure_rate: float            # 失效率 %
    avg_loss: float                # 平均亏损 %（仅亏损样本）
    avg_return: float              # 平均收益 %（所有该场景下信号）
    codes: List[str] = field(default_factory=list)  # 涉及的股票代码


@dataclass
class AdversarialReport:
    """对抗验证报告"""
    code: str
    total_signals: int
    overall_loss_rate: float              # 整体亏损率
    overall_avg_return: float             # 整体平均收益
    failure_scenarios: List[FailureScenario] = field(default_factory=list)
    outcomes: List[SignalOutcome] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"\n{'═' * 60}",
            f"  🛡️  对抗验证报告  · {self.code}",
            f"{'═' * 60}",
            f"  信号总数: {self.total_signals}",
            f"  整体亏损率: {self.overall_loss_rate:.1f}%",
            f"  整体平均收益: {self.overall_avg_return:+.2f}%",
        ]

        if not self.failure_scenarios:
            lines.append("  ✅ 未发现显著失效场景")
        else:
            lines.append(f"\n  信号失效场景 Top{len(self.failure_scenarios)}:")
            for i, sc in enumerate(self.failure_scenarios, 1):
                lines.append(
                    f"    {i}. {sc.name}"
                    f"：{sc.failure_samples}次样本"
                    f"，平均{sc.avg_return:+.1f}%"
                    f"，失效率{sc.failure_rate:.0f}%"
                )

        lines.append(f"{'═' * 60}")
        return "\n".join(lines)


# ===========================================================================
# 核心分析引擎
# ===========================================================================
def _compute_forward_returns(
    close: np.ndarray,
    dates: np.ndarray,
    entry_idx: int,
    periods: Optional[List[int]] = None,
) -> Dict[int, float]:
    if periods is None:
        periods = [5, 10, 20]
    """计算入场后N天的收益"""
    returns = {}
    entry_price = close[entry_idx]
    for p in periods:
        target_idx = entry_idx + p
        if target_idx < len(close) and entry_price > 0:
            returns[p] = round((close[target_idx] / entry_price - 1) * 100, 2)
        else:
            returns[p] = None
    return returns


def _compute_market_features(
    market_df: pd.DataFrame,
    date: str,
    lookback: int = 5,
) -> Tuple[int, float]:
    """计算大盘特征：连续下跌天数、近N日涨跌幅"""
    if market_df is None or len(market_df) == 0:
        return 0, 0.0

    market_df = market_df.sort_values('trade_date').reset_index(drop=True)
    idx = market_df.index[market_df['trade_date'] == date].tolist()
    if not idx:
        # 找最近的
        mask = market_df['trade_date'] <= date
        if mask.sum() == 0:
            return 0, 0.0
        idx = [mask.values.nonzero()[0][-1]]

    i = idx[0]

    # 连续下跌天数
    consec_down = 0
    j = i
    while j >= 0:
        pct = market_df.iloc[j].get('pct_chg', 0)
        if pd.isna(pct):
            pct = 0
        if float(pct) < 0:
            consec_down += 1
            j -= 1
        else:
            break

    # 近N日涨跌幅
    start_i = max(0, i - lookback)
    if i > start_i:
        close_series = market_df['close'].astype(float).values
        mkt_chg = (close_series[i] / close_series[start_i] - 1) * 100
    else:
        mkt_chg = 0.0

    return consec_down, round(mkt_chg, 2)


def _compute_volume_features(df: pd.DataFrame, idx: int) -> Tuple[float, float]:
    """计算成交量特征：换手率比率（近5日均量/近20日均量）、当日量比"""
    if 'vol' not in df.columns or len(df) < 25:
        return 1.0, 1.0

    vol = df['vol'].astype(float).values
    if idx < 20:
        return 1.0, 1.0

    recent_5 = np.mean(vol[max(0, idx-4):idx+1])
    recent_20 = np.mean(vol[max(0, idx-19):idx+1])

    turnover_ratio = recent_5 / recent_20 if recent_20 > 0 else 1.0
    vol_relative = vol[idx] / recent_20 if recent_20 > 0 else 1.0

    return round(turnover_ratio, 3), round(vol_relative, 3)


def _compute_pe_percentile(code: str, date: str) -> Optional[float]:
    """计算PE历史分位"""
    pe_cache = _CACHE_DIR / f"pe_{code}.csv"
    if not pe_cache.exists():
        return None

    try:
        pe_df = pd.read_csv(pe_cache, dtype={'trade_date': str})
        pe_df = pe_df.sort_values('trade_date').reset_index(drop=True)

        # 找到 date 当天或之前最近的PE
        mask = pe_df['trade_date'] <= date
        if mask.sum() < 20:
            return None

        idx = mask.values.nonzero()[0][-1]
        current_pe = pe_df.iloc[idx].get('pe_ttm', None)
        if current_pe is None or pd.isna(current_pe) or float(current_pe) <= 0:
            return None

        # 近250个交易日的分位
        window_start = max(0, idx - 250)
        window = pe_df.iloc[window_start:idx+1]['pe_ttm'].dropna()
        window = window[window > 0]
        if len(window) < 20:
            return None

        percentile = (window < float(current_pe)).sum() / len(window)
        return round(float(percentile), 4)
    except Exception:
        return None


def _compute_price_features(close: np.ndarray, idx: int) -> Tuple[bool, int]:
    """计算价格特征：是否在MA20上方、入场前个股连涨天数"""
    if idx < 20:
        return False, 0

    ma20 = np.mean(close[max(0, idx-19):idx+1])
    above_ma20 = close[idx] > ma20

    # 连涨天数
    consec_up = 0
    j = idx
    while j > 0:
        if close[j] > close[j-1]:
            consec_up += 1
            j -= 1
        else:
            break

    return bool(above_ma20), consec_up


# ===========================================================================
# 主分析函数
# ===========================================================================
def analyze_signals(
    code: str,
    df: pd.DataFrame,
    market_df: Optional[pd.DataFrame] = None,
    forward_periods: Tuple[int, ...] = (5, 10, 20),
    eval_period: int = 10,
    loss_threshold: float = 0.0,
) -> List[SignalOutcome]:
    """
    分析个股所有 entry_candidate 信号的实际结果。

    参数：
      code           : 股票代码
      df             : 个股日K数据
      market_df      : 大盘日K数据（可选）
      forward_periods: 前瞻收益计算周期
      eval_period    : 用于判定盈亏的评估周期（默认10天）
      loss_threshold : 亏损判定阈值（收益<=此值视为亏损）

    返回：SignalOutcome 列表
    """
    df = df.sort_values('trade_date').reset_index(drop=True)
    close = df['close'].astype(float).values
    dates = df['trade_date'].astype(str).values

    # 生成信号
    signals = generate_signals(df)
    sig_map = {s.date: s for s in signals}

    # MACD值
    _, _, macd_bar = calc_macd(close)

    outcomes = []

    for i in range(len(df)):
        date = dates[i]
        sig = sig_map.get(date)

        if sig is None or sig.signal_type != "entry_candidate":
            continue

        # 前瞻收益
        fwd = _compute_forward_returns(close, dates, i, list(forward_periods))

        # 评估周期收益
        eval_return = fwd.get(eval_period, None)
        if eval_return is None:
            continue  # 数据不够，跳过

        is_loss = eval_return <= loss_threshold
        worst = min(v for v in fwd.values() if v is not None) if fwd else 0.0

        # 大盘特征
        mkt_down, mkt_5d = _compute_market_features(market_df, date) if market_df is not None else (0, 0.0)

        # 成交量特征
        turnover_ratio, vol_rel = _compute_volume_features(df, i)

        # PE分位
        pe_pct = _compute_pe_percentile(code, date)

        # 价格特征
        above_ma20, consec_up = _compute_price_features(close, i)

        outcome = SignalOutcome(
            date=date,
            price=round(close[i], 2),
            signal_strength=sig.signal_strength,
            price_position=sig.price_position,
            green_peak_area=sig.green_peak_area,
            green_peak_severity=sig.green_peak_severity,
            forward_returns=fwd,
            market_consecutive_down=mkt_down,
            market_5d_chg=mkt_5d,
            turnover_ratio=turnover_ratio,
            pe_percentile=pe_pct,
            vol_relative=vol_rel,
            macd_bar_value=round(macd_bar[i], 4),
            price_above_ma20=above_ma20,
            consecutive_up_days=consec_up,
            is_loss=is_loss,
            worst_return=round(worst, 2),
        )
        outcomes.append(outcome)

    return outcomes


def find_failure_scenarios(
    outcomes: List[SignalOutcome],
    min_samples: int = 2,
    top_n: int = 5,
) -> List[FailureScenario]:
    """
    从信号结果中自动发现信号失效场景。

    按多个特征维度切片，找出亏损集中出现的条件。

    参数：
      outcomes   : analyze_signals 的输出
      min_samples: 最小样本数
      top_n      : 返回的场景数上限
    """
    if not outcomes:
        return []

    total = len(outcomes)

    # 定义特征切片规则
    # 每条规则：(场景名, 条件描述, 过滤函数)
    rules: List[Tuple[str, str, Any]] = [
        (
            "大盘连续下跌≥3天时入场",
            "market_consecutive_down >= 3",
            lambda o: o.market_consecutive_down >= 3,
        ),
        (
            "大盘连续下跌≥2天时入场",
            "market_consecutive_down >= 2",
            lambda o: o.market_consecutive_down >= 2,
        ),
        (
            "大盘近5日跌幅>3%时入场",
            "market_5d_chg < -3",
            lambda o: o.market_5d_chg < -3,
        ),
        (
            "换手率比率>1.5时入场（放量入场）",
            "turnover_ratio > 1.5",
            lambda o: o.turnover_ratio > 1.5,
        ),
        (
            "换手率比率>2.0时入场（急剧放量）",
            "turnover_ratio > 2.0",
            lambda o: o.turnover_ratio > 2.0,
        ),
        (
            "当日成交量>1.5倍均量",
            "vol_relative > 1.5",
            lambda o: o.vol_relative > 1.5,
        ),
        (
            "PE历史分位>80%时入场",
            "pe_percentile > 0.8",
            lambda o: o.pe_percentile is not None and o.pe_percentile > 0.8,
        ),
        (
            "PE历史分位>60%时入场",
            "pe_percentile > 0.6",
            lambda o: o.pe_percentile is not None and o.pe_percentile > 0.6,
        ),
        (
            "价格位置>70%时入场（高位）",
            "price_position > 0.7",
            lambda o: o.price_position > 0.7,
        ),
        (
            "信号强度<0.4时入场（弱信号）",
            "signal_strength < 0.4",
            lambda o: o.signal_strength < 0.4,
        ),
        (
            "价格在MA20上方入场",
            "price_above_ma20 == True",
            lambda o: o.price_above_ma20,
        ),
        (
            "入场前个股连涨≥3天",
            "consecutive_up_days >= 3",
            lambda o: o.consecutive_up_days >= 3,
        ),
        (
            "绿峰面积小(shallow)时入场",
            "green_peak_severity == 'shallow'",
            lambda o: o.green_peak_severity == "shallow",
        ),
        (
            "绿峰面积大(deep)时入场",
            "green_peak_severity == 'deep'",
            lambda o: o.green_peak_severity == "deep",
        ),
    ]

    scenarios = []
    for name, condition, filter_fn in rules:
        subset = [o for o in outcomes if filter_fn(o)]
        if len(subset) < min_samples:
            continue

        losses = [o for o in subset if o.is_loss]
        loss_returns = [o.forward_returns.get(10, 0) or 0 for o in losses]

        avg_return = sum(o.forward_returns.get(10, 0) or 0 for o in subset) / len(subset)
        avg_loss = sum(loss_returns) / len(loss_returns) if loss_returns else 0.0
        failure_rate = len(losses) / len(subset) * 100

        scenarios.append(FailureScenario(
            name=name,
            condition=condition,
            total_samples=total,
            failure_samples=len(losses),
            failure_rate=round(failure_rate, 1),
            avg_loss=round(avg_loss, 2),
            avg_return=round(avg_return, 2),
        ))

    # 按 avg_return 升序（最亏的排前面）
    scenarios.sort(key=lambda x: x.avg_return)

    return scenarios[:top_n]


def generate_report(
    code: str,
    df: pd.DataFrame,
    market_df: Optional[pd.DataFrame] = None,
    eval_period: int = 10,
    loss_threshold: float = 0.0,
    min_samples: int = 2,
    top_n: int = 5,
) -> AdversarialReport:
    """
    一站式：分析信号 → 发现失效场景 → 生成报告。

    参数：
      code          : 股票代码
      df            : 个股日K
      market_df     : 大盘日K（可选）
      eval_period   : 盈亏评估周期（天）
      loss_threshold: 亏损阈值
      min_samples   : 最小样本数
      top_n         : 返回Top-N场景
    """
    outcomes = analyze_signals(
        code=code, df=df, market_df=market_df,
        eval_period=eval_period, loss_threshold=loss_threshold,
    )

    scenarios = find_failure_scenarios(outcomes, min_samples=min_samples, top_n=top_n)

    total = len(outcomes)
    losses = [o for o in outcomes if o.is_loss]
    all_returns = [o.forward_returns.get(eval_period, 0) or 0 for o in outcomes]
    avg_return = sum(all_returns) / len(all_returns) if all_returns else 0.0

    return AdversarialReport(
        code=code,
        total_signals=total,
        overall_loss_rate=round(len(losses) / total * 100, 1) if total > 0 else 0.0,
        overall_avg_return=round(avg_return, 2),
        failure_scenarios=scenarios,
        outcomes=outcomes,
    )


# ===========================================================================
# 便捷入口
# ===========================================================================
def run(
    code: str,
    use_cache: bool = True,
    index_code: str = "000300",
    eval_period: int = 10,
    top_n: int = 5,
) -> AdversarialReport:
    """
    便捷入口：加载数据 → 分析 → 返回报告。

    参数：
      code       : 股票代码
      use_cache  : 是否使用缓存数据
      index_code : 基准指数代码
      eval_period: 评估周期
      top_n      : 失效场景数
    """
    df = load_daily(code, use_cache=use_cache)
    market_df = None
    try:
        market_df = load_index(index_code, days=730)
    except Exception:
        pass

    return generate_report(
        code=code, df=df, market_df=market_df,
        eval_period=eval_period, top_n=top_n,
    )


# ===========================================================================
# CLI 入口
# ===========================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="对抗验证器：自动寻找MACD信号失效场景")
    parser.add_argument("code", help="股票代码，如 002580")
    parser.add_argument("--index", default="000300", help="基准指数（默认沪深300）")
    parser.add_argument("--no-cache", action="store_true", help="不使用缓存")
    parser.add_argument("--eval-period", type=int, default=10, help="盈亏评估周期")
    parser.add_argument("--top", type=int, default=5, help="失效场景Top-N")
    args = parser.parse_args()

    report = run(
        code=args.code,
        use_cache=not args.no_cache,
        index_code=args.index,
        eval_period=args.eval_period,
        top_n=args.top,
    )
    print(report.summary())
