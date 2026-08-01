#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
恶魔股阶段识别器

功能：
- 检测MACD面积策略与当前市场的匹配度
- 用近期（20天）该策略产生的信号胜率作为匹配度指标
- 近期信号胜率<30% → "恶魔股阶段"（低匹配度，应降仓）
- 近期信号胜率30-50% → "中性阶段"
- 近期信号胜率>50% → "匹配阶段"（可正常或加仓）
- 输出StageResult数据类：阶段类型、匹配度、建议仓位调整系数
"""
from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.signals.macd_area import generate_signals, MACDAreaSignal
from src.data.loader import load_daily


class MarketStage(Enum):
    """市场阶段类型"""
    DEMON = "恶魔股阶段"       # 低匹配度，应降仓
    NEUTRAL = "中性阶段"       # 中等匹配度
    MATCHED = "匹配阶段"       # 高匹配度，可正常或加仓


@dataclass
class StageResult:
    """阶段识别结果"""
    stage: MarketStage                 # 阶段类型
    win_rate: float                    # 近期信号胜率 (%)
    total_signals: int                 # 近期信号总数
    winning_signals: int               # 盈利信号数
    position_adjustment: float         # 建议仓位调整系数 (0.0-1.5)
    signal_hold_days: int              # 信号持有天数（用于判定胜负）
    description: str = ""


def _evaluate_signal_outcome(
    df: pd.DataFrame, signal_idx: int, hold_days: int = 5
) -> Optional[float]:
    """
    评估信号在持有 hold_days 天后的收益。
    如果剩余数据不足 hold_days，至少要后续1根K线才能评估（用最后可用的一天）。

    Returns:
        收益率 (%)，如果连1根后续K线都没有返回 None
    """
    n = len(df)
    end_idx = min(signal_idx + hold_days, n - 1)
    if end_idx <= signal_idx:
        return None  # 后续无数据

    entry_price = df.iloc[signal_idx]["close"]
    exit_price = df.iloc[end_idx]["close"]
    if entry_price <= 0:
        return None
    return (exit_price / entry_price - 1) * 100


def detect_stage(
    code: str,
    recent_days: int = 20,
    signal_hold_days: int = 5,
    win_threshold: float = 0.0,
    df: Optional[pd.DataFrame] = None,
) -> StageResult:
    """
    识别个股当前所处阶段。

    逻辑：
    1. 用 MACD 面积策略在个股上生成信号
    2. 取近 recent_days 天的 entry_candidate 信号
    3. 每个信号评估持有 signal_hold_days 天后收益
    4. 收益 > win_threshold 计为"胜"
    5. 胜率 < 30% → 恶魔股阶段（仓位系数 0.5）
       胜率 30-50% → 中性阶段（仓位系数 1.0）
       胜率 > 50% → 匹配阶段（仓位系数 1.5）

    Args:
        code: 股票代码
        recent_days: 近期天数窗口
        signal_hold_days: 每个信号持有天数
        win_threshold: 胜负判定阈值（收益率%），默认0（正收益即胜）
        df: 可选，已加载的日K数据

    Returns:
        StageResult
    """
    if df is None:
        df = load_daily(code)

    if len(df) < 60:
        return StageResult(
            stage=MarketStage.NEUTRAL,
            win_rate=0.0,
            total_signals=0,
            winning_signals=0,
            position_adjustment=1.0,
            signal_hold_days=signal_hold_days,
            description=f"数据不足({len(df)}行)，默认中性阶段",
        )

    # 生成 MACD 面积信号
    signals = generate_signals(df)
    if not signals:
        return StageResult(
            stage=MarketStage.NEUTRAL,
            win_rate=0.0,
            total_signals=0,
            winning_signals=0,
            position_adjustment=1.0,
            signal_hold_days=signal_hold_days,
            description="无信号生成，默认中性阶段",
        )

    # 信号日期 -> df index 映射
    date_to_idx = {str(d): i for i, d in enumerate(df["trade_date"].values)}

    # 筛选近 recent_days 天的入场信号
    latest_date = df["trade_date"].iloc[-1]
    latest_date_dt = pd.to_datetime(latest_date, format="%Y%m%d")
    cutoff = (latest_date_dt - pd.Timedelta(days=recent_days)).strftime("%Y%m%d")

    entry_signals = [
        s for s in signals
        if s.signal_type == "entry_candidate" and s.date >= cutoff
    ]

    if not entry_signals:
        # 没有近期信号，说明策略近期没产生入场机会
        return StageResult(
            stage=MarketStage.NEUTRAL,
            win_rate=0.0,
            total_signals=0,
            winning_signals=0,
            position_adjustment=1.0,
            signal_hold_days=signal_hold_days,
            description=f"近{recent_days}天无入场信号，默认中性阶段",
        )

    # 评估每个信号
    wins = 0
    evaluated = 0
    for sig in entry_signals:
        idx = date_to_idx.get(sig.date)
        if idx is None:
            continue
        ret = _evaluate_signal_outcome(df, idx, signal_hold_days)
        if ret is not None:
            evaluated += 1
            if ret > win_threshold:
                wins += 1

    if evaluated == 0:
        return StageResult(
            stage=MarketStage.NEUTRAL,
            win_rate=0.0,
            total_signals=len(entry_signals),
            winning_signals=0,
            position_adjustment=1.0,
            signal_hold_days=signal_hold_days,
            description="信号均无法评估（数据不足），默认中性阶段",
        )

    win_rate = round(wins / evaluated * 100, 1)

    # 阶段判定
    if win_rate < 30:
        stage = MarketStage.DEMON
        adj = 0.5
        desc = f"恶魔股阶段：近期胜率仅{win_rate}%，策略与当前走势不匹配，建议降仓至50%"
    elif win_rate <= 50:
        stage = MarketStage.NEUTRAL
        adj = 1.0
        desc = f"中性阶段：近期胜率{win_rate}%，策略匹配度一般，维持标准仓位"
    else:
        stage = MarketStage.MATCHED
        adj = 1.5
        desc = f"匹配阶段：近期胜率{win_rate}%，策略高度匹配当前走势，可加仓50%"

    return StageResult(
        stage=stage,
        win_rate=win_rate,
        total_signals=evaluated,
        winning_signals=wins,
        position_adjustment=adj,
        signal_hold_days=signal_hold_days,
        description=desc,
    )


def detect_stage_batch(
    codes: List[str], recent_days: int = 20, signal_hold_days: int = 5
) -> List[StageResult]:
    """批量识别多只股票的阶段"""
    results = []
    for code in codes:
        try:
            r = detect_stage(code, recent_days, signal_hold_days)
            results.append(r)
        except Exception as e:
            print(f"  [WARN] {code} 识别失败: {e}")
            results.append(StageResult(
                stage=MarketStage.NEUTRAL,
                win_rate=0.0,
                total_signals=0,
                winning_signals=0,
                position_adjustment=1.0,
                signal_hold_days=signal_hold_days,
                description=f"识别失败: {e}",
            ))
    return results


if __name__ == "__main__":
    code = "002580"
    result = detect_stage(code)
    print(f"\n{'='*60}")
    print(f"股票 {code} 阶段识别结果：")
    print(f"  阶段类型: {result.stage.value}")
    print(f"  近期胜率: {result.win_rate}% ({result.winning_signals}/{result.total_signals})")
    print(f"  仓位系数: {result.position_adjustment}")
    print(f"  说明: {result.description}")
    print(f"{'='*60}")
