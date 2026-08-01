#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""动态出场引擎

核心逻辑（来自用户实战经验）：
出场不是固定时间（5天/10天），而是条件驱动的。

用户的真实出场条件（按优先级）：
1. 止损触发：跌破入场价X% → 立即离场（铁律）
2. 目标利润+市场环境：涨到~20% → 看连板概率/板块热度决定止盈还是继续
3. 动能衰竭：分时MACD红柱连续缩短 → 减仓
4. 高开回落：高开后回落（如圣阳18.56→16.92）→ 冲高不追就走
5. 板块走弱：持仓票所在板块明显弱于大盘 → 减仓
6. 大盘恶化：大盘系统性风险 → 降仓位

设计：每天检查所有条件，任一触发就执行对应操作（减仓/清仓）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class ExitReason(Enum):
    STOP_LOSS = "止损"
    TAKE_PROFIT = "止盈"
    MOMENTUM_FAIL = "动能衰竭"
    HIGH_OPEN_REJECT = "高开回落"
    SECTOR_WEAK = "板块走弱"
    MARKET_RISK = "大盘风险"
    HOLD = "继续持有"


@dataclass
class Position:
    """持仓状态"""
    code: str
    name: str
    entry_date: str
    entry_price: float
    shares: int
    cost_basis: float       # 总成本
    current_price: float = 0.0
    current_date: str = ""
    max_price: float = 0.0  # 持仓期间最高价
    min_price: float = 999.0 # 持仓期间最低价
    holding_days: int = 0
    max_profit_pct: float = 0.0  # 最大浮盈%
    
    def update(self, date: str, price: float):
        self.current_date = date
        self.current_price = price
        self.max_price = max(self.max_price, price)
        self.min_price = min(self.min_price, price)
        self.holding_days += 1
        self.max_profit_pct = max(self.max_profit_pct, 
                                  (self.max_price / self.entry_price - 1) * 100)
    
    @property
    def current_profit_pct(self) -> float:
        if self.entry_price <= 0:
            return 0
        return (self.current_price / self.entry_price - 1) * 100
    
    @property
    def drawdown_from_peak(self) -> float:
        """从最高点的回撤%"""
        if self.max_price <= 0:
            return 0
        return (self.current_price / self.max_price - 1) * 100


@dataclass
class ExitConfig:
    """出场参数配置"""
    # 止损
    stop_loss_pct: float = -8.0        # 跌破入场价8%止损
    
    # 止盈
    take_profit_pct: float = 20.0      # 到20%开始考虑止盈
    take_profit_full: float = 30.0     # 到30%强制止盈
    
    # 动能衰竭
    red_bar_shrink_days: int = 2       # 日线MACD红柱连续缩短N天=动能衰竭
    
    # 高开回落
    high_open_threshold: float = 5.0   # 高开超过5%后回落
    high_open_reject_pct: float = 3.0  # 从高点回落超过3%=高开回落确认
    
    # 板块走弱
    sector_underperform_days: int = 2  # 板块连续跑输大盘N天
    sector_underperform_pct: float = -2.0  # 板块跑输大盘2%
    
    # 大盘风险
    market_drop_pct: float = -2.0      # 大盘单日跌超2%=风险信号
    market_drop_days: int = 3          # 大盘连续下跌N天
    
    # 仓位调整
    partial_exit_ratio: float = 0.5    # 减仓比例（减一半）


@dataclass 
class ExitDecision:
    """出场决策"""
    action: str        # "hold" / "reduce" / "close"
    reason: ExitReason
    reduce_ratio: float = 0.0  # 减仓比例
    description: str = ""


def check_exit(position: Position, config: ExitConfig,
               macd_red_shrinking: bool = False,
               high_open_rejected: bool = False,
               sector_vs_market: float = 0.0,
               market_change: float = 0.0,
               market_consecutive_down: int = 0,
               limit_up_probability: float = 0.5,
               sector_strength: str = "neutral") -> ExitDecision:
    """
    检查是否应该出场/减仓
    
    参数：
    - position: 当前持仓状态
    - config: 出场参数
    - macd_red_shrinking: 日线MACD红柱是否连续缩短
    - high_open_rejected: 当日是否高开回落
    - sector_vs_market: 板块相对大盘表现%
    - market_change: 大盘当日涨跌%
    - market_consecutive_down: 大盘连续下跌天数
    - limit_up_probability: 当前市场连板概率（0-1）
    - sector_strength: 板块强度 strong/neutral/weak
    """
    profit = position.current_profit_pct
    drawdown = position.drawdown_from_peak
    
    # 1. 止损（铁律，最高优先级）
    if profit <= config.stop_loss_pct:
        return ExitDecision(
            action="close", reason=ExitReason.STOP_LOSS,
            description=f"止损：浮亏{profit:.1f}% ≤ {config.stop_loss_pct}%"
        )
    
    # 2. 强制止盈
    if profit >= config.take_profit_full:
        return ExitDecision(
            action="close", reason=ExitReason.TAKE_PROFIT,
            description=f"强制止盈：浮盈{profit:.1f}% ≥ {config.take_profit_full}%"
        )
    
    # 3. 目标利润区间 + 市场环境判断
    if profit >= config.take_profit_pct:
        # 到了20%目标，看市场环境决定走不走
        if limit_up_probability < 0.3 or sector_strength == "weak":
            # 连板概率低 + 板块弱 → 止盈离场
            return ExitDecision(
                action="close", reason=ExitReason.TAKE_PROFIT,
                description=f"止盈：浮盈{profit:.1f}%，连板概率低({limit_up_probability:.0%})+板块{sector_strength}"
            )
        elif limit_up_probability < 0.5 or sector_strength == "neutral":
            # 连板概率中等 → 减半仓
            return ExitDecision(
                action="reduce", reason=ExitReason.TAKE_PROFIT,
                reduce_ratio=config.partial_exit_ratio,
                description=f"减仓50%：浮盈{profit:.1f}%，连板概率中等({limit_up_probability:.0%})"
            )
    
    # 4. 动能衰竭（MACD红柱连续缩短）
    if macd_red_shrinking and profit > 5:
        if profit > 15:
            return ExitDecision(
                action="reduce", reason=ExitReason.MOMENTUM_FAIL,
                reduce_ratio=config.partial_exit_ratio,
                description=f"减仓50%：MACD红柱缩短(动能衰竭)，浮盈{profit:.1f}%"
            )
        else:
            return ExitDecision(
                action="hold", reason=ExitReason.MOMENTUM_FAIL,
                description=f"注意：MACD红柱缩短，但浮盈仅{profit:.1f}%，观察"
            )
    
    # 5. 高开回落
    if high_open_rejected and profit > 10:
        return ExitDecision(
            action="reduce", reason=ExitReason.HIGH_OPEN_REJECT,
            reduce_ratio=config.partial_exit_ratio,
            description=f"减仓50%：高开回落，浮盈{profit:.1f}%"
        )
    
    # 6. 板块走弱
    if sector_vs_market < config.sector_underperform_pct and profit > 5:
        return ExitDecision(
            action="reduce", reason=ExitReason.SECTOR_WEAK,
            reduce_ratio=config.partial_exit_ratio,
            description=f"减仓50%：板块跑输大盘{sector_vs_market:.1f}%，浮盈{profit:.1f}%"
        )
    
    # 7. 大盘风险
    if market_consecutive_down >= config.market_drop_days:
        if profit > 0:
            return ExitDecision(
                action="reduce", reason=ExitReason.MARKET_RISK,
                reduce_ratio=config.partial_exit_ratio,
                description=f"减仓50%：大盘连续下跌{market_consecutive_down}天"
            )
        elif profit < -3:
            return ExitDecision(
                action="close", reason=ExitReason.MARKET_RISK,
                description=f"清仓：大盘连续下跌{market_consecutive_down}天+浮亏{profit:.1f}%"
            )
    
    # 继续持有
    return ExitDecision(
        action="hold", reason=ExitReason.HOLD,
        description=f"持有：浮盈{profit:.1f}%，最大浮盈{position.max_profit_pct:.1f}%"
    )
