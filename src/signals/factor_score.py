#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""量化因子评分系统

基于成熟量化因子，替代主观绿峰评分。

因子分类：
1. 反转因子：超跌程度（Illiquidity、下行捕捉、最大回撤）
2. 波动率因子：蓄势程度（波动率收缩、下行半方差比）
3. 动量因子：拐头确认（MACD加速度、RoC）
4. 支撑因子：底部确认（价格分位、52周低点比率）
5. 弹性因子：反弹潜力（涨停频率、倍差、上行捕获率）
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FactorResult:
    """单个因子结果"""
    name: str
    raw_value: float  # 原始值
    score: float      # 标准化到0-1
    description: str = ""


@dataclass
class FactorScore:
    """量化因子综合评分"""
    # 各类因子得分（0-1）
    reversal: float = 0      # 反转因子
    volatility: float = 0    # 波动率因子
    momentum: float = 0      # 动量因子
    support: float = 0       # 支撑因子
    elasticity: float = 0    # 弹性因子
    
    # 各因子详情
    factors: List[FactorResult] = field(default_factory=list)
    
    # 总分（0-100）
    total: float = 0
    
    # 权重
    weights = {
        'reversal': 0.25,     # 反转最重要（超跌是前提）
        'volatility': 0.20,   # 波动率收缩是蓄势信号
        'momentum': 0.25,     # 动量拐头是入场确认
        'support': 0.15,      # 底部位置
        'elasticity': 0.15,   # 弹性决定空间
    }
    
    description: str = ""


def calc_factors(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                 volume: np.ndarray, pct_chg: np.ndarray,
                 macd_bar: np.ndarray, dif: np.ndarray) -> FactorScore:
    """计算量化因子综合评分
    
    所有因子标准化到0-1，越高越好（越适合入场）
    """
    result = FactorScore()
    factors = []
    
    n = len(close)
    if n < 60:
        return result
    
    # ===== 1. 反转因子（0-1）=====
    
    # 1a. 最大回撤（越大越超跌）
    running_max = np.maximum.accumulate(close)
    drawdown = (close / running_max - 1)  # 负值
    max_dd = abs(min(drawdown))
    dd_score = min(max_dd / 0.60, 1.0)  # 跌60%以上满分
    
    factors.append(FactorResult('max_drawdown', max_dd, dd_score, 
                                f'最大回撤{max_dd:.1%}'))
    
    # 1b. Amihud Illiquidity（下跌期间的价格冲击/成交量）
    # 取最近30天，计算 |收益率| / 成交量 的均值
    recent_ret = np.abs(pct_chg[-30:]) / 100
    recent_vol = volume[-30:]
    valid = recent_vol > 0
    if np.sum(valid) > 10:
        amihud = np.nanmean(np.where(valid, recent_ret / (recent_vol + 1e-10), np.nan))
        # Amihud越大=流动性越差=下跌中卖压消耗
        # 但太大的话可能是没人交易，适中最好
        amihud_score = min(amihud * 1e6 / 5, 1.0)  # 标准化
    else:
        amihud_score = 0.5
    
    factors.append(FactorResult('amihud_illiq', amihud, amihud_score,
                                f'流动性{amihud_score:.2f}'))
    
    # 反转综合
    result.reversal = (dd_score * 0.6 + amihud_score * 0.4)
    
    # ===== 2. 波动率因子（0-1）=====
    
    # 2a. 波动率收缩比率：近10天波动率 / 近60天波动率
    if n >= 60:
        vol_recent = np.std(pct_chg[-10:])
        vol_history = np.std(pct_chg[-60:])
        vol_ratio = vol_recent / vol_history if vol_history > 0 else 1
        # 比率<1=波动收缩=蓄势；>1=还在加速
        vol_shrink = max(0, min(1, 1 - vol_ratio))  # 0=没收缩 1=完全收缩
    else:
        vol_shrink = 0.5
    
    factors.append(FactorResult('vol_shrink_ratio', vol_ratio if n>=60 else 0, vol_shrink,
                                f'波动收缩{vol_shrink:.2f}'))
    
    # 2b. 下行半方差比率（最近30天下跌日的方差/总方差）
    recent_pct = pct_chg[-30:]
    down_days = recent_pct[recent_pct < 0]
    if len(down_days) > 5:
        downside_var = np.var(down_days)
        total_var = np.var(recent_pct) if np.var(recent_pct) > 0 else 1
        ds_ratio = downside_var / total_var  # 越小=下行风险在收敛
        ds_score = max(0, 1 - ds_ratio)
    else:
        ds_score = 0.5
    
    factors.append(FactorResult('downside_semivar', ds_ratio if len(down_days)>5 else 0, ds_score,
                                f'下行收敛{ds_score:.2f}'))
    
    result.volatility = (vol_shrink * 0.6 + ds_score * 0.4)
    
    # ===== 3. 动量因子（0-1）=====
    
    # 3a. MACD柱加速度（二阶导）
    if len(macd_bar) >= 3:
        velocity = macd_bar[-1] - macd_bar[-2]  # 一阶导（速度）
        acceleration = (macd_bar[-1] - macd_bar[-2]) - (macd_bar[-2] - macd_bar[-3])  # 二阶导
        # 速度>0=红柱放大，加速度>0=加速放大
        if macd_bar[-1] > 0 and velocity > 0:
            macd_mom = 0.8 + min(abs(acceleration) * 2, 0.2)  # 红柱放大+加速
        elif macd_bar[-1] > 0 and velocity <= 0:
            macd_mom = 0.5  # 红柱但减速
        elif macd_bar[-1] <= 0 and velocity > 0:
            macd_mom = 0.3  # 绿柱但缩短（接近翻红）
        else:
            macd_mom = 0.0  # 绿柱放大
    else:
        macd_mom = 0.5
    
    factors.append(FactorResult('macd_acceleration', acceleration if len(macd_bar)>=3 else 0, macd_mom,
                                f'MACD动量{macd_mom:.2f}'))
    
    # 3b. RoC：12日收益率变化率
    if n >= 12:
        roc = (close[-1] / close[-12] - 1) * 100
        # 正且加速=动量向上
        roc_prev = (close[-2] / close[-14] - 1) * 100 if n >= 14 else 0
        roc_change = roc - roc_prev
        if roc > 0 and roc_change > 0:
            roc_score = 0.8 + min(abs(roc_change) / 5, 0.2)
        elif roc > 0:
            roc_score = 0.5
        elif roc > -5 and roc_change > 0:
            roc_score = 0.3  # 跌幅收窄
        else:
            roc_score = 0.1
    else:
        roc_score = 0.5
    
    factors.append(FactorResult('roc_12d', roc if n>=12 else 0, roc_score,
                                f'RoC{roc_score:.2f}'))
    
    result.momentum = (macd_mom * 0.6 + roc_score * 0.4)
    
    # ===== 4. 支撑因子（0-1）=====
    
    # 4a. 价格分位（当前价格在过去252日的分位）
    if n >= 60:
        percentile = np.sum(close < close[-1]) / n
        # 分位越低=越接近底部
        pos_score = max(0, 1 - percentile * 2)  # 底部50%才有分
    else:
        pos_score = 0.5
    
    factors.append(FactorResult('price_percentile', percentile if n>=60 else 0, pos_score,
                                f'价格分位{(1-pos_score)*50:.0f}%'))
    
    # 4b. 距离52周低点的比率
    year_low = min(low)
    year_high = max(high)
    dist_to_low = (close[-1] - year_low) / close[-1] * 100 if close[-1] > 0 else 100
    # 距离低点<10%=接近底部
    low_score = max(0, min(1, (30 - dist_to_low) / 30))
    
    factors.append(FactorResult('dist_to_52w_low', dist_to_low, low_score,
                                f'距低点{dist_to_low:.1f}%'))
    
    result.support = (pos_score * 0.5 + low_score * 0.5)
    
    # ===== 5. 弹性因子（0-1）=====
    
    # 5a. 涨停频率（过去1年）
    limit_ups = np.sum(pct_chg >= 9.8)
    lu_freq = limit_ups / n * 252  # 年化涨停频率
    lu_score = min(lu_freq / 15, 1.0)  # 年化15次满分
    
    factors.append(FactorResult('limit_up_freq', lu_freq, lu_score,
                                f'涨停{limit_ups}次/年化{lu_freq:.1f}'))
    
    # 5b. 高低价倍差（波动弹性）
    ratio_val = 0
    if year_low > 0:
        ratio_val = year_high / year_low
        ratio_score = min(max(0, (ratio_val - 1) / 4), 1.0)  # 倍差5倍满分
    else:
        ratio_score = 0
    
    factors.append(FactorResult('price_range_ratio', ratio_val, ratio_score,
                                f'倍差{ratio_val:.1f}'))
    
    # 5c. 上行捕获率（上涨日的平均涨幅 / 下跌日的平均跌幅）
    up_days = pct_chg[pct_chg > 0]
    down_days_pct = pct_chg[pct_chg < 0]
    if len(up_days) > 5 and len(down_days_pct) > 5:
        avg_up = np.mean(up_days)
        avg_down = abs(np.mean(down_days_pct))
        up_capture = avg_up / avg_down if avg_down > 0 else 1
        uc_score = min(up_capture / 1.5, 1.0)  # 1.5倍满分
    else:
        uc_score = 0.5
    
    factors.append(FactorResult('upside_capture', up_capture if len(up_days)>5 else 0, uc_score,
                                f'上行捕获{uc_score:.2f}'))
    
    result.elasticity = (lu_score * 0.4 + ratio_score * 0.3 + uc_score * 0.3)
    
    # ===== 总分 =====
    result.total = (
        result.reversal * result.weights['reversal'] +
        result.volatility * result.weights['volatility'] +
        result.momentum * result.weights['momentum'] +
        result.support * result.weights['support'] +
        result.elasticity * result.weights['elasticity']
    ) * 100
    
    result.factors = factors
    result.description = f"反转{result.reversal:.2f} 波动{result.volatility:.2f} 动量{result.momentum:.2f} 支撑{result.support:.2f} 弹性{result.elasticity:.2f}"
    
    return result
