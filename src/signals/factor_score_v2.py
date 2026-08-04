#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""量化因子评分系统 v2

基于Barra框架+MACD面积战法，引入成熟量化因子：
1. 反转因子（25%）：最大回撤 + Illiquidity + RSI超卖
2. 波动率因子（20%）：波动率收缩 + 下行半方差 + VaR
3. 动量因子（25%）：MACD加速度 + RoC + RSI拐头 + 价格突破52周低点距离
4. 支撑因子（15%）：价格分位 + 距52周低点 + OBV背离
5. 弹性因子（15%）：涨停频率 + 倍差 + 上行捕获率 + Beta

新增因子：
- RSI（超卖+拐头）
- OBV背离（底部量价背离）
- Beta（市场敏感度）
- VaR（尾部风险）
- CCI（顺势指标）
- Aroon（趋势拐点）
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple


@dataclass
class FactorResult:
    name: str
    raw_value: float
    score: float
    description: str = ""


@dataclass
class FactorScore:
    reversal: float = 0
    volatility: float = 0
    momentum: float = 0
    support: float = 0
    elasticity: float = 0
    factors: List[FactorResult] = field(default_factory=list)
    total: float = 0
    description: str = ""
    
    WEIGHTS = {
        'reversal': 0.25,
        'volatility': 0.20,
        'momentum': 0.25,
        'support': 0.15,
        'elasticity': 0.15,
    }


# ===== 辅助函数：计算技术指标 =====

def calc_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI相对强弱指数"""
    if len(close) < period + 1:
        return np.full(len(close), 50.0)
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.zeros(len(close))
    avg_loss = np.zeros(len(close))
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])
    
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period
    
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
    rsi = 100 - 100 / (1 + rs)
    rsi[:period] = 50
    return rsi


def calc_obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """OBV能量潮"""
    obv = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]
    return obv


def calc_beta(close: np.ndarray, market_close: np.ndarray = None) -> float:
    """简化Beta：用价格序列波动率代理"""
    if len(close) < 20:
        return 1.0
    ret = np.diff(np.log(close[-60:])) if len(close) >= 60 else np.diff(np.log(close))
    return float(np.std(ret) * np.sqrt(252) / 0.16)  # 0.16≈大盘年化波动率16%


def calc_cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20) -> np.ndarray:
    """CCI顺势指标"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    tp = (high + low + close) / 3
    cci = np.zeros(n)
    for i in range(period - 1, n):
        ma = np.mean(tp[i-period+1:i+1])
        md = np.mean(np.abs(tp[i-period+1:i+1] - ma))
        if md > 0:
            cci[i] = (tp[i] - ma) / (0.015 * md)
    return cci


def calc_aroon(high: np.ndarray, low: np.ndarray, period: int = 25) -> Tuple[float, float]:
    """Aroon指标，返回(Aroon Up, Aroon Down)"""
    n = len(high)
    if n < period + 1:
        return 50.0, 50.0
    
    recent_high = high[-period-1:]
    recent_low = low[-period-1:]
    
    high_idx = np.argmax(recent_high)
    low_idx = np.argmin(recent_low)
    
    aroon_up = (period - high_idx) / period * 100
    aroon_down = (period - low_idx) / period * 100
    return aroon_up, aroon_down


def calc_var(pct_chg: np.ndarray, confidence: float = 0.05) -> float:
    """历史VaR"""
    if len(pct_chg) < 30:
        return -5.0
    return float(np.percentile(pct_chg[-60:], confidence * 100))


# ===== 主评分函数 =====

def calc_factors(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                 volume: np.ndarray, pct_chg: np.ndarray,
                 macd_bar: np.ndarray, dif: np.ndarray) -> FactorScore:
    """计算量化因子综合评分 v2"""
    
    result = FactorScore()
    factors = []
    n = len(close)
    if n < 60:
        return result
    
    # ===== 预计算技术指标 =====
    rsi = calc_rsi(close, 14)
    obv = calc_obv(close, volume)
    beta = calc_beta(close)
    cci = calc_cci(high, low, close, 20)
    aroon_up, aroon_down = calc_aroon(high, low, 25)
    var_95 = calc_var(pct_chg, 0.05)
    
    # ===== 1. 反转因子（0-1）=====
    
    # 1a. 最大回撤
    running_max = np.maximum.accumulate(close)
    drawdown = close / running_max - 1
    max_dd = abs(min(drawdown))
    dd_score = min(max_dd / 0.60, 1.0)
    factors.append(FactorResult('max_drawdown', max_dd, dd_score, f'回撤{max_dd:.1%}'))
    
    # 1b. Amihud Illiquidity
    recent_ret = np.abs(pct_chg[-30:]) / 100
    recent_vol = volume[-30:]
    valid = recent_vol > 0
    if np.sum(valid) > 10:
        amihud = float(np.nanmean(np.where(valid, recent_ret / (recent_vol + 1e-10), np.nan)))
        amihud_score = min(amihud * 1e6 / 5, 1.0)
    else:
        amihud_score = 0.5
    factors.append(FactorResult('amihud_illiq', amihud if 'amihud' in dir() else 0, amihud_score, f'Illiq{amihud_score:.2f}'))
    
    # 1c. RSI超卖反弹（新增）
    rsi_current = rsi[-1]
    rsi_prev = rsi[-5] if len(rsi) > 5 else rsi[-1]
    if rsi_current < 25:
        rsi_score = 0.9  # 极度超卖
    elif rsi_current < 35:
        rsi_score = 0.7  # 超卖
    elif rsi_current < 45:
        rsi_score = 0.5
    elif rsi_current < 55:
        rsi_score = 0.4
    else:
        rsi_score = 0.2  # 中性偏高
    
    # RSI拐头加分
    if rsi_current > rsi_prev and rsi_prev < 40:
        rsi_score = min(rsi_score + 0.2, 1.0)
        factors.append(FactorResult('rsi_oversold', rsi_current, rsi_score, 
                                     f'RSI{rsi_current:.0f}拐头✅'))
    else:
        factors.append(FactorResult('rsi_oversold', rsi_current, rsi_score,
                                     f'RSI{rsi_current:.0f}'))
    
    result.reversal = dd_score * 0.4 + amihud_score * 0.25 + rsi_score * 0.35
    
    # ===== 2. 波动率因子（0-1）=====
    
    # 2a. 波动率收缩
    if n >= 60:
        vol_recent = float(np.std(pct_chg[-10:]))
        vol_history = float(np.std(pct_chg[-60:]))
        vol_ratio = vol_recent / vol_history if vol_history > 0 else 1
        vol_shrink = max(0, min(1, 1 - vol_ratio))
    else:
        vol_shrink = 0.5
    factors.append(FactorResult('vol_shrink', vol_ratio if n >= 60 else 0, vol_shrink,
                                 f'波动收缩{vol_shrink:.2f}'))
    
    # 2b. 下行半方差
    recent_pct = pct_chg[-30:]
    down_days = recent_pct[recent_pct < 0]
    if len(down_days) > 5:
        downside_var = float(np.var(down_days))
        total_var = float(np.var(recent_pct)) if np.var(recent_pct) > 0 else 1
        ds_ratio = downside_var / total_var
        ds_score = max(0, 1 - ds_ratio)
    else:
        ds_score = 0.5
    factors.append(FactorResult('downside_semivar', ds_ratio if len(down_days) > 5 else 0, ds_score,
                                 f'下行收敛{ds_score:.2f}'))
    
    # 2c. VaR尾部风险（新增）
    var_score = max(0, min(1, (10 + var_95) / 10))  # VaR=-10%→0分，VaR=0%→1分
    factors.append(FactorResult('var_95', var_95, var_score, f'VaR{var_95:.1f}%'))
    
    result.volatility = vol_shrink * 0.4 + ds_score * 0.3 + var_score * 0.3
    
    # ===== 3. 动量因子（0-1）=====
    
    # 3a. MACD柱加速度
    if len(macd_bar) >= 3:
        velocity = macd_bar[-1] - macd_bar[-2]
        acceleration = velocity - (macd_bar[-2] - macd_bar[-3])
        if macd_bar[-1] > 0 and velocity > 0:
            macd_mom = 0.8 + min(abs(acceleration) * 2, 0.2)
        elif macd_bar[-1] > 0 and velocity <= 0:
            macd_mom = 0.5
        elif macd_bar[-1] <= 0 and velocity > 0:
            macd_mom = 0.3
        else:
            macd_mom = 0.0
    else:
        macd_mom = 0.5
    factors.append(FactorResult('macd_accel', acceleration if len(macd_bar) >= 3 else 0, macd_mom,
                                 f'MACD动量{macd_mom:.2f}'))
    
    # 3b. RoC 12日
    if n >= 12:
        roc = (close[-1] / close[-12] - 1) * 100
        roc_prev = (close[-2] / close[-14] - 1) * 100 if n >= 14 else 0
        roc_change = roc - roc_prev
        if roc > 0 and roc_change > 0:
            roc_score = 0.8 + min(abs(roc_change) / 5, 0.2)
        elif roc > 0:
            roc_score = 0.5
        elif roc > -5 and roc_change > 0:
            roc_score = 0.3
        else:
            roc_score = 0.1
    else:
        roc_score = 0.5
    factors.append(FactorResult('roc_12d', roc if n >= 12 else 0, roc_score,
                                 f'RoC{roc_score:.2f}'))
    
    # 3c. CCI（新增）：CCI<-100超卖，从-100以下回升=拐头
    cci_current = cci[-1]
    cci_prev = cci[-2] if len(cci) > 1 else 0
    if cci_current > cci_prev and cci_prev < -100:
        cci_score = 0.9  # 超卖拐头
    elif cci_current > 0:
        cci_score = 0.7  # 正向
    elif cci_current > -100:
        cci_score = 0.4
    else:
        cci_score = 0.1  # 极弱
    factors.append(FactorResult('cci', cci_current, cci_score, f'CCI{cci_current:.0f}'))
    
    # 3d. Aroon（新增）：Aroon Up > Aroon Down = 趋势向上
    if aroon_up > 70 and aroon_down < 30:
        aroon_score = 0.9  # 强势上涨趋势
    elif aroon_up > aroon_down:
        aroon_score = 0.6
    elif aroon_up < 30 and aroon_down > 70:
        aroon_score = 0.1  # 强势下跌
    else:
        aroon_score = 0.4  # 中性
    factors.append(FactorResult('aroon', aroon_up, aroon_score, 
                                 f'Aroon U{aroon_up:.0f}/D{aroon_down:.0f}'))
    
    result.momentum = macd_mom * 0.35 + roc_score * 0.25 + cci_score * 0.2 + aroon_score * 0.2
    
    # ===== 4. 支撑因子（0-1）=====
    
    # 4a. 价格分位
    if n >= 60:
        percentile = float(np.sum(close < close[-1])) / n
        pos_score = max(0, 1 - percentile * 2)
    else:
        pos_score = 0.5
    factors.append(FactorResult('price_percentile', percentile if n >= 60 else 0, pos_score,
                                 f'分位{(1-pos_score)*50:.0f}%'))
    
    # 4b. 距52周低点
    year_low = float(min(low))
    year_high = float(max(high))
    dist_to_low = (close[-1] - year_low) / close[-1] * 100 if close[-1] > 0 else 100
    low_score = max(0, min(1, (30 - dist_to_low) / 30))
    factors.append(FactorResult('dist_to_52w_low', dist_to_low, low_score,
                                 f'距低点{dist_to_low:.1f}%'))
    
    # 4c. OBV背离（新增）：价格创新低但OBV不创新低=底部背离
    if n >= 60:
        price_window = close[-60:]
        obv_window = obv[-60:]
        price_low_idx = np.argmin(price_window)
        # 检查是否有二次探底背离
        if len(price_window) > 20:
            first_low_idx = np.argmin(price_window[:30])
            second_low_idx = 30 + np.argmin(price_window[30:])
            if second_low_idx > first_low_idx:
                price_div = price_window[second_low_idx] <= price_window[first_low_idx] * 1.02
                obv_div = obv_window[second_low_idx] > obv_window[first_low_idx]
                if price_div and obv_div:
                    obv_score = 1.0  # 完美底部背离
                elif obv_div:
                    obv_score = 0.7  # OBV背离但价格未确认
                else:
                    obv_score = 0.3
            else:
                obv_score = 0.3
        else:
            obv_score = 0.3
    else:
        obv_score = 0.3
    factors.append(FactorResult('obv_divergence', 0, obv_score, f'OBV背离{obv_score:.2f}'))
    
    result.support = pos_score * 0.35 + low_score * 0.35 + obv_score * 0.30
    
    # ===== 5. 弹性因子（0-1）=====
    
    # 5a. 涨停频率
    limit_ups = int(np.sum(pct_chg >= 9.8))
    lu_freq = limit_ups / n * 252
    lu_score = min(lu_freq / 15, 1.0)
    factors.append(FactorResult('limit_up_freq', lu_freq, lu_score,
                                 f'涨停{limit_ups}次'))
    
    # 5b. 倍差
    if year_low > 0:
        ratio_val = year_high / year_low
        ratio_score = min(max(0, (ratio_val - 1) / 4), 1.0)
    else:
        ratio_val = 0
        ratio_score = 0
    factors.append(FactorResult('price_range_ratio', ratio_val, ratio_score,
                                 f'倍差{ratio_val:.1f}'))
    
    # 5c. 上行捕获率
    up_days_arr = pct_chg[pct_chg > 0]
    down_days_arr = pct_chg[pct_chg < 0]
    if len(up_days_arr) > 5 and len(down_days_arr) > 5:
        avg_up = float(np.mean(up_days_arr))
        avg_down = float(abs(np.mean(down_days_arr)))
        up_capture = avg_up / avg_down if avg_down > 0 else 1
        uc_score = min(up_capture / 1.5, 1.0)
    else:
        uc_score = 0.5
    factors.append(FactorResult('upside_capture', up_capture if len(up_days_arr) > 5 else 0, uc_score,
                                 f'上行捕获{uc_score:.2f}'))
    
    # 5d. Beta（新增）：低Beta=防守，高Beta=进攻。底部反转需要适度Beta
    if beta < 0.8:
        beta_score = 0.5  # 防守但弹性不足
    elif beta < 1.2:
        beta_score = 0.7  # 适中
    elif beta < 2.0:
        beta_score = 1.0  # 高弹性
    else:
        beta_score = 0.4  # 过于激进
    factors.append(FactorResult('beta', beta, beta_score, f'Beta{beta:.2f}'))
    
    result.elasticity = lu_score * 0.3 + ratio_score * 0.25 + uc_score * 0.25 + beta_score * 0.20
    
    # ===== 总分 =====
    result.total = (
        result.reversal * result.WEIGHTS['reversal'] +
        result.volatility * result.WEIGHTS['volatility'] +
        result.momentum * result.WEIGHTS['momentum'] +
        result.support * result.WEIGHTS['support'] +
        result.elasticity * result.WEIGHTS['elasticity']
    ) * 100
    
    result.factors = factors
    
    # 描述
    strong = []
    if result.reversal >= 0.7: strong.append('深度超跌')
    if result.volatility >= 0.6: strong.append('波动收敛')
    if result.momentum >= 0.8: strong.append('动量强劲')
    if result.support >= 0.7: strong.append('底部确认')
    if result.elasticity >= 0.7: strong.append('高弹性')
    result.description = ' | '.join(strong) if strong else '中性'
    
    return result
