#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""连板预警器

独立于因子v2选股，专门筛选"可能即将连板"的票。

基于537次连板≥3天事件的前20天统计特征：
1. DIF持续上升（零轴上方≥15天）
2. 红峰面积持续放大
3. D-4~D-6有突然加速（单日涨1.5%+）
4. D-1~D-2价格趋于平静（涨跌<0.5%）
5. RSI在45-55（不超买不超卖）
6. 近5天均量 > 近20天均量（放量趋势）

评分维度（总分100）：
- DIF趋势分（25）：DIF持续上升+零轴上方
- 红峰积累分（20）：红峰面积大+在放大
- 加速信号分（20）：D-4~D-6有单日大涨
- 平静前夜分（15）：D-1~D-2缩量平静
- 量能分（10）：近5天/近20天量比
- 位置分（10）：不是极端高位
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List


def calc_macd(close, fast=12, slow=26, signal=9):
    if len(close) < slow + signal:
        return np.zeros(len(close)), np.zeros(len(close)), np.zeros(len(close))
    ema_f = np.zeros(len(close)); ema_s = np.zeros(len(close))
    ema_f[:fast] = close[0]; ema_s[:slow] = close[0]
    kf = 2/(fast+1); ks = 2/(slow+1)
    for i in range(1, len(close)):
        ema_f[i] = ema_f[i-1]*(1-kf) + close[i]*kf
        ema_s[i] = ema_s[i-1]*(1-ks) + close[i]*ks
    dif = ema_f - ema_s
    dea = np.zeros(len(close))
    kd = 2/(signal+1); dea[0] = dif[0]
    for i in range(1, len(close)):
        dea[i] = dea[i-1]*(1-kd) + dif[i]*kd
    return dif, dea, 2*(dif-dea)


def calc_rsi(close, period=14):
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.convolve(gain, np.ones(period)/period, mode='valid')
    avg_loss = np.convolve(loss, np.ones(period)/period, mode='valid')
    avg_loss = np.where(avg_loss > 0, avg_loss, 1e-10)
    rsi = 100 - 100/(1 + avg_gain/avg_loss)
    padded = np.full(len(close), 50.0)
    padded[-len(rsi):] = rsi
    return padded


@dataclass
class StreakAlert:
    code: str
    name: str
    close: float
    score: float
    dif_score: float
    red_score: float
    accel_score: float
    calm_score: float
    vol_score: float
    pos_score: float
    dif_current: float
    dif_trend: str
    rsi: float
    macd_bar: float
    red_peak: float
    green_peak: float
    recent_d4: float  # D-4涨跌幅
    recent_d1: float  # D-1涨跌幅
    vol_ratio: float
    price_pos: float
    days_above_zero: int  # DIF零轴上方天数
    description: str


def detect_streak_alert(close, high, low, vol, pct) -> dict:
    """检测连板预警信号（价位差异化阈值）"""
    if len(close) < 40:
        return None

    dif, dea, macd_bar = calc_macd(close)
    rsi = calc_rsi(close)

    n = len(close)
    price = close[-1]
    dif_20 = dif[-20:]
    macd_20 = macd_bar[-20:]
    pct_20 = pct[-20:]
    vol_20 = vol[-20:]
    rsi_20 = rsi[-20:]

    # 价位分档
    if price < 5:
        tier = 'low'      # <5元：超跌反弹型
    elif price < 20:
        tier = 'mid_low'  # 5-20元：主力型
    elif price < 50:
        tier = 'mid_high' # 20-50元：趋势型
    else:
        tier = 'high'     # >50元：高价型

    # 价位统计阈值（来自537次连板分析）
    tier_params = {
        'low':     {'min_dif_days': 5,  'rsi_range': (35,55), 'min_ret20': -10, 'max_ret20': 15},
        'mid_low': {'min_dif_days': 8,  'rsi_range': (40,58), 'min_ret20': -5,  'max_ret20': 20},
        'mid_high':{'min_dif_days': 12, 'rsi_range': (48,65), 'min_ret20': 5,   'max_ret20': 35},
        'high':    {'min_dif_days': 15, 'rsi_range': (50,65), 'min_ret20': 10,  'max_ret20': 50},
    }
    params = tier_params[tier]

    # === 1. DIF趋势分（25分）— 价位差异化 ===
    dif_current = dif[-1]
    dif_above_zero = sum(1 for d in dif_20 if d > 0)
    dif_rising = dif_20[-1] > dif_20[0]
    dif_change = dif_20[-1] - dif_20[0]
    min_days = params['min_dif_days']

    dif_score = 0
    # 零上天数（按价位标准）
    if dif_above_zero >= min_days + 5: dif_score += 12
    elif dif_above_zero >= min_days: dif_score += 10
    elif dif_above_zero >= min_days - 3: dif_score += 6
    elif dif_above_zero >= min_days - 6: dif_score += 3

    if dif_rising: dif_score += 8
    if dif_change > 0.05: dif_score += 3
    if dif_change > 0.15: dif_score += 2
    dif_score = min(dif_score, 25)

    # === 2. 红峰积累分（20分）===
    # 计算当前红峰面积
    red_peaks = []; cur_r = 0
    for b in macd_bar:
        if b > 0: cur_r += b
        else:
            if cur_r > 0: red_peaks.append(cur_r)
            cur_r = 0
    if cur_r > 0: red_peaks.append(cur_r)
    current_red = red_peaks[-1] if red_peaks else 0

    # 绿峰
    green_peaks = []; cur_g = 0
    for b in macd_bar:
        if b < 0: cur_g += abs(b)
        else:
            if cur_g > 0: green_peaks.append(cur_g)
            cur_g = 0
    if cur_g > 0: green_peaks.append(cur_g)
    current_green = max(green_peaks) if green_peaks else 0

    # 红峰是否在放大（近5天 vs 前15天）
    red_recent = sum(macd_20[-5:])
    red_earlier = sum(macd_20[:15])
    red_expanding = red_recent > red_earlier * 0.4  # 近5天红柱 > 前15天的40%

    red_score = 0
    if current_red >= 8: red_score += 10
    elif current_red >= 5: red_score += 7
    elif current_red >= 3: red_score += 4

    if red_expanding: red_score += 6
    if macd_20[-1] > macd_20[-5]: red_score += 4  # 红柱在放大
    red_score = min(red_score, 20)

    # === 3. 加速信号分（20分）===
    # D-4~D-6有单日大涨（≥1.5%）
    recent_4_6 = pct_20[-7:-4]  # D-6,D-5,D-4
    max_accel = max(recent_4_6) if len(recent_4_6) > 0 else 0

    accel_score = 0
    if max_accel >= 3: accel_score += 15
    elif max_accel >= 2: accel_score += 12
    elif max_accel >= 1.5: accel_score += 10
    elif max_accel >= 1: accel_score += 6
    elif max_accel >= 0.5: accel_score += 3

    # D-3~D-1减速（加速后趋缓）
    recent_1_3 = pct_20[-3:]
    avg_recent = np.mean(recent_1_3)
    if max_accel > 1 and avg_recent < max_accel * 0.5:
        accel_score += 5  # 加速后减速=蓄力
    accel_score = min(accel_score, 20)

    # === 4. 平静前夜分（15分）===
    d1 = pct_20[-1]
    d2 = pct_20[-2] if len(pct_20) > 1 else 0

    calm_score = 0
    if abs(d1) < 0.5: calm_score += 8
    elif abs(d1) < 1.0: calm_score += 5
    elif abs(d1) < 2.0: calm_score += 2

    if abs(d2) < 1.0: calm_score += 4
    elif abs(d2) < 2.0: calm_score += 2
    calm_score = min(calm_score, 15)

    # === 5. 量能分（10分）===
    vol_recent_5 = np.mean(vol_20[-5:])
    vol_earlier = np.mean(vol_20[:15])
    vol_ratio = vol_recent_5 / vol_earlier if vol_earlier > 0 else 1

    vol_score = 0
    if vol_ratio >= 2.0: vol_score += 10
    elif vol_ratio >= 1.5: vol_score += 8
    elif vol_ratio >= 1.2: vol_score += 6
    elif vol_ratio >= 1.0: vol_score += 4
    elif vol_ratio >= 0.8: vol_score += 2
    vol_score = min(vol_score, 10)

    # === 6. 位置分（10分）— 价位差异化 ===
    year_h = max(high[-60:]) if len(high) >= 60 else max(high)
    year_l = min(low[-60:]) if len(low) >= 60 else min(low)
    price_pos = (close[-1] - year_l) / (year_h - year_l) * 100 if year_h > year_l else 50

    # 前20天涨幅
    ret_20d = (close[-1] / close[-21] - 1) * 100 if len(close) >= 21 else 0

    pos_score = 0
    # 价格位置评分（中高位最佳，但允许低价股在底部）
    if tier == 'low':
        if 30 <= price_pos <= 70: pos_score += 7
        elif 20 <= price_pos <= 80: pos_score += 5
        else: pos_score += 2
    elif tier == 'mid_low':
        if 40 <= price_pos <= 70: pos_score += 8
        elif 30 <= price_pos <= 80: pos_score += 6
        else: pos_score += 3
    else:  # mid_high / high
        if 50 <= price_pos <= 80: pos_score += 8
        elif 40 <= price_pos <= 85: pos_score += 6
        else: pos_score += 3

    # 前20天涨幅是否在统计范围内
    if params['min_ret20'] <= ret_20d <= params['max_ret20']:
        pos_score += 2
    pos_score = min(pos_score, 10)

    # RSI加分（在统计范围内）
    rsi_lo, rsi_hi = params['rsi_range']
    if rsi_lo <= rsi[-1] <= rsi_hi:
        pos_score = min(pos_score + 0, pos_score)  # 已经在位置分里了

    # 总分
    total = dif_score + red_score + accel_score + calm_score + vol_score + pos_score

    # 描述
    flags = []
    if dif_score >= 20: flags.append('DIF强势上升')
    if red_score >= 15: flags.append('红峰积累')
    if accel_score >= 15: flags.append('加速信号')
    if calm_score >= 10: flags.append('暴风雨前夜')
    if vol_score >= 8: flags.append('放量')
    tier_cn = {'low':'低价','mid_low':'中低','mid_high':'中高','high':'高价'}[tier]
    flags.append(f'{tier_cn}档')
    description = ' | '.join(flags) if flags else '信号偏弱'

    return {
        'score': round(total, 1),
        'dif_score': round(dif_score, 1),
        'red_score': round(red_score, 1),
        'accel_score': round(accel_score, 1),
        'calm_score': round(calm_score, 1),
        'vol_score': round(vol_score, 1),
        'pos_score': round(pos_score, 1),
        'dif_current': round(dif_current, 2),
        'dif_trend': '上升' if dif_rising else '下降',
        'rsi': round(rsi[-1], 0),
        'macd_bar': round(macd_bar[-1], 2),
        'red_peak': round(current_red, 1),
        'green_peak': round(current_green, 1),
        'recent_d4': round(max_accel, 1),
        'recent_d1': round(d1, 1),
        'vol_ratio': round(vol_ratio, 2),
        'price_pos': round(price_pos, 0),
        'days_above_zero': dif_above_zero,
        'description': description,
    }
