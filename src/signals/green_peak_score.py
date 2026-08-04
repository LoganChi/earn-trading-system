#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""绿峰综合评分系统

评分维度：
1. 面积分（0-30）：绿柱累积面积大小
2. 连续性分（0-20）：连续绿柱 vs 断续绿柱
3. 价格响应分（0-25）：价格跌幅与绿柱面积的匹配度
4. 递减模式分（0-15）：多个绿峰是否递减（衰竭信号）
5. 翻红确认分（0-10）：最近翻红的力度

总分100，≥50分入池
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GreenPeakDetail:
    """单个绿峰详情"""
    start_idx: int
    end_idx: int
    area: float  # 面积
    duration: int  # 持续天数
    price_drop: float  # 价格跌幅%
    efficiency: float  # 卖压效率 = 面积/跌幅


@dataclass 
class GreenPeakScore:
    """绿峰综合评分"""
    # 各维度得分
    area_score: float = 0      # /30
    continuity_score: float = 0  # /20
    price_score: float = 0     # /25
    decrease_score: float = 0  # /15
    reversal_score: float = 0  # /10
    
    # 总分
    total: float = 0
    
    # 详情
    peaks: List[GreenPeakDetail] = None
    max_area: float = 0
    total_area: float = 0
    peak_count: int = 0
    price_drop_total: float = 0  # 最近一年总跌幅
    is_decreasing: bool = False
    latest_red_bar: float = 0
    description: str = ""


def score_green_peaks(close: np.ndarray, high: np.ndarray, low: np.ndarray, 
                      macd_bar: np.ndarray, dif: np.ndarray) -> GreenPeakScore:
    """计算绿峰综合评分
    
    Args:
        close: 收盘价序列
        high: 最高价序列
        low: 最低价序列
        macd_bar: MACD柱序列
        dif: DIF线序列
    """
    score = GreenPeakScore()
    
    # 1. 找所有绿峰
    peaks = []
    i = 0
    while i < len(macd_bar):
        if macd_bar[i] < 0:
            start = i
            area = 0
            while i < len(macd_bar) and macd_bar[i] < 0:
                area += abs(macd_bar[i])
                i += 1
            end = i - 1
            if end > start:  # 至少2天
                # 价格跌幅
                price_start = close[start]
                price_end = min(close[start:end+1])  # 绿柱期间的最低价
                price_drop = (price_start / price_end - 1) * 100 if price_end > 0 else 0
                
                # 卖压效率（面积/跌幅，值越大=卖压越强但价格没跌多少=有人在接）
                efficiency = area / max(abs(price_drop), 0.1)
                
                peaks.append(GreenPeakDetail(
                    start_idx=start, end_idx=end,
                    area=round(area, 2),
                    duration=end - start + 1,
                    price_drop=round(price_drop, 1),
                    efficiency=round(efficiency, 2),
                ))
        else:
            i += 1
    
    if not peaks:
        score.description = "无绿峰"
        return score
    
    score.peaks = peaks
    score.peak_count = len(peaks)
    score.max_area = max(p.area for p in peaks)
    score.total_area = sum(p.area for p in peaks)
    
    # 最近一年的总跌幅
    if len(close) >= 2:
        year_high = max(high)
        current = close[-1]
        score.price_drop_total = round((year_high / current - 1) * 100, 1)
    
    # ===== 1. 面积分（0-30）=====
    # 最大单个绿峰面积
    max_a = score.max_area
    if max_a >= 50:
        score.area_score = 30
    elif max_a >= 20:
        score.area_score = 25
    elif max_a >= 10:
        score.area_score = 20
    elif max_a >= 5:
        score.area_score = 15
    elif max_a >= 3:
        score.area_score = 10
    else:
        score.area_score = 5
    
    # ===== 2. 连续性分（0-20）=====
    # 最长连续绿柱天数 vs 断续的对比
    max_duration = max(p.duration for p in peaks)
    avg_duration = sum(p.duration for p in peaks) / len(peaks)
    
    if max_duration >= 20:
        score.continuity_score = 20
    elif max_duration >= 10:
        score.continuity_score = 15
    elif max_duration >= 5:
        score.continuity_score = 10
    else:
        score.continuity_score = 5
    
    # ===== 3. 价格响应分（0-25）=====
    # 核心逻辑：绿柱面积要和价格跌幅匹配
    # 如果绿柱面积很大但价格没跌多少 = 有人接盘（好事，底更实）
    # 如果绿柱面积小但价格跌很多 = 流动性差/闪崩（危险）
    # 如果绿柱面积大+价格跌很多 = 正常的深跌消耗
    
    latest_peak = peaks[-1]
    if latest_peak.price_drop >= 20:
        score.price_score = 25  # 深跌
    elif latest_peak.price_drop >= 10:
        score.price_score = 20
    elif latest_peak.price_drop >= 5:
        score.price_score = 15
    else:
        score.price_score = 8
    
    # 卖压效率修正：面积/跌幅的比值
    # efficiency高 = 面积大但跌幅小 = 筹码在交换 = 可能有人在接
    if latest_peak.efficiency >= 5 and latest_peak.price_drop >= 5:
        score.price_score += 0  # 不额外加分，但要标记
        score.description += "有接盘迹象 "
    
    # ===== 4. 递减模式分（0-15）=====
    # 取最近3个绿峰看是否递减
    recent_peaks = peaks[-3:] if len(peaks) >= 3 else peaks
    
    if len(recent_peaks) >= 2:
        areas = [p.area for p in recent_peaks]
        prices = [p.price_drop for p in recent_peaks]
        
        # 面积递减 + 价格跌幅也递减 = 典型衰竭
        area_decreasing = all(areas[i] > areas[i+1] for i in range(len(areas)-1))
        price_decreasing = all(prices[i] > prices[i+1] for i in range(len(prices)-1))
        
        # 面积递减但价格不创新低 = 底部确认
        price_not_new_low = len(recent_peaks) >= 2 and \
            min(close[recent_peaks[-1].start_idx:recent_peaks[-1].end_idx+1]) >= \
            min(close[recent_peaks[-2].start_idx:recent_peaks[-2].end_idx+1])
        
        if area_decreasing and price_not_new_low:
            score.decrease_score = 15
            score.is_decreasing = True
            score.description += "面积递减+价格不创新低=底部确认 "
        elif area_decreasing:
            score.decrease_score = 10
            score.description += "面积递减 "
        elif price_not_new_low:
            score.decrease_score = 8
            score.description += "价格不创新低 "
        else:
            score.decrease_score = 3
    
    # ===== 5. 翻红确认分（0-10）=====
    # 最近MACD柱状态
    score.latest_red_bar = macd_bar[-1] if macd_bar[-1] > 0 else 0
    
    if macd_bar[-1] > 0:
        prev_bar = macd_bar[-2] if len(macd_bar) > 1 else 0
        if macd_bar[-1] > prev_bar:
            score.reversal_score = 10  # 红柱放大
            score.description += "红柱放大✅"
        else:
            score.reversal_score = 7  # 红柱但缩短
            score.description += "红柱缩短"
    elif macd_bar[-1] < 0:
        prev_bar = macd_bar[-2] if len(macd_bar) > 1 else 0
        if macd_bar[-1] > prev_bar:
            score.reversal_score = 5  # 绿柱缩短（接近翻红）
            score.description += "绿柱缩短"
        else:
            score.reversal_score = 0  # 绿柱放大
            score.description += "绿柱放大❌"
    
    # 总分
    score.total = (score.area_score + score.continuity_score + 
                   score.price_score + score.decrease_score + score.reversal_score)
    
    return score
