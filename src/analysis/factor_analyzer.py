#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""因子有效性分析器

基于阿瑞"三巨头"方法论：因子分组分析+IC检验

功能：
1. 全市场因子IC检验（Spearman相关）
2. 分组收益差（10组Q10-Q1 spread）
3. 因子方向检测（正向/反向/无效）
4. 按子池分析（涨停≥5 vs 全市场 vs 0涨停）
5. Regime分析（牛市/熊市/震荡因子方向变化）
6. 因子权重建议（根据IC自动生成）

输出：因子IC报告+权重建议+方向标记
"""

import numpy as np
import pandas as pd
import json
from collections import defaultdict
from scipy.stats import spearmanr
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class FactorResult:
    factor_name: str
    ic_mean: float
    ic_std: float
    ic_win_rate: float
    ic_ir: float  # IC信息比率 = IC均值/IC标准差
    q1_ret: float  # 最低组收益
    q10_ret: float  # 最高组收益
    spread: float  # Q10-Q1
    direction: str  # 正向/反向/无效
    effective: bool  # 是否有效
    weight_suggestion: float  # 权重建议
    description: str = ''


class FactorAnalyzer:
    """因子有效性分析器"""
    
    def __init__(self, all_daily: pd.DataFrame, forward_days: int = 5):
        self.all_daily = all_daily.copy()
        self.all_daily['trade_date'] = self.all_daily['trade_date'].astype(str)
        self.forward_days = forward_days
        self.trade_dates = sorted(self.all_daily['trade_date'].unique())
        
        # 计算月度调仓日
        self.rebalance_dates = self._get_rebalance_dates()
        
    def _get_rebalance_dates(self) -> List[str]:
        """获取每月最后一个交易日"""
        month_ends = []
        prev_month = ''
        prev_date = ''
        for d in self.trade_dates:
            m = d[:6]
            if m != prev_month and prev_month:
                month_ends.append(prev_date)
            prev_month = m
            prev_date = d
        if prev_date and prev_date not in month_ends:
            month_ends.append(prev_date)
        return month_ends
    
    def _calc_macd(self, close, fast=12, slow=26, signal=9):
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
    
    def _calc_rsi(self, close, period=14):
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
    
    def _calc_cci(self, close, high, low, period=20):
        tp = (high + low + close) / 3
        cci = np.zeros(len(close))
        for i in range(period-1, len(close)):
            ma = np.mean(tp[i-period+1:i+1])
            md = np.mean(np.abs(tp[i-period+1:i+1] - ma))
            if md > 0:
                cci[i] = (tp[i] - ma) / (0.015 * md)
        return cci
    
    def _calc_factors_for_stock(self, group: pd.DataFrame, idx: int) -> Dict[str, float]:
        """计算单只股票在idx位置的所有因子值"""
        if idx < 30:
            return {}
        close = group['close'].values[:idx+1].astype(float)
        high = group['high'].values[:idx+1].astype(float)
        low = group['low'].values[:idx+1].astype(float)
        vol = group['vol'].values[:idx+1].astype(float) if 'vol' in group.columns else np.ones(idx+1)
        pct = group['pct_chg'].values[:idx+1].astype(float)
        
        if len(close) < 35:
            return {}
        
        dif, dea, macd_bar = self._calc_macd(close)
        rsi = self._calc_rsi(close)
        cci = self._calc_cci(close, high, low)
        
        # 因子值
        year_h = max(high); year_l = min(low)
        max_dd = 0; peak = close[0]
        for c in close:
            peak = max(peak, c); max_dd = min(max_dd, (c/peak-1)*100)
        
        return {
            'rsi': float(rsi[-1]),
            'roc_12d': float(close[-1]/close[-13]-1)*100 if len(close)>13 else 0,
            'momentum_20d': float(close[-1]/close[-21]-1)*100 if len(close)>21 else 0,
            'volatility_20d': float(np.std(pct[-20:])),
            'turnover_ratio': float(np.mean(vol[-5:])/np.mean(vol[-20:])) if np.mean(vol[-20:])>0 else 1,
            'price_position': float((close[-1]-year_l)/(year_h-year_l)*100) if year_h>year_l else 50,
            'max_drawdown': float(max_dd),
            'macd_bar': float(macd_bar[-1]),
            'dif': float(dif[-1]),
            'cci': float(cci[-1]),
        }
    
    def analyze_factors(self, stock_filter: Optional[set] = None, 
                        min_limit_ups: int = 0,
                        label: str = '全市场') -> List[FactorResult]:
        """
        分析因子有效性
        
        Args:
            stock_filter: 限制股票池（ts_code集合）
            min_limit_ups: 最低涨停次数
            label: 分析标签
        """
        print(f'\n分析: {label} (min_limit_ups={min_limit_ups})')
        
        results = defaultdict(list)
        
        for mi, me_date in enumerate(self.rebalance_dates):
            me_data = self.all_daily[self.all_daily['trade_date'] == me_date]
            if len(me_data) < 100: continue
            
            future_dates = sorted([d for d in self.trade_dates if d > me_date])[:self.forward_days]
            if len(future_dates) < self.forward_days: continue
            end_date = future_dates[-1]
            
            future_data = self.all_daily[self.all_daily['trade_date'] == end_date][['ts_code','close']].rename(columns={'close':'future_close'})
            me_future = me_data.merge(future_data, on='ts_code', how='inner')
            me_future['ret'] = (me_future['future_close'] / me_future['close'] - 1) * 100
            me_future = me_future[(me_future['close'] > 1) & (me_future['ret'].abs() < 30)]
            
            count = 0
            for ts_code in me_future['ts_code'].unique():
                if stock_filter and ts_code not in stock_filter:
                    continue
                group = self.all_daily[self.all_daily['ts_code']==ts_code].sort_values('trade_date').reset_index(drop=True)
                me_idx_list = group.index[group['trade_date']==me_date].tolist()
                if not me_idx_list: continue
                idx = me_idx_list[0]
                if idx < 30: continue
                
                # 涨停次数过滤
                if min_limit_ups > 0:
                    pct_before = group['pct_chg'].values[:idx+1].astype(float)
                    if sum(1 for p in pct_before if p >= 9.8) < min_limit_ups:
                        continue
                
                factors = self._calc_factors_for_stock(group, idx)
                if not factors: continue
                
                row = me_future[me_future['ts_code']==ts_code].iloc[0]
                for fname, fval in factors.items():
                    results[fname].append({
                        'date': me_date, 'factor_val': fval, 'ret': row['ret']
                    })
                count += 1
            
            print(f'  {mi+1}/{len(self.rebalance_dates)} {me_date}: {count}只')
        
        # 分析每个因子
        factor_results = []
        for factor_name, data in results.items():
            df = pd.DataFrame(data)
            if len(df) < 200: continue
            
            # IC
            ic_by_date = []
            for d, group in df.groupby('date'):
                if len(group) > 50:
                    corr, _ = spearmanr(group['factor_val'], group['ret'])
                    if not np.isnan(corr):
                        ic_by_date.append(corr)
            
            ic_mean = np.mean(ic_by_date) if ic_by_date else 0
            ic_std = np.std(ic_by_date) if ic_by_date else 0.1
            ic_win = (np.array(ic_by_date) > 0).mean() * 100 if ic_by_date else 0
            ic_ir = ic_mean / ic_std if ic_std > 0 else 0
            
            # 分组
            df['quintile'] = pd.qcut(df['factor_val'], 10, labels=False, duplicates='drop')
            group_ret = df.groupby('quintile')['ret'].mean()
            q1_ret = group_ret.iloc[0]
            q10_ret = group_ret.iloc[-1]
            spread = q10_ret - q1_ret
            
            # 方向判断
            if abs(ic_mean) < 0.02:
                direction = '❌无效'
                effective = False
                weight = 0
            elif ic_mean > 0 and spread > 0:
                direction = '✅正向(高→涨)'
                effective = True
                weight = min(abs(ic_mean) * 200, 30)  # IC越大权重越高
            elif ic_mean < 0 and spread < 0:
                direction = '⚠️反转(低→涨)'
                effective = True
                weight = min(abs(ic_mean) * 200, 30)
            else:
                direction = '❌混乱'
                effective = False
                weight = 0
            
            factor_results.append(FactorResult(
                factor_name=factor_name,
                ic_mean=round(ic_mean, 4),
                ic_std=round(ic_std, 4),
                ic_win_rate=round(ic_win, 0),
                ic_ir=round(ic_ir, 3),
                q1_ret=round(q1_ret, 2),
                q10_ret=round(q10_ret, 2),
                spread=round(spread, 2),
                direction=direction,
                effective=effective,
                weight_suggestion=round(weight, 1),
                description=f'{factor_name}: IC={ic_mean:+.4f}, {direction}'
            ))
        
        return factor_results
    
    def compare_pools(self) -> Dict[str, List[FactorResult]]:
        """对比不同股票池的因子有效性"""
        return {
            '全市场': self.analyze_factors(label='全市场'),
            '涨停≥5': self.analyze_factors(min_limit_ups=5, label='涨停≥5次'),
            '涨停≥10': self.analyze_factors(min_limit_ups=10, label='涨停≥10次'),
        }
    
    def print_report(self, results: List[FactorResult], label: str = ''):
        print(f'\n{"="*80}')
        if label:
            print(f'  因子有效性报告: {label}')
        else:
            print(f'  因子有效性报告')
        print(f'{"="*80}')
        print(f'{"因子":18s} {"IC":>8s} {"IC胜率":>6s} {"IC_IR":>6s} {"Q1":>7s} {"Q10":>7s} {"差":>7s} {"权重":>5s} 方向')
        print('-' * 100)
        for r in sorted(results, key=lambda x: abs(x.ic_mean), reverse=True):
            print(f'{r.factor_name:18s} {r.ic_mean:+.4f}  {r.ic_win_rate:5.0f}%  {r.ic_ir:+.3f}  {r.q1_ret:+5.2f}%  {r.q10_ret:+5.2f}%  {r.spread:+5.2f}%  {r.weight_suggestion:4.1f}%  {r.direction}')
    
    def to_dict_list(self, results: List[FactorResult]) -> List[dict]:
        return [
            {
                'factor': r.factor_name, 'ic': r.ic_mean, 'ic_std': r.ic_std,
                'ic_win': r.ic_win_rate, 'ic_ir': r.ic_ir,
                'q1_ret': r.q1_ret, 'q10_ret': r.q10_ret, 'spread': r.spread,
                'direction': r.direction, 'effective': r.effective,
                'weight': r.weight_suggestion
            }
            for r in results
        ]


if __name__ == '__main__':
    # 示例用法
    all_daily = pd.read_csv('/tmp/screen_cache_full.csv', dtype={'ts_code': str})
    
    analyzer = FactorAnalyzer(all_daily, forward_days=5)
    
    # 全市场
    r1 = analyzer.analyze_factors(label='全市场')
    analyzer.print_report(r1, '全市场')
    
    # 涨停≥5
    r2 = analyzer.analyze_factors(min_limit_ups=5, label='涨停≥5次')
    analyzer.print_report(r2, '涨停≥5次')
    
    # 涨停≥10
    r3 = analyzer.analyze_factors(min_limit_ups=10, label='涨停≥10次')
    analyzer.print_report(r3, '涨停≥10次')
    
    # 保存
    with open('/tmp/factor_ic_analysis_v2.json', 'w') as f:
        json.dump({
            '全市场': analyzer.to_dict_list(r1),
            '涨停≥5': analyzer.to_dict_list(r2),
            '涨停≥10': analyzer.to_dict_list(r3),
        }, f, ensure_ascii=False, indent=2)
    
    print('\n保存: /tmp/factor_ic_analysis_v2.json')
