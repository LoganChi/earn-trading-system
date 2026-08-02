#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""月度滚动选股回测

每月末用过去1年数据选股，下个月用分时回测验证。
严格无前视偏差：选股数据截止上月末，回测数据为当月。

流程：
  1. 拉取截至上月末的1年日K数据
  2. 筛选：MACD绿峰≥5+翻红+底部位置+主板+低价+有涨停+倍差≥2
  3. 对筛选结果跑当月分时回测
  4. 统计胜率/收益
"""
import sys, os, json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_daily, load_index, _init_tushare
from src.signals.macd_area import calc_macd, find_green_peaks, calc_price_position
from src.signals.macd_area_v2 import generate_signals_v2
from src.backtest.intraday_simulator import load_minute_data, run_intraday_backtest


def screen_stocks_monthly(end_date: str, lookback_days: int = 250) -> list:
    """用截至end_date的数据选股，严格无前视偏差"""
    pro = _init_tushare()
    if pro is None:
        return []
    
    # 获取股票列表
    stocks = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,industry')
    stocks = stocks[~stocks['name'].str.contains('ST|退|\\*', na=False, regex=True)]
    stocks = stocks[~stocks['ts_code'].str.endswith('.BJ')]
    stocks = stocks[~stocks['symbol'].str.startswith(('300', '301', '688'))]
    
    stock_map = {row['ts_code']: row for _, row in stocks.iterrows()}
    
    # 计算开始日期（往前推lookback_days个日历天）
    end_dt = datetime.strptime(end_date, '%Y%m%d')
    start_dt = end_dt - timedelta(days=lookback_days)
    start_date = start_dt.strftime('%Y%m%d')
    
    # 批量拉日K
    cache = Path(__file__).parent.parent / 'data' / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    
    # 按交易日批量拉
    cal = pro.trade_cal(exchange='SSE', start_date=start_date, end_date=end_date)
    trade_dates = sorted(cal[cal['is_open'] == 1]['cal_date'].tolist())
    
    cache_file = cache / f'monthly_screen_{start_date}_{end_date}.csv'
    if cache_file.exists():
        all_daily = pd.read_csv(cache_file, dtype={'ts_code': str})
    else:
        all_data = []
        valid_codes = set(stock_map.keys())
        for i, d in enumerate(trade_dates):
            try:
                df = pro.daily(trade_date=d)
                if df is not None and len(df) > 0:
                    df = df[df['ts_code'].isin(valid_codes)]
                    all_data.append(df)
                if (i+1) % 30 == 0:
                    print(f'  [{i+1}/{len(trade_dates)}]')
            except:
                pass
        all_daily = pd.concat(all_data, ignore_index=True)
        all_daily.to_csv(cache_file, index=False)
    
    print(f'  日K数据: {len(all_daily):,}条, {all_daily["ts_code"].nunique()}只')
    
    # 逐只筛选
    results = []
    for ts_code, group in all_daily.groupby('ts_code'):
        group = group.sort_values('trade_date')
        if len(group) < 60:
            continue
        
        close = group['close'].values
        if close[-1] >= 10 or close[-1] < 1:  # 10元以下，但不排除仙股
            continue
        
        # MACD
        dif, dea, macd_bar = calc_macd(close)
        
        # 条件1: MACD翻红或接近翻红（最近5根有红柱）
        recent_bars = macd_bar[-5:]
        has_red = any(b > 0 for b in recent_bars)
        if not has_red:
            continue
        
        # 条件2: 绿峰面积≥5
        dates = group['trade_date'].values
        green_peaks = find_green_peaks(macd_bar, dif, dates, close)
        if not green_peaks:
            continue
        last_peak = green_peaks[-1]
        if last_peak.area < 5:
            continue
        
        # 条件3: 价格位置≤60%
        price_pos = calc_price_position(close, len(close)-1)
        if price_pos > 0.60:
            continue
        
        # 条件4: 倍差≥2（用过去1年的高低点）
        year_high = group['high'].max()
        year_low = group['low'].min()
        if year_low <= 0:
            continue
        ratio = year_high / year_low
        if ratio < 2.0:
            continue
        
        # 条件5: 排除ST股 + 有过正常涨停
        # ST判断：如果选股截止日附近5天的涨跌幅都在±5%以内，说明当前是ST
        last5_pct = group['pct_chg'].tail(5).abs()
        is_currently_st = (last5_pct <= 5.5).all() and (last5_pct >= 4.5).any()
        if is_currently_st:
            continue  # 当前是ST状态，跳过
        
        # 有过涨停（正常时期10%）
        limit_ups = group[group['pct_chg'] >= 9.8]
        if len(limit_ups) == 0:
            continue
        
        # 当前价格位置（相对年内高低）
        current_pos = (close[-1] - year_low) / (year_high - year_low) if year_high > year_low else 0.5
        if current_pos > 0.50:  # 底部50%以下（放宽）
            continue
        
        info = stock_map.get(ts_code, {})
        results.append({
            'code': ts_code.split('.')[0],
            'ts_code': ts_code,
            'name': info.get('name', ''),
            'industry': info.get('industry', ''),
            'close': round(close[-1], 2),
            'green_area': round(last_peak.area, 1),
            'macd_bar': round(macd_bar[-1], 4),
            'ratio': round(ratio, 1),
            'limit_ups': len(limit_ups),
            'price_pos': round(current_pos, 2),
        })
    
    results.sort(key=lambda x: -x['green_area'])
    return results


def backtest_month(codes: list, bt_start: str, bt_end: str, market_df=None) -> dict:
    """对筛选结果跑当月分时回测"""
    all_trades = []
    
    for code in codes[:30]:  # 最多取前30只（按绿峰面积排序）
        ts_code = code['ts_code']
        try:
            # 拉前3个月+当月的日线，确保MACD信号能计算
            bt_start_dt = datetime.strptime(bt_start, '%Y%m%d')
            signal_start = (bt_start_dt - timedelta(days=90)).strftime('%Y%m%d')
            daily = load_daily(code['code'], start_date=signal_start, end_date=bt_end, use_cache=True)
            minute = load_minute_data(code['code'], bt_start, bt_end)
            
            if len(daily) < 35 or minute.empty:
                continue
            
            trades = run_intraday_backtest(
                daily, minute, code['code'], code['name'],
                market_df=market_df,
                intraday_red_shrink_bars=2,
                intraday_min_profit_for_exit=5.0,
                intraday_pullback_from_peak=3.0,
            )
            all_trades.extend(trades)
        except Exception as e:
            pass
    
    # 统计
    completed = [t for t in all_trades if t.exit_reason]
    if not completed:
        return {
            'trades': 0, 'win_rate': 0, 'total_pnl': 0,
            'avg_win': 0, 'avg_loss': 0, 'stops': 0, 'macd_exits': 0,
            'stop_rate': 0, 'details': []
        }
    
    wins = [t for t in completed if t.pnl_pct > 0]
    losses = [t for t in completed if t.pnl_pct <= 0]
    stops = [t for t in completed if '止损' in t.exit_reason]
    macd_exits = [t for t in completed if '红柱' in t.exit_reason]
    
    total_pnl = sum(t.pnl_pct for t in completed)
    win_rate = len(wins) / len(completed) * 100
    avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0
    
    return {
        'trades': len(completed),
        'win_rate': round(win_rate, 1),
        'total_pnl': round(total_pnl, 2),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'stops': len(stops),
        'macd_exits': len(macd_exits),
        'stop_rate': round(len(stops)/len(completed)*100, 0) if completed else 0,
        'details': completed,
    }


def run_monthly_backtest():
    """运行6个月滚动选股回测"""
    
    # 每月的选股截止日和回测区间
    months = [
        ('20251231', '20260101', '20260131', '1月'),
        ('20260131', '20260201', '20260228', '2月'),
        ('20260228', '20260301', '20260331', '3月'),
        ('20260331', '20260401', '20260430', '4月'),
        ('20260430', '20260501', '20260531', '5月'),
        ('20260531', '20260601', '20260630', '6月'),
    ]
    
    # 加载大盘数据
    print('加载大盘数据...')
    mkt = load_index('000300', days=400)
    
    all_results = []
    
    for screen_end, bt_start, bt_end, label in months:
        print(f'\n{"="*60}')
        print(f'  {label}: 选股截止{screen_end} → 回测{bt_start}~{bt_end}')
        print(f'{"="*60}')
        
        # Step1: 选股（严格用screen_end之前的数据）
        print(f'选股中（用{screen_end}之前1年数据）...')
        screened = screen_stocks_monthly(screen_end, lookback_days=365)
        print(f'  筛选出: {len(screened)}只')
        
        if len(screened) == 0:
            print(f'  无标的，跳过')
            all_results.append({
                'month': label, 'screened': 0, 'trades': 0,
                'win_rate': 0, 'total_pnl': 0, 'stops': 0, 'macd_exits': 0
            })
            continue
        
        # 打印Top10
        for s in screened[:10]:
            print(f'    {s["code"]} {s["name"]:8s} {s["industry"]:10s} {s["close"]:5.2f} 绿峰{s["green_area"]:5.1f} 倍差{s["ratio"]:4.1f} 涨停{s["limit_ups"]:2d}次')
        
        # Step2: 当月回测
        print(f'回测中（{bt_start}~{bt_end}）...')
        result = backtest_month(screened, bt_start, bt_end, market_df=mkt)
        
        print(f'  交易: {result["trades"]}笔 | 胜率: {result["win_rate"]}% | 收益: {result["total_pnl"]:+.2f}%')
        print(f'  止损: {result["stops"]}笔({result.get("stop_rate",0):.0f}%) | 分时红柱拐头: {result["macd_exits"]}笔')
        
        all_results.append({
            'month': label,
            'screen_end': screen_end,
            'bt_start': bt_start,
            'bt_end': bt_end,
            'screened': len(screened),
            'trades': result['trades'],
            'win_rate': result['win_rate'],
            'total_pnl': result['total_pnl'],
            'avg_win': result['avg_win'],
            'avg_loss': result['avg_loss'],
            'stops': result['stops'],
            'macd_exits': result['macd_exits'],
            'stop_rate': result.get('stop_rate', 0),
        })
    
    # 汇总
    print(f'\n{"="*70}')
    print(f'  月度滚动选股回测汇总（严格无前视偏差）')
    print(f'{"="*70}')
    print(f'{"月份":6s} {"选股":>4s} {"交易":>4s} {"胜率":>6s} {"收益":>8s} {"止损":>6s} {"红柱出场":>6s}')
    print('-' * 50)
    
    total_trades = 0
    total_pnl = 0
    total_wins = 0
    total_completed = 0
    
    for r in all_results:
        print(f'{r["month"]:6s} {r["screened"]:4d} {r["trades"]:4d} {r["win_rate"]:5.1f}% {r["total_pnl"]:+7.2f}% {r["stops"]:4d}笔 {r["macd_exits"]:4d}笔')
        total_trades += r['trades']
        total_pnl += r['total_pnl']
        total_wins += int(r['win_rate'] / 100 * r['trades']) if r['trades'] > 0 else 0
        total_completed += r['trades']
    
    overall_wr = total_wins / total_completed * 100 if total_completed > 0 else 0
    print(f'\n  总计: {total_trades}笔 | 胜率{overall_wr:.1f}% | 累计收益{total_pnl:+.2f}%')
    
    return all_results


if __name__ == '__main__':
    results = run_monthly_backtest()
