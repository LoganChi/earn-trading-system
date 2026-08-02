#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v2信号月度回测：周K趋势+多绿峰评分制入场"""
import sys, json, numpy as np
from datetime import datetime, timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_daily, load_index, _init_tushare
from src.signals.macd_area_v2 import generate_signals_v2 as _gen_v2_original
from src.signals.price_zone import analyze_price_zone


def generate_signals_v2(daily_df):
    """v2信号 + 价格区间整合"""
    # 先调原始v2
    sigs = _gen_v2_original(daily_df)
    if not sigs:
        return sigs
    
    df = daily_df.sort_values('trade_date').reset_index(drop=True)
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    vol = df['vol'].values if 'vol' in df.columns else None
    
    # 对每个信号加上价格区间分析
    for sig in sigs:
        # 找sig在数据中的位置
        sig_rows = df[df['trade_date'].astype(str) == sig.date]
        if len(sig_rows) == 0:
            continue
        sig_idx = sig_rows.index[0]
        
        # 价格区间分析（用截至sig日的60天数据）
        lookback = min(60, sig_idx + 1)
        if lookback < 20:
            continue
        
        zone = analyze_price_zone(
            close[max(0, sig_idx-lookback+1):sig_idx+1],
            high[max(0, sig_idx-lookback+1):sig_idx+1],
            low[max(0, sig_idx-lookback+1):sig_idx+1],
            vol[max(0, sig_idx-lookback+1):sig_idx+1] if vol is not None else None,
            lookback=lookback,
        )
        
        if zone is None:
            continue
        
        # 价格区间加分逻辑
        zone_bonus = 0.0
        
        if zone.current_position == "lower" and zone.strength > 0.5:
            # 价格在区间下沿 + 区间强度高 = 强支撑
            zone_bonus = 0.15
        
        if zone.current_position == "below" and zone.breakout_direction == "down":
            # 跌破区间下沿 = 弱势，减分
            zone_bonus = -0.10
        
        if zone.current_position == "upper" and zone.strength > 0.5:
            # 在区间上沿 = 接近阻力，减分
            zone_bonus = -0.05
        
        if zone.current_position == "above" and zone.breakout_direction == "up":
            # 突破区间上沿 = 强势，但可能是假突破
            if zone.touches_upper >= 3:
                zone_bonus = 0.10  # 多次试探后突破=真突破概率高
            else:
                zone_bonus = -0.05  # 试探次数少=假突破概率高
        
        # 整理充分+区间强度高 = 加分
        if zone.consolidation_days > 20 and zone.strength > 0.6:
            zone_bonus += 0.05
        
        # 更新信号
        original_strength = sig.signal_strength
        sig.signal_strength = max(0, min(1.0, original_strength + zone_bonus))
        
        # 如果价格区间减分导致从entry降到wait
        if sig.signal_type == "entry" and sig.signal_strength < 0.45:
            sig.signal_type = "wait"
        
        # 如果价格区间加分导致从wait升到entry
        if sig.signal_type == "wait" and sig.signal_strength >= 0.55 and original_strength >= 0.40:
            sig.signal_type = "entry"
        
        # 更新描述
        sig.description += f" | 区间{zone.lower}-{zone.upper} 位置{zone.current_position} 强度{zone.strength:.0%}"
    
    return sigs
from src.backtest.intraday_simulator import load_minute_data


def simulate_exit(name, entry_date, entry_price, daily_sorted, entry_idx):
    """简化出场模拟：止损/止盈/回落/超时"""
    days = daily_sorted.iloc[entry_idx:]
    if len(days) == 0:
        return None
    
    stop_loss = -8.0
    take_profit = 20.0
    max_hold = 20
    peak_profit = 0
    
    for di, (_, row) in enumerate(days.iterrows()):
        if di > max_hold:
            return {'name': name, 'entry_date': entry_date, 'entry_price': entry_price,
                    'exit_reason': '超时平仓', 'exit_price': row['close'],
                    'pnl_pct': round((row['close']/entry_price - 1)*100, 2)}
        
        profit = (row['close'] / entry_price - 1) * 100
        peak_profit = max(peak_profit, profit)
        
        if profit <= stop_loss:
            return {'name': name, 'entry_date': entry_date, 'entry_price': entry_price,
                    'exit_reason': '止损', 'exit_price': row['close'],
                    'pnl_pct': round(profit, 2)}
        
        if profit >= take_profit:
            return {'name': name, 'entry_date': entry_date, 'entry_price': entry_price,
                    'exit_reason': '目标止盈', 'exit_price': row['close'],
                    'pnl_pct': round(profit, 2)}
        
        # 分时红柱拐头：涨超5%后日K回落
        if di > 0 and profit >= 5.0:
            prev_profit = (days.iloc[di-1]['close'] / entry_price - 1) * 100
            if profit < prev_profit and peak_profit > profit + 1:
                return {'name': name, 'entry_date': entry_date, 'entry_price': entry_price,
                        'exit_reason': '分时红柱拐头', 'exit_price': row['close'],
                        'pnl_pct': round(profit, 2)}
    
    last = days.iloc[-1]
    return {'name': name, 'entry_date': entry_date, 'entry_price': entry_price,
            'exit_reason': '超时平仓', 'exit_price': last['close'],
            'pnl_pct': round((last['close']/entry_price-1)*100, 2)}


def screen_stocks_monthly(end_date, lookback_days=365):
    """选股（同monthly_walk_forward）"""
    pro = _init_tushare()
    stocks = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,industry')
    stocks = stocks[~stocks['name'].str.contains('ST|退|\\*', na=False, regex=True)]
    stocks = stocks[~stocks['ts_code'].str.endswith('.BJ')]
    stocks = stocks[~stocks['symbol'].str.startswith(('300', '301', '688'))]
    stock_map = {row['ts_code']: row for _, row in stocks.iterrows()}
    
    end_dt = datetime.strptime(end_date, '%Y%m%d')
    start_date = (end_dt - timedelta(days=lookback_days)).strftime('%Y%m%d')
    
    cache = Path(__file__).parent.parent / 'data' / 'cache'
    cache_file = cache / f'monthly_screen_{start_date}_{end_date}.csv'
    if cache_file.exists():
        all_daily = __import__('pandas').read_csv(cache_file, dtype={'ts_code': str})
    else:
        cal = pro.trade_cal(exchange='SSE', start_date=start_date, end_date=end_date)
        trade_dates = sorted(cal[cal['is_open'] == 1]['cal_date'].tolist())
        all_data = []
        for d in trade_dates:
            df = pro.daily(trade_date=d)
            if df is not None and len(df) > 0:
                df = df[df['ts_code'].isin(stock_map.keys())]
                all_data.append(df)
        all_daily = __import__('pandas').concat(all_data, ignore_index=True)
        all_daily.to_csv(cache_file, index=False)
    
    from src.signals.macd_area import calc_macd, find_green_peaks, calc_price_position
    results = []
    for ts_code, group in all_daily.groupby('ts_code'):
        group = group.sort_values('trade_date')
        if len(group) < 60:
            continue
        close = group['close'].values
        if close[-1] >= 10 or close[-1] < 1:
            continue
        dif, dea, macd_bar = calc_macd(close)
        recent_bars = macd_bar[-5:]
        if not any(b > 0 for b in recent_bars):
            continue
        dates_arr = group['trade_date'].values
        peaks = find_green_peaks(macd_bar, dif, dates_arr, close)
        if not peaks or peaks[-1].area < 5:
            continue
        price_pos = calc_price_position(close, len(close)-1)
        if price_pos > 0.60:
            continue
        year_high = group['high'].max()
        year_low = group['low'].min()
        if year_low <= 0:
            continue
        ratio = year_high / year_low
        if ratio < 2.0:
            continue
        # ST排除
        last5_pct = group['pct_chg'].tail(5).abs()
        if (last5_pct <= 5.5).all() and (last5_pct >= 4.5).any():
            continue
        limit_ups = group[group['pct_chg'] >= 9.8]
        if len(limit_ups) == 0:
            continue
        current_pos = (close[-1] - year_low) / (year_high - year_low)
        if current_pos > 0.50:
            continue
        info = stock_map.get(ts_code, {})
        results.append({
            'code': ts_code.split('.')[0], 'ts_code': ts_code,
            'name': info.get('name', ''), 'industry': info.get('industry', ''),
            'close': round(close[-1], 2),
        })
    return results


def run():
    months = [
        ('20251231', '20260101', '20260131', '1月'),
        ('20260131', '20260201', '20260228', '2月'),
        ('20260228', '20260301', '20260331', '3月'),
        ('20260331', '20260401', '20260430', '4月'),
        ('20260430', '20260501', '20260531', '5月'),
        ('20260531', '20260601', '20260630', '6月'),
    ]
    
    all_results = []
    all_trades_detail = []
    
    for screen_end, bt_start, bt_end, label in months:
        print(f'\n{"="*60}')
        print(f'  {label}: 选股截止{screen_end} → 回测{bt_start}~{bt_end}')
        print(f'{"="*60}')
        
        screened = screen_stocks_monthly(screen_end, lookback_days=365)
        print(f'筛选出: {len(screened)}只')
        
        if not screened:
            all_results.append({'month': label, 'screened': 0, 'trades': 0, 'win_rate': 0, 'total_pnl': 0, 'stops': 0, 'macd_exits': 0})
            continue
        
        month_trades = []
        last_entry_per_stock = {}  # 冷却：每只票出场后3天内不再入场
        
        for s in screened[:20]:
            code = s['code']
            try:
                bt_start_dt = datetime.strptime(bt_start, '%Y%m%d')
                signal_start = (bt_start_dt - timedelta(days=120)).strftime('%Y%m%d')
                daily = load_daily(code, start_date=signal_start, end_date=bt_end, use_cache=True)
                
                if len(daily) < 60:
                    continue
                
                # v2信号
                sigs = generate_signals_v2(daily)
                entry_sigs = [s2 for s2 in sigs if s2.signal_type == 'entry' and bt_start <= s2.date <= bt_end]
                
                if not entry_sigs:
                    continue
                
                daily_sorted = daily.sort_values('trade_date').reset_index(drop=True)
                
                for es in entry_sigs:
                    ed = es.date
                    # 冷却检查
                    if code in last_entry_per_stock:
                        last_ed = last_entry_per_stock[code]
                        # 找两个日期之间隔了几个交易日
                        try:
                            idx1 = list(daily_sorted['trade_date'].astype(str)).index(last_ed)
                            idx2 = list(daily_sorted['trade_date'].astype(str)).index(ed)
                            if idx2 - idx1 < 3:
                                continue
                        except ValueError:
                            pass
                    
                    ed_rows = daily_sorted[daily_sorted['trade_date'].astype(str) == ed]
                    if len(ed_rows) == 0:
                        continue
                    entry_idx = ed_rows.index[0]
                    entry_price = daily_sorted.iloc[entry_idx]['close']
                    
                    trade = simulate_exit(s['name'], ed, entry_price, daily_sorted, entry_idx)
                    if trade:
                        trade['code'] = code
                        trade['industry'] = s.get('industry','')
                        trade['signal_strength'] = es.signal_strength
                        month_trades.append(trade)
                        last_entry_per_stock[code] = ed
            except Exception as e:
                print(f'  {code} 错误: {str(e)[:50]}')
        
        # 统计
        completed = [t for t in month_trades if t.get('exit_reason')]
        wins = [t for t in completed if t['pnl_pct'] > 0]
        stops = [t for t in completed if '止损' in t.get('exit_reason','')]
        macd_exits = [t for t in completed if '红柱' in t.get('exit_reason','')]
        
        total_pnl = sum(t['pnl_pct'] for t in completed)
        win_rate = len(wins) / len(completed) * 100 if completed else 0
        
        print(f'  交易: {len(completed)}笔 | 胜率: {win_rate:.1f}% | 收益: {total_pnl:+.2f}%')
        print(f'  止损: {len(stops)}笔 | 分时红柱拐头: {len(macd_exits)}笔')
        
        for t in completed:
            flag = '✅' if t['pnl_pct'] > 0 else '❌'
            print(f'    {t["code"]:8s} {t["name"]:8s} 入{t["entry_date"]}@{t["entry_price"]:5.2f} 出{t["exit_reason"]:8s} {t["pnl_pct"]:+6.2f}% 信号{t["signal_strength"]:.0%} {flag}')
        
        all_trades_detail.extend(completed)
        all_results.append({
            'month': label, 'screened': len(screened), 'trades': len(completed),
            'win_rate': round(win_rate,1), 'total_pnl': round(total_pnl,2),
            'stops': len(stops), 'macd_exits': len(macd_exits),
        })
    
    # 汇总
    print(f'\n{"="*70}')
    print(f'  v2信号（周K+多绿峰评分制）月度回测汇总')
    print(f'{"="*70}')
    print(f'{"月份":6s} {"选股":>4s} {"交易":>4s} {"胜率":>6s} {"收益":>8s} {"止损":>4s} {"红柱":>4s}')
    print('-'*50)
    for r in all_results:
        print(f'{r["month"]:6s} {r["screened"]:4d} {r["trades"]:4d} {r["win_rate"]:5.1f}% {r["total_pnl"]:+7.2f}% {r["stops"]:4d} {r["macd_exits"]:4d}')
    total_t = sum(r['trades'] for r in all_results)
    total_p = sum(r['total_pnl'] for r in all_results)
    print(f'\n总计: {total_t}笔 | 收益{total_p:+.2f}%')
    
    # 出场原因统计
    from collections import Counter
    reason_cnt = Counter(t['exit_reason'] for t in all_trades_detail)
    print(f'\n出场原因: {dict(reason_cnt)}')


if __name__ == '__main__':
    run()
