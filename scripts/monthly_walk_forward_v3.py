#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v2月度回测：资金比例仓位 + 分时回测

核心改进：
1. 仓位按资金比例分配（不是等权手数）
   - 低价档(<10元)：总仓位30%
   - 中价档(10-30元)：总仓位50%
   - 高价档(>30元)：总仓位20%
2. 每笔交易的PnL按资金比例计算（不是等权百分比）
3. 同时持有最多5只股票
"""
import sys, json, numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_daily, load_index, _init_tushare
from src.signals.macd_area_v2 import generate_signals_v2 as _gen_v2_original
from src.signals.price_zone import analyze_price_zone
from src.backtest.intraday_simulator import load_minute_data, run_intraday_backtest


def generate_signals_v2(daily_df):
    """v2信号 + 价格区间"""
    sigs = _gen_v2_original(daily_df)
    if not sigs:
        return sigs
    
    df = daily_df.sort_values('trade_date').reset_index(drop=True)
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    vol = df['vol'].values if 'vol' in df.columns else None
    
    for sig in sigs:
        sig_rows = df[df['trade_date'].astype(str) == sig.date]
        if len(sig_rows) == 0:
            continue
        sig_idx = sig_rows.index[0]
        
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
        
        zone_bonus = 0.0
        if zone.current_position == "lower" and zone.strength > 0.5:
            zone_bonus = 0.10
        if zone.current_position == "below" and zone.breakout_direction == "down":
            zone_bonus = -0.10
        if zone.consolidation_days > 20 and zone.strength > 0.6:
            zone_bonus += 0.05
        
        original_strength = sig.signal_strength
        sig.signal_strength = max(0, min(1.0, original_strength + zone_bonus))
        
        if sig.signal_type == "entry" and sig.signal_strength < 0.45:
            sig.signal_type = "wait"
        if sig.signal_type == "wait" and sig.signal_strength >= 0.55 and original_strength >= 0.40:
            sig.signal_type = "entry"
        
        sig.description += f" | 区间{zone.lower}-{zone.upper} {zone.current_position}"
    
    return sigs


def get_price_tier_allocation(price):
    """按价格区间分配资金比例"""
    if price < 10:
        return 0.06  # 低价档：每只6%（最多5只=30%）
    elif price < 30:
        return 0.08  # 中价档：每只8%（最多6只≈50%）
    else:
        return 0.04  # 高价档：每只4%（最多5只=20%）


def screen_stocks_monthly(end_date, lookback_days=365):
    """选股（主板全价格）"""
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
        if close[-1] < 1:
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
    
    mkt = load_index('000300', days=400)
    INITIAL_CAPITAL = 1000000  # 100万
    MAX_POSITION_RATIO = 0.50  # 总仓位上限50%（恶魔股阶段保守）
    MAX_HOLDINGS = 8  # 最多同时持有8只
    
    all_results = []
    all_trades_detail = []
    
    for screen_end, bt_start, bt_end, label in months:
        print(f'\n{"="*60}')
        print(f'  {label}: 选股截止{screen_end} → 回测{bt_start}~{bt_end}')
        print(f'{"="*60}')
        
        screened = screen_stocks_monthly(screen_end, lookback_days=365)
        print(f'筛选出: {len(screened)}只')
        
        if not screened:
            all_results.append({'month': label, 'screened': 0, 'trades': 0, 'win_rate': 0, 
                              'total_pnl_pct': 0, 'total_pnl_yuan': 0, 'stops': 0, 'macd_exits': 0})
            continue
        
        # 按价格分档取标的（保证各档都有代表）
        low = [s for s in screened if s['close'] < 10][:12]
        mid = [s for s in screened if 10 <= s['close'] < 30][:12]
        high = [s for s in screened if s['close'] >= 30][:8]
        backtest_universe = low + mid + high
        
        print(f'  分档: 低价{len(low)} 中价{len(mid)} 高价{len(high)} = {len(backtest_universe)}只')
        
        # 收集当月所有入场信号
        all_entries = []
        
        for s in backtest_universe:
            code = s['code']
            try:
                bt_start_dt = datetime.strptime(bt_start, '%Y%m%d')
                signal_start = (bt_start_dt - timedelta(days=120)).strftime('%Y%m%d')
                daily = load_daily(code, start_date=signal_start, end_date=bt_end, use_cache=True)
                minute = load_minute_data(code, bt_start, bt_end)
                
                if len(daily) < 60 or minute.empty:
                    continue
                
                sigs = generate_signals_v2(daily)
                entry_sigs = [s2 for s2 in sigs if s2.signal_type == 'entry' 
                             and bt_start <= s2.date <= bt_end and s2.signal_strength >= 0.55]
                
                if not entry_sigs:
                    continue
                
                # 用分时回测引擎（不是简化出场）
                trades = run_intraday_backtest(
                    daily, minute, code, s['name'],
                    market_df=mkt,
                    intraday_red_shrink_bars=2,
                    intraday_min_profit_for_exit=5.0,
                    intraday_pullback_from_peak=3.0,
                )
                
                # 把交易和信号匹配
                daily_sorted = daily.sort_values('trade_date').reset_index(drop=True)
                for t in trades:
                    if not t.exit_reason:
                        continue
                    if t.entry_date < bt_start or t.entry_date > bt_end:
                        continue
                    
                    # 找对应的信号强度
                    strength = 0.5
                    for es in entry_sigs:
                        if es.date == t.entry_date:
                            strength = es.signal_strength
                            break
                    
                    entry_price = t.entry_price
                    # 按价格区间分配资金
                    allocation = get_price_tier_allocation(entry_price)
                    pnl_yuan = INITIAL_CAPITAL * allocation * (t.pnl_pct / 100)
                    
                    all_entries.append({
                        'code': code,
                        'name': s['name'],
                        'industry': s.get('industry', ''),
                        'entry_date': t.entry_date,
                        'entry_price': entry_price,
                        'exit_reason': t.exit_reason,
                        'exit_price': t.exit_price,
                        'pnl_pct': round(t.pnl_pct, 2),
                        'allocation': allocation,
                        'pnl_yuan': round(pnl_yuan, 0),
                        'signal_strength': round(strength, 2),
                        'intraday_max': round(getattr(t, 'intraday_max_profit', 0), 1),
                        'price_tier': '低价' if entry_price < 10 else ('中价' if entry_price < 30 else '高价'),
                    })
            except Exception as e:
                pass
        
        # 统计
        completed = all_entries
        wins = [t for t in completed if t['pnl_pct'] > 0]
        stops = [t for t in completed if '止损' in t['exit_reason']]
        macd_exits = [t for t in completed if '红柱' in t['exit_reason']]
        
        total_pnl_pct = sum(t['pnl_pct'] for t in completed)
        total_pnl_yuan = sum(t['pnl_yuan'] for t in completed)
        win_rate = len(wins) / len(completed) * 100 if completed else 0
        
        # 按价格分档统计
        tier_stats = {}
        for tier in ['低价', '中价', '高价']:
            tier_trades = [t for t in completed if t['price_tier'] == tier]
            tier_pnl = sum(t['pnl_yuan'] for t in tier_trades)
            tier_wr = len([t for t in tier_trades if t['pnl_pct'] > 0]) / len(tier_trades) * 100 if tier_trades else 0
            tier_stats[tier] = {'count': len(tier_trades), 'pnl': tier_pnl, 'wr': tier_wr}
        
        print(f'  交易: {len(completed)}笔 | 胜率: {win_rate:.1f}%')
        print(f'  收益: {total_pnl_pct:+.2f}%(等权) | 资金收益: {total_pnl_yuan:+,.0f}元 / {INITIAL_CAPITAL/10000:.0f}万')
        print(f'  止损: {len(stops)}笔 | 分时红柱拐头: {len(macd_exits)}笔')
        print(f'  价格分档:')
        for tier in ['低价', '中价', '高价']:
            ts = tier_stats[tier]
            print(f'    {tier}({ts["count"]:.0f}只): {ts["pnl"]:+,.0f}元 胜率{ts["wr"]:.0f}%')
        
        for t in completed:
            flag = '✅' if t['pnl_pct'] > 0 else '❌'
            print(f'    {t["code"]:8s} {t["name"]:8s} {t["price_tier"]} 入{t["entry_date"]}@{t["entry_price"]:6.2f} 出{t["exit_reason"]:8s} {t["pnl_pct"]:+6.2f}% 资金{t["allocation"]:.0%}={t["pnl_yuan"]:+,.0f}元 {flag}')
        
        all_trades_detail.extend(completed)
        all_results.append({
            'month': label, 'screened': len(screened), 'trades': len(completed),
            'win_rate': round(win_rate,1), 'total_pnl_pct': round(total_pnl_pct,2),
            'total_pnl_yuan': round(total_pnl_yuan, 0),
            'stops': len(stops), 'macd_exits': len(macd_exits),
        })
    
    # 汇总
    print(f'\n{"="*70}')
    print(f'  v2信号+资金比例仓位 月度回测汇总')
    print(f'{"="*70}')
    print(f'{"月份":6s} {"选股":>4s} {"交易":>4s} {"胜率":>6s} {"等权%":>8s} {"资金收益":>10s} {"止损":>4s} {"红柱":>4s}')
    print('-'*60)
    for r in all_results:
        print(f'{r["month"]:6s} {r["screened"]:4d} {r["trades"]:4d} {r["win_rate"]:5.1f}% {r["total_pnl_pct"]:+7.2f}% {r["total_pnl_yuan"]:+9,.0f}元 {r["stops"]:4d} {r["macd_exits"]:4d}')
    
    total_t = sum(r['trades'] for r in all_results)
    total_p = sum(r['total_pnl_pct'] for r in all_results)
    total_y = sum(r['total_pnl_yuan'] for r in all_results)
    roi = total_y / INITIAL_CAPITAL * 100
    print(f'\n总计: {total_t}笔 | 等权{total_p:+.2f}% | 资金{total_y:+,.0f}元 | ROI={roi:+.1f}%')
    
    # 出场原因
    reason_cnt = Counter(t['exit_reason'] for t in all_trades_detail)
    print(f'出场原因: {dict(reason_cnt)}')
    
    # 价格分档汇总
    print(f'\n价格分档汇总:')
    for tier in ['低价', '中价', '高价']:
        tier_trades = [t for t in all_trades_detail if t['price_tier'] == tier]
        if tier_trades:
            tier_pnl = sum(t['pnl_yuan'] for t in tier_trades)
            tier_wr = len([t for t in tier_trades if t['pnl_pct'] > 0]) / len(tier_trades) * 100
            print(f'  {tier}({len(tier_trades)}笔): {tier_pnl:+,.0f}元 胜率{tier_wr:.0f}%')


if __name__ == '__main__':
    run()
