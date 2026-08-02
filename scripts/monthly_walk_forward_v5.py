#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v5月度回测：持仓周期模拟器

核心改进（vs v4）：
1. 一只票建仓一次，出场后周期结束（不再重复买入）
2. 出场完全条件驱动（止损/止盈/动能衰竭/分时红柱拐头/高开回落）
3. 没有固定持仓天数
4. 分时MACD红柱拐头精确到分钟
5. 每天的出场判断是综合判断，不是单一条件
"""
import sys, json, numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, Counter
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_daily, load_index, _init_tushare
from src.signals.macd_area import calc_macd, find_green_peaks, calc_price_position
from src.signals.macd_area_v2 import generate_signals_v2
from src.signals.price_zone import analyze_price_zone
from src.backtest.position_simulator import simulate_position
from src.backtest.intraday_simulator import load_minute_data


def get_price_tier_allocation(price):
    if price < 10: return 0.06
    elif price < 30: return 0.08
    else: return 0.04


def screen_stocks_monthly(end_date, lookback_days=365):
    """选股：主板全价格，不要求当前翻红"""
    pro = _init_tushare()
    import time as _time
    for attempt in range(3):
        try:
            stocks = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,industry')
            break
        except:
            _time.sleep(3)
    else:
        return []
    
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
            try:
                df = pro.daily(trade_date=d)
                if df is not None and len(df) > 0:
                    df = df[df['ts_code'].isin(stock_map.keys())]
                    all_data.append(df)
            except: pass
        all_daily = __import__('pandas').concat(all_data, ignore_index=True)
        all_daily.to_csv(cache_file, index=False)
    
    results = []
    for ts_code, group in all_daily.groupby('ts_code'):
        group = group.sort_values('trade_date')
        if len(group) < 60: continue
        close = group['close'].values
        if close[-1] < 1: continue
        dif, dea, macd_bar = calc_macd(close)
        recent10 = macd_bar[-10:]
        has_red = any(b > 0 for b in recent10)
        last3 = macd_bar[-3:]
        green_acc = all(last3[i] < last3[i-1] for i in range(1, len(last3))) and all(b < 0 for b in last3)
        if not has_red and green_acc: continue
        dates_arr = group['trade_date'].values
        peaks = find_green_peaks(macd_bar, dif, dates_arr, close)
        if not peaks: continue
        current_green_area = 0
        for b in reversed(macd_bar):
            if b < 0: current_green_area += abs(b)
            else: break
        max_peak = max(p.area for p in peaks)
        recent_combined = peaks[-1].area + current_green_area
        effective = max(max_peak, recent_combined)
        if effective < 5: continue
        price_pos = calc_price_position(close, len(close)-1)
        if price_pos > 0.60: continue
        year_high = group['high'].max()
        year_low = group['low'].min()
        if year_low <= 0: continue
        ratio = year_high / year_low
        if ratio < 2.0: continue
        last5_pct = group['pct_chg'].tail(5).abs()
        if (last5_pct <= 5.5).all() and (last5_pct >= 4.5).any(): continue
        limit_ups = group[group['pct_chg'] >= 9.8]
        if len(limit_ups) == 0: continue
        current_pos = (close[-1] - year_low) / (year_high - year_low)
        if current_pos > 0.50: continue
        info = stock_map.get(ts_code, {})
        results.append({
            'code': ts_code.split('.')[0], 'ts_code': ts_code,
            'name': info.get('name', ''), 'industry': info.get('industry', ''),
            'close': round(close[-1], 2),
        })
    return results


def load_verify_data():
    verify = {}
    for i in range(1, 5):
        try:
            with open(f'/tmp/verify_batch{i}.md') as f:
                for line in f:
                    if line.startswith('|') and '---' not in line and '代码' not in line and '评级' not in line:
                        parts = [p.strip() for p in line.split('|')]
                        if len(parts) >= 8:
                            code = parts[1]
                            rating = parts[7] if len(parts) > 7 else 'B'
                            if 'A' in rating and 'C' not in rating: rating = 'A'
                            elif 'C' in rating: rating = 'C'
                            else: rating = 'B'
                            verify[code] = {'positive': parts[5][:40], 'negative': parts[6][:40], 'rating': rating}
        except: pass
    return verify


def run():
    months_config = [
        ('20251231', '20260101', '20260131', '1月'),
        ('20260131', '20260201', '20260228', '2月'),
        ('20260228', '20260301', '20260331', '3月'),
        ('20260331', '20260401', '20260430', '4月'),
        ('20260430', '20260501', '20260531', '5月'),
        ('20260531', '20260601', '20260630', '6月'),
    ]
    
    verify_data = load_verify_data()
    INITIAL = 1000000
    cross_month_trades = {}
    all_trades = []
    
    for screen_end, bt_start, bt_end, label in months_config:
        print(f'\n{"="*60}')
        print(f'  {label}: 选股截止{screen_end} → 回测{bt_start}~{bt_end}')
        print(f'{"="*60}')
        
        import time
        for attempt in range(3):
            try:
                screened = screen_stocks_monthly(screen_end, lookback_days=365)
                break
            except Exception as e:
                print(f'  选股失败({attempt+1}): {str(e)[:40]}')
                time.sleep(5)
        else:
            continue
        
        low = [s for s in screened if s['close'] < 10][:12]
        mid = [s for s in screened if 10 <= s['close'] < 30][:12]
        high = [s for s in screened if s['close'] >= 30][:8]
        universe = low + mid + high
        print(f'  选股: {len(screened)}只 → 分档参与: {len(universe)}只')
        
        month_trades = []
        traded_codes = set()  # 当月已交易的票
        
        for s in universe:
            code = s['code']
            if code in traded_codes:
                continue
            
            try:
                bt_start_dt = datetime.strptime(bt_start, '%Y%m%d')
                signal_start = (bt_start_dt - timedelta(days=120)).strftime('%Y%m%d')
                daily = load_daily(code, start_date=signal_start, end_date=bt_end, use_cache=True)
                minute = load_minute_data(code, bt_start, bt_end)
                if len(daily) < 60 or minute.empty: continue
                
                prior = cross_month_trades.get(code, [])
                sigs = generate_signals_v2(daily, prior_trades=prior)
                entry_sigs = [s2 for s2 in sigs if s2.signal_type == 'entry' 
                             and bt_start <= s2.date <= bt_end and s2.signal_strength >= 0.55]
                if not entry_sigs: continue
                
                # 一只票只取第一个信号（仓位周期）
                first_entry = entry_sigs[0]
                traded_codes.add(code)
                
                # 用新的持仓周期模拟器（日K定信号日，分时定买入时机）
                pos = simulate_position(
                    code=code, name=s['name'],
                    entry_date=first_entry.date,
                    daily_df=daily,
                    minute_df=minute,
                    allocation=get_price_tier_allocation(first_entry.price),
                    signal_strength=first_entry.signal_strength,
                )
                
                if pos is None or not pos.exit_reason:
                    continue
                
                allocation = get_price_tier_allocation(pos.entry_price)
                pnl_yuan = INITIAL * allocation * (pos.pnl_pct / 100)
                v = verify_data.get(code, {})
                
                trade = {
                    'month': label, 'code': code, 'name': s['name'],
                    'tier': pos.tier,
                    'entry_date': pos.entry_date, 'entry_price': pos.entry_price,
                    'exit_time': pos.exit_time, 'exit_price': pos.exit_price,
                    'exit_reason': pos.exit_reason,
                    'pnl_pct': round(pos.pnl_pct, 2),
                    'strength': f"{int(pos.signal_strength*100)}%",
                    'allocation': f"{int(allocation*100)}%",
                    'pnl_yuan': round(pnl_yuan, 0),
                    'intraday_max': round(pos.peak_profit, 1),
                    'intraday_max_price': round(pos.peak_price, 2),
                    'holding_days': pos.holding_days,
                    'holding_minutes': pos.holding_minutes,
                    'rating': v.get('rating', '?'),
                    'positive': v.get('positive', '未验证')[:40],
                    'negative': v.get('negative', '未验证')[:40],
                }
                month_trades.append(trade)
                all_trades.append(trade)
                
                # 记录跨月
                if code not in cross_month_trades:
                    cross_month_trades[code] = []
                cross_month_trades[code].append({
                    'entry_date': pos.entry_date,
                    'exit_date': pos.exit_date,
                    'pnl_pct': pos.pnl_pct,
                })
            except Exception as e:
                pass
        
        wins = [t for t in month_trades if t['pnl_pct'] > 0]
        total_pnl = sum(t['pnl_yuan'] for t in month_trades)
        wr = len(wins)/len(month_trades)*100 if month_trades else 0
        print(f'  交易: {len(month_trades)}笔 | 胜率: {wr:.0f}% | 资金: {total_pnl:+,.0f}元')
        for t in month_trades:
            flag = '✅' if t['pnl_pct'] > 0 else '❌'
            print(f'    {t["code"]:8s} {t["name"]:8s} {t["tier"]} 入{t["entry_date"]}@{t["entry_price"]:6.2f} 出{t["exit_time"]} {t["exit_reason"]:8s} {t["pnl_pct"]:+6.2f}% 持{t["holding_days"]}天 {flag}')
    
    # 生成HTML
    by_month = defaultdict(list)
    for t in all_trades:
        by_month[t['month']].append(t)
    data = {m: by_month[m] for m in ['1月','2月','3月','4月','5月','6月'] if m in by_month}
    data_json = json.dumps(data, ensure_ascii=False)
    
    total_pnl = sum(t['pnl_yuan'] for t in all_trades)
    wins = len([t for t in all_trades if t['pnl_pct'] > 0])
    rating_cnt = Counter(t['rating'] for t in all_trades)
    win_color = '#3fb950' if total_pnl > 0 else '#f85149'
    
    html = f'''<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>v5持仓周期回测</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,sans-serif;background:#0d1117;color:#c9d1d9;padding:16px}}
h1{{color:#58a6ff;margin-bottom:16px;font-size:20px}}
.summary{{background:#161b22;padding:16px;border-radius:8px;margin-bottom:20px;display:flex;gap:20px;flex-wrap:wrap;justify-content:center}}
.stat{{text-align:center;min-width:70px}}.stat-num{{font-size:22px;font-weight:700}}.stat-label{{font-size:11px;color:#8b949e}}
.month-section{{margin-bottom:32px}}
.month-title{{color:#f0c6ff;font-size:16px;margin-bottom:8px;padding-bottom:6px;border-bottom:1px solid #30363d}}
table{{width:100%;border-collapse:collapse;font-size:11px}}
th{{background:#161b22;color:#8b949e;padding:8px 4px;text-align:left;border-bottom:2px solid #30363d;white-space:nowrap}}
td{{padding:6px 4px;border-bottom:1px solid #21262d;white-space:nowrap}}
tr:hover td{{background:#161b22}}
.win{{color:#3fb950;font-weight:600}}.loss{{color:#f85149;font-weight:600}}
.r-A{{background:#1a3a1a;color:#3fb950;padding:2px 6px;border-radius:4px;font-size:10px}}
.r-B{{background:#2a2515;color:#d29922;padding:2px 6px;border-radius:4px;font-size:10px}}
.r-C{{background:#3a1515;color:#f85149;padding:2px 6px;border-radius:4px;font-size:10px}}
.pos{{color:#3fb950}}.neg{{color:#f85149}}
.t-low{{color:#d29922}}.t-mid{{color:#58a6ff}}.t-high{{color:#bc8cff}}
.exit-macd{{color:#3fb950;font-weight:600}}.exit-stop{{color:#f85149;font-weight:600}}.exit-timeout{{color:#8b949e}}.exit-profit{{color:#58a6ff;font-weight:600}}.exit-exhaust{{color:#d29922;font-weight:600}}.exit-pullback{{color:#bc8cff;font-weight:600}}
</style></head><body>
<h1>📊 v5持仓周期回测（建仓一次+条件驱动出场+分时精确）</h1>
<div class="summary">
<div class="stat"><div class="stat-num" style="color:#58a6ff">{len(all_trades)}</div><div class="stat-label">总交易</div></div>
<div class="stat"><div class="stat-num" style="color:#3fb950">{wins}</div><div class="stat-label">盈利</div></div>
<div class="stat"><div class="stat-num" style="color:#f85149">{len(all_trades)-wins}</div><div class="stat-label">亏损</div></div>
<div class="stat"><div class="stat-num" style="color:#3fb950">{rating_cnt.get("A",0)}</div><div class="stat-label">A级</div></div>
<div class="stat"><div class="stat-num" style="color:#d29922">{rating_cnt.get("B",0)}</div><div class="stat-label">B级</div></div>
<div class="stat"><div class="stat-num" style="color:#f85149">{rating_cnt.get("C",0)}</div><div class="stat-label">C级</div></div>
<div class="stat"><div class="stat-num" style="color:{win_color}">{total_pnl:+,.0f}</div><div class="stat-label">资金收益(元)</div></div>
</div>
<div id="content"></div>
<script>
const D={data_json};
function ec(r){{return r.includes('红柱')?'exit-macd':r.includes('止损')?'exit-stop':r.includes('止盈')?'exit-profit':r.includes('衰竭')?'exit-exhaust':r.includes('回落')?'exit-pullback':'exit-timeout'}}
function tc(t){{return t==='低价'?'t-low':t==='中价'?'t-mid':'t-high'}}
let h='';
for(const[m,ts]of Object.entries(D)){{
const w=ts.filter(t=>t.pnl_pct>0).length;
const p=ts.reduce((s,t)=>s+t.pnl_yuan,0);
h+='<div class="month-section"><div class="month-title">📅 '+m+'（'+ts.length+'笔|胜率'+(w/ts.length*100).toFixed(0)+'%|'+(p>0?'+':'')+p.toFixed(0)+'元）</div>';
h+='<table><thead><tr><th>代码</th><th>名称</th><th>档</th><th>买入日</th><th>买入价</th><th>卖出时间</th><th>卖出价</th><th>出场原因</th><th>收益</th><th>盘中最高</th><th>峰值价</th><th>持有天</th><th>信号</th><th>仓位</th><th>盈亏</th><th>评级</th><th>正面</th><th>负面</th></tr></thead><tbody>';
for(const t of ts){{
const pc=t.pnl_pct>0?'win':'loss';
h+='<tr><td>'+t.code+'</td><td>'+t.name+'</td><td class="'+tc(t.tier)+'">'+t.tier+'</td>';
h+='<td>'+t.entry_date+'</td><td>'+t.entry_price.toFixed(2)+'</td>';
h+='<td>'+t.exit_time+'</td><td>'+t.exit_price.toFixed(2)+'</td>';
h+='<td class="'+ec(t.exit_reason)+'">'+t.exit_reason+'</td>';
h+='<td class="'+pc+'">'+(t.pnl_pct>0?'+':'')+t.pnl_pct.toFixed(2)+'%</td>';
h+='<td class="'+(t.intraday_max>0?'win':'')+'">'+(t.intraday_max>0?'+':'')+t.intraday_max.toFixed(1)+'%</td>';
h+='<td>'+(t.intraday_max_price>0?t.intraday_max_price.toFixed(2):'-')+'</td>';
h+='<td>'+t.holding_days+'</td>';
h+='<td>'+t.strength+'</td><td>'+t.allocation+'</td>';
h+='<td class="'+pc+'">'+(t.pnl_yuan>0?'+':'')+t.pnl_yuan.toFixed(0)+'元</td>';
h+='<td><span class="r-'+t.rating+'">'+t.rating+'</span></td>';
h+='<td class="pos" style="max-width:150px;overflow:hidden;text-overflow:ellipsis">'+t.positive+'</td>';
h+='<td class="neg" style="max-width:150px;overflow:hidden;text-overflow:ellipsis">'+t.negative+'</td></tr>';
}}
h+='</tbody></table></div>';
}}
document.getElementById('content').innerHTML=h;
</script></body></html>'''
    
    with open('/tmp/v5_dashboard.html', 'w', encoding='utf-8') as f:
        f.write(html)
    
    # 汇总
    print(f'\n{"="*70}')
    print(f'  v5回测汇总（持仓周期模拟器）')
    print(f'{"="*70}')
    print(f'{"月份":6s} {"交易":>4s} {"胜率":>6s} {"资金收益":>10s}')
    print('-'*40)
    for m in ['1月','2月','3月','4月','5月','6月']:
        ts = by_month.get(m, [])
        if ts:
            w = len([t for t in ts if t['pnl_pct']>0])
            p = sum(t['pnl_yuan'] for t in ts)
            print(f'{m:6s} {len(ts):4d} {w/len(ts)*100:5.0f}% {p:+9,.0f}元')
    print(f'\n总计: {len(all_trades)}笔 | 资金收益{total_pnl:+,.0f}元')
    
    reason_cnt = Counter(t['exit_reason'] for t in all_trades)
    print(f'出场原因: {dict(reason_cnt)}')


if __name__ == '__main__':
    run()
