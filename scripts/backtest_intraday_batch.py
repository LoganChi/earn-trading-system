#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""分时级别批量回测器

对多只股票运行分时级别回测（调用intraday_simulator.run_intraday_backtest），
然后与日K回测结果对比汇总分析。

核心流程：
  1. get_stock_universe() 获取股票池
  2. 每只股票：load_daily → load_minute_data → run_intraday_backtest
  3. 同时跑日K回测做对比
  4. 汇总统计：胜率/收益/出场原因/利润回吐改善

用法：
  /usr/bin/python3.12 scripts/backtest_intraday_batch.py --sample 20 --html
  /usr/bin/python3.12 scripts/backtest_intraday_batch.py --codes 002580,601727
  /usr/bin/python3.12 scripts/backtest_intraday_batch.py --sample 20 --start 20250101 --end 20250630
"""
import sys
import os
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np

from src.data.loader import load_daily, load_index, _init_tushare, _to_ts_code
from src.backtest.intraday_simulator import (
    load_minute_data, run_intraday_backtest, IntradayTrade, print_intraday_results
)
from src.backtest.simulator import run_backtest, BacktestResult
from src.risk.dynamic_exit import ExitConfig

# 复用日K回测框架
sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_strategy import get_stock_universe, backtest_single, analyze_results as analyze_daily_results


# ============================================================
#  分时回测单只
# ============================================================

def backtest_intraday_single(code, name, start_date="20260101", end_date="",
                             stop_loss_pct=-8.0, take_profit_pct=20.0, take_profit_full=30.0,
                             verbose=False):
    """对单只股票运行分时级别回测"""
    try:
        # 1. 拉日线
        daily_df = load_daily(code, start_date=start_date, end_date=end_date)
        if len(daily_df) < 60:
            return None, f"日线不足({len(daily_df)}天)"

        # 2. 拉分钟数据
        minute_df = load_minute_data(code, start_date=start_date, end_date=end_date)
        if minute_df is None or minute_df.empty:
            return None, "无分钟数据"

        # 3. 运行分时回测
        trades = run_intraday_backtest(
            daily_df, minute_df,
            code=code, name=name,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            take_profit_full=take_profit_full,
        )
        return trades, None
    except Exception as e:
        return None, str(e)[:100]


# ============================================================
#  批量分时回测 + 日K对比
# ============================================================

def run_batch_intraday(universe, start_date="20260101", end_date="",
                       stop_loss_pct=-8.0, take_profit_pct=20.0, take_profit_full=30.0,
                       verbose=False):
    """批量运行分时回测 + 日K对比回测"""
    all_intraday_trades = []
    all_daily_trades = []
    errors = []
    per_stock = []  # 每只股票的统计

    total = len(universe)
    t0 = time.time()

    for i, (code, name) in enumerate(universe):
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (total - i - 1) if i > 0 else 0
        print(f"  [{i+1}/{total}] {code} {name}... (已用{elapsed:.0f}s, 预计还需{eta:.0f}s)", end=" ", flush=True)

        # === 分时回测 ===
        intraday_trades, err = backtest_intraday_single(
            code, name, start_date, end_date,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            take_profit_full=take_profit_full,
        )

        # === 日K回测对比（同一只股票同一时间段）===
        daily_result, daily_err = backtest_single(code, name, start_date, end_date)

        if err:
            errors.append({"code": code, "name": name, "error": err})
            print(f"❌ {err}")
            continue

        if intraday_trades and len(intraday_trades) > 0:
            all_intraday_trades.extend(intraday_trades)

            # 分时统计
            completed = [t for t in intraday_trades if t.exit_reason]
            wins = [t for t in completed if t.pnl_pct > 0]
            wr = len(wins) / len(completed) * 100 if completed else 0
            avg_pnl = np.mean([t.pnl_pct for t in completed]) if completed else 0
            avg_hold = np.mean([t.holding_days for t in completed]) if completed else 0

            print(f"✅ 分时{len(intraday_trades)}笔 胜率{wr:.0f}% 均{avg_pnl:+.1f}% 持{avg_hold:.0f}天", end="")
        else:
            print(f"⚪ 无分时信号", end="")

        # 日K统计
        if daily_result and daily_result.trades:
            all_daily_trades.extend(daily_result.trades)
            print(f" | 日K{len(daily_result.trades)}笔 胜率{daily_result.win_rate:.0f}% 均{daily_result.avg_profit_pct:+.1f}%")
        else:
            print(f" | 日K无信号")

        # 每只股票记录
        per_stock.append({
            "code": code, "name": name,
            "intraday_trades": len(intraday_trades) if intraday_trades else 0,
            "daily_trades": len(daily_result.trades) if daily_result else 0,
            "daily_error": daily_err or "",
        })

    elapsed = time.time() - t0
    print(f"\n  总耗时 {elapsed:.0f}s ({elapsed/total:.1f}s/只)")

    return all_intraday_trades, all_daily_trades, per_stock, errors


# ============================================================
#  分析汇总
# ============================================================

def analyze_intraday_vs_daily(intraday_trades, daily_trades, per_stock):
    """分析分时vs日K对比结果"""

    # === 分时统计 ===
    intraday_completed = [t for t in intraday_trades if t.exit_reason]
    intraday_wins = [t for t in intraday_completed if t.pnl_pct > 0]
    intraday_losses = [t for t in intraday_completed if t.pnl_pct <= 0]

    intraday_win_rate = len(intraday_wins) / len(intraday_completed) * 100 if intraday_completed else 0
    intraday_avg_profit = np.mean([t.pnl_pct for t in intraday_completed]) if intraday_completed else 0
    intraday_avg_win = np.mean([t.pnl_pct for t in intraday_wins]) if intraday_wins else 0
    intraday_avg_loss = np.mean([t.pnl_pct for t in intraday_losses]) if intraday_losses else 0
    intraday_pl_ratio = abs(intraday_avg_win / intraday_avg_loss) if intraday_avg_loss != 0 else 0
    intraday_avg_hold = np.mean([t.holding_days for t in intraday_completed]) if intraday_completed else 0
    intraday_total_return = sum(t.pnl_pct for t in intraday_completed)

    # 分时利润回吐（截断到100%，避免亏损交易拉爆均值）
    givebacks = [min(t.profit_giveback, 100) for t in intraday_completed if t.profit_giveback > 0]
    intraday_avg_giveback = np.mean(givebacks) if givebacks else 0

    # 分时出场原因分布
    intraday_reasons = {}
    for t in intraday_completed:
        r = t.exit_reason
        if r not in intraday_reasons:
            intraday_reasons[r] = {"count": 0, "total_pnl": 0, "wins": 0}
        intraday_reasons[r]["count"] += 1
        intraday_reasons[r]["total_pnl"] += t.pnl_pct
        if t.pnl_pct > 0:
            intraday_reasons[r]["wins"] += 1

    # === 日K统计 ===
    daily_completed = [t for t in daily_trades if t.exit_date]
    daily_wins = [t for t in daily_completed if t.pnl_pct > 0]
    daily_losses = [t for t in daily_completed if t.pnl_pct <= 0]

    daily_win_rate = len(daily_wins) / len(daily_completed) * 100 if daily_completed else 0
    daily_avg_profit = np.mean([t.pnl_pct for t in daily_completed]) if daily_completed else 0
    daily_avg_win = np.mean([t.pnl_pct for t in daily_wins]) if daily_wins else 0
    daily_avg_loss = np.mean([t.pnl_pct for t in daily_losses]) if daily_losses else 0
    daily_pl_ratio = abs(daily_avg_win / daily_avg_loss) if daily_avg_loss != 0 else 0
    daily_avg_hold = np.mean([t.holding_days for t in daily_completed]) if daily_completed else 0
    daily_total_return = sum(t.pnl_pct for t in daily_completed)

    # 日K利润回吐：用 max_profit_pct 和 pnl_pct 估算（截断到100%）
    daily_givebacks = []
    for t in daily_completed:
        if t.max_profit_pct > 0 and t.pnl_pct < t.max_profit_pct:
            gb = min((t.max_profit_pct - t.pnl_pct) / t.max_profit_pct * 100, 100)
            daily_givebacks.append(gb)
    daily_avg_giveback = np.mean(daily_givebacks) if daily_givebacks else 0

    # 日K出场原因
    daily_reasons = {}
    for t in daily_completed:
        r = t.exit_reason
        if r not in daily_reasons:
            daily_reasons[r] = {"count": 0, "total_pnl": 0, "wins": 0}
        daily_reasons[r]["count"] += 1
        daily_reasons[r]["total_pnl"] += t.pnl_pct
        if t.pnl_pct > 0:
            daily_reasons[r]["wins"] += 1

    return {
        "intraday": {
            "total_trades": len(intraday_completed),
            "win_count": len(intraday_wins),
            "loss_count": len(intraday_losses),
            "win_rate": round(intraday_win_rate, 1),
            "avg_profit": round(intraday_avg_profit, 2),
            "avg_win": round(intraday_avg_win, 2),
            "avg_loss": round(intraday_avg_loss, 2),
            "pl_ratio": round(intraday_pl_ratio, 2),
            "avg_holding_days": round(intraday_avg_hold, 1),
            "total_return": round(intraday_total_return, 2),
            "avg_giveback": round(intraday_avg_giveback, 1),
            "max_single_profit": round(max((t.pnl_pct for t in intraday_completed), default=0), 2),
            "max_single_loss": round(min((t.pnl_pct for t in intraday_completed), default=0), 2),
            "reasons": {k: {
                "count": v["count"],
                "win_rate": round(v["wins"] / v["count"] * 100, 1),
                "avg_pnl": round(v["total_pnl"] / v["count"], 2),
            } for k, v in intraday_reasons.items()},
        },
        "daily": {
            "total_trades": len(daily_completed),
            "win_count": len(daily_wins),
            "loss_count": len(daily_losses),
            "win_rate": round(daily_win_rate, 1),
            "avg_profit": round(daily_avg_profit, 2),
            "avg_win": round(daily_avg_win, 2),
            "avg_loss": round(daily_avg_loss, 2),
            "pl_ratio": round(daily_pl_ratio, 2),
            "avg_holding_days": round(daily_avg_hold, 1),
            "total_return": round(daily_total_return, 2),
            "avg_giveback": round(daily_avg_giveback, 1),
            "max_single_profit": round(max((t.pnl_pct for t in daily_completed), default=0), 2),
            "max_single_loss": round(min((t.pnl_pct for t in daily_completed), default=0), 2),
            "reasons": {k: {
                "count": v["count"],
                "win_rate": round(v["wins"] / v["count"] * 100, 1),
                "avg_pnl": round(v["total_pnl"] / v["count"], 2),
            } for k, v in daily_reasons.items()},
        },
        "comparison": {
            "win_rate_diff": round(intraday_win_rate - daily_win_rate, 1),
            "profit_diff": round(intraday_avg_profit - daily_avg_profit, 2),
            "holding_reduction": round(daily_avg_hold - intraday_avg_hold, 1),
            "giveback_improvement": round(daily_avg_giveback - intraday_avg_giveback, 1),
            "pl_ratio_diff": round(intraday_pl_ratio - daily_pl_ratio, 2),
        },
        "stocks_tested": len(per_stock),
        "stocks_with_intraday": len([s for s in per_stock if s["intraday_trades"] > 0]),
        "stocks_with_daily": len([s for s in per_stock if s["daily_trades"] > 0]),
    }


# ============================================================
#  打印文本结果
# ============================================================

def print_comparison(analysis):
    """打印分时vs日K对比"""
    intra = analysis["intraday"]
    daily = analysis["daily"]
    cmp = analysis["comparison"]

    print(f"\n{'='*60}")
    print(f"  战法分时回测 vs 日K回测对比")
    print(f"{'='*60}")

    print(f"\n  {'指标':16s} {'分时':>12s} {'日K':>12s} {'差异':>12s}")
    print(f"  {'-'*56}")
    print(f"  {'总交易数':16s} {intra['total_trades']:>12d} {daily['total_trades']:>12d} {intra['total_trades']-daily['total_trades']:>+12d}")
    print(f"  {'胜率':16s} {intra['win_rate']:>11.1f}% {daily['win_rate']:>11.1f}% {cmp['win_rate_diff']:>+11.1f}pp")
    print(f"  {'平均收益':16s} {intra['avg_profit']:>+11.2f}% {daily['avg_profit']:>+11.2f}% {cmp['profit_diff']:>+11.2f}%")
    print(f"  {'平均盈利':16s} {intra['avg_win']:>+11.2f}% {daily['avg_win']:>+11.2f}% {intra['avg_win']-daily['avg_win']:>+11.2f}%")
    print(f"  {'平均亏损':16s} {intra['avg_loss']:>+11.2f}% {daily['avg_loss']:>+11.2f}% {intra['avg_loss']-daily['avg_loss']:>+11.2f}%")
    print(f"  {'盈亏比':16s} {intra['pl_ratio']:>12.2f} {daily['pl_ratio']:>12.2f} {cmp['pl_ratio_diff']:>+12.2f}")
    print(f"  {'平均持有天数':16s} {intra['avg_holding_days']:>11.1f}天 {daily['avg_holding_days']:>11.1f}天 {cmp['holding_reduction']:>+11.1f}天")
    print(f"  {'利润回吐':16s} {intra['avg_giveback']:>11.1f}% {daily['avg_giveback']:>11.1f}% {cmp['giveback_improvement']:>+11.1f}%")
    print(f"  {'累计收益':16s} {intra['total_return']:>+11.2f}% {daily['total_return']:>+11.2f}% {intra['total_return']-daily['total_return']:>+11.2f}%")

    print(f"\n  分时出场原因分布:")
    for reason, s in sorted(intra["reasons"].items(), key=lambda x: -x[1]["count"]):
        pct = s["count"] / intra["total_trades"] * 100 if intra["total_trades"] > 0 else 0
        print(f"    {reason:12s} {s['count']:>3d}次 ({pct:>4.0f}%)  胜率{s['win_rate']:.0f}%  均{s['avg_pnl']:+.2f}%")

    print(f"\n  日K出场原因分布:")
    for reason, s in sorted(daily["reasons"].items(), key=lambda x: -x[1]["count"]):
        pct = s["count"] / daily["total_trades"] * 100 if daily["total_trades"] > 0 else 0
        print(f"    {reason:12s} {s['count']:>3d}次 ({pct:>4.0f}%)  胜率{s['win_rate']:.0f}%  均{s['avg_pnl']:+.2f}%")

    print(f"\n  📌 关键改善:")
    print(f"    利润回吐: {daily['avg_giveback']:.0f}% → {intra['avg_giveback']:.0f}%  (改善{cmp['giveback_improvement']:.0f}%)")
    print(f"    平均持有: {daily['avg_holding_days']:.0f}天 → {intra['avg_holding_days']:.0f}天  (缩短{cmp['holding_reduction']:.0f}天)")
    print(f"    胜率: {daily['win_rate']:.0f}% → {intra['win_rate']:.0f}%  ({cmp['win_rate_diff']:+.0f}pp)")

    print(f"\n  覆盖: {analysis['stocks_tested']}只股票 | 有分时信号{analysis['stocks_with_intraday']}只 | 有日K信号{analysis['stocks_with_daily']}只")
    print(f"{'='*60}")


# ============================================================
#  HTML报告
# ============================================================

def generate_intraday_html_report(analysis, output_path):
    """生成HTML报告（分时vs日K对比版）"""
    data_json = json.dumps(analysis, ensure_ascii=False, indent=2)

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>分时级别回测报告</title>
<style>
:root{{--bg:#0f1117;--card:#1a1d29;--text:#e4e7ef;--muted:#8892a6;--accent:#4fc3f7;--green:#26a69a;--red:#ef5350;--yellow:#ffc107;--purple:#ab47bc;--orange:#ff9800;--border:#2a2e3e}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:var(--bg);color:var(--text);font-family:-apple-system,'PingFang SC','Microsoft YaHei',sans-serif;padding:20px;max-width:960px;margin:0 auto;line-height:1.7}}
h1{{text-align:center;font-size:1.5rem;margin-bottom:4px;background:linear-gradient(135deg,var(--accent),var(--purple));-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.subtitle{{text-align:center;color:var(--muted);font-size:0.85rem;margin-bottom:20px}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px;margin-bottom:16px}}
.card h2{{font-size:1.1rem;color:var(--accent);margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid var(--border)}}
.stat-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px}}
.stat{{background:rgba(79,195,247,0.08);border-radius:8px;padding:12px;text-align:center}}
.stat .val{{font-size:1.3rem;font-weight:700}}
.stat .lbl{{font-size:0.75rem;color:var(--muted);margin-top:2px}}
.green{{color:var(--green)}} .red{{color:var(--red)}} .yellow{{color:var(--yellow)}} .orange{{color:var(--orange)}}
table{{width:100%;border-collapse:collapse;font-size:0.9rem;margin:8px 0}}
th{{text-align:left;padding:8px;color:var(--accent);border-bottom:2px solid var(--border)}}
td{{padding:8px;border-bottom:1px solid var(--border)}}
.col-intra{{color:var(--accent)}} .col-daily{{color:var(--orange)}}
.highlight{{background:rgba(79,195,247,0.05);border-left:3px solid var(--accent);padding:8px 12px;margin:6px 0;border-radius:0 6px 6px 0}}
.improve{{background:rgba(38,166,154,0.08);border-left:3px solid var(--green);padding:8px 12px;margin:6px 0;border-radius:0 6px 6px 0}}
.disclaimer{{text-align:center;color:var(--muted);font-size:0.8rem;margin-top:20px;padding:12px;border-top:1px solid var(--border)}}
</style>
</head>
<body>
<h1>🕐 分时级别回测报告</h1>
<div class="subtitle">分时MACD精准出场 vs 日K回测对比 · <span id="date"></span></div>
<div id="app"></div>
<div class="disclaimer">⚠️ 以上为量化回测结果，不构成投资建议。回测收益不代表未来表现。</div>
<script>
const D = {data_json};
const app = document.getElementById('app');
document.getElementById('date').textContent = new Date().toLocaleDateString('zh-CN');

function colorVal(v, suffix='%') {{
  const cls = v > 0 ? 'green' : v < 0 ? 'red' : '';
  return `<span class="${{cls}}">${{v > 0 ? '+' : ''}}${{v.toFixed(1)}}${{suffix}}</span>`;
}}

let html = '';

// === 对比表 ===
html += `<div class="card"><h2>📊 分时 vs 日K 回测对比</h2>`;
html += `<table>`;
html += `<tr><th>指标</th><th class="col-intra">🕐 分时</th><th class="col-daily">📈 日K</th><th>差异</th></tr>`;

const rows = [
  ['总交易数', D.intraday.total_trades, D.daily.total_trades, D.intraday.total_trades - D.daily.total_trades, ''],
  ['胜率', D.intraday.win_rate + '%', D.daily.win_rate + '%', D.comparison.win_rate_diff.toFixed(1) + 'pp', ''],
  ['平均收益', D.intraday.avg_profit, D.daily.avg_profit, D.comparison.profit_diff, '%'],
  ['平均盈利', D.intraday.avg_win, D.daily.avg_win, D.intraday.avg_win - D.daily.avg_win, '%'],
  ['平均亏损', D.intraday.avg_loss, D.daily.avg_loss, D.intraday.avg_loss - D.daily.avg_loss, '%'],
  ['盈亏比', D.intraday.pl_ratio, D.daily.pl_ratio, D.comparison.pl_ratio_diff, ''],
  ['平均持有', D.intraday.avg_holding_days + '天', D.daily.avg_holding_days + '天', D.comparison.holding_reduction.toFixed(1) + '天', ''],
  ['利润回吐', D.intraday.avg_giveback + '%', D.daily.avg_giveback + '%', D.comparison.giveback_improvement.toFixed(1) + '%', ''],
  ['累计收益', D.intraday.total_return, D.daily.total_return, D.intraday.total_return - D.daily.total_return, '%'],
];

for (const [label, intra, daily, diff, suffix] of rows) {{
  let diffStr = diff;
  let diffCls = '';
  if (typeof diff === 'number') {{
    diffStr = (diff > 0 ? '+' : '') + diff.toFixed(suffix === '%' ? 2 : 1) + (suffix || '');
    if (label === '平均持有' || label === '利润回吐') {{
      diffCls = diff < 0 ? 'green' : 'red';  // 持有天数/回吐减少是好事
    }} else if (label === '胜率' || label === '平均收益' || label === '盈亏比' || label === '累计收益') {{
      diffCls = diff > 0 ? 'green' : 'red';
    }} else {{
      diffCls = diff > 0 ? 'green' : diff < 0 ? 'red' : '';
    }}
  }}
  let intraStr = typeof intra === 'number' ? (suffix === '%' ? (intra > 0 ? '+' : '') + intra.toFixed(1) + '%' : intra.toFixed(2)) : intra;
  let dailyStr = typeof daily === 'number' ? (suffix === '%' ? (daily > 0 ? '+' : '') + daily.toFixed(1) + '%' : daily.toFixed(2)) : daily;
  html += `<tr><td>${{label}}</td><td class="col-intra">${{intraStr}}</td><td class="col-daily">${{dailyStr}}</td><td class="${{diffCls}}">${{diffStr}}</td></tr>`;
}}
html += `</table>`;
html += `</div>`;

// === 关键改善 ===
html += `<div class="card"><h2>🎯 分时出场精度提升</h2>`;
html += `<div class="improve"><b>利润回吐</b>：${{D.daily.avg_giveback.toFixed(0)}}% → ${{D.intraday.avg_giveback.toFixed(0)}}%（改善${{D.comparison.giveback_improvement.toFixed(0)}}%）</div>`;
html += `<div class="improve"><b>平均持有</b>：${{D.daily.avg_holding_days.toFixed(0)}}天 → ${{D.intraday.avg_holding_days.toFixed(0)}}天（缩短${{D.comparison.holding_reduction.toFixed(0)}}天）</div>`;
html += `<div class="improve"><b>胜率</b>：${{D.daily.win_rate.toFixed(0)}}% → ${{D.intraday.win_rate.toFixed(0)}}%（${{D.comparison.win_rate_diff > 0 ? '↑' : '↓'}}${{Math.abs(D.comparison.win_rate_diff).toFixed(0)}}pp）</div>`;
html += `</div>`;

// === 分时统计卡片 ===
html += `<div class="card"><h2>🕐 分时回测详情</h2>`;
html += `<div class="stat-grid">`;
html += `<div class="stat"><div class="val ${{D.intraday.win_rate > 50 ? 'green' : 'red'}}">${{D.intraday.win_rate}}%</div><div class="lbl">分时胜率</div></div>`;
html += `<div class="stat"><div class="val ${{D.intraday.avg_profit > 0 ? 'green' : 'red'}}">${{D.intraday.avg_profit > 0 ? '+' : ''}}${{D.intraday.avg_profit}}%</div><div class="lbl">分时均收益</div></div>`;
html += `<div class="stat"><div class="val green">${{D.intraday.avg_win}}%</div><div class="lbl">分时均盈</div></div>`;
html += `<div class="stat"><div class="val red">${{D.intraday.avg_loss}}%</div><div class="lbl">分时均亏</div></div>`;
html += `<div class="stat"><div class="val">${{D.intraday.total_trades}}</div><div class="lbl">交易数</div></div>`;
html += `<div class="stat"><div class="val">${{D.intraday.avg_holding_days}}天</div><div class="lbl">平均持有</div></div>`;
html += `</div></div>`;

// === 日K统计卡片 ===
html += `<div class="card"><h2>📈 日K回测详情</h2>`;
html += `<div class="stat-grid">`;
html += `<div class="stat"><div class="val ${{D.daily.win_rate > 50 ? 'green' : 'red'}}">${{D.daily.win_rate}}%</div><div class="lbl">日K胜率</div></div>`;
html += `<div class="stat"><div class="val ${{D.daily.avg_profit > 0 ? 'green' : 'red'}}">${{D.daily.avg_profit > 0 ? '+' : ''}}${{D.daily.avg_profit}}%</div><div class="lbl">日K均收益</div></div>`;
html += `<div class="stat"><div class="val green">${{D.daily.avg_win}}%</div><div class="lbl">日K均盈</div></div>`;
html += `<div class="stat"><div class="val red">${{D.daily.avg_loss}}%</div><div class="lbl">日K均亏</div></div>`;
html += `<div class="stat"><div class="val">${{D.daily.total_trades}}</div><div class="lbl">交易数</div></div>`;
html += `<div class="stat"><div class="val">${{D.daily.avg_holding_days}}天</div><div class="lbl">平均持有</div></div>`;
html += `</div></div>`;

// === 分时出场原因 ===
html += `<div class="card"><h2>🚪 分时出场原因分布</h2>`;
html += `<table><tr><th>原因</th><th>次数</th><th>占比</th><th>胜率</th><th>平均收益</th></tr>`;
for (const [reason, s] of Object.entries(D.intraday.reasons)) {{
  const pct = D.intraday.total_trades > 0 ? (s.count / D.intraday.total_trades * 100).toFixed(0) : 0;
  html += `<tr><td>${{reason}}</td><td>${{s.count}}</td><td>${{pct}}%</td><td>${{s.win_rate}}%</td><td>${{colorVal(s.avg_pnl)}}</td></tr>`;
}}
html += `</table></div>`;

// === 日K出场原因 ===
html += `<div class="card"><h2>🚪 日K出场原因分布</h2>`;
html += `<table><tr><th>原因</th><th>次数</th><th>占比</th><th>胜率</th><th>平均收益</th></tr>`;
for (const [reason, s] of Object.entries(D.daily.reasons)) {{
  const pct = D.daily.total_trades > 0 ? (s.count / D.daily.total_trades * 100).toFixed(0) : 0;
  html += `<tr><td>${{reason}}</td><td>${{s.count}}</td><td>${{pct}}%</td><td>${{s.win_rate}}%</td><td>${{colorVal(s.avg_pnl)}}</td></tr>`;
}}
html += `</table></div>`;

// === 覆盖范围 ===
html += `<div class="card"><h2>📋 覆盖范围</h2>`;
html += `<p>测试股票数: ${{D.stocks_tested}} | 有分时信号: ${{D.stocks_with_intraday}} | 有日K信号: ${{D.stocks_with_daily}}</p>`;
html += `</div>`;

app.innerHTML = html;
</script>
</body>
</html>'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return output_path


# ============================================================
#  主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="分时级别批量回测器")
    parser.add_argument("--sample", type=int, default=20, help="抽样股票数（默认20）")
    parser.add_argument("--codes", type=str, help="指定股票代码（逗号分隔）")
    parser.add_argument("--start", type=str, default="20260101", help="开始日期")
    parser.add_argument("--end", type=str, default="", help="结束日期")
    parser.add_argument("--stop-loss", type=float, default=-8.0, help="止损百分比（默认-8）")
    parser.add_argument("--take-profit", type=float, default=20.0, help="目标止盈（默认20）")
    parser.add_argument("--take-profit-full", type=float, default=30.0, help="强制止盈（默认30）")
    parser.add_argument("--html", action="store_true", help="生成HTML报告")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  分时级别批量回测器")
    print(f"  分时MACD精准出场 vs 日K对比")
    print(f"{'='*60}")

    # 获取股票池
    if args.codes:
        codes = [c.strip() for c in args.codes.split(",")]
        universe = [(c, f"stock_{c}") for c in codes]
        print(f"\n指定股票: {len(universe)}只")
    else:
        universe = get_stock_universe(mode="sample", n=args.sample)
        print(f"\n抽样: {len(universe)}只（每行业取样）")

    print(f"回测区间: {args.start} ~ {args.end or 'today'}")
    print(f"出场参数: 止损{args.stop_loss}% 止盈{args.take_profit}% 强制止{args.take_profit_full}%")
    print()

    # 批量回测
    print(">>> 开始分时+日K双回测...\n")
    intraday_trades, daily_trades, per_stock, errors = run_batch_intraday(
        universe,
        start_date=args.start,
        end_date=args.end,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        take_profit_full=args.take_profit_full,
        verbose=args.verbose,
    )

    print(f"\n>>> 回测完成：分时{len(intraday_trades)}笔 日K{len(daily_trades)}笔 {len(errors)}只出错")

    if errors:
        print(f"\n出错股票:")
        for e in errors:
            print(f"  {e['code']} {e['name']}: {e['error']}")

    # 分析
    analysis = analyze_intraday_vs_daily(intraday_trades, daily_trades, per_stock)

    # 打印对比
    print_comparison(analysis)

    # HTML报告
    if args.html:
        reports_dir = Path(__file__).resolve().parents[1] / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = reports_dir / f"intraday_backtest_{date_str}.html"
        generate_intraday_html_report(analysis, str(html_path))
        print(f"\n>>> HTML报告: {html_path}")

    # 保存JSON
    json_dir = Path(__file__).resolve().parents[1] / "reports"
    json_dir.mkdir(parents=True, exist_ok=True)
    json_path = json_dir / f"intraday_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2, default=str)
    print(f">>> JSON数据: {json_path}")

    return analysis


if __name__ == "__main__":
    main()
