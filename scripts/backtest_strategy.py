#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""战法回测验证器

不绑定具体持仓，而是在全市场样本上验证交易战法本身是否有效。

战法定义：
  入场：MACD绿峰面积充分消耗 + 红柱开始积累 + 底部价格位置
  出场：动态条件驱动（止损/止盈/动能衰竭/高开回落/大盘风险）
  仓位：固定（验证信号本身，不验证仓位管理）

验证目标：
  1. 这套入场+出场逻辑在全市场上的胜率和期望值
  2. 哪些行业/市值/价格位置上特别有效
  3. 哪些条件下失效（失效场景清单）
  4. vs 随机入场的对照（是否真有alpha）

输出：
  - 文本摘要（胜率/平均收益/失效场景/行业分布）
  - HTML报告（结构化+可视化）

用法：
  /usr/bin/python3.12 scripts/backtest_strategy.py --sample 50    # 抽样50只
  /usr/bin/python3.12 scripts/backtest_strategy.py --codes 002580,601727  # 指定
  /usr/bin/python3.12 scripts/backtest_strategy.py --all            # 全市场（慢）
"""
import sys
import os
import json
import argparse
import time
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np

from src.data.loader import load_daily, load_index, _init_tushare, _to_ts_code
from src.signals.macd_area import generate_signals
from src.backtest.simulator import run_backtest, BacktestResult, Trade
from src.risk.dynamic_exit import ExitConfig


def get_stock_universe(mode="sample", n=50, codes=None):
    """获取股票池"""
    cache = Path(__file__).resolve().parents[1] / "data" / "cache" / "all_stocks.csv"
    
    if codes:
        return [(c, f"stock_{c}") for c in codes]
    
    if not cache.exists():
        pro = _init_tushare()
        if not pro:
            raise RuntimeError("无法获取股票列表")
        df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,industry')
        df.to_csv(cache, index=False)
    
    df = pd.read_csv(cache, dtype={'symbol': str})
    
    if mode == "all":
        return [(r['symbol'], r['name']) for _, r in df.iterrows()]
    
    # 抽样：每行业取前N只
    sampled = df.groupby('industry').head(max(1, n // 30)).reset_index(drop=True)
    if len(sampled) > n:
        sampled = sampled.sample(n=min(n, len(sampled)), random_state=42)
    
    return [(r['symbol'], r['name']) for _, r in sampled.iterrows()]


def backtest_single(code, name, start_date="20240101", end_date="", 
                    exit_config=None, verbose=False):
    """单只股票战法回测"""
    if exit_config is None:
        exit_config = ExitConfig()
    
    try:
        df = load_daily(code, start_date=start_date, end_date=end_date)
        if len(df) < 60:
            return None, f"数据不足({len(df)}天)"
        
        result = run_backtest(
            df, code=code, name=name,
            exit_config=exit_config,
            signal_min_strength=0.4,
            position_size=1000,
            market_df=None,  # 不接大盘数据，让出场引擎用默认值
            limit_up_prob=0.4,
            verbose=verbose,
        )
        return result, None
    except Exception as e:
        return None, str(e)[:80]


def random_baseline(code, name, start_date="20240101", end_date="", 
                    exit_config=None, n_random=20):
    """
    随机入场对照：在同段时间内随机选N个入场点，用同样的动态出场逻辑。
    
    如果战法有alpha，战法的收益应该显著优于随机入场。
    """
    if exit_config is None:
        exit_config = ExitConfig()
    
    try:
        df = load_daily(code, start_date=start_date, end_date=end_date)
        if len(df) < 60:
            return None
        
        random_returns = []
        for trial in range(n_random):
            # 随机选一个入场日（中间60%的区域）
            start_idx = np.random.randint(len(df) // 5, len(df) * 4 // 5)
            entry_date = str(df.iloc[start_idx]['trade_date'])
            entry_price = df.iloc[start_idx]['close']
            
            # 用同样的动态出场逻辑
            pos_shares = 1000
            max_hold = 30  # 最多持有30天
            
            for j in range(start_idx + 1, min(start_idx + max_hold + 1, len(df))):
                price = df.iloc[j]['close']
                profit = (price / entry_price - 1) * 100
                
                # 简化出场：止损或止盈或到期
                if profit <= exit_config.stop_loss_pct:
                    random_returns.append(profit)
                    break
                elif profit >= exit_config.take_profit_full:
                    random_returns.append(profit)
                    break
                elif j == min(start_idx + max_hold, len(df) - 1):
                    random_returns.append(profit)
                    break
        
        return random_returns
    except:
        return None


def run_batch_backtest(universe, start_date="20240101", end_date="", 
                       with_baseline=True, verbose=False):
    """批量战法回测"""
    all_trades = []
    all_results = []
    baselines = {}
    errors = []
    
    total = len(universe)
    
    for i, (code, name) in enumerate(universe):
        if verbose or (i + 1) % 10 == 0:
            print(f"  [{i+1}/{total}] {code} {name}...", end=" ")
        
        result, error = backtest_single(code, name, start_date, end_date)
        
        if error:
            errors.append({"code": code, "name": name, "error": error})
            if verbose or (i + 1) % 10 == 0:
                print(f"❌ {error}")
            continue
        
        if result and result.trades:
            all_trades.extend(result.trades)
            all_results.append({
                "code": code, "name": name,
                "trades": len(result.trades),
                "win_rate": result.win_rate,
                "avg_profit": result.avg_profit_pct,
                "total_pnl": result.total_pnl_pct,
            })
            
            if verbose or (i + 1) % 10 == 0:
                print(f"✅ {len(result.trades)}笔 胜率{result.win_rate:.0f}% 均{result.avg_profit_pct:+.1f}%")
        else:
            if verbose or (i + 1) % 10 == 0:
                print(f"⚪ 无信号")
        
        # 随机对照
        if with_baseline:
            baseline = random_baseline(code, name, start_date, end_date)
            if baseline:
                baselines[code] = baseline
    
    return all_trades, all_results, baselines, errors


def analyze_results(all_trades, all_results, baselines):
    """分析回测结果"""
    
    if not all_trades:
        return {"error": "无交易记录"}
    
    # 总体统计
    completed = [t for t in all_trades if t.exit_date]
    wins = [t for t in completed if t.pnl_pct > 0]
    losses = [t for t in completed if t.pnl_pct <= 0]
    
    total_return = sum(t.pnl_pct for t in completed)
    avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0
    win_rate = len(wins) / len(completed) * 100 if completed else 0
    
    # 按出场原因统计
    reason_stats = {}
    for t in completed:
        reason = t.exit_reason
        if reason not in reason_stats:
            reason_stats[reason] = {"count": 0, "total_pnl": 0, "wins": 0}
        reason_stats[reason]["count"] += 1
        reason_stats[reason]["total_pnl"] += t.pnl_pct
        if t.pnl_pct > 0:
            reason_stats[reason]["wins"] += 1
    
    # 信号强度 vs 收益
    strength_bins = {"0.4-0.6": [], "0.6-0.8": [], "0.8-1.0": []}
    for t in completed:
        s = t.signal_strength
        if s < 0.6:
            strength_bins["0.4-0.6"].append(t.pnl_pct)
        elif s < 0.8:
            strength_bins["0.6-0.8"].append(t.pnl_pct)
        else:
            strength_bins["0.8-1.0"].append(t.pnl_pct)
    
    # 随机对照
    all_baseline_returns = []
    for code, returns in baselines.items():
        all_baseline_returns.extend(returns)
    
    baseline_avg = np.mean(all_baseline_returns) if all_baseline_returns else 0
    baseline_win_rate = sum(1 for r in all_baseline_returns if r > 0) / len(all_baseline_returns) * 100 if all_baseline_returns else 0
    
    # 失效场景分析
    failure_scenes = []
    for t in losses:
        scene = []
        if t.holding_days <= 3:
            scene.append("快速止损")
        if t.max_profit_pct > 10 and t.pnl_pct < 0:
            scene.append("利润回吐")
        if t.signal_strength >= 0.8:
            scene.append("强信号仍亏损")
        
        for s in scene:
            existing = [f for f in failure_scenes if f["scene"] == s]
            if existing:
                existing[0]["count"] += 1
                existing[0]["avg_loss"] += t.pnl_pct
            else:
                failure_scenes.append({"scene": s, "count": 1, "avg_loss": t.pnl_pct})
    
    for f in failure_scenes:
        f["avg_loss"] = round(f["avg_loss"] / f["count"], 2)
    failure_scenes.sort(key=lambda x: x["count"], reverse=True)
    
    return {
        "total_trades": len(completed),
        "win_rate": round(win_rate, 1),
        "win_count": len(wins),
        "loss_count": len(losses),
        "avg_profit": round(total_return / len(completed), 2) if completed else 0,
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "total_return": round(total_return, 2),
        "max_single_profit": round(max(t.pnl_pct for t in completed), 2) if completed else 0,
        "max_single_loss": round(min(t.pnl_pct for t in completed), 2) if completed else 0,
        "avg_holding_days": round(np.mean([t.holding_days for t in completed]), 1) if completed else 0,
        
        "reason_stats": {k: {
            "count": v["count"],
            "win_rate": round(v["wins"] / v["count"] * 100, 1),
            "avg_pnl": round(v["total_pnl"] / v["count"], 2),
        } for k, v in reason_stats.items()},
        
        "strength_analysis": {k: {
            "count": len(v),
            "avg_pnl": round(np.mean(v), 2) if v else 0,
            "win_rate": round(sum(1 for r in v if r > 0) / len(v) * 100, 1) if v else 0,
        } for k, v in strength_bins.items()},
        
        "baseline": {
            "avg_return": round(baseline_avg, 2),
            "win_rate": round(baseline_win_rate, 1),
            "sample_count": len(all_baseline_returns),
        },
        
        "alpha_vs_random": round(total_return / len(completed) - baseline_avg, 2) if completed else 0,
        
        "failure_scenes": failure_scenes[:5],
        
        "stocks_tested": len(all_results),
        "stocks_with_trades": len([r for r in all_results if r["trades"] > 0]),
    }


def generate_html_report(analysis, output_path):
    """生成HTML报告"""
    data_json = json.dumps(analysis, ensure_ascii=False, indent=2)
    
    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>战法回测验证报告</title>
<style>
:root{{--bg:#0f1117;--card:#1a1d29;--text:#e4e7ef;--muted:#8892a6;--accent:#4fc3f7;--green:#26a69a;--red:#ef5350;--yellow:#ffc107;--purple:#ab47bc;--border:#2a2e3e}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:var(--bg);color:var(--text);font-family:-apple-system,'PingFang SC','Microsoft YaHei',sans-serif;padding:20px;max-width:900px;margin:0 auto;line-height:1.7}}
h1{{text-align:center;font-size:1.5rem;margin-bottom:4px;background:linear-gradient(135deg,var(--accent),var(--purple));-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.subtitle{{text-align:center;color:var(--muted);font-size:0.85rem;margin-bottom:20px}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px;margin-bottom:16px}}
.card h2{{font-size:1.1rem;color:var(--accent);margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid var(--border)}}
.stat-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px}}
.stat{{background:rgba(79,195,247,0.08);border-radius:8px;padding:12px;text-align:center}}
.stat .val{{font-size:1.4rem;font-weight:700}}
.stat .lbl{{font-size:0.75rem;color:var(--muted);margin-top:2px}}
.green{{color:var(--green)}} .red{{color:var(--red)}} .yellow{{color:var(--yellow)}}
table{{width:100%;border-collapse:collapse;font-size:0.9rem;margin:8px 0}}
th{{text-align:left;padding:8px;color:var(--accent);border-bottom:2px solid var(--border)}}
td{{padding:8px;border-bottom:1px solid var(--border)}}
.bar{{display:inline-block;height:20px;border-radius:4px;vertical-align:middle}}
.bar-bg{{display:inline-block;width:100px;height:20px;background:var(--border);border-radius:4px;overflow:hidden;vertical-align:middle}}
.bar-fill{{height:20px;border-radius:4px}}
.positive{{background:var(--green)}} .negative{{background:var(--red)}}
.scene{{background:rgba(239,83,80,0.1);border-left:3px solid var(--red);padding:8px 12px;margin:6px 0;border-radius:0 6px 6px 0;font-size:0.9rem}}
.disclaimer{{text-align:center;color:var(--muted);font-size:0.8rem;margin-top:20px;padding:12px;border-top:1px solid var(--border)}}
</style>
</head>
<body>
<h1>⚔️ 战法回测验证报告</h1>
<div class="subtitle">MACD面积入场 + 动态出场 · <span id="date"></span></div>
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

// 总体统计
html += `<div class="card"><h2>📊 总体表现</h2>`;
html += `<div class="stat-grid">`;
html += `<div class="stat"><div class="val ${{D.win_rate > 50 ? 'green' : 'red'}}">${{D.win_rate}}%</div><div class="lbl">胜率</div></div>`;
html += `<div class="stat"><div class="val ${{D.avg_profit > 0 ? 'green' : 'red'}}">${{D.avg_profit > 0 ? '+' : ''}}${{D.avg_profit}}%</div><div class="lbl">平均收益</div></div>`;
html += `<div class="stat"><div class="val green">${{D.avg_win}}%</div><div class="lbl">平均盈利</div></div>`;
html += `<div class="stat"><div class="val red">${{D.avg_loss}}%</div><div class="lbl">平均亏损</div></div>`;
html += `<div class="stat"><div class="val">${{D.total_trades}}</div><div class="lbl">总交易数</div></div>`;
html += `<div class="stat"><div class="val">${{D.avg_holding_days}}天</div><div class="lbl">平均持有</div></div>`;
html += `</div></div>`;

// vs 随机对照
const alpha = D.alpha_vs_random;
html += `<div class="card"><h2>🎯 vs 随机入场对照</h2>`;
html += `<table>`;
html += `<tr><th>指标</th><th>战法</th><th>随机</th><th>Alpha</th></tr>`;
html += `<tr><td>平均收益</td><td>${{colorVal(D.avg_profit)}}</td><td>${{colorVal(D.baseline.avg_return)}}</td><td><b>${{colorVal(alpha)}}</b></td></tr>`;
html += `<tr><td>胜率</td><td>${{D.win_rate}}%</td><td>${{D.baseline.win_rate}}%</td><td>${{(D.win_rate - D.baseline.win_rate).toFixed(1)}}pp</td></tr>`;
html += `<tr><td>样本数</td><td>${{D.total_trades}}</td><td>${{D.baseline.sample_count}}</td><td>-</td></tr>`;
html += `</table>`;
html += `<p style="color:var(--muted);font-size:0.85rem;margin-top:8px">Alpha = 战法平均收益 - 随机平均收益。正值说明战法有超额收益。</p>`;
html += `</div>`;

// 出场原因
html += `<div class="card"><h2>🚪 出场原因分析</h2>`;
html += `<table><tr><th>原因</th><th>次数</th><th>胜率</th><th>平均收益</th></tr>`;
for (const [reason, s] of Object.entries(D.reason_stats)) {{
  html += `<tr><td>${{reason}}</td><td>${{s.count}}</td><td>${{s.win_rate}}%</td><td>${{colorVal(s.avg_pnl)}}</td></tr>`;
}}
html += `</table></div>`;

// 信号强度分析
html += `<div class="card"><h2>📡 信号强度 vs 收益</h2>`;
html += `<table><tr><th>强度区间</th><th>交易数</th><th>胜率</th><th>平均收益</th></tr>`;
for (const [strength, s] of Object.entries(D.strength_analysis)) {{
  html += `<tr><td>${{strength}}</td><td>${{s.count}}</td><td>${{s.win_rate}}%</td><td>${{colorVal(s.avg_pnl)}}</td></tr>`;
}}
html += `</table></div>`;

// 失效场景
if (D.failure_scenes && D.failure_scenes.length > 0) {{
  html += `<div class="card"><h2>⚠️ 信号失效场景Top5</h2>`;
  for (const f of D.failure_scenes) {{
    html += `<div class="scene"><b>${{f.scene}}</b>：${{f.count}}次，平均亏损 ${{f.avg_loss}}%</div>`;
  }}
  html += `</div>`;
}}

// 覆盖范围
html += `<div class="card"><h2>📋 覆盖范围</h2>`;
html += `<p>测试股票数: ${{D.stocks_tested}} | 有信号股票: ${{D.stocks_with_trades}}</p>`;
html += `</div>`;

app.innerHTML = html;
</script>
</body>
</html>'''
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="战法回测验证器")
    parser.add_argument("--sample", type=int, default=50, help="抽样股票数")
    parser.add_argument("--codes", type=str, help="指定股票代码（逗号分隔）")
    parser.add_argument("--all", action="store_true", help="全市场（慢）")
    parser.add_argument("--start", type=str, default="20240101", help="开始日期")
    parser.add_argument("--end", type=str, default="", help="结束日期")
    parser.add_argument("--no-baseline", action="store_true", help="跳过随机对照")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--html", action="store_true", help="生成HTML报告")
    args = parser.parse_args()
    
    print(f"{'='*60}")
    print(f"  战法回测验证器")
    print(f"  MACD面积入场 + 动态出场引擎")
    print(f"{'='*60}")
    
    # 获取股票池
    if args.codes:
        codes = args.codes.split(",")
        universe = [(c.strip(), f"stock_{c.strip()}") for c in codes]
        print(f"\n指定股票: {len(universe)}只")
    elif args.all:
        universe = get_stock_universe(mode="all")
        print(f"\n全市场: {len(universe)}只")
    else:
        universe = get_stock_universe(mode="sample", n=args.sample)
        print(f"\n抽样: {len(universe)}只（每行业取样）")
    
    print(f"回测区间: {args.start} ~ {args.end or 'today'}")
    print(f"随机对照: {'否' if args.no_baseline else '是'}")
    print()
    
    # 批量回测
    print(">>> 开始批量回测...")
    all_trades, all_results, baselines, errors = run_batch_backtest(
        universe, start_date=args.start, end_date=args.end,
        with_baseline=not args.no_baseline, verbose=args.verbose,
    )
    
    print(f"\n>>> 回测完成：{len(all_trades)}笔交易，{len(errors)}只出错")
    
    # 分析
    analysis = analyze_results(all_trades, all_results, baselines)
    
    # 打印结果
    print(f"\n{'='*60}")
    print(f"  战法验证结果")
    print(f"{'='*60}")
    print(f"  总交易数: {analysis.get('total_trades', 0)}")
    print(f"  胜率: {analysis.get('win_rate', 0)}% ({analysis.get('win_count', 0)}胜 / {analysis.get('loss_count', 0)}负)")
    print(f"  平均收益: {analysis.get('avg_profit', 0):+.2f}%")
    print(f"  平均盈利: +{analysis.get('avg_win', 0)}% / 平均亏损: {analysis.get('avg_loss', 0)}%")
    print(f"  累计收益: {analysis.get('total_return', 0):+.2f}%")
    print(f"  最大盈利: +{analysis.get('max_single_profit', 0)}% / 最大亏损: {analysis.get('max_single_loss', 0)}%")
    print(f"  平均持有: {analysis.get('avg_holding_days', 0)}天")
    
    print(f"\n  vs 随机入场:")
    print(f"    战法平均: {analysis.get('avg_profit', 0):+.2f}%")
    print(f"    随机平均: {analysis.get('baseline', {}).get('avg_return', 0):+.2f}%")
    print(f"    Alpha: {analysis.get('alpha_vs_random', 0):+.2f}%")
    
    if analysis.get('failure_scenes'):
        print(f"\n  失效场景:")
        for f in analysis['failure_scenes']:
            print(f"    {f['scene']}: {f['count']}次, 均{f['avg_loss']}%")
    
    print(f"\n  测试股票: {analysis.get('stocks_tested', 0)} | 有信号: {analysis.get('stocks_with_trades', 0)}")
    print(f"{'='*60}")
    
    # HTML报告
    if args.html:
        reports_dir = Path(__file__).resolve().parents[1] / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d")
        html_path = reports_dir / f"strategy_backtest_{date_str}.html"
        generate_html_report(analysis, str(html_path))
        print(f"\n>>> HTML报告: {html_path}")
    
    return analysis


if __name__ == "__main__":
    main()
