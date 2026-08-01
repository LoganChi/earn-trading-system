#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""每日盘前分析系统

对用户持仓列表逐个分析：
  1. MACD 面积信号 (src/signals/macd_area.py)
  2. 成交量分布信号 (src/signals/volume_profile.py)
  3. 五因子仓位评分 (src/position/five_factor.py)
  4. 恶魔股阶段 (src/position/stage_detector.py)
  5. 市场整体情绪 (src/position/limit_up_stats.py)

输出：
  - 控制台结构化文本
  - HTML 卡片式报告 reports/daily_analysis_YYYYMMDD.html
"""
from __future__ import annotations

import json
import sys
import os
from datetime import datetime
from pathlib import Path

# ---- 项目路径 ------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("TUSHARE_TOKEN", os.environ.get("TUSHARE_TOKEN", ""))

import numpy as np
import pandas as pd

from src.data.loader import load_daily
from src.signals.macd_area import generate_signals as gen_macd_signals, calc_macd
from src.signals.volume_profile import generate_vp_signals, calc_volume_profile
from src.position.five_factor import evaluate as five_factor_evaluate
from src.position.stage_detector import detect_stage
from src.position.limit_up_stats import compute_limit_up_stats, load_cached_stats


# ===========================================================================
# 配置
# ===========================================================================
PORTFOLIO = [
    {"code": "601727", "name": "上海电气"},
    {"code": "600839", "name": "四川长虹"},
    {"code": "601127", "name": "赛力斯"},
    {"code": "600460", "name": "士兰微"},
]

REPORTS_DIR = _PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# 分析单只股票
# ===========================================================================
def analyze_stock(code: str, name: str) -> dict:
    """对单只股票运行所有分析，返回结构化结果 dict。"""
    print(f"\n{'='*60}")
    print(f"  分析 {name} {code}")
    print(f"{'='*60}")

    result = {
        "code": code,
        "name": name,
        "errors": [],
    }

    # ---- 加载数据 ----
    try:
        df = load_daily(code)
        result["trade_date"] = str(df["trade_date"].iloc[-1])
        result["close"] = round(float(df["close"].iloc[-1]), 2)
        result["pct_chg"] = round(float(df["pct_chg"].iloc[-1]), 2)
    except Exception as e:
        result["errors"].append(f"数据加载失败: {e}")
        print(f"  [ERROR] 数据加载失败: {e}")
        return result

    # ---- MACD 面积信号 ----
    try:
        macd_signals = gen_macd_signals(df)
        if macd_signals:
            latest = macd_signals[-1]
            # 取最近绿峰面积
            from src.signals.macd_area import find_green_peaks
            dif, dea, macd_bar = calc_macd(df["close"].values)
            green_peaks = find_green_peaks(macd_bar, dif, df["trade_date"].values, df["close"].values)
            last_green_area = green_peaks[-1].area if green_peaks else 0.0
            last_green_severity = green_peaks[-1].severity if green_peaks else "N/A"

            result["macd"] = {
                "signal_type": latest.signal_type,
                "signal_strength": round(latest.signal_strength, 2),
                "green_area": round(last_green_area, 2),
                "green_severity": last_green_severity,
                "description": latest.description or "无描述",
                "dif": round(float(dif[-1]), 4),
                "dea": round(float(dea[-1]), 4),
                "macd_bar": round(float(macd_bar[-1]), 4),
            }
        else:
            result["macd"] = {"description": "MACD信号不足"}
            result["errors"].append("MACD信号列表为空")
    except Exception as e:
        result["macd"] = {"description": f"MACD分析失败: {e}"}
        result["errors"].append(f"MACD: {e}")
        print(f"  [ERROR] MACD: {e}")

    # ---- VP 信号 ----
    try:
        vp_signals = generate_vp_signals(df)
        if vp_signals:
            latest_vp = vp_signals[-1]
            vp = calc_volume_profile(df.tail(20))
            result["vp"] = {
                "signal_type": latest_vp.signal_type,
                "signal_strength": round(latest_vp.signal_strength, 2),
                "poc_price": vp.poc_price,
                "value_area_high": vp.value_area_high,
                "value_area_low": vp.value_area_low,
                "at_edge": latest_vp.at_edge,
                "edge_breakout": latest_vp.edge_breakout,
                "description": latest_vp.description or "无描述",
            }
        else:
            result["vp"] = {"description": "VP信号不足"}
            result["errors"].append("VP信号列表为空")
    except Exception as e:
        result["vp"] = {"description": f"VP分析失败: {e}"}
        result["errors"].append(f"VP: {e}")
        print(f"  [ERROR] VP: {e}")

    # ---- 五因子仓位评分 ----
    try:
        ff = five_factor_evaluate(code)
        result["five_factor"] = {
            "long_votes": ff.long_votes,
            "short_votes": ff.short_votes,
            "neutral_votes": ff.neutral_votes,
            "position_advice": ff.position_advice,
            "position_ratio": ff.position_ratio,
            "factors": [
                {"name": "估值", "direction": ff.valuation.direction.value, "strength": round(ff.valuation.strength, 2), "detail": ff.valuation.detail},
                {"name": "资金", "direction": ff.capital_flow.direction.value, "strength": round(ff.capital_flow.strength, 2), "detail": ff.capital_flow.detail},
                {"name": "技术", "direction": ff.technical.direction.value, "strength": round(ff.technical.strength, 2), "detail": ff.technical.detail},
                {"name": "情绪", "direction": ff.sentiment.direction.value, "strength": round(ff.sentiment.strength, 2), "detail": ff.sentiment.detail},
                {"name": "基本面", "direction": ff.fundamental.direction.value, "strength": round(ff.fundamental.strength, 2), "detail": ff.fundamental.detail},
            ],
        }
    except Exception as e:
        result["five_factor"] = {"position_advice": "评估失败"}
        result["errors"].append(f"五因子: {e}")
        print(f"  [ERROR] 五因子: {e}")

    # ---- 恶魔股阶段 ----
    try:
        stage = detect_stage(code, df=df)
        result["stage"] = {
            "stage": stage.stage.value,
            "win_rate": stage.win_rate,
            "total_signals": stage.total_signals,
            "winning_signals": stage.winning_signals,
            "position_adjustment": stage.position_adjustment,
            "description": stage.description,
        }
    except Exception as e:
        result["stage"] = {"stage": "未知", "description": f"阶段识别失败: {e}"}
        result["errors"].append(f"阶段: {e}")
        print(f"  [ERROR] 阶段: {e}")

    # ---- 止损位计算（基于近期低点 + ATR）----
    try:
        recent = df.tail(20)
        low_20 = float(recent["low"].min())
        close_last = float(df["close"].iloc[-1])
        # 简单ATR
        if len(df) >= 20:
            tr_list = []
            for i in range(-20, 0):
                hi = float(df["high"].iloc[i])
                lo = float(df["low"].iloc[i])
                pc = float(df["close"].iloc[i - 1]) if i > -20 else float(df["close"].iloc[i])
                tr_list.append(max(hi - lo, abs(hi - pc), abs(lo - pc)))
            atr = float(np.mean(tr_list)) if tr_list else close_last * 0.05
        else:
            atr = close_last * 0.05
        stop_loss = round(min(low_20, close_last - 1.5 * atr), 2)
        result["stop_loss"] = stop_loss
        result["low_20"] = round(low_20, 2)
        result["atr"] = round(atr, 3)
    except Exception as e:
        result["stop_loss"] = round(result.get("close", 0) * 0.92, 2)
        result["errors"].append(f"止损: {e}")

    # ---- 综合建议 ----
    result["action"] = _derive_action(result)

    print(f"  ✓ 完成: {result['action']}")
    return result


def _derive_action(r: dict) -> str:
    """综合所有信号得出操作建议。"""
    ff = r.get("five_factor", {})
    macd = r.get("macd", {})
    vp = r.get("vp", {})
    stage = r.get("stage", {})

    long_votes = ff.get("long_votes", 0)
    short_votes = ff.get("short_votes", 0)
    macd_type = macd.get("signal_type", "neutral")
    stage_name = stage.get("stage", "")

    # 恶魔股阶段 → 偏防守
    if "恶魔" in stage_name:
        if short_votes >= 2:
            return "减仓"
        return "关注"

    # 出场信号
    if macd_type == "exit_warning" or short_votes >= 3:
        return "减仓"

    # 强入场信号
    if macd_type == "entry_candidate" and vp.get("edge_breakout"):
        return "持有（可加仓）"

    if long_votes >= 3:
        return "持有"
    elif long_votes == 2:
        return "持有"
    elif short_votes >= 2:
        return "关注"
    else:
        return "持有"


# ===========================================================================
# 市场情绪
# ===========================================================================
def get_market_sentiment() -> dict:
    """获取市场整体情绪（连板统计）。"""
    sentiment = {
        "promotion_1to2": 0.0,
        "promotion_2to3": 0.0,
        "limit_up_count": 0,
        "avg_next_day_return": 0.0,
        "thermometer": "冰点",
        "description": "",
    }

    try:
        stats_list = compute_limit_up_stats(lookback_days=10, use_cache=True)
        if stats_list:
            latest = stats_list[-1]
            sentiment["promotion_1to2"] = latest.promotion_1to2
            sentiment["promotion_2to3"] = latest.promotion_2to3
            sentiment["limit_up_count"] = latest.limit_up_count
            sentiment["avg_next_day_return"] = latest.avg_next_day_return

            p12 = latest.promotion_1to2
            if p12 > 25:
                sentiment["thermometer"] = "过热"
            elif p12 > 15:
                sentiment["thermometer"] = "温热"
            elif p12 > 8:
                sentiment["thermometer"] = "正常"
            elif p12 > 3:
                sentiment["thermometer"] = "偏冷"
            else:
                sentiment["thermometer"] = "冰点"

            sentiment["description"] = (
                f"涨停{latest.limit_up_count}只，"
                f"1→2晋级率{p12}%，"
                f"2→3晋级率{latest.promotion_2to3}%，"
                f"涨停股次日均收{latest.avg_next_day_return}%"
            )
    except Exception as e:
        # 尝试从缓存读
        try:
            cache_df = load_cached_stats()
            if len(cache_df) > 0:
                latest_row = cache_df.iloc[-1]
                sentiment["promotion_1to2"] = float(latest_row.get("promotion_1to2", 0))
                sentiment["promotion_2to3"] = float(latest_row.get("promotion_2to3", 0))
                sentiment["limit_up_count"] = int(latest_row.get("limit_up_count", 0))
                sentiment["avg_next_day_return"] = float(latest_row.get("avg_next_day_return", 0))
                sentiment["description"] = f"(缓存) 涨停{sentiment['limit_up_count']}只，1→2:{sentiment['promotion_1to2']}%"
            else:
                sentiment["description"] = f"连板数据获取失败: {e}"
        except Exception as e2:
            sentiment["description"] = f"连板数据不可用: {e} / {e2}"

    return sentiment


# ===========================================================================
# 建议总仓位
# ===========================================================================
def calc_total_position(results: list, sentiment: dict) -> float:
    """根据各股评分和市场情绪计算建议总仓位。"""
    if not results:
        return 0.3

    ratios = []
    for r in results:
        ff = r.get("five_factor", {})
        ratio = ff.get("position_ratio", 0.3)
        stage = r.get("stage", {})
        adj = stage.get("position_adjustment", 1.0)
        adjusted = ratio * adj
        ratios.append(min(1.0, adjusted))

    avg_ratio = float(np.mean(ratios)) if ratios else 0.3

    # 市场情绪调整
    p12 = sentiment.get("promotion_1to2", 0)
    if p12 < 5:
        avg_ratio *= 0.7
    elif p12 < 10:
        avg_ratio *= 0.85
    elif p12 > 25:
        avg_ratio = min(0.95, avg_ratio * 1.1)

    return round(min(0.95, avg_ratio), 2)


# ===========================================================================
# 文本报告
# ===========================================================================
def generate_text_report(results: list, sentiment: dict, total_position: float, date_str: str) -> str:
    """生成控制台文本报告。"""
    lines = []
    lines.append(f"📊 每日盘前分析 {date_str}")
    lines.append("")
    lines.append(f"市场情绪：连板晋级率{sentiment.get('promotion_1to2', 0)}%，温度计{sentiment.get('thermometer', 'N/A')}")
    lines.append(f"  {sentiment.get('description', '')}")
    lines.append(f"建议总仓位：{total_position:.0%}")
    lines.append("")
    lines.append("─" * 60)

    for r in results:
        name = r.get("name", "")
        code = r.get("code", "")
        lines.append(f"\n{name} {code}")

        if r.get("errors") and not r.get("macd"):
            lines.append(f"  ❌ 分析失败: {r['errors'][0]}")
            continue

        # 最新价
        close = r.get("close", 0)
        pct = r.get("pct_chg", 0)
        lines.append(f"  最新价：{close}元 ({pct:+.2f}%)")

        # MACD
        macd = r.get("macd", {})
        lines.append(f"  信号：MACD面积 {macd.get('green_area', 'N/A')}({macd.get('green_severity', '')}) | "
                      f"VP：{macd.get('signal_type', 'N/A')}")
        vp = r.get("vp", {})
        if vp.get("poc_price"):
            lines.append(f"  VP详情：POC={vp['poc_price']} 价值区[{vp.get('value_area_low', '?')}~{vp.get('value_area_high', '?')}] "
                          f"{'边缘突破!' if vp.get('edge_breakout') else vp.get('signal_type', '')}")

        # 五因子
        ff = r.get("five_factor", {})
        lines.append(f"  仓位评分：{ff.get('long_votes', 0)}票多/{ff.get('short_votes', 0)}票空 → {ff.get('position_advice', 'N/A')}({ff.get('position_ratio', 0):.0%})")
        for f in ff.get("factors", []):
            lines.append(f"    {f['name']}: {f['direction']} ({f['strength']:.0%}) {f['detail']}")

        # 阶段
        stage = r.get("stage", {})
        lines.append(f"  阶段：{stage.get('stage', 'N/A')}（近期胜率{stage.get('win_rate', 0)}%, {stage.get('winning_signals', 0)}/{stage.get('total_signals', 0)}）")

        # 建议
        lines.append(f"  建议：{r.get('action', 'N/A')}")
        lines.append(f"  止损位：{r.get('stop_loss', 'N/A')}元 (近20日低{r.get('low_20', '?')})")

    lines.append("\n" + "═" * 60)
    lines.append("⚠️  以上为量化信号参考，不构成投资建议。请结合基本面和自身判断操作。")
    return "\n".join(lines)


# ===========================================================================
# HTML 报告
# ===========================================================================
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>📊 每日盘前分析 __DATE__</title>
<style>
:root {{
  --bg: #0f1117;
  --card-bg: #1a1d29;
  --card-hover: #22263a;
  --text: #e4e7ef;
  --text-muted: #8892a6;
  --accent: #4fc3f7;
  --green: #26a69a;
  --red: #ef5350;
  --yellow: #ffc107;
  --purple: #ab47bc;
  --border: #2a2e3e;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, 'PingFang SC', 'Microsoft YaHei', sans-serif;
  padding: 20px;
  max-width: 1000px;
  margin: 0 auto;
}}
h1 {{
  text-align: center;
  font-size: 1.6rem;
  margin-bottom: 5px;
  background: linear-gradient(135deg, var(--accent), var(--purple));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}}
.subtitle {{ text-align: center; color: var(--text-muted); font-size: 0.9rem; margin-bottom: 20px; }}

.summary-bar {{
  display: flex;
  gap: 15px;
  margin-bottom: 25px;
  flex-wrap: wrap;
}}
.summary-card {{
  flex: 1;
  min-width: 180px;
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px 20px;
  text-align: center;
}}
.summary-card .label {{ color: var(--text-muted); font-size: 0.8rem; margin-bottom: 5px; }}
.summary-card .value {{ font-size: 1.5rem; font-weight: 700; }}

.stock-card {{
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 20px;
  margin-bottom: 18px;
  transition: border-color 0.2s;
}}
.stock-card:hover {{ border-color: var(--accent); }}
.stock-header {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 15px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--border);
}}
.stock-name {{ font-size: 1.3rem; font-weight: 700; }}
.stock-code {{ color: var(--text-muted); font-size: 0.9rem; margin-left: 8px; }}
.stock-price {{ font-size: 1.1rem; }}
.stock-price .pct {{ margin-left: 8px; font-size: 0.9rem; }}

.signal-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 12px;
  margin-bottom: 15px;
}}
.signal-box {{
  background: rgba(255,255,255,0.03);
  border-radius: 8px;
  padding: 12px;
}}
.signal-box .title {{
  font-size: 0.75rem;
  color: var(--text-muted);
  text-transform: uppercase;
  margin-bottom: 6px;
}}
.signal-box .content {{ font-size: 0.95rem; }}

.factor-row {{
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 6px 0;
  font-size: 0.9rem;
}}
.factor-badge {{
  display: inline-block;
  width: 40px;
  text-align: center;
  border-radius: 4px;
  font-size: 0.8rem;
  font-weight: 600;
  padding: 2px 0;
}}
.badge-long {{ background: rgba(38,166,154,0.2); color: var(--green); border: 1px solid var(--green); }}
.badge-short {{ background: rgba(239,83,80,0.2); color: var(--red); border: 1px solid var(--red); }}
.badge-neutral {{ background: rgba(136,146,166,0.2); color: var(--text-muted); border: 1px solid var(--text-muted); }}

.action-box {{
  display: flex;
  gap: 15px;
  margin-top: 12px;
  flex-wrap: wrap;
}}
.action-tag {{
  padding: 8px 16px;
  border-radius: 8px;
  font-weight: 600;
  font-size: 0.95rem;
}}
.tag-hold {{ background: rgba(38,166,154,0.15); color: var(--green); border: 1px solid var(--green); }}
.tag-reduce {{ background: rgba(239,83,80,0.15); color: var(--red); border: 1px solid var(--red); }}
.tag-watch {{ background: rgba(255,193,7,0.15); color: var(--yellow); border: 1px solid var(--yellow); }}
.tag-add {{ background: rgba(79,195,247,0.15); color: var(--accent); border: 1px solid var(--accent); }}

.stop-loss {{ color: var(--red); font-weight: 600; }}

.stage-badge {{
  display: inline-block;
  padding: 3px 10px;
  border-radius: 6px;
  font-size: 0.8rem;
  font-weight: 600;
}}
.stage-demon {{ background: rgba(239,83,80,0.2); color: var(--red); }}
.stage-neutral {{ background: rgba(136,146,166,0.2); color: var(--text-muted); }}
.stage-matched {{ background: rgba(38,166,154,0.2); color: var(--green); }}

.strength-bar {{
  display: inline-block;
  width: 50px;
  height: 6px;
  background: var(--border);
  border-radius: 3px;
  margin-left: 5px;
  vertical-align: middle;
  overflow: hidden;
}}
.strength-fill {{
  height: 100%;
  border-radius: 3px;
  transition: width 0.3s;
}}

.disclaimer {{
  text-align: center;
  color: var(--text-muted);
  font-size: 0.8rem;
  margin-top: 30px;
  padding: 15px;
  border-top: 1px solid var(--border);
}}

@media (max-width: 600px) {{
  .stock-header {{ flex-direction: column; align-items: flex-start; gap: 8px; }}
  .signal-grid {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>

<h1>📊 每日盘前分析</h1>
<p class="subtitle">__DATE__ | earn-trading-system</p>

<div id="app"></div>

<p class="disclaimer">⚠️ 以上为量化信号参考，不构成投资建议。请结合基本面和自身判断操作。<br>
Generated by daily_analysis.py</p>

<script>
const REPORT_DATA = __DATA_JSON__;

function badgeClass(dir) {{
  if (dir === '多') return 'badge-long';
  if (dir === '空') return 'badge-short';
  return 'badge-neutral';
}}

function actionTagClass(action) {{
  if (action.includes('加仓')) return 'tag-add';
  if (action.includes('减仓')) return 'tag-reduce';
  if (action.includes('关注')) return 'tag-watch';
  return 'tag-hold';
}}

function stageClass(stage) {{
  if (stage.includes('恶魔')) return 'stage-demon';
  if (stage.includes('匹配')) return 'stage-matched';
  return 'stage-neutral';
}}

function strengthColor(s) {{
  if (s > 0.6) return '#26a69a';
  if (s > 0.3) return '#ffc107';
  return '#ef5350';
}}

function pctColor(pct) {{
  return pct >= 0 ? '#26a69a' : '#ef5350';
}}

function renderFactor(f) {{
  const sc = strengthColor(f.strength);
  return `<div class="factor-row">
    <span class="factor-badge ${{badgeClass(f.direction)}}">${{f.direction}}</span>
    <span style="width:50px;color:var(--text-muted)">${{f.name}}</span>
    <span style="flex:1;font-size:0.85rem;color:var(--text-muted)">${{f.detail}}</span>
    <span class="strength-bar"><span class="strength-fill" style="width:${{f.strength*100}}%;background:${{sc}}"></span></span>
  </div>`;
}}

function renderStock(r) {{
  const macd = r.macd || {{}};
  const vp = r.vp || {{}};
  const ff = r.five_factor || {{}};
  const stage = r.stage || {{}};

  let factorsHtml = (ff.factors || []).map(renderFactor).join('');

  let actionHtml = '';
  if (r.action) {{
    actionHtml = `<div class="action-box">
      <span class="action-tag ${{actionTagClass(r.action)}}">📋 ${{r.action}}</span>
      <span class="action-tag tag-reduce">🛑 止损: <span class="stop-loss">${{r.stop_loss || '?'}}元</span></span>
    </div>`;
  }}

  const stageStr = stage.stage || '未知';
  const winRate = stage.win_rate != null ? `${{stage.win_rate}}%` : 'N/A';

  return `<div class="stock-card">
    <div class="stock-header">
      <div>
        <span class="stock-name">${{r.name}}</span>
        <span class="stock-code">${{r.code}}</span>
      </div>
      <div class="stock-price">
        ${{r.close || '?'}}元
        <span class="pct" style="color:${{pctColor(r.pct_chg || 0)}}">${{(r.pct_chg || 0) > 0 ? '+' : ''}}${{(r.pct_chg || 0).toFixed(2)}}%</span>
      </div>
    </div>

    <div class="signal-grid">
      <div class="signal-box">
        <div class="title">MACD 面积</div>
        <div class="content">
          绿峰面积: <strong>${{macd.green_area || 'N/A'}}</strong> (${{macd.green_severity || ''}})<br>
          <span style="color:var(--text-muted);font-size:0.85rem">${{macd.description || ''}}</span>
        </div>
      </div>
      <div class="signal-box">
        <div class="title">成交量分布</div>
        <div class="content">
          POC: <strong>${{vp.poc_price || 'N/A'}}</strong><br>
          价值区: ${{vp.value_area_low || '?'}} ~ ${{vp.value_area_high || '?'}}<br>
          <span style="color:var(--text-muted);font-size:0.85rem">${{vp.description || ''}}</span>
        </div>
      </div>
      <div class="signal-box">
        <div class="title">阶段识别</div>
        <div class="content">
          <span class="stage-badge ${{stageClass(stageStr)}}">${{stageStr}}</span><br>
          胜率: ${{winRate}} (${{stage.winning_signals || 0}}/${{stage.total_signals || 0}})
        </div>
      </div>
    </div>

    <div style="margin:10px 0;">
      <div style="font-size:0.75rem;color:var(--text-muted);text-transform:uppercase;margin-bottom:8px;">
        五因子投票: ${{ff.long_votes || 0}}多 / ${{ff.short_votes || 0}}空 / ${{ff.neutral_votes || 0}}中性 → ${{ff.position_advice || 'N/A'}} (${{((ff.position_ratio || 0)*100).toFixed(0)}}%)
      </div>
      ${{factorsHtml}}
    </div>

    ${{actionHtml}}
  </div>`;
}}

function render() {{
  const app = document.getElementById('app');
  let html = '';

  // 摘要栏
  const s = REPORT_DATA.sentiment || {{}};
  html += `<div class="summary-bar">
    <div class="summary-card">
      <div class="label">🌡️ 市场温度</div>
      <div class="value" style="color:var(--yellow)">${{s.thermometer || 'N/A'}}</div>
    </div>
    <div class="summary-card">
      <div class="label">连板晋级率</div>
      <div class="value" style="color:var(--accent)">${{s.promotion_1to2 || 0}}%</div>
    </div>
    <div class="summary-card">
      <div class="label">涨停股数</div>
      <div class="value">${{s.limit_up_count || 0}}</div>
    </div>
    <div class="summary-card">
      <div class="label">建议总仓位</div>
      <div class="value" style="color:var(--green)">${{((REPORT_DATA.total_position || 0)*100).toFixed(0)}}%</div>
    </div>
  </div>`;

  // 股票卡片
  for (const r of REPORT_DATA.results) {{
    html += renderStock(r);
  }}

  app.innerHTML = html;
}}

render();
</script>
</body>
</html>
"""


def generate_html_report(results: list, sentiment: dict, total_position: float, date_str: str) -> str:
    """生成HTML报告并返回文件路径。"""
    report_data = {
        "date": date_str,
        "sentiment": sentiment,
        "total_position": total_position,
        "results": results,
    }

    data_json = json.dumps(report_data, ensure_ascii=False, default=str)
    html = HTML_TEMPLATE.replace("__DATE__", date_str).replace("__DATA_JSON__", data_json)

    html_path = REPORTS_DIR / f"daily_analysis_{date_str.replace('-', '')}.html"
    html_path.write_text(html, encoding="utf-8")
    return str(html_path)


# ===========================================================================
# 主函数
# ===========================================================================
def main():
    date_str = datetime.now().strftime("%Y-%m-%d")
    print(f"╔{'═'*58}╗")
    print(f"║  📊 每日盘前分析系统  {date_str}{' '*24}║")
    print(f"╚{'═'*58}╝")

    # 1. 市场情绪
    print("\n>>> 1/3 获取市场情绪...")
    sentiment = get_market_sentiment()
    print(f"  ✓ {sentiment.get('description', 'N/A')}")

    # 2. 逐股分析
    print(f"\n>>> 2/3 分析 {len(PORTFOLIO)} 只持仓股...")
    results = []
    for stock in PORTFOLIO:
        r = analyze_stock(stock["code"], stock["name"])
        results.append(r)

    # 3. 计算总仓位
    total_position = calc_total_position(results, sentiment)

    # 4. 生成文本报告
    text_report = generate_text_report(results, sentiment, total_position, date_str)
    print(f"\n{'═'*60}")
    print(text_report)

    # 5. 生成HTML报告
    print(f"\n{'═'*60}")
    print(">>> 3/3 生成HTML报告...")
    html_path = generate_html_report(results, sentiment, total_position, date_str)
    print(f"  ✓ HTML报告已保存: {html_path}")

    # 打印路径（方便脚本提取）
    print(f"\n[HTML_PATH]{html_path}[END_HTML_PATH]")

    return html_path


if __name__ == "__main__":
    main()
