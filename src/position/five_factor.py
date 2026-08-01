#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""五因子仓位评分器

基于"做量化的西蒙"v16期五因子投票框架：
  1. 估值因子  — PE历史分位（近1年）
  2. 资金因子  — 换手率趋势（近5日 vs 近20日均值的比率）
  3. 技术因子  — MACD面积信号强度（复用 src/signals/macd_area.py）
  4. 情绪因子  — 连板晋级率（全市场涨停数据，近5日）
  5. 基本面因子 — 板块强度（个股当日涨跌幅 vs 大盘涨跌幅）

每个因子输出：方向（多/空/中性）+ 强度（0-1）
五因子投票：≥3票多 → 满仓；2票多 → 半仓；≤1票多 → 空仓/轻仓
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# ---- 项目内模块 ----------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from data.loader import load_daily, load_index, _init_tushare, _to_ts_code, _CACHE_DIR
from signals.macd_area import generate_signals


# ===========================================================================
# 数据结构
# ===========================================================================
class Direction(str, Enum):
    LONG = "多"
    SHORT = "空"
    NEUTRAL = "中性"


@dataclass
class FactorScore:
    """单个因子的评分"""
    name: str
    direction: Direction
    strength: float          # 0-1
    raw_value: float = 0.0   # 原始指标值（便于调试）
    detail: str = ""


@dataclass
class FiveFactorResult:
    """五因子评分总结果"""
    code: str
    trade_date: str

    valuation: FactorScore        # 估值因子
    capital_flow: FactorScore     # 资金因子
    technical: FactorScore        # 技术因子
    sentiment: FactorScore        # 情绪因子
    fundamental: FactorScore      # 基本面因子

    long_votes: int = 0           # 多票数
    short_votes: int = 0          # 空票数
    neutral_votes: int = 0        # 中性票数
    position_advice: str = ""     # "满仓" / "半仓" / "空仓"
    position_ratio: float = 0.0   # 建议仓位比例 0-1

    def summary(self) -> str:
        lines = [
            f"═══ 五因子仓位评分  {self.code}  {self.trade_date} ═══",
            f"  估值因子  : {self.valuation.direction.value:2s}  强度 {self.valuation.strength:.0%}  "
            f"| PE分位 {self.valuation.raw_value:.1%}  {self.valuation.detail}",
            f"  资金因子  : {self.capital_flow.direction.value:2s}  强度 {self.capital_flow.strength:.0%}  "
            f"| 换手率比率 {self.capital_flow.raw_value:.2f}  {self.capital_flow.detail}",
            f"  技术因子  : {self.technical.direction.value:2s}  强度 {self.technical.strength:.0%}  "
            f"| {self.technical.detail}",
            f"  情绪因子  : {self.sentiment.direction.value:2s}  强度 {self.sentiment.strength:.0%}  "
            f"| 连板晋级率 {self.sentiment.raw_value:.1%}  {self.sentiment.detail}",
            f"  基本面因子: {self.fundamental.direction.value:2s}  强度 {self.fundamental.strength:.0%}  "
            f"| 超额 {self.fundamental.raw_value:+.2f}%  {self.fundamental.detail}",
            f"  ─────────────────────────────────",
            f"  投票  多:{self.long_votes}  空:{self.short_votes}  中性:{self.neutral_votes}",
            f"  ➤ 仓位建议: {self.position_advice}  ({self.position_ratio:.0%})",
        ]
        return "\n".join(lines)


# ===========================================================================
# 工具：tushare 扩展数据拉取（带缓存，复用 loader 的缓存目录）
# ===========================================================================
def _fetch_daily_basic_pe(code: str, lookback_days: int = 400) -> pd.DataFrame:
    """拉取个股 daily_basic 中的 PE（ttm）序列，近1年+缓冲。

    返回: trade_date, pe_ttm  （按 trade_date 升序）
    """
    cache_file = _CACHE_DIR / f"pe_{code}.csv"
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y%m%d")

    if cache_file.exists():
        df = pd.read_csv(cache_file, dtype={"trade_date": str})
        df = df[(df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)]
        if len(df) > 30:
            return df.sort_values("trade_date").reset_index(drop=True)

    pro = _init_tushare()
    if pro is None:
        raise RuntimeError("TUSHARE_TOKEN 未配置，无法拉取PE数据")

    ts_code = _to_ts_code(code)
    raw = pro.daily_basic(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        fields="ts_code,trade_date,pe_ttm",
    )
    if raw is None or len(raw) == 0:
        raise RuntimeError(f"拉取 {code} PE(daily_basic) 失败")

    df = raw[["trade_date", "pe_ttm"]].copy()
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values("trade_date").reset_index(drop=True)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_file, index=False)
    return df


def _fetch_limit_list_recent(days: int = 7) -> pd.DataFrame:
    """拉取最近N天全市场涨停列表。

    返回: trade_date, ts_code, name, limit（涨停类型 U/D）
    """
    cache_file = _CACHE_DIR / "limit_list_recent.csv"
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=days + 10)).strftime("%Y%m%d")  # +10 buffer for weekends

    if cache_file.exists():
        df = pd.read_csv(cache_file, dtype={"trade_date": str})
        df = df[(df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)]
        if len(df) > 10:
            return df.sort_values("trade_date").reset_index(drop=True)

    pro = _init_tushare()
    if pro is None:
        raise RuntimeError("TUSHARE_TOKEN 未配置，无法拉取涨停数据")

    frames = []
    cur = datetime.now() - timedelta(days=days + 10)
    # limit_list_d 按日期逐日拉取
    while cur <= datetime.now():
        d = cur.strftime("%Y%m%d")
        try:
            raw = pro.limit_list_d(trade_date=d, limit_type="U")
            if raw is not None and len(raw) > 0:
                frames.append(raw)
        except Exception:
            pass
        cur += timedelta(days=1)

    if not frames:
        return pd.DataFrame(columns=["trade_date", "ts_code", "name", "limit_times"])

    df = pd.concat(frames, ignore_index=True)
    # limit_times: 连板数（1=首板, 2=二板...）
    keep = [c for c in ["trade_date", "ts_code", "name", "limit_times"] if c in df.columns]
    df = df[keep].copy()
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values("trade_date").reset_index(drop=True)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_file, index=False)
    return df


# ===========================================================================
# 因子1: 估值因子 — PE历史分位
# ===========================================================================
def score_valuation(pe_df: pd.DataFrame) -> FactorScore:
    """PE历史分位越低 → 越便宜 → 看多。

    分位 < 20%  → 多（强度高）
    分位 20-40% → 多（强度中）
    分位 40-60% → 中性
    分位 60-80% → 空
    分位 > 80%  → 空（强度高）
    """
    pe_series = pe_df["pe_ttm"] if "pe_ttm" in pe_df.columns else pe_df.iloc[:, 1]
    pe_vals = pe_series.dropna()
    pe_vals = pe_vals[pe_vals > 0]  # 负PE（亏损）剔除

    if len(pe_vals) < 20:
        return FactorScore("估值", Direction.NEUTRAL, 0.0, 0.0, "PE数据不足")

    current_pe = pe_vals.iloc[-1]
    # 近1年（约250个交易日）分位
    window = pe_vals.iloc[-250:] if len(pe_vals) > 250 else pe_vals
    percentile = (window < current_pe).sum() / len(window)

    if percentile < 0.20:
        strength = 0.5 + (0.20 - percentile) / 0.20 * 0.5  # 0.5-1.0
        direction = Direction.LONG
        detail = f"PE={current_pe:.1f} 极低估值"
    elif percentile < 0.40:
        strength = 0.3 + (0.40 - percentile) / 0.20 * 0.2  # 0.3-0.5
        direction = Direction.LONG
        detail = f"PE={current_pe:.1f} 偏低估"
    elif percentile <= 0.60:
        strength = 0.2
        direction = Direction.NEUTRAL
        detail = f"PE={current_pe:.1f} 估值合理"
    elif percentile <= 0.80:
        strength = 0.3 + (percentile - 0.60) / 0.20 * 0.2  # 0.3-0.5
        direction = Direction.SHORT
        detail = f"PE={current_pe:.1f} 偏高估"
    else:
        strength = 0.5 + (percentile - 0.80) / 0.20 * 0.5  # 0.5-1.0
        direction = Direction.SHORT
        detail = f"PE={current_pe:.1f} 极高估值"

    return FactorScore("估值", direction, round(strength, 3), round(percentile, 4), detail)


# ===========================================================================
# 因子2: 资金因子 — 换手率趋势
# ===========================================================================
def score_capital_flow(daily_df: pd.DataFrame) -> FactorScore:
    """换手率比率 = 近5日均值 / 近20日均值

    ratio > 1.3 → 资金涌入 → 多
    ratio 1.0-1.3 → 偏多
    ratio 0.8-1.0 → 中性/偏空
    ratio < 0.8 → 资金撤退 → 空

    注：tushare daily 接口 vol 单位为手，无 turnover_rate 字段时用 vol 近似。
    若有 daily_basic 的 turnover_rate 更精确，这里优先用 vol。
    """
    if "vol" not in daily_df.columns or len(daily_df) < 25:
        return FactorScore("资金", Direction.NEUTRAL, 0.0, 1.0, "成交量数据不足")

    vol = daily_df["vol"].astype(float).values
    recent_5 = np.mean(vol[-5:])
    recent_20 = np.mean(vol[-20:])

    if recent_20 <= 0:
        return FactorScore("资金", Direction.NEUTRAL, 0.0, 1.0, "成交量为零")

    ratio = recent_5 / recent_20

    if ratio > 1.5:
        strength = min(1.0, 0.5 + (ratio - 1.5) * 0.5)
        direction = Direction.LONG
        detail = "量能急剧放大"
    elif ratio > 1.2:
        strength = 0.3 + (ratio - 1.2) * 0.5  # 0.3-0.4
        direction = Direction.LONG
        detail = "放量温和"
    elif ratio > 0.9:
        strength = 0.2
        direction = Direction.NEUTRAL
        detail = "量能平稳"
    elif ratio > 0.7:
        strength = 0.3 + (0.9 - ratio) * 0.5  # 0.3-0.4
        direction = Direction.SHORT
        detail = "缩量"
    else:
        strength = min(1.0, 0.5 + (0.7 - ratio) * 0.8)
        direction = Direction.SHORT
        detail = "量能急剧萎缩"

    return FactorScore("资金", direction, round(strength, 3), round(ratio, 4), detail)


# ===========================================================================
# 因子3: 技术因子 — MACD面积信号强度
# ===========================================================================
def score_technical(daily_df: pd.DataFrame) -> FactorScore:
    """从 macd_area.generate_signals 取最新信号。

    signal_type:
      entry_candidate → 多（用 signal_strength）
      exit_warning    → 空
      neutral         → 中性 / 根据最新MACD柱方向微调
    """
    signals = generate_signals(daily_df)
    if not signals:
        return FactorScore("技术", Direction.NEUTRAL, 0.0, 0.0, "MACD信号不足")

    latest = signals[-1]

    if latest.signal_type == "entry_candidate":
        return FactorScore(
            "技术", Direction.LONG,
            round(latest.signal_strength, 3),
            round(latest.signal_strength, 3),
            latest.description or "MACD入场信号",
        )
    elif latest.signal_type == "exit_warning":
        return FactorScore(
            "技术", Direction.SHORT,
            round(latest.signal_strength, 3),
            round(latest.signal_strength, 3),
            latest.description or "MACD减仓信号",
        )
    else:
        # neutral: 看最新MACD柱方向做轻量判断
        from signals.macd_area import calc_macd
        close = daily_df["close"].astype(float).values
        _, _, macd_bar = calc_macd(close)
        last_bar = macd_bar[-1]
        if last_bar > 0:
            return FactorScore("技术", Direction.NEUTRAL, 0.3, 0.3,
                               f"MACD中性，红柱运行 bar={last_bar:.3f}")
        else:
            return FactorScore("技术", Direction.NEUTRAL, 0.2, 0.2,
                               f"MACD中性，绿柱运行 bar={last_bar:.3f}")


# ===========================================================================
# 因子4: 情绪因子 — 连板晋级率
# ===========================================================================
def score_sentiment(limit_df: pd.DataFrame) -> FactorScore:
    """连板晋级率 = 近5日中，次日仍涨停的股票数 / 前日涨停股数，取平均。

    晋级率高 → 市场情绪好（追板意愿强）→ 多
    晋级率低 → 市场情绪差（打板容易被砸）→ 空

    如果拿不到涨停数据，回退为中性。
    """
    if limit_df is None or len(limit_df) == 0 or "trade_date" not in limit_df.columns:
        return FactorScore("情绪", Direction.NEUTRAL, 0.2, 0.0, "无涨停数据")

    df = limit_df.copy()
    df["trade_date"] = df["trade_date"].astype(str)
    dates = sorted(df["trade_date"].unique())
    if len(dates) < 2:
        return FactorScore("情绪", Direction.NEUTRAL, 0.2, 0.0, "涨停数据天数不足")

    # 统计每个日期的涨停股数
    daily_counts = df.groupby("trade_date")["ts_code"].nunique().to_dict()

    # 连板晋级率：如果有 limit_times 列，计算 limit_times>=2 的占比
    # 否则用相邻日涨停重叠率近似
    promote_rates = []
    if "limit_times" in df.columns:
        # 近5个交易日，当日涨停中 limit_times>=2 的占比 = 晋级率
        recent_dates = dates[-5:]
        for d in recent_dates:
            day_stocks = df[df["trade_date"] == d]
            total = len(day_stocks)
            promoted = (day_stocks["limit_times"].fillna(1).astype(float) >= 2).sum()
            if total > 0:
                promote_rates.append(promoted / total)
    else:
        # 相邻日涨停重叠近似
        recent_dates = dates[-6:]
        for i in range(1, len(recent_dates)):
            prev_stocks = set(df[df["trade_date"] == recent_dates[i - 1]]["ts_code"])
            curr_stocks = set(df[df["trade_date"] == recent_dates[i]]["ts_code"])
            if prev_stocks:
                overlap = len(prev_stocks & curr_stocks)
                promote_rates.append(overlap / len(prev_stocks))

    if not promote_rates:
        return FactorScore("情绪", Direction.NEUTRAL, 0.2, 0.0, "无法计算晋级率")

    avg_rate = float(np.mean(promote_rates))

    if avg_rate > 0.30:
        strength = min(1.0, 0.4 + (avg_rate - 0.30) * 2)
        direction = Direction.LONG
        detail = "市场情绪亢奋"
    elif avg_rate > 0.18:
        strength = 0.3 + (avg_rate - 0.18) * 1.5
        direction = Direction.LONG
        detail = "情绪偏暖"
    elif avg_rate > 0.10:
        strength = 0.2
        direction = Direction.NEUTRAL
        detail = "情绪一般"
    elif avg_rate > 0.05:
        strength = 0.3 + (0.10 - avg_rate) * 2
        direction = Direction.SHORT
        detail = "情绪偏冷"
    else:
        strength = min(1.0, 0.5 + (0.05 - avg_rate) * 5)
        direction = Direction.SHORT
        detail = "情绪冰点"

    return FactorScore("情绪", direction, round(strength, 3), round(avg_rate, 4), detail)


# ===========================================================================
# 因子5: 基本面因子 — 板块强度（个股 vs 大盘）
# ===========================================================================
def score_fundamental(stock_pct: float, index_pct: float) -> FactorScore:
    """超额收益 = 个股涨跌幅 - 大盘涨跌幅

    超额 > +3%  → 强势 → 多
    超额 +1~+3% → 偏多
    超额 -1~+1% → 中性
    超额 -3~-1% → 偏空
    超额 < -3%  → 弱势 → 空
    """
    excess = stock_pct - index_pct

    if excess > 5:
        strength = min(1.0, 0.5 + (excess - 5) * 0.1)
        direction = Direction.LONG
        detail = "远超大盘"
    elif excess > 2:
        strength = 0.3 + (excess - 2) * 0.1  # 0.3-0.6
        direction = Direction.LONG
        detail = "跑赢大盘"
    elif excess > -2:
        strength = 0.2
        direction = Direction.NEUTRAL
        detail = "同步大盘"
    elif excess > -5:
        strength = 0.3 + (-2 - excess) * 0.1  # 0.3-0.6
        direction = Direction.SHORT
        detail = "跑输大盘"
    else:
        strength = min(1.0, 0.5 + (-5 - excess) * 0.1)
        direction = Direction.SHORT
        detail = "远弱于大盘"

    return FactorScore("基本面", direction, round(strength, 3), round(excess, 4), detail)


# ===========================================================================
# 主函数：五因子评分
# ===========================================================================
def evaluate(
    code: str,
    index_code: str = "000300",
    use_cache: bool = True,
) -> FiveFactorResult:
    """对个股进行五因子评分，返回 FiveFactorResult。"""
    # --- 拉数据 ---
    daily_df = load_daily(code, use_cache=use_cache)
    index_df = load_index(index_code, days=400)

    # 最新交易日
    trade_date = str(daily_df["trade_date"].iloc[-1])

    # 因子1: 估值（PE分位）
    try:
        pe_df = _fetch_daily_basic_pe(code)
        f_valuation = score_valuation(pe_df)
    except Exception as e:
        f_valuation = FactorScore("估值", Direction.NEUTRAL, 0.0, 0.0, f"PE拉取失败: {e}")

    # 因子2: 资金（换手率/成交量趋势）
    f_capital = score_capital_flow(daily_df)

    # 因子3: 技术（MACD面积）
    f_technical = score_technical(daily_df)

    # 因子4: 情绪（连板晋级率）
    try:
        limit_df = _fetch_limit_list_recent(days=7)
        f_sentiment = score_sentiment(limit_df)
    except Exception as e:
        f_sentiment = FactorScore("情绪", Direction.NEUTRAL, 0.2, 0.0, f"涨停拉取失败: {e}")

    # 因子5: 基本面（板块强度）
    stock_pct = float(daily_df["pct_chg"].iloc[-1])
    if len(index_df) > 0:
        # 匹配最新交易日，找不到就用最后一条
        idx_row = index_df[index_df["trade_date"] == trade_date]
        if len(idx_row) == 0:
            idx_row = index_df.iloc[-1:]
        index_pct = float(idx_row["pct_chg"].iloc[0])
    else:
        index_pct = 0.0
    f_fundamental = score_fundamental(stock_pct, index_pct)

    # --- 投票 ---
    factors = [f_valuation, f_capital, f_technical, f_sentiment, f_fundamental]
    long_votes = sum(1 for f in factors if f.direction == Direction.LONG)
    short_votes = sum(1 for f in factors if f.direction == Direction.SHORT)
    neutral_votes = sum(1 for f in factors if f.direction == Direction.NEUTRAL)

    if long_votes >= 3:
        position_advice = "满仓"
        position_ratio = 0.8 + min(0.2, (long_votes - 3) * 0.1)  # 3票=0.8, 4票=0.9, 5票=1.0
    elif long_votes == 2:
        position_advice = "半仓"
        position_ratio = 0.5
    else:
        position_advice = "空仓/轻仓"
        # 如果空头票多，仓位更低
        position_ratio = max(0.0, 0.2 - short_votes * 0.05)

    result = FiveFactorResult(
        code=code,
        trade_date=trade_date,
        valuation=f_valuation,
        capital_flow=f_capital,
        technical=f_technical,
        sentiment=f_sentiment,
        fundamental=f_fundamental,
        long_votes=long_votes,
        short_votes=short_votes,
        neutral_votes=neutral_votes,
        position_advice=position_advice,
        position_ratio=round(position_ratio, 2),
    )
    return result


# ===========================================================================
# CLI 入口
# ===========================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="五因子仓位评分器")
    parser.add_argument("code", help="股票代码，如 002580")
    parser.add_argument("--index", default="000300", help="基准指数代码（默认沪深300）")
    parser.add_argument("--no-cache", action="store_true", help="不使用缓存")
    args = parser.parse_args()

    res = evaluate(args.code, index_code=args.index, use_cache=not args.no_cache)
    print(res.summary())
