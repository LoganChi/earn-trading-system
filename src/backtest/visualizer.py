#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""回测可视化模块

在K线图上标注：
- 入场点（绿色向上三角 ▲）
- 出场点（红色圆点 ●）
- 分批减仓点（橙色菱形 ◆）
- MACD面积信号（柱状图 + 绿峰/红峰高亮）
- 持仓区间阴影

用 matplotlib 画图，输出 PNG。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 无头模式
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from typing import List, Optional
from pathlib import Path

# 中文字体（matplotlib 默认不支持中文）
import matplotlib.font_manager as fm

# 尝试加载中文字体
_CN_FONT = None
for font_name in ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC',
                   'SimHei', 'Microsoft YaHei', 'PingFang SC', 'Heiti SC',
                   'Source Han Sans CN', 'AR PL UMing CN']:
    try:
        fp = fm.findfont(fm.FontProperties(family=font_name))
        if fp and 'LastResort' not in fp:
            _CN_FONT = font_name
            break
    except Exception:
        continue

if _CN_FONT:
    plt.rcParams['font.sans-serif'] = [_CN_FONT, 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
else:
    # 没有中文字体就用英文标注
    plt.rcParams['axes.unicode_minus'] = False

HAS_CN_FONT = _CN_FONT is not None


import sys
sys.path.insert(0, ".")
from src.signals.macd_area import calc_macd, generate_signals
from src.backtest.simulator import Trade


def _fmt_date(date_str: str) -> str:
    """20250101 → 2025-01-01"""
    s = str(date_str)
    if len(s) == 8:
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    return s


def plot_backtest(
    df: pd.DataFrame,
    trades: List[Trade],
    code: str,
    output_path: str,
    title: str = "",
    figsize: tuple = (16, 10),
    show_macd: bool = True,
    show_volume: bool = True,
    max_candles: int = 250,
) -> str:
    """
    在K线图上标注入场/出场/减仓点 + MACD面积信号

    参数：
    - df: 日K数据 (trade_date, open, high, low, close, vol)
    - trades: Trade 列表
    - code: 股票代码
    - output_path: PNG输出路径
    - title: 图表标题
    - figsize: 图表大小
    - show_macd: 是否显示MACD子图
    - show_volume: 是否显示成交量子图
    - max_candles: 最大显示K线数（截取最近的）

    返回：output_path
    """
    if df is None or len(df) == 0:
        raise ValueError("df is empty")

    df = df.sort_values('trade_date').reset_index(drop=True)

    # 截取最近N根
    if len(df) > max_candles:
        df = df.iloc[-max_candles:].reset_index(drop=True)

    dates = df['trade_date'].values
    opens = df['open'].values if 'open' in df.columns else df['close'].values
    highs = df['high'].values if 'high' in df.columns else df['close'].values
    lows = df['low'].values if 'low' in df.columns else df['close'].values
    closes = df['close'].values
    vols = df['vol'].values if 'vol' in df.columns else np.zeros(len(df))

    close_arr = df['close'].values.astype(float)
    dif, dea, macd_bar = calc_macd(close_arr)

    # 构建日期轴
    x = np.arange(len(df))
    date_labels = [_fmt_date(d) for d in dates]

    # 过滤在本图日期范围内的交易
    min_date = str(dates[0])
    max_date = str(dates[-1])
    visible_trades = [
        t for t in trades
        if t.entry_date >= min_date and t.entry_date <= max_date
    ]

    # === 布局 ===
    n_rows = 1
    if show_macd:
        n_rows += 1
    if show_volume:
        n_rows += 1

    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=figsize,
        gridspec_kw={'height_ratios': [3] + [1] * (n_rows - 1)},
        sharex=True,
    )
    if n_rows == 1:
        axes = [axes]

    ax_price = axes[0]

    # === 1. K线图 ===
    up = closes >= opens
    down = closes < opens

    # 画K线
    for i in range(len(df)):
        color = '#ef4444' if up[i] else '#22c55e'  # A股：红涨绿跌
        # 影线
        ax_price.vlines(x[i], lows[i], highs[i], color=color, linewidth=0.8)
        # 实体
        body_low = min(opens[i], closes[i])
        body_height = abs(closes[i] - opens[i])
        if body_height < 0.001:
            body_height = closes[i] * 0.003  # 十字星最小高度
        rect = Rectangle(
            (x[i] - 0.3, body_low), 0.6, body_height,
            facecolor=color, edgecolor=color, linewidth=0.5,
        )
        ax_price.add_patch(rect)

    # === 标注入场/出场点 ===
    for trade in visible_trades:
        entry_idx_arr = np.where(dates.astype(str) == trade.entry_date)[0]
        if len(entry_idx_arr) == 0:
            # 尝试近似匹配
            entry_idx_arr = np.where([str(d) >= trade.entry_date for d in dates])[0]
            if len(entry_idx_arr) == 0:
                continue
        entry_idx = entry_idx_arr[0]

        # 入场点：绿色三角
        ax_price.scatter(
            entry_idx, trade.entry_price * 0.985,
            marker='^', s=150, c='#22c55e', edgecolors='white',
            linewidths=1, zorder=10, label='Entry' if entry_idx == visible_trades[0].entry_date else '',
        )

        # 信号强度标注
        if trade.signal_strength > 0:
            label_text = f"IN {trade.signal_strength:.0%}"
            ax_price.annotate(
                label_text,
                (entry_idx, trade.entry_price * 0.975),
                fontsize=7, color='#22c55e', ha='center',
            )

        # 持仓区间阴影
        if trade.exit_date:
            exit_idx_arr = np.where(dates.astype(str) == trade.exit_date)[0]
            if len(exit_idx_arr) > 0:
                exit_idx = exit_idx_arr[0]
            else:
                exit_idx = len(df) - 1
        else:
            exit_idx = len(df) - 1

        ax_price.axvspan(
            entry_idx, exit_idx,
            alpha=0.08, color='#3b82f6',
        )

        # 出场点：红色圆点
        if trade.exit_date and trade.exit_price > 0:
            ax_price.scatter(
                exit_idx, trade.exit_price * 1.015,
                marker='o', s=120, c='#ef4444', edgecolors='white',
                linewidths=1, zorder=10,
            )
            exit_label = f"OUT {trade.pnl_pct:+.1f}%"
            ax_price.annotate(
                exit_label,
                (exit_idx, trade.exit_price * 1.025),
                fontsize=7, color='#ef444e', ha='center',
            )

        # 分批减仓点：橙色菱形
        for pe in trade.partial_exits:
            pe_date = pe.get('date', '')
            pe_idx_arr = np.where(dates.astype(str) == pe_date)[0]
            if len(pe_idx_arr) == 0:
                continue
            pe_idx = pe_idx_arr[0]
            ax_price.scatter(
                pe_idx, pe.get('price', closes[pe_idx]) * 1.01,
                marker='D', s=80, c='#f59e0b', edgecolors='white',
                linewidths=1, zorder=10,
            )
            pe_label = f"-{pe.get('shares', 0)}sh"
            ax_price.annotate(
                pe_label,
                (pe_idx, pe.get('price', closes[pe_idx]) * 1.02),
                fontsize=6, color='#f59e0b', ha='center',
            )

    ax_price.set_ylabel('Price')
    if not title:
        title = f"{code} Backtest"
    ax_price.set_title(title, fontsize=14, fontweight='bold')

    # 自定义图例
    legend_elements = [
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#22c55e',
                   markersize=10, label='Entry'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ef4444',
                   markersize=10, label='Exit'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#f59e0b',
                   markersize=9, label='Partial Exit'),
    ]
    ax_price.legend(handles=legend_elements, loc='upper right', fontsize=8)

    row = 1

    # === 2. MACD 子图 ===
    if show_macd:
        ax_macd = axes[row]
        row += 1

        # MACD柱（红绿）
        macd_colors = ['#ef4444' if b >= 0 else '#22c55e' for b in macd_bar]
        ax_macd.bar(x, macd_bar, color=macd_colors, width=0.6, alpha=0.7)

        # DIF / DEA 线
        ax_macd.plot(x, dif, color='#3b82f6', linewidth=1, label='DIF')
        ax_macd.plot(x, dea, color='#f59e0b', linewidth=1, label='DEA')
        ax_macd.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax_macd.set_ylabel('MACD')
        ax_macd.legend(loc='upper right', fontsize=7)

        # 标注绿峰面积区域
        in_green = False
        green_start = 0
        for i in range(len(macd_bar)):
            if macd_bar[i] < 0 and not in_green:
                in_green = True
                green_start = i
            elif macd_bar[i] >= 0 and in_green:
                in_green = False
                area = sum(abs(macd_bar[j]) for j in range(green_start, i))
                if area > 1.0:
                    ax_macd.axvspan(
                        green_start, i,
                        alpha=0.15, color='#22c55e',
                    )
                    mid = (green_start + i) // 2
                    ax_macd.text(
                        mid, min(macd_bar[green_start:i]) * 0.8,
                        f'A={area:.1f}', fontsize=6, ha='center', color='#16a34a',
                    )

    # === 3. 成交量子图 ===
    if show_volume:
        ax_vol = axes[row]
        vol_colors = ['#ef4444' if closes[i] >= opens[i] else '#22c55e' for i in range(len(df))]
        ax_vol.bar(x, vols, color=vol_colors, width=0.6, alpha=0.5)
        ax_vol.set_ylabel('Volume')

    # X轴格式
    ax_last = axes[-1]
    n = len(dates)
    step = max(1, n // 15)
    tick_positions = list(range(0, n, step))
    tick_labels = [date_labels[i] if i < n else '' for i in tick_positions]
    ax_last.set_xticks(tick_positions)
    ax_last.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)

    plt.tight_layout()

    # 输出
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return output_path
