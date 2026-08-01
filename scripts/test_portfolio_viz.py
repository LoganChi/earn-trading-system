#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试 portfolio.py + visualizer.py — 用601727(上海电气)"""
import sys
sys.path.insert(0, ".")

from src.data.loader import load_daily, load_index
from src.backtest.simulator import run_backtest, print_result
from src.backtest.portfolio import run_portfolio_backtest, print_portfolio_result
from src.backtest.visualizer import plot_backtest

# === 加载601727数据 ===
print("=== 加载601727(上海电气)日K ===")
df_727 = load_daily('601727')
print(f"  数据范围: {df_727['trade_date'].iloc[0]} ~ {df_727['trade_date'].iloc[-1]}, {len(df_727)}天")

# === 单股回测 ===
print("\n=== 单股回测 ===")
result = run_backtest(df=df_727, code='601727', name='上海电气', verbose=True)
print_result(result, '601727')

# === 组合回测（用601727模拟多股票场景：截取不同时段作为"多只票"） ===
print("\n=== 组合回测 ===")
# 分两段作为两个"股票"（实际场景会传不同code）
df_part1 = df_727.iloc[:240].copy()
df_part2 = df_727.iloc[200:].copy()

stocks = [
    {"code": "601727", "name": "上海电气(全段)", "df": df_727},
]

pr = run_portfolio_backtest(
    stocks=stocks,
    capital=1_000_000,
    max_total_position_pct=0.8,
    max_single_position_pct=0.3,
    verbose=False,
)
print_portfolio_result(pr)

# === 可视化 ===
print("\n=== 生成可视化图表 ===")
output = plot_backtest(
    df=df_727,
    trades=result.trades,
    code='601727',
    output_path='reports/backtest_601727.png',
    title='601727 上海电气 - 动态持仓回测',
)
print(f"  图表已保存: {output}")

# === 组合多股票测试（加载更多股票如果有缓存）===
print("\n=== 多股票组合测试 ===")
try:
    df_2580 = load_daily('002580')
    stocks_multi = [
        {"code": "601727", "name": "上海电气", "df": df_727},
        {"code": "002580", "name": "圣阳股份", "df": df_2580},
    ]
    pr_multi = run_portfolio_backtest(
        stocks=stocks_multi,
        capital=1_000_000,
        max_total_position_pct=0.8,
        max_single_position_pct=0.3,
        verbose=False,
    )
    print_portfolio_result(pr_multi)
except Exception as e:
    print(f"  多股票测试跳过: {e}")

print("\n✅ 测试完成")
