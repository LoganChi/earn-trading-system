#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试 learning 模块：交易日志 + 对抗验证器，用 002580(圣阳股份) 测试"""
import sys
sys.path.insert(0, ".")

from src.data.loader import load_daily, load_index
from src.learning.trade_journal import record, query, review, print_review, match_entry_exit
from src.learning.adversarial import run as run_adversarial


CODE = "002580"
NAME = "圣阳股份"

# =========================================================
# 模块1：交易日志记录器测试
# =========================================================
print("=" * 60)
print("  模块1：交易日志记录器 测试")
print("=" * 60)

# 加载数据
df = load_daily(CODE)
print(f"\n加载数据: {CODE} {NAME}")
print(f"  数据范围: {df['trade_date'].iloc[0]} ~ {df['trade_date'].iloc[-1]}, {len(df)}天")

# --- 记录几笔模拟交易 ---
print("\n--- 记录模拟交易 ---")

# 交易1：盈利
record(
    code=CODE, name=NAME, action="entry",
    date="20250107", price=9.65,
    reason="MACD绿峰面积15.2(deep)后翻红，DIF拐头向上，价格底部位置",
    market_env="大盘震荡走平，储能板块回暖",
    macd_area_desc="绿峰面积15.2 deep 红柱放大 DIF拐头 价格位置15%",
    vp_desc="VP上沿放量突破，POC=8.8",
    confidence=0.75, shares=2000,
)
record(
    code=CODE, name=NAME, action="exit",
    date="20250210", price=11.20,
    reason="红柱连续缩短，MACD动能衰竭，VP触及上沿无量",
    market_env="大盘开始回调",
    macd_area_desc="红柱连续3天缩短",
    vp_desc="VP上沿无量测试",
    confidence=0.6, shares=2000,
    pnl_pct=16.06,
)

# 交易2：亏损
record(
    code=CODE, name=NAME, action="entry",
    date="20250315", price=10.50,
    reason="MACD绿峰面积8.5(moderate)后翻红，信号偏弱",
    market_env="大盘连续下跌3天，市场情绪较差",
    macd_area_desc="绿峰面积8.5 moderate 信号强度35%",
    vp_desc="VP中性区，无明确方向",
    confidence=0.35, shares=1000,
)
record(
    code=CODE, name=NAME, action="exit",
    date="20250325", price=9.80,
    reason="跌破止损位-5%，动能持续走弱",
    market_env="大盘加速下跌",
    macd_area_desc="MACD再次翻绿",
    vp_desc="VP下沿测试",
    confidence=0.3, shares=1000,
    pnl_pct=-6.67,
)

# 交易3：减仓+清仓
record(
    code=CODE, name=NAME, action="entry",
    date="20250520", price=12.30,
    reason="MACD绿峰面积22.0(deep)后翻红，底部反转信号",
    market_env="大盘企稳反弹",
    macd_area_desc="绿峰面积22.0 deep DIF拐头 价格位置20%",
    vp_desc="POC附近良好入场位",
    confidence=0.8, shares=3000,
)
record(
    code=CODE, name=NAME, action="reduce",
    date="20250605", price=14.50,
    reason="涨到VP上沿，减仓锁利50%",
    market_env="大盘走强",
    macd_area_desc="红柱开始缩短",
    vp_desc="VP上沿触及",
    confidence=0.5, shares=1500,
    pnl_pct=17.89,
)
record(
    code=CODE, name=NAME, action="exit",
    date="20250620", price=13.80,
    reason="红柱连续缩短，止盈剩余仓位",
    market_env="大盘震荡",
    macd_area_desc="红柱动能衰竭",
    vp_desc="VP回归中性",
    confidence=0.5, shares=1500,
    pnl_pct=12.20,
)

print("  ✅ 已记录5笔交易操作（2个完整交易 + 1个减仓）")

# --- 查询测试 ---
print("\n--- 查询测试 ---")

# 按股票查询
records = query(code=CODE)
print(f"  查询 {CODE}: 找到 {len(records)} 条记录")

# 按操作类型查询
entries = query(action="entry")
print(f"  查询 entry: 找到 {len(entries)} 条")

exits = query(action="exit")
print(f"  查询 exit: 找到 {len(exits)} 条")

# 按时间段查询
range_records = query(start_date="20250301", end_date="20250331")
print(f"  查询 202503: 找到 {len(range_records)} 条")

# --- 配对交易 ---
print("\n--- 配对交易（entry→exit） ---")
matched = match_entry_exit()
for t in matched:
    print(f"  {t.code} {t.entry_date}→{t.exit_date} "
          f"@{t.entry_price:.2f}→@{t.exit_price:.2f} "
          f"持有{t.holding_days}天 "
          f"收益{t.pnl_pct:+.2f}%")
    print(f"    入场理由: {t.entry_reason[:60]}")

# --- 回顾统计 ---
print("\n--- 回顾统计 ---")
stats = review(code=CODE)
print_review(stats)


# =========================================================
# 模块2：对抗验证器测试
# =========================================================
print("\n\n" + "=" * 60)
print("  模块2：对抗验证器 测试")
print("=" * 60)

report = run_adversarial(code=CODE, eval_period=10, top_n=5)
print(report.summary())

# 打印所有信号详情
print("\n--- 所有entry_candidate信号详情 ---")
for o in report.outcomes:
    ret_10 = o.forward_returns.get(10, None)
    ret_20 = o.forward_returns.get(20, None)
    pe_str = f"PE分位{o.pe_percentile:.0%}" if o.pe_percentile else "PE无"
    print(f"  {o.date} @{o.price:.2f} | "
          f"强度{o.signal_strength:.0%} | "
          f"位置{o.price_position:.0%} | "
          f"绿峰{o.green_peak_area:.1f}({o.green_peak_severity}) | "
          f"大盘连跌{o.market_consecutive_down}天 | "
          f"换手率比{o.turnover_ratio:.2f} | "
          f"{pe_str} | "
          f"10天收益{ret_10:+.2f}%" if ret_10 is not None else f"  {o.date} @{o.price:.2f} | 数据不足")
    if ret_20 is not None:
        print(f"    └─ 20天收益{ret_20:+.2f}% {'🔴亏损' if ret_10 is not None and ret_10 <= 0 else '🟢盈利'}")

print("\n✅ 测试完成")
