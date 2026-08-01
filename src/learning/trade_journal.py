#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""交易日志记录器（Trade Journal）

灵感来源：
- 用户实战经验：交易日志是持续改进的基础——记录入场/出场理由，而非只记价格
- 桥水PAT：持续学习飞轮（记录 → 回顾 → benchmark → 改进 → 全员受益）
-LOOP文章：每笔交易需要有完整的决策上下文

记录字段：
  code, name, action(entry/exit/reduce), date, price, reason,
  market_env, macd_area_desc, vp_desc, confidence

持久化格式：JSONL（每行一个JSON对象），文件 data/trade_journal.jsonl

功能：
  1. record()        — 记录一笔交易操作
  2. query()         — 按股票/时间段/操作类型查询
  3. review()        — 周期性回顾：胜率、最大盈亏、最常亏钱场景
  4. match_entry_exit() — 将entry和exit配对，计算每笔完整交易的盈亏
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any

# 项目根目录
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_JOURNAL_PATH = _PROJECT_ROOT / "data" / "trade_journal.jsonl"


# ===========================================================================
# 数据结构
# ===========================================================================
@dataclass
class TradeRecord:
    """单笔交易操作日志"""
    code: str                         # 股票代码 "002580"
    name: str                         # 股票名称
    action: str                       # "entry" / "exit" / "reduce"
    date: str                         # 交易日期 "YYYYMMDD"
    price: float                      # 交易价格
    reason: str                       # 决策理由（自然语言）
    market_env: str = ""              # 市场环境描述（如"大盘连涨3日，板块走强"）
    macd_area_desc: str = ""          # MACD面积信号描述
    vp_desc: str = ""                 # 成交量分布信号描述
    confidence: float = 0.0           # 信心度 0-1
    shares: int = 0                   # 交易股数（可选）
    pnl_pct: Optional[float] = None   # 出场时盈亏百分比（exit/reduce时填）
    tags: List[str] = field(default_factory=list)  # 自定义标签
    timestamp: str = ""               # 记录写入时间戳

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TradeRecord":
        # 兼容缺失字段
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


@dataclass
class MatchedTrade:
    """一笔配对的完整交易（entry → exit/reduce）"""
    code: str
    name: str
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    pnl_pct: float
    holding_days: int
    entry_reason: str
    exit_reason: str
    market_env: str
    macd_area_desc: str
    confidence: float


# ===========================================================================
# 核心功能
# ===========================================================================
def record(
    code: str,
    name: str,
    action: str,
    date: str,
    price: float,
    reason: str,
    market_env: str = "",
    macd_area_desc: str = "",
    vp_desc: str = "",
    confidence: float = 0.0,
    shares: int = 0,
    pnl_pct: Optional[float] = None,
    tags: Optional[List[str]] = None,
    journal_path: Optional[str] = None,
) -> TradeRecord:
    """
    记录一笔交易操作到 JSONL 日志。

    参数：
      code         : 股票代码
      name         : 股票名称
      action       : "entry" / "exit" / "reduce"
      date         : 交易日期 "YYYYMMDD"
      price        : 交易价格
      reason       : 决策理由
      market_env   : 市场环境描述
      macd_area_desc: MACD面积信号描述
      vp_desc      : 成交量分布信号描述
      confidence   : 信心度 0-1
      shares       : 交易股数
      pnl_pct      : 出场盈亏%
      tags         : 自定义标签列表
      journal_path : 自定义日志路径（默认 data/trade_journal.jsonl）

    返回：TradeRecord 对象
    """
    assert action in ("entry", "exit", "reduce"), f"action 必须是 entry/exit/reduce，得到 {action}"

    rec = TradeRecord(
        code=code, name=name, action=action, date=date, price=price,
        reason=reason, market_env=market_env,
        macd_area_desc=macd_area_desc, vp_desc=vp_desc,
        confidence=round(confidence, 3),
        shares=shares, pnl_pct=pnl_pct,
        tags=tags or [],
    )

    path = Path(journal_path) if journal_path else _JOURNAL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")

    return rec


def load_all(journal_path: Optional[str] = None) -> List[TradeRecord]:
    """加载全部交易记录"""
    path = Path(journal_path) if journal_path else _JOURNAL_PATH
    if not path.exists():
        return []

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                records.append(TradeRecord.from_dict(d))
            except (json.JSONDecodeError, TypeError):
                continue
    return records


def query(
    code: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    action: Optional[str] = None,
    journal_path: Optional[str] = None,
) -> List[TradeRecord]:
    """
    查询交易记录。

    参数均可选，为 None 表示不过滤：
      code       : 按股票代码过滤
      start_date : 起始日期 "YYYYMMDD"（含）
      end_date   : 结束日期 "YYYYMMDD"（含）
      action     : 按操作类型过滤 "entry"/"exit"/"reduce"
    """
    records = load_all(journal_path)

    result = []
    for r in records:
        if code and r.code != code:
            continue
        if start_date and r.date < start_date:
            continue
        if end_date and r.date > end_date:
            continue
        if action and r.action != action:
            continue
        result.append(r)

    return result


def match_entry_exit(journal_path: Optional[str] = None) -> List[MatchedTrade]:
    """
    将 entry 和 exit/reduce 记录配对，计算每笔完整交易的盈亏。

    配对逻辑：按股票代码分组，每个 entry 匹配之后第一个 exit。
    reduce 视为部分出场，不影响 entry→exit 配对。

    返回：MatchedTrade 列表
    """
    records = load_all(journal_path)

    # 按股票分组
    by_code: Dict[str, List[TradeRecord]] = {}
    for r in records:
        by_code.setdefault(r.code, []).append(r)

    matched = []
    for code, recs in by_code.items():
        # 按日期排序
        recs.sort(key=lambda x: (x.date, x.timestamp))

        open_entry: Optional[TradeRecord] = None
        for r in recs:
            if r.action == "entry":
                if open_entry is not None:
                    # 前一个entry没有配对exit，跳过（或可视为未平仓）
                    pass
                open_entry = r
            elif r.action == "exit" and open_entry is not None:
                pnl = r.pnl_pct
                if pnl is None and open_entry.price > 0:
                    pnl = (r.price / open_entry.price - 1) * 100

                holding_days = _count_trading_days(open_entry.date, r.date)

                matched.append(MatchedTrade(
                    code=code,
                    name=open_entry.name,
                    entry_date=open_entry.date,
                    entry_price=open_entry.price,
                    exit_date=r.date,
                    exit_price=r.price,
                    pnl_pct=round(pnl, 2) if pnl is not None else 0.0,
                    holding_days=holding_days,
                    entry_reason=open_entry.reason,
                    exit_reason=r.reason,
                    market_env=open_entry.market_env,
                    macd_area_desc=open_entry.macd_area_desc,
                    confidence=open_entry.confidence,
                ))
                open_entry = None

    return matched


def review(
    code: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    journal_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    周期性回顾统计。

    返回 dict 包含：
      total_trades    : 完整交易数
      win_rate        : 胜率 %
      avg_pnl         : 平均盈亏 %
      max_profit      : 最大单笔盈利 %
      max_loss        : 最大单笔亏损 %
      total_pnl       : 累计盈亏 %
      avg_holding_days: 平均持有天数
      loss_scenarios  : 最常亏钱的场景列表（按 reason 分组）
      win_scenarios   : 最常赚钱的场景列表
    """
    matched = match_entry_exit(journal_path)

    # 过滤
    filtered = []
    for t in matched:
        if code and t.code != code:
            continue
        if start_date and t.entry_date < start_date:
            continue
        if end_date and t.entry_date > end_date:
            continue
        filtered.append(t)

    if not filtered:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_pnl": 0.0,
            "max_profit": 0.0,
            "max_loss": 0.0,
            "total_pnl": 0.0,
            "avg_holding_days": 0.0,
            "loss_scenarios": [],
            "win_scenarios": [],
            "message": "无完整交易记录",
        }

    wins = [t for t in filtered if t.pnl_pct > 0]
    losses = [t for t in filtered if t.pnl_pct <= 0]

    # 亏钱场景聚类：按 entry_reason 关键词分组
    loss_scenarios = _cluster_by_reason(losses)
    win_scenarios = _cluster_by_reason(wins)

    return {
        "total_trades": len(filtered),
        "win_trades": len(wins),
        "loss_trades": len(losses),
        "win_rate": round(len(wins) / len(filtered) * 100, 1),
        "avg_pnl": round(sum(t.pnl_pct for t in filtered) / len(filtered), 2),
        "max_profit": round(max(t.pnl_pct for t in filtered), 2),
        "max_loss": round(min(t.pnl_pct for t in filtered), 2),
        "total_pnl": round(sum(t.pnl_pct for t in filtered), 2),
        "avg_holding_days": round(sum(t.holding_days for t in filtered) / len(filtered), 1),
        "loss_scenarios": loss_scenarios[:5],
        "win_scenarios": win_scenarios[:5],
    }


def print_review(stats: Dict[str, Any]) -> None:
    """格式化打印回顾统计"""
    print(f"\n{'═' * 60}")
    print(f"  📊 交易日志回顾统计")
    print(f"{'═' * 60}")

    if stats["total_trades"] == 0:
        print(f"  {stats.get('message', '无数据')}")
        print(f"{'═' * 60}")
        return

    print(f"  完整交易数: {stats['total_trades']} ({stats.get('win_trades', 0)}胜 / {stats.get('loss_trades', 0)}负)")
    print(f"  胜率: {stats['win_rate']}%")
    print(f"  平均盈亏: {stats['avg_pnl']:+.2f}%")
    print(f"  最大单笔盈利: {stats['max_profit']:+.2f}%")
    print(f"  最大单笔亏损: {stats['max_loss']:+.2f}%")
    print(f"  累计盈亏: {stats['total_pnl']:+.2f}%")
    print(f"  平均持有天数: {stats['avg_holding_days']:.1f}天")

    if stats.get("loss_scenarios"):
        print(f"\n  🔴 最常亏钱的场景 Top{len(stats['loss_scenarios'])}:")
        for i, s in enumerate(stats["loss_scenarios"], 1):
            print(f"    {i}. {s['reason'][:50]}")
            print(f"       {s['count']}次，平均{s['avg_pnl']:+.2f}%，累计{s['total_pnl']:+.2f}%")

    if stats.get("win_scenarios"):
        print(f"\n  🟢 最常赚钱的场景 Top{len(stats['win_scenarios'])}:")
        for i, s in enumerate(stats["win_scenarios"], 1):
            print(f"    {i}. {s['reason'][:50]}")
            print(f"       {s['count']}次，平均{s['avg_pnl']:+.2f}%，累计{s['total_pnl']:+.2f}%")

    print(f"{'═' * 60}")


# ===========================================================================
# 辅助函数
# ===========================================================================
def _count_trading_days(start: str, end: str) -> int:
    """粗略估算交易日天数（日历天 × 5/7）"""
    try:
        d1 = datetime.strptime(start, "%Y%m%d")
        d2 = datetime.strptime(end, "%Y%m%d")
        calendar_days = (d2 - d1).days
        return max(0, int(calendar_days * 5 / 7))
    except (ValueError, TypeError):
        return 0


def _cluster_by_reason(trades: List[MatchedTrade]) -> List[Dict[str, Any]]:
    """按入场理由聚类，返回按亏损/盈利排序的场景列表"""
    clusters: Dict[str, List[MatchedTrade]] = {}
    for t in trades:
        # 用 entry_reason 的前20个字符作为聚类key（简化）
        key = t.entry_reason[:20] if t.entry_reason else "(无理由)"
        clusters.setdefault(key, []).append(t)

    result = []
    for reason, group in clusters.items():
        result.append({
            "reason": reason,
            "count": len(group),
            "avg_pnl": round(sum(t.pnl_pct for t in group) / len(group), 2),
            "total_pnl": round(sum(t.pnl_pct for t in group), 2),
            "codes": list(set(t.code for t in group)),
        })

    # 按平均亏损从小到大排序（亏损场景：最负的排前面）
    result.sort(key=lambda x: x["avg_pnl"])
    return result


# ===========================================================================
# CLI 入口
# ===========================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="交易日志记录器")
    sub = parser.add_subparsers(dest="cmd")

    # record 子命令
    p_rec = sub.add_parser("record", help="记录一笔交易")
    p_rec.add_argument("--code", required=True)
    p_rec.add_argument("--name", default="")
    p_rec.add_argument("--action", required=True, choices=["entry", "exit", "reduce"])
    p_rec.add_argument("--date", required=True)
    p_rec.add_argument("--price", type=float, required=True)
    p_rec.add_argument("--reason", required=True)
    p_rec.add_argument("--market-env", default="")
    p_rec.add_argument("--macd-desc", default="")
    p_rec.add_argument("--vp-desc", default="")
    p_rec.add_argument("--confidence", type=float, default=0.0)

    # query 子命令
    p_qry = sub.add_parser("query", help="查询交易记录")
    p_qry.add_argument("--code", default=None)
    p_qry.add_argument("--start", default=None)
    p_qry.add_argument("--end", default=None)
    p_qry.add_argument("--action", default=None, choices=["entry", "exit", "reduce"])

    # review 子命令
    p_rev = sub.add_parser("review", help="回顾统计")
    p_rev.add_argument("--code", default=None)
    p_rev.add_argument("--start", default=None)
    p_rev.add_argument("--end", default=None)

    args = parser.parse_args()

    if args.cmd == "record":
        rec = record(
            code=args.code, name=args.name, action=args.action,
            date=args.date, price=args.price, reason=args.reason,
            market_env=args.market_env, macd_area_desc=args.macd_desc,
            vp_desc=args.vp_desc, confidence=args.confidence,
        )
        print(f"✅ 已记录: {rec.action} {rec.code} @{rec.price} on {rec.date}")

    elif args.cmd == "query":
        records = query(code=args.code, start_date=args.start,
                        end_date=args.end, action=args.action)
        print(f"找到 {len(records)} 条记录:")
        for r in records:
            pnl_str = f" 盈亏{r.pnl_pct:+.2f}%" if r.pnl_pct is not None else ""
            print(f"  {r.date} {r.action:6s} {r.code} @{r.price:.2f}{pnl_str}")
            print(f"    理由: {r.reason}")

    elif args.cmd == "review":
        stats = review(code=args.code, start_date=args.start, end_date=args.end)
        print_review(stats)

    else:
        parser.print_help()
