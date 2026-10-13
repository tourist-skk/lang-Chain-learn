"""第五、六节的套餐 MCP 服务：同时提供 Tool、Resource 和 Prompt。

套餐数据摘自第十章 knowledge.txt 的“二、套餐说明”（2026 Q1 版），仅用于教学。
"""

import argparse
import logging
import os
import sys
from typing import Literal

from fastmcp import FastMCP


mcp = FastMCP(name="PlanService")
logger = logging.getLogger("plan_service")

PLANS = {
    "试用版": {
        "价格": "0 元，注册后 14 天",
        "成员人数上限": "1 人",
        "月度AI问答额度": "200 次",
        "API调用额度": "不支持",
    },
    "基础版": {
        "价格": "99 元 / 用户 / 月",
        "成员人数上限": "10 人",
        "月度AI问答额度": "5000 次",
        "API调用额度": "每月 10000 次",
    },
    "专业版": {
        "价格": "199 元 / 用户 / 月",
        "成员人数上限": "50 人",
        "月度AI问答额度": "30000 次",
        "API调用额度": "每月 80000 次",
    },
    "企业版": {
        "价格": "按年签约，不公开标价",
        "成员人数上限": "按合同约定",
        "月度AI问答额度": "按合同约定",
        "API调用额度": "按合同约定",
    },
}


@mcp.tool
def get_plan(plan: Literal["试用版", "基础版", "专业版", "企业版"]) -> dict:
    """查询一个套餐的价格、成员人数上限、月度 AI 问答额度和 API 调用额度。"""
    logger.info("get_plan: plan=%s", plan)
    return {"套餐": plan, **PLANS[plan]}


@mcp.resource("plans://catalog", mime_type="text/markdown")
def plan_catalog() -> str:
    """四档标准订阅套餐的对比表。"""
    logger.info("read_resource: plans://catalog")
    fields = list(PLANS["试用版"])
    lines = [
        "| 套餐 | " + " | ".join(fields) + " |",
        "|---" * (len(fields) + 1) + "|",
    ]
    for name, plan in PLANS.items():
        lines.append(f"| {name} | " + " | ".join(plan[field] for field in fields) + " |")
    return "\n".join(lines)


@mcp.prompt
def plan_answer(question: str, style: str = "简洁") -> str:
    """按客服口径回答套餐问题的提示词模板。"""
    logger.info("get_prompt: style=%s", style)
    return (
        "你是 atguigu助手的客服。请只依据套餐资料回答用户的问题；"
        "资料没有写明的内容，说明需要人工确认。\n"
        f"回答风格：{style}。\n"
        f"用户问题：{question}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    # 与计算服务的 8000 错开，便于两个 HTTP 服务同时运行。
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--path", default="/mcp")
    options = parser.parse_args()

    # stdout 留给 stdio 协议；服务端的运行信息统一写到 stderr。
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(name)s: %(message)s",
    )
    logger.info("server_pid=%s transport=%s", os.getpid(), options.transport)

    if options.transport == "stdio":
        mcp.run(transport="stdio", show_banner=False)
    else:
        mcp.run(
            transport="http",
            host=options.host,
            port=options.port,
            path=options.path,
            show_banner=False,
        )


if __name__ == "__main__":
    main()
