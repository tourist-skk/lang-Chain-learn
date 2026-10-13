"""第三、四节的计算 MCP 服务：支持 stdio 与 Streamable HTTP。"""

import argparse
import logging
import os
import sys
from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field


mcp = FastMCP(name="Calculator")
logger = logging.getLogger("calculator")


@mcp.tool
def add(a: int, b: int) -> int:
    """计算两个整数之和。"""
    logger.info("add: a=%s b=%s", a, b)
    return a + b


@mcp.tool
def divide(
    numerator: float,
    denominator: float,
    precision: Annotated[
        int, Field(ge=0, le=6, description="保留的小数位数，范围为 0 到 6。")
    ] = 2,
) -> float:
    """计算分子除以分母，并按指定精度四舍五入。"""
    logger.info(
        "divide: numerator=%s denominator=%s precision=%s",
        numerator,
        denominator,
        precision,
    )
    if denominator == 0:
        raise ValueError("除数不能为 0")
    return round(numerator / denominator, precision)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
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
