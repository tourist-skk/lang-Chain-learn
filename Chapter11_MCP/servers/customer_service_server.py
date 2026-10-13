"""第十二章综合实战的客服 MCP 服务：知识检索、模拟订单与模拟工单。

检索沿用第十章的知识库与嵌入模型，切分采用第十章补充中的“按章节切分”；
订单与工单是教学用的模拟数据，工单只保存在服务进程的内存中。
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

from dotenv import load_dotenv
from fastmcp import FastMCP
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_FILE = PROJECT_ROOT / "Chapter10_RAG" / "knowledge.txt"
ORDERS_FILE = PROJECT_ROOT / "Chapter12_skill" / "data" / "mock-orders.json"
EMBED_MODEL_NAME = "bge-large-zh-v1.5"

mcp = FastMCP(name="CustomerService")
logger = logging.getLogger("customer_service")
knowledge_store: InMemoryVectorStore | None = None
tickets: list[dict] = []


def split_by_section(text: str) -> list[Document]:
    """先按章节切开，章节名加在每块开头，章节内部再递归切分。"""
    # 把这 30 个等号中间夹着的这一行文字单独提取出来
    # 1. \n: 换行符
    # 2. ={30}: 正好 30 个 =（量词 {30} 表示前面的字符重复恰好 30 次）
    # 3. \n: 换行符
    # 4. (.+?): (.+?) 里的 +? 是"非贪婪"匹配：+ 表示"一个或多个"，? 让它"尽量少地匹配"。因为一行标题后面还跟着 \n====...，非贪婪才能在第一个换行处就停住，只捕获 一、套餐说明 这一行，而不是贪婪地一直吃到后面很远。
    # . 默认不匹配换行符，所以 .+? 只会捕获"同一行内的文字"，天然就是标题那一行。
    # 分割后: parts = [前言, 标题1, 正文1, 标题2, 正文2, 标题3, 正文3, ...]
    #                  ↑ 第一个"====="出现之前的全部文字
    parts = re.split(r"\n={30}\n(.+?)\n={30}\n", text)
    # parts[1::2]：从下标 1 开始、步长 2 —— 取出所有标题
    # parts[2::2]：从下标 2 开始、步长 2 —— 取出所有正文。
    # zip(标题们, 正文们) 两两配对成 [(标题1, 正文1), (标题2, 正文2), ...]
    # 最前面再加一项 ("文档说明", parts[0])：第一个 ===== 之前的内容（文档开头说明）也当成一个章节，标题命名为"文档说明"。
    sections = [("文档说明", parts[0])] + list(zip(parts[1::2], parts[2::2]))
    # 创建 LangChain 的递归字符切分器，用来做"章节内部再切分"
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=50, separators=["\n\n", "\n", "。", ""]
    )
    chunks = []
    for title, body in sections:
        # 把这一章正文按上面的规则切成多个 piece（小段文本），内层逐段处理。
        for piece in splitter.split_text(body.strip()):
            # 每个小段包装成一个 Document：
            chunks.append(Document(
                page_content=f"【{title.strip()}】\n{piece}",
                metadata={"section": title.strip(), "chunk_id": len(chunks)},
            ))
    return chunks


def build_knowledge_store() -> InMemoryVectorStore:
    load_dotenv(PROJECT_ROOT / "conf" / ".env")
    # 向量模型
    embeddings = OpenAIEmbeddings(
        model=EMBED_MODEL_NAME,
        api_key=os.environ["embedding_model_key"],
        base_url=os.environ["embedding_model_base"],
        check_embedding_ctx_length=False,
    )
    # Document数组
    chunks = split_by_section(KNOWLEDGE_FILE.read_text(encoding="utf-8"))
    store = InMemoryVectorStore(embedding=embeddings)
    # add_documents(chunks)：对每个块调用嵌入模型算出向量，存进内存索引（这一步较慢，就是 start_service 注释里"建索引要调用嵌入接口"的原因）。
    store.add_documents(chunks)
    logger.info("knowledge index: %d chunks", len(chunks))
    return store

# @mcp.tool 装饰器：把下面这个普通函数注册成 MCP 工具。FastMCP 会读取函数签名和类型注解，自动生成这个工具的 JSON Schema，供 MCP 客户端/模型调用。
@mcp.tool
def search_knowledge(
    query: Annotated[str, Field(description="要检索的问题，例如“专业版的退款规则”。")],
    k: Annotated[int, Field(ge=1, le=5, description="返回的片段数量。")] = 4,
) -> str:
    """在 atguigu助手客服知识库中检索套餐、额度、退款、发票等规则，返回带片段编号和章节名的原文。"""
    if knowledge_store is None:
        raise RuntimeError("知识库索引尚未建立")
    hits = knowledge_store.similarity_search_with_score(query, k=k)
    logger.info("search_knowledge: query=%s chunks=%s", query,
                [doc.metadata["chunk_id"] for doc, _ in hits])
    return "\n\n".join(
        f"[片段{doc.metadata['chunk_id']} | {doc.metadata['section']} | 相似度 {score:.3f}]\n"
        f"{doc.page_content}"
        for doc, score in hits
    )


@mcp.tool
def get_mock_order(order_id: str) -> dict:
    """查询模拟订单：套餐、购买人数、实付金额、付款日期、距付款天数、本月 AI 问答已用次数、发票与升级情况。"""
    # 读取数据文件
    data = json.loads(ORDERS_FILE.read_text(encoding="utf-8"))
    for order in data["订单"]:
        if order["订单号"] == order_id:
            days = (date.fromisoformat(data["查询日期"])
                    - date.fromisoformat(order["付款日期"])).days
            logger.info("get_mock_order: %s", order_id)
            return {**order, "查询日期": data["查询日期"], "距付款天数": days}
    raise ValueError(f"订单 {order_id} 不存在")


@mcp.tool
def create_mock_ticket(
    order_id: str,
    decision: Literal["全额退款", "退还 50%"],
    refund_amount: Annotated[float, Field(ge=0, description="退款金额，单位元。")],
    reason: str,
    evidence: Annotated[str, Field(description="依据的规则片段编号，例如“片段18,片段19”。")],
) -> dict:
    """创建一张待人工处理的模拟退款工单，返回工单号与状态。只在审核结论为可以退款时调用。"""
    ticket = {
        "工单号": f"T{len(tickets) + 1:04d}",
        "订单号": order_id,
        "结论": decision,
        "退款金额": refund_amount,
        "理由": reason,
        "依据": evidence,
        "状态": "待处理",
    }
    tickets.append(ticket)
    logger.info("create_mock_ticket: %s order=%s", ticket["工单号"], order_id)
    return ticket


@mcp.resource("tickets://all", mime_type="application/json")
def all_tickets() -> str:
    """已创建的全部模拟工单，供应用代码核对。"""
    return json.dumps(tickets, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    # 与计算服务的 8000、套餐服务的 8001 错开。
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--path", default="/mcp")
    options = parser.parse_args()

    # stdout 留给 stdio 协议；服务端的运行信息统一写到 stderr。
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(name)s: %(message)s",
    )
    logger.info("server_pid=%s transport=%s", os.getpid(), options.transport)

    global knowledge_store
    # 启动后自动构建知识库
    knowledge_store = build_knowledge_store()

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
