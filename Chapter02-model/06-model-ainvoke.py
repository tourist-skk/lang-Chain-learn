"""
@Author:shkstart
@Desc: 
"""
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os
import asyncio
import time


# 从.env文件中加载环境变量
load_dotenv(override=True, verbose=True,dotenv_path="conf/.env")

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_API_BASE = os.environ["OPENAI_API_BASE"]  # 与本项目 .env 中的名称一致

model = init_chat_model(
    model="openai:gpt-6-luna",
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE
)

async def demo_async_invoke():
    print("=== 演示：ainvoke 的异步（非阻塞）效果 ===")
    start_time = time.perf_counter()  # 记录开始时间

    print("程序开始...")

    # 1. 创建任务 (Task)
    print(">>> 发起异步模型调用 (ainvoke)...")
    async_task = asyncio.create_task(model.ainvoke("用一句话解释人工智能。"))

    # 2. 并行执行其他任务
    print(">>> 模型请求已在后台发送，继续执行本地逻辑...")
    for i in range(3):
        await asyncio.sleep(1)  # 使用异步等待，释放控制权
        print(f">>> 正在执行第{i + 1}个任务... (已耗时 {time.perf_counter() - start_time:.2f}s)")

    # 3. 获取模型结果
    print(">>> 本地任务完成，检查模型状态...")
    response = await async_task

    end_time = time.perf_counter()
    print(f">>> 模型返回: {response.content}")
    print(f"=== 总运行耗时: {end_time - start_time:.2f}s ===")
# === 演示：invoke 的同步（阻塞）效果 ===
# 程序开始...
# >>> 发起同步模型调用 (invoke)...
# >>> 模型返回，开始执行本地逻辑...
# >>> 正在执行第1个任务... (已耗时 4.28s)
# >>> 正在执行第2个任务... (已耗时 5.28s)
# >>> 正在执行第3个任务... (已耗时 6.28s)
# >>> 模型返回: 人工智能是让机器通过数据学习、推理并完成通常需要人类智能的任务的技术。
# === 总运行耗时: 6.28s ===
# === 演示：ainvoke 的异步（非阻塞）效果 ===
# 程序开始...
# >>> 发起异步模型调用 (ainvoke)...
# >>> 模型请求已在后台发送，继续执行本地逻辑...
# >>> 正在执行第1个任务... (已耗时 1.00s)
# >>> 正在执行第2个任务... (已耗时 2.00s)
# >>> 正在执行第3个任务... (已耗时 3.00s)
# >>> 本地任务完成，检查模型状态...
# >>> 模型返回: 人工智能是让计算机执行通常需要人类智能才能完成的任务的技术，例如学习、推理和理解语言。
# === 总运行耗时: 3.37s ===




def demo_sync_invoke():
    """同步（阻塞）版本：与 demo_async_invoke 对比用"""
    print("=== 演示：invoke 的同步（阻塞）效果 ===")
    start_time = time.perf_counter()  # 记录开始时间

    print("程序开始...")

    # 1. 同步调用模型（阻塞）：必须等模型返回结果，代码才能继续往下走
    print(">>> 发起同步模型调用 (invoke)...")
    response = model.invoke("用一句话解释人工智能。")

    # 2. 模型返回后，才开始执行本地任务（无法与模型请求并行）
    print(">>> 模型返回，开始执行本地逻辑...")
    for i in range(3):
        time.sleep(1)  # 同步等待，阻塞整个线程
        print(f">>> 正在执行第{i + 1}个任务... (已耗时 {time.perf_counter() - start_time:.2f}s)")

    print(f">>> 模型返回: {response.content}")
    print(f"=== 总运行耗时: {time.perf_counter() - start_time:.2f}s ===")


async def main():
    """主函数"""
    demo_sync_invoke()
    await demo_async_invoke()


if __name__ == "__main__":
    asyncio.run(main())
