"""
@Author:shkstart
@Desc: 
"""
import asyncio
import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import time
from langchain_openai import ChatOpenAI

# 从.env文件中加载环境变量
load_dotenv(override=True, verbose=True,dotenv_path="conf/.env")

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_API_BASE = os.environ["OPENAI_API_BASE"]  # 与本项目 .env 中的名称一致

model = ChatOpenAI(
    model="gpt-6-luna",  # 当前配置已验证可用；原 gpt-5.5 返回模型权限 403
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE,
)

# === 演示：abatch 的异步（非阻塞）效果 ===
# 程序开始...                                                                                                                               >>> 发起异步批量调用 (abatch)...
# >>> 批量任务已在后台运行，主程序继续执行...
# >>> 正在执行第1个任务... (已耗时 1.00s)
# >>> 正在执行第2个任务... (已耗时 2.00s)
# >>> 正在执行第3个任务... (已耗时 3.00s)
# >>> 其他任务已完成，现在获取后台批量任务的结果...                                                                                         >>> 响应内容: 深度学习通过多层神经网络从数据中自动学习特征，传统机器学习则通常依赖人工设计特征再进行建模。
# >>> 响应内容: 日本首相是石破茂，他于2024年10月就任。
# === 总运行耗时: 6.81s ===
async def demo_async_batch():
    """演示异步批量的非阻塞特性"""
    print("=== 演示：abatch 的异步（非阻塞）效果 ===")
    start_time = time.perf_counter()  # 记录开始时间

    print("程序开始...")

    # 准备批量输入
    questions = ["用一句话说明深度学习与传统机器学习的区别", "日本首相是谁？"]

    # 1. 发起异步批量请求
    # 关键修改：使用 create_task 让协程立即在后台执行
    print(">>> 发起异步批量调用 (abatch)...")
    batch_task = asyncio.create_task(model.abatch(questions))

    # 2. 在等待批量处理的同时，执行其他任务
    print(">>> 批量任务已在后台运行，主程序继续执行...")
    for i in range(3):
        # 关键修改：使用 asyncio.sleep 允许后台任务获取 CPU 时间片进行网络请求
        await asyncio.sleep(1)
        print(f">>> 正在执行第{i + 1}个任务... (已耗时 {time.perf_counter() - start_time:.2f}s)")

    # 3. 等待批量处理结果
    print(">>> 其他任务已完成，现在获取后台批量任务的结果...")
    # 此时 batch_task 可能已经完成，或者我们在这里等待它完成
    responses = await batch_task

    end_time = time.perf_counter()

    for response in responses:
        content = response.content if hasattr(response, 'content') else str(response)
        print(f">>> 响应内容: {content}")

    print(f"=== 总运行耗时: {end_time - start_time:.2f}s ===")




def demo_sync_batch():
    """同步（阻塞）版本：与 demo_async_batch 对比用"""
    print("=== 演示：batch 的同步（阻塞）效果 ===")
    start_time = time.perf_counter()  # 记录开始时间

    print("程序开始...")

    # 准备批量输入
    questions = ["用一句话说明深度学习与传统机器学习的区别", "日本首相是谁？"]

    # 1. 同步批量调用（阻塞）：必须等所有问题都返回结果，代码才能继续往下走
    print(">>> 发起同步批量调用 (batch)...")
    responses = model.batch(questions)

    # 2. 批量结果全部返回后，才开始执行本地任务（无法与批量请求并行）
    print(">>> 批量返回，开始执行本地逻辑...")
    for i in range(3):
        time.sleep(1)  # 同步等待，阻塞整个线程
        print(f">>> 正在执行第{i + 1}个任务... (已耗时 {time.perf_counter() - start_time:.2f}s)")

    for response in responses:
        content = response.content if hasattr(response, 'content') else str(response)
        print(f">>> 响应内容: {content}")

    print(f"=== 总运行耗时: {time.perf_counter() - start_time:.2f}s ===")
# === 演示：batch 的同步（阻塞）效果 ===
# 程序开始...                                                                                                                               >>> 发起同步批量调用 (batch)...
# >>> 批量返回，开始执行本地逻辑...
# >>> 正在执行第1个任务... (已耗时 10.76s)
# >>> 正在执行第2个任务... (已耗时 11.76s)
# >>> 正在执行第3个任务... (已耗时 12.76s)
# >>> 响应内容: 深度学习是机器学习的一种，能通过多层神经网络自动学习数据特征，而传统机器学习通常需要人工设计特征再进行建模。
# >>> 响应内容: 截至 2026 年 1 月，日本首相是高市早苗。
# === 总运行耗时: 12.76s ===
# === 演示：abatch 的异步（非阻塞）效果 ===
# 程序开始...
# >>> 发起异步批量调用 (abatch)...
# >>> 批量任务已在后台运行，主程序继续执行...                                                                                               >>> 正在执行第1个任务... (已耗时 1.00s)
# >>> 正在执行第2个任务... (已耗时 2.00s)
# >>> 正在执行第3个任务... (已耗时 3.00s)
# >>> 其他任务已完成，现在获取后台批量任务的结果...
# >>> 响应内容: 深度学习是机器学习的一种，擅长从大量数据中自动学习多层次特征，而传统机器学习通常需要人工设计特征。
# >>> 响应内容: 截至2026年8月，日本首相是高市早苗。
# === 总运行耗时: 7.98s ===

async def main():
    """主函数"""
    demo_sync_batch()
    await demo_async_batch()
    
    print()


if __name__ == "__main__":
    asyncio.run(main())
