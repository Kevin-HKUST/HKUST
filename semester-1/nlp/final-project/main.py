#!/usr/bin/env python3
"""
智能搜索引擎主程序（方案 A：文件路径 | 问题）
新增功能：批量处理Word文档中问题，输出含完整指标的JSON结果
"""

import argparse
import logging
import os
import sys
import time
from typing import List, Dict, Any
import signal
import json
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, ensure_runtime_dirs
from multimodal.multimodal_router import MultimodalRouter
from multimodal.docx_processor import DocxProcessor
from core.workflow_engine import WorkflowEngine
from core.llm_client import llm_client

# Logging writes into logs/, so the directory has to exist before the handler
# is attached. Importing a module no longer creates it as a side effect.
ensure_runtime_dirs()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/app.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

print("启动 main.py...")


def signal_handler(sig, frame):
    logger.info("用户中断 (Ctrl+C)，程序退出。")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def load_test_questions(docx_path: str) -> List[str]:
    """优化问题提取逻辑，支持所有中英文问题（含重复/无编号/英文编号）"""
    processor = DocxProcessor()
    analysis_result = processor.analyze_docx(docx_path)

    # 优先使用analyze_docx的结构化问题提取（准确率更高）
    if "questions" in analysis_result and len(analysis_result["questions"]) > 0:
        questions = [
            q["text"].strip() for q in analysis_result["questions"]
            if q["text"].strip()
               and ("?" in q["text"] or "？" in q["text"])  # 确保是问题（中英文问号）
        ]
        # 保留所有问题（不去重）
        return questions

    # 备用方案：从纯文本中提取（增强英文格式支持）
    content = processor.extract_text(docx_path)
    # 优化正则：匹配 中英文编号 + 无编号问题，支持所有格式
    improved_pattern = r'^\s*(?:\d+[.\)、]|[a-zA-Z]+[.\)]|\(\d+\)|\([a-zA-Z]\)|\u2460-\u2473|\u2776-\u2793|[\u4e00-\u9fa5]+[\.、)])?\s*([A-Za-z0-9\s\w\W]+?[?？])'
    matches = re.findall(improved_pattern, content, re.MULTILINE | re.DOTALL)

    questions = []
    for match in matches:
        text = match.strip()
        if text:
            questions.append(text)

    # 保留所有问题（不主动去重，完全遵循测试集原始问题）
    return list(filter(None, questions))


def calculate_quality(answer: str) -> float:
    """优化质量评分逻辑，更精准判断成功回答"""
    if not answer or answer.strip() == "":
        return 0.0

    # 失败关键词列表（含中英文）
    failure_keywords = ["无法提供", "无法获取", "无法回答", "cannot answer", "sorry", "抱歉", "未找到",
                        "no information", "couldn't find"]
    if any(kw in answer.lower() for kw in failure_keywords):
        return 0.0

    # 成功回答判断：长度达标 + 无推测性表述（含中英文）
    char_len = len(answer.strip())
    speculative_keywords = ["可能", "估算", "约", "推断", "建议查看", "may", "might", "approximately", "suggest"]
    has_speculative = any(kw in answer.lower() for kw in speculative_keywords)

    if char_len >= 15 and not has_speculative:
        return 1.0
    elif char_len >= 8 and has_speculative:
        return 0.5
    else:
        return 0.0


def answer_document_questions(doc_path: str, engine: WorkflowEngine) -> Dict[str, Any]:
    """批量处理问题，计算完整统计指标"""
    processor = DocxProcessor()
    questions = load_test_questions(doc_path)

    if not questions:
        logger.warning("未从文档中提取到有效问题（含中英文）")
        return {
            "stats": {
                "total_questions": 0,
                "success_count": 0,
                "total_time_seconds": 0.0,
                "avg_time_per_question_seconds": 0.0,
                "success_rate": 0.0
            },
            "questions": []
        }

    answers = []
    total_processing_time = 0.0
    success_count = 0

    # 批量执行工作流（支持中英文问题，保留重复问题）
    for idx, q in enumerate(questions, 1):
        logger.info(f"处理问题 {idx}/{len(questions)}: {q[:50]}...")
        start_time = time.time()

        # 执行工作流获取答案（IntentRecognizer已支持英文关键词）
        result = engine.execute_workflow(q)
        final_answer = result.get("final_answer", {})
        answer_content = final_answer.get("answer", "").strip()
        confidence = final_answer.get("confidence", 0.0)

        # 计算耗时和质量评分
        processing_time = time.time() - start_time
        quality_score = calculate_quality(answer_content)
        is_success = quality_score >= 0.5  # 评分≥0.5视为成功回答

        # 累计统计数据
        total_processing_time += processing_time
        if is_success:
            success_count += 1

        # 保存单个问题结果（保留问题原始文本，不做去重）
        answers.append({
            "question_id": idx,
            "question": q,
            "answer": answer_content,
            "confidence": round(confidence, 2),
            "quality_score": round(quality_score, 2),
            "processing_time_seconds": round(processing_time, 3),
            "is_success": is_success
        })

    # 计算综合统计指标
    total_questions = len(questions)
    avg_time_per_question = total_processing_time / total_questions if total_questions > 0 else 0.0
    success_rate = (success_count / total_questions) * 100 if total_questions > 0 else 0.0

    return {
        "metadata": {
            "input_file": os.path.basename(doc_path),
            "processing_start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "processing_end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        },
        "stats": {
            "total_questions": total_questions,
            "success_count": success_count,
            "failure_count": total_questions - success_count,
            "total_time_seconds": round(total_processing_time, 3),
            "avg_time_per_question_seconds": round(avg_time_per_question, 3),
            "success_rate_percent": round(success_rate, 2)
        },
        "questions": answers
    }


def main():
    router = MultimodalRouter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true", help="交互模式")
    parser.add_argument("--batch", nargs=2, metavar=("INPUT_DOCX", "OUTPUT_JSON"),
                        help="批量模式：输入Word文档路径 输出JSON路径")
    parser.add_argument("--query", type=str, help="单查询模式")
    parser.add_argument("--evaluate", action="store_true", help="评估模式")
    parser.add_argument("--rebuild_rag", action="store_true", help="Rebuild the FAISS index from data/knowledge_base")
    args = parser.parse_args()

    if args.rebuild_rag:
        from core.vector_store import VectorStore
        n_chunks = VectorStore().rebuild()
        print(f"FAISS index rebuilt ({n_chunks} chunks) from data/knowledge_base.")
        return

    print("进入 main()...")

    if not Config.has_llm_credentials():
        print("未配置 HKGAI_API_KEY。请复制 .env.example 为 .env 并填入密钥。")
        logger.error("HKGAI_API_KEY 未配置")
        return

    # 初始化工作流引擎
    try:
        engine = WorkflowEngine()
        print("工作流引擎初始化成功")
    except Exception as e:
        print(f"引擎初始化失败: {e}")
        logger.error(f"引擎初始化失败: {e}")
        return

    # 测试LLM连接
    try:
        test_result = llm_client.chat("You are a helpful AI assistant.", "Hello, can you help me?")  # 测试英文交互
        if not test_result.get("content"):
            print("LLM服务连接失败")
            logger.error("LLM服务连接失败")
            return
    except Exception as e:
        print(f"LLM测试失败: {e}")
        logger.error(f"LLM测试失败: {e}")
        return

    print("知识库初始化成功（支持中英文查询）")

    # 评估模式
    if args.evaluate:
        from evaluation.evaluator import SearchEvaluator

        evaluator = SearchEvaluator(engine)

        docx_dir = "data/test_questions"
        test_docs = sorted(
            f for f in os.listdir(docx_dir) if f.endswith(".docx")
        ) if os.path.isdir(docx_dir) else []

        if not test_docs:
            print(f"未在 {docx_dir} 找到 .docx 测试集，无法评估。")
            return

        for doc in test_docs:
            questions = load_test_questions(os.path.join(docx_dir, doc))
            if not questions:
                print(f"跳过 {doc}：未提取到问题。")
                continue
            evaluator.evaluate_test_set(doc, questions)

        if evaluator.results:
            evaluator.generate_report()
        else:
            print("没有可用的评估结果。")
        return

    # 批量处理模式（核心新增功能）
    if args.batch:
        doc_path, out_path = args.batch
        if not os.path.exists(doc_path):
            print(f"错误：输入文件不存在 - {doc_path}")
            logger.error(f"输入文件不存在 - {doc_path}")
            return

        print(f"开始批量处理文档: {doc_path}（含中英文/重复问题）")
        logger.info(f"开始批量处理文档: {doc_path}（含中英文/重复问题）")

        # 执行批量问答
        batch_result = answer_document_questions(doc_path, engine)

        # 保存JSON结果
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(batch_result, f, ensure_ascii=False, indent=2)
            print(f"批量处理完成！结果已保存至: {out_path}")
            logger.info(f"批量处理完成！结果已保存至: {out_path}")

            # 打印关键统计信息
            stats = batch_result["stats"]
            print("\n=== 批量处理统计报告 ===")
            print(f"总问题数: {stats['total_questions']}（含中英文/重复问题）")
            print(f"成功回答数: {stats['success_count']}")
            print(f"失败回答数: {stats['failure_count']}")
            print(f"总处理时间: {stats['total_time_seconds']} 秒")
            print(f"单问题平均时间: {stats['avg_time_per_question_seconds']} 秒")
            print(f"成功率: {stats['success_rate_percent']}%")
            print("=======================")

        except Exception as e:
            print(f"保存JSON结果失败: {e}")
            logger.error(f"保存JSON结果失败: {e}")
        return

    # 单查询模式
    if args.query:
        result = engine.execute_workflow(args.query)
        fa = result.get("final_answer", {})
        print("回答:", fa.get("answer"))
        print("置信度:", round(fa.get("confidence", 0.0), 2))
        return

    # 交互模式
    if args.interactive:
        print("""
============================
智能搜索引擎 - 交互模式
支持:
  1. 普通问答（中英文）
  2. 输入文件路径自动解析
  3. 输入 “文件路径 | 问题” 同时对图文进行推理
============================
""")

        while True:
            try:
                query = input("[QUERY] ").strip()
                if query.lower() in ["quit", "退出"]:
                    break

                # 处理"文件路径 | 问题"格式
                if "|" in query:
                    file_path, user_q = map(str.strip, query.split("|", 1))
                    if os.path.exists(file_path):
                        mm = router.load(file_path)
                        extracted = mm.get("text", "")
                        full_query = extracted + "\n\n" + user_q
                        result = engine.execute_workflow(full_query)
                        fa = result.get("final_answer", {})
                        print("回答:", fa.get("answer"))
                        print("置信度:", round(fa.get("confidence", 0.0), 2))
                    continue

                # 处理单文件解析
                if os.path.exists(query):
                    mm = router.load(query)
                    extracted = mm.get("text", "")
                    print("文件解析成功，内容预览：")
                    print(extracted[:300], "...\n")
                    result = engine.execute_workflow(extracted)
                    fa = result.get("final_answer", {})
                    print("回答:", fa.get("answer"))
                    print("置信度:", round(fa.get("confidence", 0.0), 2))
                    continue

                # 普通问答（支持中英文）
                result = engine.execute_workflow(query)
                fa = result.get("final_answer", {})
                print("回答:", fa.get("answer"))
                print("置信度:", round(fa.get("confidence", 0.0), 2))

            except Exception as e:
                print(f"处理失败: {e}")
                logger.error(f"交互模式处理失败: {e}")

    input("Press Enter to exit...")


if __name__ == "__main__":
    main()