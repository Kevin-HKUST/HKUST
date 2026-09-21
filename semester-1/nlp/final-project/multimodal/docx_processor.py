import os
import docx
import logging
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class DocxProcessor:
    """
    DOCX 文本与问题抽取器（修复版）
    - 移除“必须以问号结尾”的限制
    - 改为“编号识别 + 跨行合并”
    - 适配 Test Set 1, 2, 3 全部题型
    """

    def __init__(self):
        self.supported_formats = [".docx"]

    # ------------------------------------------------------------------
    # 纯文本提取
    # ------------------------------------------------------------------
    def extract_text(self, docx_path: str) -> str:
        if not os.path.exists(docx_path):
            logger.error(f"[DocxProcessor] 文件不存在: {docx_path}")
            return ""

        try:
            doc = docx.Document(docx_path)
            parts = []

            for para in doc.paragraphs:
                t = para.text.strip()
                if t:
                    parts.append(t)

            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join([cell.text.strip() for cell in row.cells if cell.text.strip()])
                    if row_text:
                        parts.append(row_text)

            return "\n".join(parts)

        except Exception as e:
            logger.error(f"[DocxProcessor] 提取失败: {e}")
            return ""

    # ------------------------------------------------------------------
    # 结构化问题抽取（核心修复）
    # ------------------------------------------------------------------
    def analyze_docx(self, docx_path: str) -> Dict[str, Any]:
        if not os.path.exists(docx_path):
            return {"error": f"文件不存在: {docx_path}"}

        try:
            doc = docx.Document(docx_path)

            # 收集所有文本行
            lines = []
            for para in doc.paragraphs:
                t = para.text.strip()
                if t:
                    lines.append(t)

            for table in doc.tables:
                for row in table.rows:
                    row_text = " ".join([cell.text.strip() for cell in row.cells if cell.text.strip()])
                    if row_text:
                        lines.append(row_text)

            # ---------------------------
            # 题目识别规则：捕获编号开头的行
            # ---------------------------
            # 支持：
            # 1. 英文编号：1. 1) (1) A. a)
            # 2. 中文编号：1、 一、 ① ②
            # 3. Input 标签不影响识别
            numbering_regex = re.compile(
                r"""^
                \s*
                (
                    \d+[.\)]      |        # 1. 1)
                    \(\d+\)       |        # (1)
                    [A-Za-z][.\)] |        # A. a)
                    [一二三四五六七八九十]+[、.] | # 一、 二.
                    [\u2460-\u2473]        # ① ②
                )
                """,
                re.VERBOSE
            )

            questions = []
            buffer = ""

            for line in lines:
                # 归一化 Input 标签（中文/英文）
                line = re.sub(r"^\s*\[(Input|輸入)[^\]]*\]\s*", "", line)

                if numbering_regex.match(line):
                    # 若 buffer 非空，说明上一题结束
                    if buffer.strip():
                        questions.append(buffer.strip())
                    buffer = numbering_regex.sub("", line).strip()
                else:
                    # 跨行内容，追加到上一题
                    if buffer:
                        buffer += " " + line.strip()

            # 最后一题写入
            if buffer.strip():
                questions.append(buffer.strip())

            # 转结构化
            result = {
                "questions": [{"id": i + 1, "text": q} for i, q in enumerate(questions)],
                "paragraph_count": len(doc.paragraphs),
                "table_count": len(doc.tables),
                "full_text": "\n".join(lines),
            }

            logger.info(f"[DocxProcessor] 成功抽取 {len(questions)} 个问题（修复编号+跨行+无问号）")

            return result

        except Exception as e:
            logger.error(f"[DocxProcessor] 分析失败: {e}")
            return {"error": str(e)}

    # ------------------------------------------------------------------
    def is_supported_format(self, file_path: str) -> bool:
        return any(file_path.lower().endswith(ext) for ext in self.supported_formats)