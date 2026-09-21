import os
import logging
from typing import Dict, Any, List
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class PDFProcessor:
    """轻量 + 稳定版 PDF 文本提取器"""

    def __init__(self):
        self.supported_formats = ['.pdf']

    # ------------------------------------------------------
    # 纯文本提取（主功能）
    # ------------------------------------------------------
    def extract_text(self, pdf_path: str) -> str:
        if not os.path.exists(pdf_path):
            logger.error(f"[PDFProcessor] 文件不存在: {pdf_path}")
            return ""

        try:
            doc = fitz.open(pdf_path)
            texts = []
            for page in doc:
                texts.append(page.get_text().strip())
            doc.close()

            content = "\n\n".join(texts)
            logger.info(f"[PDFProcessor] 提取文本成功: {len(content)} 字符")
            return content

        except Exception as e:
            logger.error(f"[PDFProcessor] 文本提取失败: {e}")
            return ""

    # ------------------------------------------------------
    # 元数据 + 文本（轻量）
    # ------------------------------------------------------
    def extract_text_with_metadata(self, pdf_path: str) -> Dict[str, Any]:
        if not os.path.exists(pdf_path):
            return {"error": f"文件不存在: {pdf_path}"}

        try:
            doc = fitz.open(pdf_path)
            pages_data = []
            full_text = []

            for page_no, page in enumerate(doc, 1):
                txt = page.get_text()
                pages_data.append({
                    "page": page_no,
                    "text": txt.strip(),
                    "bbox": list(page.rect)
                })
                full_text.append(txt.strip())

            result = {
                "page_count": len(pages_data),
                "pages": pages_data,
                "full_text": "\n\n".join(full_text)
            }

            doc.close()
            return result

        except Exception as e:
            logger.error(f"[PDFProcessor] 元数据提取失败: {e}")
            return {"error": str(e)}

    # ------------------------------------------------------
    # 简单分析（轻量版）
    # ------------------------------------------------------
    def analyze_pdf(self, pdf_path: str) -> Dict[str, Any]:
        if not os.path.exists(pdf_path):
            return {"error": f"文件不存在: {pdf_path}"}

        try:
            doc = fitz.open(pdf_path)
            summaries = []
            full = []

            for page_no, page in enumerate(doc, 1):
                txt = page.get_text().strip()
                full.append(txt)
                summaries.append(f"第 {page_no} 页: {txt[:200]}...")

            doc.close()

            return {
                "page_count": len(summaries),
                "summary": "\n".join(summaries),
                "full_text": "\n\n".join(full)
            }

        except Exception as e:
            logger.error(f"[PDFProcessor] 分析失败: {e}")
            return {"error": str(e)}

    # ------------------------------------------------------
    # PDF 图像提取（轻量且正确处理 CMYK）
    # ------------------------------------------------------
    def extract_images(self, pdf_path: str, output_dir: str = None) -> List[str]:
        if not os.path.exists(pdf_path):
            return []

        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(pdf_path), "pdf_images")
        os.makedirs(output_dir, exist_ok=True)

        try:
            doc = fitz.open(pdf_path)
            saved_paths = []

            for page_no, page in enumerate(doc, 1):
                for img_index, img in enumerate(page.get_images(), 1):
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)

                    # convert CMYK to RGB
                    if pix.n > 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)

                    out_path = os.path.join(
                        output_dir,
                        f"{os.path.basename(pdf_path)}_p{page_no}_{img_index}.png"
                    )

                    pix.save(out_path)
                    pix = None
                    saved_paths.append(out_path)

            doc.close()
            return saved_paths

        except Exception as e:
            logger.error(f"[PDFProcessor] 图像提取失败: {e}")
            return []

    # ------------------------------------------------------
    def is_supported_format(self, file_path: str) -> bool:
        return any(file_path.lower().endswith(ext) for ext in self.supported_formats)