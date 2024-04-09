# -*- coding: utf-8 -*-
"""A PDF reader service function."""
import os
from typing import List

import fitz
import pdfplumber


def read_pdf(pdf_path: str) -> str:
    """Read a PDF file and return the text content."""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            print(page.extract_text() + "\n")
            # 处理表格
            # for table in page.extract_tables():
            #     print(table)  # 可改为将表格数据写入到文件或数据库
    return ""


read_pdf("/Users/david/Downloads/2305.10601.pdf")


def convert_pdf_to_image(pdf_path: str, target_dir: str) -> List[str]:
    """Convert a PDF file to multiple images, and return the paths to the
    images.

    Args:
        pdf_path (`str`):
            The path to the PDF file.
        target_dir (`str`):
            The directory to save the images.

    Returns:
        `List[str]`: A list of paths to the images.
    """

    # check if the target dir exists
    os.makedirs(target_dir, exist_ok=True)

    paths = []
    with fitz.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)  # start from 0

            # 设置图片的DPI，x和y的DPI一样即可。这里设置为300DPI，通常印刷品的标准。
            # 提高这个值会使图片更清晰，但文件大小也会增加。
            zoom_x = 300 / 72  # 72是PDF的默认DPI
            zoom_y = 300 / 72
            mat = fitz.Matrix(zoom_x, zoom_y)  # 创建变换矩阵

            pix = page.get_pixmap(matrix=mat)
            output_path = os.path.join(
                target_dir,
                f"output_page_{page_num}.png",
            )
            # Save locally
            pix.save(output_path)

            paths.append(output_path)

    return paths
