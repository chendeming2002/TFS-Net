#!/usr/bin/env python3
"""
read_pdf.py — 提取 PDF 论文文本内容
用法: python read_pdf.py <pdf_path> [--out <out.txt>] [--max-chars <N>]
依赖: pdftotext (poppler-utils) 或 pdfplumber
"""
import subprocess
import sys
import os
import shutil


def extract_with_pdftotext(pdf_path, out_path=None, layout=True):
    """用 pdftotext (poppler-utils) 提取，保留版面。"""
    cmd = ["pdftotext"]
    if layout:
        cmd.append("-layout")
    cmd.extend([pdf_path, out_path or "-"])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"pdftotext failed: {result.stderr}")
    return result.stdout if out_path is None else None


def extract_with_pdfplumber(pdf_path, max_pages=None):
    """用 pdfplumber 提取（需 pip install pdfplumber）。"""
    import pdfplumber
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            if max_pages and i >= max_pages:
                break
            t = page.extract_text() or ""
            texts.append(f"--- Page {i+1} ---\n{t}")
    return "\n\n".join(texts)


def extract_pdf(pdf_path, out_path=None, method="auto", max_chars=None):
    """主入口：自动选择可用方法提取 PDF 文本。"""
    if method == "auto":
        if shutil.which("pdftotext"):
            method = "pdftotext"
        else:
            method = "pdfplumber"

    if method == "pdftotext":
        text = extract_with_pdftotext(pdf_path, out_path=None)
    elif method == "pdfplumber":
        text = extract_with_pdfplumber(pdf_path)
    else:
        raise ValueError(f"Unknown method: {method}")

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved to {out_path} ({len(text)} chars)")

    if max_chars:
        text = text[:max_chars]
    return text


def search_keywords(text, keywords, context=2):
    """在提取的文本中搜索关键词并返回上下文行。"""
    lines = text.split("\n")
    results = []
    for i, line in enumerate(lines):
        for kw in keywords:
            if kw.lower() in line.lower():
                start = max(0, i - context)
                end = min(len(lines), i + context + 1)
                results.append({
                    "keyword": kw,
                    "line": i + 1,
                    "context": "\n".join(lines[start:end]),
                })
                break
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract text from PDF")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--out", default=None, help="Output text file path")
    parser.add_argument("--method", default="auto", choices=["auto", "pdftotext", "pdfplumber"])
    parser.add_argument("--max-chars", type=int, default=None, help="Max chars to print")
    parser.add_argument("--search", nargs="+", default=None, help="Keywords to search")
    args = parser.parse_args()

    text = extract_pdf(args.pdf_path, args.out, args.method, args.max_chars)

    if args.search:
        results = search_keywords(text, args.search)
        for r in results:
            print(f"\n=== '{r['keyword']}' at line {r['line']} ===")
            print(r["context"])
    else:
        print(text)
