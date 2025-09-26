#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Smart OCR Pipeline
- Decide "OCR or Skip" using:
  * PDF text layer presence (skip OCR if selectable text exists)
  * Fast text detector (EAST) for images/scanned pages
  * Resolution threshold: skip if W*H < 0.3MP
- Selective pages: front 3 + last (default) or all / firstN+lastN
- OCR engine: Tesseract (LSTM)
- Redaction:
  * Images: draw filled rectangles over matched OCR words
  * PDFs: true redaction using PyMuPDF redaction annotations
- Sensitive keywords: combine "관리자 입력 DB + OCR DB" style lists (editable below)

Requirements:
  pip install pytesseract Pillow pymupdf opencv-python numpy python-magic-bin tqdm
And Tesseract OCR engine installed in OS.

EAST model:
  Provide --east-model path/to/frozen_east_text_detection.pb
"""

import os
import io
import re
import sys
import json
import mimetypes
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

import pytesseract
import fitz  # PyMuPDF
import cv2

# For robust MIME sniffing (best-effort). On Linux/mac use python-magic.
try:
    import magic  # type: ignore
    HAVE_MAGIC = True
except Exception:
    HAVE_MAGIC = False

# --------------------------
# Configurable constants
# --------------------------

MIN_MP = 0.3  # 0.3 Megapixel threshold
EAST_CONF_THRESH = 0.5
EAST_NMS_THRESH = 0.4

# Keywords – freely extend/modify (관리자DB + OCR DB 통합 예시)
HEADER_KEYWORDS = [
    # 계약/비밀
    "비밀유지계약서", "NDA", "계약서", "합의서", "제1조", "갑", "을", "대외비",
    "INTERNAL USE ONLY", "CONFIDENTIAL",
    # 세무/영수
    "세금계산서", "전자세금계산서", "공급자", "공급받는자", "등록번호", "합계금액",
    # 신분/증빙
    "주민등록등본", "주민등록초본", "재직증명서", "사업자등록증", "운전면허증", "여권",
    # 급여/거래
    "급여명세서", "사번", "지급총액", "공제총액", "실지급액", "은행거래내역서",
    "거래일자", "적요", "입금액", "출금액", "잔액",
    # 의료
    "진단서", "처방전", "의료기관", "환자명", "진단명",
]

FIELD_KEYWORDS = [
    # 개인
    "성명", "생년월일", "주민등록번호", "주소", "연락처", "이메일", "사번",
    # 금융
    "계좌번호", "은행명", "예금주", "카드번호", "유효기간", "CVV",
    # 사업/법인
    "사업자등록번호", "법인등록번호", "대표자",
    # 의료
    "환자명", "진단명", "차트번호",
]

# (선택) 숫자 패턴 정규식 예시 – 필요시 강화
REGEX_PATTERNS = {
    "rrn": re.compile(r"\b\d{6}-\d{7}\b"),  # 주민등록번호(형식 예시)
    "card": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),  # 카드번호 대략
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
}

# --------------------------
# Utils
# --------------------------

def set_tesseract_cmd(cmd: Optional[str]):
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd

def sniff_mime(path: Path) -> str:
    if HAVE_MAGIC:
        try:
            return magic.from_file(str(path), mime=True)  # type: ignore
        except Exception:
            pass
    # fallback
    mt, _ = mimetypes.guess_type(str(path))
    return mt or "application/octet-stream"

def is_pdf_mime(m: str) -> bool:
    return m == "application/pdf" or ("/pdf" in m)

def is_image_mime(m: str) -> bool:
    return m.startswith("image/")

def mpixels_of_img(pil_img: Image.Image) -> float:
    w, h = pil_img.size
    return (w * h) / 1_000_000.0

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# --------------------------
# EAST text detector
# --------------------------

def load_east(model_path: Optional[str]):
    if not model_path:
        return None
    if not Path(model_path).exists():
        raise FileNotFoundError(f"EAST model not found: {model_path}")
    net = cv2.dnn.readNet(model_path)
    return net

def east_has_text(net, pil_img: Image.Image, conf_thresh=EAST_CONF_THRESH) -> bool:
    """Fast check: returns True if any text-like boxes detected."""
    if net is None:
        # If no model provided, assume we cannot decide -> proceed with OCR path
        return True

    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    H, W = img.shape[:2]

    # EAST expects 320x320 or 640x640 multiples of 32
    inpW = 320
    inpH = 320
    blob = cv2.dnn.blobFromImage(img, 1.0, (inpW, inpH),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid",
                                      "feature_fusion/concat_3"])

    rects = []
    confidences = []

    numRows, numCols = scores.shape[2:4]
    for y in range(numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(numCols):
            score = scoresData[x]
            if score < conf_thresh:
                continue

            offsetX = x * 4.0
            offsetY = y * 4.0

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(float(score))

    if not rects:
        return False

    boxes = cv2.dnn.NMSBoxes(
        [(*r, 1, 1) for r in rects], confidences, conf_thresh, EAST_NMS_THRESH
    )
    return len(boxes) > 0

# --------------------------
# OCR + Sensitive detection
# --------------------------

def preprocess_for_ocr(pil_img: Image.Image, do_deskew=True) -> Image.Image:
    """Lightweight, fast, safe defaults."""
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # (옵션) 간단한 deskew
    if do_deskew:
        try:
            gray_inv = cv2.bitwise_not(gray)
            thr = cv2.threshold(gray_inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            coords = np.column_stack(np.where(thr > 0))
            if coords.size > 0:
                angle = cv2.minAreaRect(coords)[-1]
                if angle < -45:
                    angle = -(90 + angle)
                else:
                    angle = -angle
                (h, w) = gray.shape[:2]
                M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_REPLICATE)
        except Exception:
            pass

    # 약한 노이즈 제거 + 가벼운 이진화
    gray = cv2.bilateralFilter(gray, 5, 40, 40)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 31, 10)
    out = Image.fromarray(th)
    return out

def tesseract_ocr(pil_img: Image.Image, lang="kor+eng", psm=3, oem=1,
                  tessdata_dir: Optional[str] = None) -> Tuple[str, Dict]:
    cfg = f"--psm {psm} --oem {oem}"
    if tessdata_dir:
        cfg += f" --tessdata-dir {tessdata_dir}"
    text = pytesseract.image_to_string(pil_img, lang=lang, config=cfg)
    data = pytesseract.image_to_data(pil_img, lang=lang, config=cfg,
                                     output_type=pytesseract.Output.DICT)
    return text, data

def match_sensitive(text: str) -> Dict[str, List[str]]:
    found = {"headers": [], "fields": [], "regex": []}
    t = text  # 그대로 포함 검색 (한국어 경계 이슈 방지)

    for k in HEADER_KEYWORDS:
        if k in t:
            found["headers"].append(k)
    for k in FIELD_KEYWORDS:
        if k in t:
            found["fields"].append(k)
    for name, rx in REGEX_PATTERNS.items():
        if rx.search(t):
            found["regex"].append(name)
    return found

def words_to_mask_by_keywords(ocr_data: Dict, keywords: List[str]) -> List[Tuple[int,int,int,int]]:
    """Return word boxes to mask if word text contains any keyword (very literal)."""
    boxes = []
    texts = ocr_data.get("text", [])
    lefts = ocr_data.get("left", [])
    tops = ocr_data.get("top", [])
    widths = ocr_data.get("width", [])
    heights = ocr_data.get("height", [])
    for i, w in enumerate(texts):
        if not w or not str(w).strip():
            continue
        word = str(w)
        if any(k in word for k in keywords):
            x, y = int(lefts[i]), int(tops[i])
            boxes.append((x, y, x + int(widths[i]), y + int(heights[i])))
    return boxes

# --------------------------
# Redaction
# --------------------------

def redact_image(pil_img: Image.Image, boxes: List[Tuple[int,int,int,int]]) -> Image.Image:
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    for (x1, y1, x2, y2) in boxes:
        draw.rectangle([x1, y1, x2, y2], fill="black")
    return img

def redact_pdf_page(page: fitz.Page, boxes: List[fitz.Rect]):
    for rect in boxes:
        page.add_redact_annot(rect, fill=(0, 0, 0))
    # 실제로 제거
    page.apply_redactions()

# --------------------------
# PDF helpers
# --------------------------

def pdf_page_has_text(page: fitz.Page) -> bool:
    txt = page.get_text("text") or ""
    return bool(txt.strip())

def pdf_search_boxes_for_keywords(page: fitz.Page, keywords: List[str]) -> List[fitz.Rect]:
    rects = []
    for kw in keywords:
        try:
            rects.extend(page.search_for(kw, quads=False))
        except Exception:
            pass
    return rects

# --------------------------
# Page policy
# --------------------------

def pick_pages(total: int, policy: str, first_n: int = 3, last_n: int = 1) -> List[int]:
    if policy == "all":
        return list(range(total))
    if policy == "sample":
        picked = list(range(min(first_n, total)))
        last_idx = total - 1
        if last_idx >= 0 and last_idx not in picked:
            picked.append(last_idx)
        return picked
    # "firstN+lastN"
    picked = list(range(min(first_n, total)))
    for j in range(1, last_n+1):
        idx = total - j
        if 0 <= idx < total and idx not in picked:
            picked.append(idx)
    picked.sort()
    return picked

# --------------------------
# Main handlers
# --------------------------

def handle_image(path: Path, args, east_net):
    # Load and quick rules
    pil = Image.open(path).convert("RGB")

    # Skip tiny resolution
    if mpixels_of_img(pil) < args.min_mp:
        return {"file": str(path), "skipped": "small_resolution"}

    # Quick text detection (EAST)
    if not east_has_text(east_net, pil, conf_thresh=args.east_conf):
        return {"file": str(path), "skipped": "no_text_detected"}

    # Preprocess → OCR
    if not args.no_pre:
        pil_for_ocr = preprocess_for_ocr(pil, do_deskew=not args.no_deskew)
    else:
        pil_for_ocr = pil
    text, data = tesseract_ocr(pil_for_ocr, lang=args.lang, psm=args.psm,
                               oem=args.oem, tessdata_dir=args.tessdata)

    found = match_sensitive(text)

    # Save outputs
    stem = path.stem
    outdir = Path(args.out); ensure_dir(outdir)
    if args.save in ("txt", "both"):
        (outdir / f"{stem}.txt").write_text(text, encoding="utf-8")
    if args.save in ("json", "both"):
        (outdir / f"{stem}.json").write_text(json.dumps({"text": text, "ocr": data, "found": found}, ensure_ascii=False, indent=2), encoding="utf-8")

    # Redaction (optional)
    if args.redact:
        boxes = []
        if args.redact == "all":
            # mask all words
            boxes = words_to_mask_by_keywords(data, [w for w in data.get("text", []) if str(w).strip()])
        elif args.redact == "keywords":
            keys = list(set(HEADER_KEYWORDS + FIELD_KEYWORDS))
            boxes = words_to_mask_by_keywords(data, keys)
        if boxes:
            red = redact_image(pil, boxes)
            red.save(outdir / f"{stem}.redacted.png")

    return {"file": str(path), "status": "ok", "found": found}

def handle_pdf(path: Path, args, east_net):
    outdir = Path(args.out); ensure_dir(outdir)
    results = {"file": str(path), "pages": []}

    with fitz.open(path) as doc:
        total = len(doc)
        pages = pick_pages(total, args.page_policy, args.first_n, args.last_n)

        # Pass 1: redact text-layer pages immediately if keywords present (no OCR)
        if args.redact:
            # We'll create a working copy in memory then write out
            work = fitz.open(stream=doc.tobytes(), filetype="pdf")
            for i in range(total):
                p = work.load_page(i)
                if pdf_page_has_text(p):
                    # search for keywords in PDF text layer
                    rects = pdf_search_boxes_for_keywords(p, list(set(HEADER_KEYWORDS + FIELD_KEYWORDS)))
                    if rects:
                        redact_pdf_page(p, rects)
            # save as *_textlayer.redacted.pdf
            work.save(outdir / f"{path.stem}.textlayer.redacted.pdf", deflate=True)
            work.close()

        # Pass 2: OCR only selected pages that have NO text layer & (EAST says text-like) & not too small
        for pi in tqdm(pages, desc=f"OCR {path.name}", unit="page"):
            page = doc.load_page(pi)
            has_text = pdf_page_has_text(page)
            rec = {"page": pi+1, "skipped": None}

            if has_text:
                rec["skipped"] = "pdf_text_layer_present"
                results["pages"].append(rec); continue

            # Render raster for EAST + OCR
            zoom = args.dpi / 72.0
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # 0.3MP skip
            if mpixels_of_img(pil) < args.min_mp:
                rec["skipped"] = "small_resolution"
                results["pages"].append(rec); continue

            # EAST quick check
            if not east_has_text(east_net, pil, conf_thresh=args.east_conf):
                rec["skipped"] = "no_text_detected"
                results["pages"].append(rec); continue

            # OCR
            if not args.no_pre:
                pil_for_ocr = preprocess_for_ocr(pil, do_deskew=not args.no_deskew)
            else:
                pil_for_ocr = pil

            text, data = tesseract_ocr(pil_for_ocr, lang=args.lang, psm=args.psm,
                                       oem=args.oem, tessdata_dir=args.tessdata)
            found = match_sensitive(text)
            rec["found"] = found

            # Save page-wise outputs
            base = f"{path.stem}_p{pi+1:04d}"
            if args.save in ("txt", "both"):
                (outdir / f"{base}.txt").write_text(text, encoding="utf-8")
            if args.save in ("json", "both"):
                (outdir / f"{base}.json").write_text(json.dumps({"text": text, "ocr": data, "found": found}, ensure_ascii=False, indent=2), encoding="utf-8")

            # Redaction on raster (optional) – produced as PNG snapshot of that page
            if args.redact:
                boxes = []
                if args.redact == "all":
                    boxes = words_to_mask_by_keywords(data, [w for w in data.get("text", []) if str(w).strip()])
                elif args.redact == "keywords":
                    keys = list(set(HEADER_KEYWORDS + FIELD_KEYWORDS))
                    boxes = words_to_mask_by_keywords(data, keys)
                if boxes:
                    red = redact_image(pil, boxes)
                    red.save(outdir / f"{base}.redacted.png")

            results["pages"].append(rec)

    return results

# --------------------------
# CLI
# --------------------------

def main():
    ap = argparse.ArgumentParser(description="Smart selective OCR with EAST + Tesseract + Redaction")
    ap.add_argument("input", type=str, help="file or directory")
    ap.add_argument("--out", type=str, default="./out", help="output dir")
    ap.add_argument("--lang", type=str, default="kor+eng", help="tesseract languages")
    ap.add_argument("--psm", type=int, default=3, help="tesseract PSM (3:auto, 6:block, 11:sparse)")
    ap.add_argument("--oem", type=int, default=1, help="tesseract OEM (1:LSTM only, 3:default)")
    ap.add_argument("--tessdata", type=str, default=None, help="tessdata dir")
    ap.add_argument("--tess-cmd", type=str, default=None, help="tesseract executable path, if needed")

    ap.add_argument("--east-model", type=str, default=None, help="frozen_east_text_detection.pb path")
    ap.add_argument("--east-conf", type=float, default=EAST_CONF_THRESH, help="EAST confidence threshold")
    ap.add_argument("--min-mp", type=float, default=MIN_MP, help="skip images below this megapixel")

    ap.add_argument("--dpi", type=int, default=220, help="PDF rasterize DPI")
    ap.add_argument("--page-policy", type=str, default="sample", choices=["all", "sample", "firstN+lastN"], help="which pages to OCR")
    ap.add_argument("--first-n", type=int, default=3, help="for sample/firstN+lastN")
    ap.add_argument("--last-n", type=int, default=1, help="for firstN+lastN")

    ap.add_argument("--save", type=str, default="both", choices=["txt", "json", "both"], help="what to save")
    ap.add_argument("--redact", type=str, default=None, choices=[None, "keywords", "all"], help="draw black boxes (images) / true redaction (PDF text-layer pages)")

    ap.add_argument("--no-pre", action="store_true", help="disable preprocessing")
    ap.add_argument("--no-deskew", action="store_true", help="disable deskew")
    ap.add_argument("--content-type", type=str, default=None, help="override MIME if known from network (e.g., image/png, application/pdf)")

    args = ap.parse_args()

    set_tesseract_cmd(args.tess_cmd)

    east_net = load_east(args.east_model)

    in_path = Path(args.input)
    files: List[Path] = []
    if in_path.is_dir():
        for p in in_path.rglob("*"):
            if p.is_file():
                files.append(p)
    else:
        files.append(in_path)

    ensure_dir(Path(args.out))

    summary = []
    for f in files:
        try:
            # Decide by content-type (packet header → content_type)
            mime = args.content_type or sniff_mime(f)

            if is_pdf_mime(mime):
                res = handle_pdf(f, args, east_net)
                summary.append(res)
            elif is_image_mime(mime):
                res = handle_image(f, args, east_net)
                summary.append(res)
            else:
                summary.append({"file": str(f), "skipped": f"unsupported_mime:{mime}"})
        except Exception as e:
            summary.append({"file": str(f), "error": str(e)})

    # Write run summary
    (Path(args.out) / "_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
