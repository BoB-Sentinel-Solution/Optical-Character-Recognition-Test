# Optical-Character-Recognition-Test
OCR 간단 테스트

## 1 이미지
```
python smart_ocr.py .\scan.jpg --east-model .\models\frozen_east_text_detection.pb \
  --lang kor+eng --out .\out
```

#### 결과
결과: out/scan.txt, out/scan.json

텍스트 탐지(EAST) 실패/해상도 0.3MP 미만이면 자동 스킵.

## 2 PDF (앞 3장 + 마지막 1장)
```
python smart_ocr.py .\doc.pdf --east-model .\models\frozen_east_text_detection.pb \
  --page-policy sample --first-n 3 --last-n 1 --dpi 220 --out .\out
```

#### 결과
PDF 페이지에 텍스트 레이어가 있으면 해당 페이지는 OCR 스킵.

OCR한 페이지만 out/doc_p0001.txt/json 형태로 저장.

## 3 폴더 전체(하위 포함) 일괄 처리
```
python smart_ocr.py .\in --east-model .\models\frozen_east_text_detection.pb \
  --out .\out
```

#### 결과
폴더 내 이미지/PDF 자동 판별 → 파일별 결과 생성.

요약: out/_summary.json
