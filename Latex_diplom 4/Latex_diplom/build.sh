#!/bin/bash
# =============================================
#  LaTeX Build Script — алдаагүй compile хийнэ
#  Ашиглах: ./build.sh
# =============================================

set -e  # Алдаа гарвал шууд зогсоно

MAIN="main"
LOG="${MAIN}.log"
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# main.out болон main.aux гэмтсэн байвал устгана
rm -f "${MAIN}.out" "${MAIN}.aux"

echo -e "${YELLOW}[1/4] pdflatex — эхний pass...${NC}"
pdflatex -interaction=nonstopmode -halt-on-error "${MAIN}.tex"
if [ $? -ne 0 ]; then
    echo -e "${RED}✗ pdflatex алдаа гарлаа. Log шалгана уу:${NC}"
    grep -A 3 "^!" "${LOG}" | head -30
    exit 1
fi
echo -e "${GREEN}✓ Эхний pass амжилттай${NC}"

echo -e "${YELLOW}[2/4] biber — ном зүй боловсруулж байна...${NC}"
# main.bcf байхгүй бол pdflatex нэмж ажиллуулна
if [ ! -f "${MAIN}.bcf" ]; then
    echo -e "${YELLOW}⚠ main.bcf байхгүй — pdflatex дахин ажиллуулж байна...${NC}"
    pdflatex -interaction=nonstopmode "${MAIN}.tex" > /dev/null 2>&1
fi
biber "${MAIN}"
if [ $? -ne 0 ]; then
    echo -e "${RED}✗ biber алдаа гарлаа${NC}"
    exit 1
fi
echo -e "${GREEN}✓ biber амжилттай${NC}"

echo -e "${YELLOW}[3/4] pdflatex — хоёрдахь pass...${NC}"
pdflatex -interaction=nonstopmode -halt-on-error "${MAIN}.tex"
if [ $? -ne 0 ]; then
    echo -e "${RED}✗ pdflatex алдаа гарлаа. Log шалгана уу:${NC}"
    grep -A 3 "^!" "${LOG}" | head -30
    exit 1
fi
echo -e "${GREEN}✓ Хоёрдахь pass амжилттай${NC}"

echo -e "${YELLOW}[4/4] pdflatex — гуравдахь pass (cross-reference)...${NC}"
pdflatex -interaction=nonstopmode -halt-on-error "${MAIN}.tex"
if [ $? -ne 0 ]; then
    echo -e "${RED}✗ pdflatex алдаа гарлаа. Log шалгана уу:${NC}"
    grep -A 3 "^!" "${LOG}" | head -30
    exit 1
fi
echo -e "${GREEN}✓ Гуравдахь pass амжилттай${NC}"

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}✓ Build амжилттай! PDF нээж байна...${NC}"
echo -e "${GREEN}============================================${NC}"

# PDF нээх (macOS)
open "${MAIN}.pdf"