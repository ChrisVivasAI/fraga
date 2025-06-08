import cv2
import numpy as np
import re
import streamlit as st
import logging

logger = logging.getLogger(__name__)

@st.cache_data(ttl=3600, show_spinner=False)
def preprocess_image(image, enhance_resolution=True):
    """Convert to grayscale, binarize, denoise and optionally upscale."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
    if enhance_resolution:
        return cv2.resize(denoised, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    return denoised

@st.cache_data(ttl=3600, show_spinner=False)
def clean_ocr_text(text):
    """Remove common OCR artefacts and normalise whitespace."""
    if not text:
        return ""
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'(?<!\w)([a-zA-Z])(?!\w)', ' ', text)
    text = text.replace('|', 'I').replace('0', 'O')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(\.)([a-z])', r'\1 \2', text)
    return text.strip()

@st.cache_data(ttl=3600, show_spinner=False)
def standardize_measurements(text: str) -> str:
    """Normalise measurement expressions in technical text."""
    if not text:
        return ""
    text = re.sub(r'(\d+)-(\d+)/(\d+)"', lambda m: f"{int(m.group(1)) + int(m.group(2))/int(m.group(3))}\"", text)
    text = re.sub(r'(\d+)\'(\s*)(\d+)"', r"\1'-\3\"", text)
    units = ['mm', 'cm', 'in', 'ft', 'psf', 'psi', 'ksi', 'pcf', 'sq ft', 'kg']
    for unit in units:
        text = re.sub(rf'(\d+)\s+{unit}', rf'\1{unit}', text)
    return text

@st.cache_data(ttl=3600, show_spinner=False)
def standardize_code_references(text: str) -> str:
    """Standardise references to building codes."""
    if not text:
        return ""
    text = re.sub(r'(ASTM|ANSI|ACI|AISI|IBC|IPC)\s+([A-Z])\s+(\d+)', r'\1 \2\3', text)
    text = re.sub(r'(?i)(sec|sect)\.?\s+(\d+\.\d+)', r'Section \2', text)
    return text

@st.cache_data(ttl=3600, show_spinner=False)
def fix_technical_terminology(text: str) -> str:
    """Fix known OCR errors in engineering terminology."""
    if not text:
        return ""
    corrections = {
        "relnforced": "reinforced",
        "concreie": "concrete",
        "concrele": "concrete",
        "structurai": "structural",
        "sieel": "steel",
        "steei": "steel",
        "specificaiions": "specifications",
        "lnsulation": "insulation",
        "lnstallation": "installation",
        "fastenlng": "fastening",
    }
    for error, correction in corrections.items():
        text = re.sub(rf'\b{error}\b', correction, text, flags=re.IGNORECASE)
    return text

@st.cache_data(ttl=3600, show_spinner=False)
def format_specifications(text: str) -> str:
    """Normalise lists and bullet points."""
    if not text:
        return ""
    text = re.sub(r'(?<!\d)(\d+)(?!\d|\.)(\s+[A-Z])', r'\1.\2', text)
    text = re.sub(r'(?<=\n)[\*\-•⦁◦] ?', '• ', text)
    return text

@st.cache_data(ttl=3600, show_spinner=False)
def preprocess_technical_text(text: str) -> str:
    """Apply all technical text cleaning steps."""
    if not text:
        return ""
    text = standardize_measurements(text)
    text = standardize_code_references(text)
    text = fix_technical_terminology(text)
    text = format_specifications(text)
    return text

@st.cache_data(ttl=3600, show_spinner=False)
def chunk_text_with_context(text, chunk_size=1500, overlap=300):
    """Split text into overlapping chunks while keeping context."""
    if not text:
        return []
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = []
    current_size = 0
    for paragraph in paragraphs:
        words = paragraph.split()
        size = len(words)
        if size > chunk_size:
            if current_size > 0:
                chunks.append(" ".join(current_chunk))
                overlap_start = max(0, len(current_chunk) - overlap)
                current_chunk = current_chunk[overlap_start:]
                current_size = len(current_chunk)
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            for sentence in sentences:
                sentence_words = sentence.split()
                sentence_size = len(sentence_words)
                if current_size + sentence_size > chunk_size and current_size > 0:
                    chunks.append(" ".join(current_chunk))
                    overlap_start = max(0, len(current_chunk) - overlap)
                    current_chunk = current_chunk[overlap_start:]
                    current_size = len(current_chunk)
                current_chunk.extend(sentence_words)
                current_size += sentence_size
        else:
            if current_size + size > chunk_size and current_size > 0:
                chunks.append(" ".join(current_chunk))
                overlap_start = max(0, len(current_chunk) - overlap)
                current_chunk = current_chunk[overlap_start:]
                current_size = len(current_chunk)
            current_chunk.extend(words)
            current_size += size
    if current_size > 0:
        chunks.append(" ".join(current_chunk))
    return [chunk for chunk in chunks if len(chunk.split()) > 30]
