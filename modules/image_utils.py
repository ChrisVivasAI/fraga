import io
import logging
from typing import List, Tuple, Dict, Any
from PIL import Image
import fitz

logger = logging.getLogger(__name__)


def extract_embedded_images(page: fitz.Page, pdf_document: fitz.Document, page_num: int) -> List[Tuple[Image.Image, bytes, Dict[str, Any], int]]:
    """Extract embedded images from a PDF page."""
    results = []
    image_list = page.get_images(full=True)
    if not image_list:
        return results
    for img_index, img_info in enumerate(image_list):
        xref = img_info[0]
        if xref == 0:
            continue
        try:
            base_image = pdf_document.extract_image(xref)
            if not base_image:
                continue
            image_bytes = base_image["image"]
            pil_image = Image.open(io.BytesIO(image_bytes))
            position_data = {
                "x": 0,
                "y": 0,
                "width": pil_image.width,
                "height": pil_image.height,
                "source": "embedded",
                "xref": xref,
            }
            results.append((pil_image, image_bytes, position_data, page_num))
        except Exception as err:
            logger.error(f"Error extracting embedded image xref {xref} on page {page_num+1}: {err}")
    return results


def render_full_page(page: fitz.Page, page_num: int, dpi: int) -> Tuple[Image.Image, bytes, Dict[str, Any], int]:
    """Render a full PDF page to an image."""
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    page_image_bytes = pix.tobytes("png")
    page_pil_image = Image.open(io.BytesIO(page_image_bytes))
    position_data = {
        "x": 0,
        "y": 0,
        "width": page_pil_image.width,
        "height": page_pil_image.height,
        "source": "full_page_render",
        "xref": 0,
    }
    return page_pil_image, page_image_bytes, position_data, page_num
