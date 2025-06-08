import io
import json
import base64
import uuid
import logging
from typing import Dict, Any, List, Optional
from PIL import Image
import streamlit as st

from .shared_resources import get_supabase_client

logger = logging.getLogger(__name__)
supabase = get_supabase_client()
STORAGE_BUCKET_NAME = 'project-images'


def store_image_in_db(project_id: int, file_name: str, image_data: bytes, page_num: int, position_data: Dict[str, Any]) -> Optional[int]:
    """Upload an image to Supabase storage and record metadata in the database."""
    if not supabase or not image_data:
        logger.error("Supabase client or image data missing.")
        return None

    image_url = None
    unique_filename = f"project_{project_id}/image_{uuid.uuid4()}.png"
    try:
        pil_img = Image.open(io.BytesIO(image_data))
        img_byte_arr = io.BytesIO()
        pil_img.save(img_byte_arr, format='PNG')
        clean_img_bytes = img_byte_arr.getvalue()
        supabase.storage.from_(STORAGE_BUCKET_NAME).upload(unique_filename, clean_img_bytes)
        image_url = supabase.storage.from_(STORAGE_BUCKET_NAME).get_public_url(unique_filename)
    except Exception as err:
        logger.error(f"Failed to upload image: {err}")
        return None

    try:
        image_data_b64 = base64.b64encode(clean_img_bytes).decode('utf-8')
    except Exception:
        image_data_b64 = base64.b64encode(image_data).decode('utf-8')

    db_payload = {
        "project_id": project_id,
        "file_name": file_name,
        "image_url": image_url,
        "image_data": image_data_b64,
        "page_num": page_num,
        "position_data": json.dumps(position_data),
    }
    try:
        response = supabase.table('images').insert(db_payload).execute()
        if hasattr(response, 'data') and response.data:
            image_id = response.data[0]['id']
            logger.info(f"Stored image record ID: {image_id} for project {project_id}")
            return image_id
    except Exception as e:
        logger.error(f"Error storing image metadata: {e}")
    return None


def get_images_from_db(project_id: int, file_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve stored image metadata for a project."""
    if not supabase:
        logger.error("Supabase client not available.")
        return []
    try:
        query = supabase.table('images').select('id, project_id, file_name, page_num, position_data, analysis, image_url, image_data').eq('project_id', project_id)
        if file_name:
            query = query.eq('file_name', file_name)
        response = query.execute()
        if hasattr(response, 'data') and response.data:
            images = []
            for record in response.data:
                record['image_data_b64'] = record.pop('image_data', None)
                if isinstance(record.get('position_data'), str):
                    record['position_data'] = json.loads(record['position_data'])
                images.append(record)
            return images
    except Exception as e:
        logger.error(f"Error retrieving image metadata: {e}")
    return []


def update_image_analysis(image_id: int, analysis: str) -> bool:
    """Update the analysis text for a stored image."""
    if not supabase:
        return False
    try:
        response = supabase.table('images').update({"analysis": analysis}).eq('id', image_id).execute()
        return hasattr(response, 'data') and bool(response.data)
    except Exception as e:
        logger.error(f"Error updating image analysis: {e}")
        return False


def check_images_processed(project_id: int, file_name: str) -> bool:
    """Check if images for this file have already been processed."""
    if not supabase:
        return False
    try:
        response = supabase.table('images').select('id', count='exact').eq('project_id', project_id).eq('file_name', file_name).execute()
        return hasattr(response, 'count') and response.count > 0
    except Exception as e:
        logger.error(f"Error checking processed images: {e}")
        return False
