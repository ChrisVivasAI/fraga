import fitz
from PIL import Image
import streamlit as st
import io
from typing import List, Tuple, Dict, Optional, Any
import base64
import logging
import cv2
import numpy as np
import json
import time
import uuid  # Import uuid for unique filenames

from .image_db import (
    store_image_in_db,
    get_images_from_db,
    update_image_analysis,
    check_images_processed,
)
from .image_utils import extract_embedded_images, render_full_page

# Import shared resources for Ollama client and model name
from .shared_resources import (
    get_ollama_client,
    get_ollama_model_name,
    get_supabase_client
)

logger = logging.getLogger(__name__)
ollama_client = get_ollama_client()
ollama_model_name = get_ollama_model_name()

supabase = get_supabase_client()

# Define bucket name (make sure this matches your Supabase bucket)
STORAGE_BUCKET_NAME = 'project-images'

# Remove caching for this version as direct extraction is fast
# @st.cache_data(ttl=3600, show_spinner=False)
def extract_images_from_pdf(pdf_file: io.BytesIO, min_area_ratio=0.02, dpi=150) -> List[Tuple[Image.Image, bytes, Dict[str, Any], int]]:
    """Extracts images directly embedded in the PDF using Fitz and also creates full-page renderings.
       Args:
           pdf_file: BytesIO object of the PDF file.
           min_area_ratio: Not used in this method.
           dpi: Dots per inch for rendering full-page snapshots.
       
       Returns:
           List of tuples containing (PIL Image, bytes data, position info, page number)
           for both embedded images and full-page renderings.
    """
    extracted_data = []
    try:
        pdf_file.seek(0)
        pdf_content = pdf_file.read()
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        logger.info(f"Processing PDF with {pdf_document.page_count} pages for image extraction (embedded + full page renders at {dpi} DPI).")

        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]

            # Embedded images
            extracted_data.extend(extract_embedded_images(page, pdf_document, page_num))

            # Full page render
            try:
                logger.info(f"Rendering full page {page_num + 1} at {dpi} DPI...")
                extracted_data.append(render_full_page(page, page_num, dpi))
            except Exception as page_render_err:
                logger.error(f"Error rendering full page {page_num + 1}: {page_render_err}", exc_info=True)

        pdf_document.close()
        logger.info(f"Image extraction (embedded + full page) finished. Found {len(extracted_data)} images in total.")
        return extracted_data
        
    except Exception as e:
        # Log general errors during PDF processing
        st.error(f"Error during PDF processing for image extraction: {str(e)}")
        logger.error(f"Error during direct image extraction process: {e}", exc_info=True)
        return []

# Process PDF and store images in the database
def process_pdf_images(pdf_file: io.BytesIO, project_id: int) -> bool:
    """
    Process a PDF file, extract images, and store them in the database.
    
    Args:
        pdf_file: PDF file to process
        project_id: Project ID to associate with the images
        
    Returns:
        True if processing was successful, False otherwise
    """
    file_name = getattr(pdf_file, 'name', 'Unknown File')
    
    # Check if images for this file are already processed
    if check_images_processed(project_id, file_name):
        logger.info(f"Images already processed for project {project_id}, file {file_name}. Skipping extraction.")
        return True
        
    # Extract images from the PDF
    with st.spinner(f"Extracting images from {file_name}..."):
        extracted_images_data = extract_images_from_pdf(pdf_file)
    
    if not extracted_images_data:
        logger.info(f"No images found in file: {file_name}")
        return False
        
    # Store images in the database
    with st.spinner(f"Storing {len(extracted_images_data)} images in database..."):
        stored_count = 0
        for _, img_bytes, position_data, page_num in extracted_images_data:
            image_id = store_image_in_db(
                project_id=project_id,
                file_name=file_name,
                image_data=img_bytes,
                page_num=page_num,
                position_data=position_data
            )
            if image_id:
                stored_count += 1
                
        logger.info(f"Stored {stored_count}/{len(extracted_images_data)} images in database for project {project_id}, file {file_name}")
        return stored_count > 0

# --- Unified Image Analysis Function --- 
def analyze_image(
    image_id: Optional[int] = None,
    image_base64: Optional[str] = None,
    image_url: Optional[str] = None
) -> str:
    """
    Sends image data to the selected AI model (Ollama or Gemini) for analysis.
    Prioritizes using the image URL over base64 data.
    """
    
    model_preference = st.session_state.get('model_preference', 'local') # Default to local
    google_api_key = st.session_state.get('google_api_key', None)
    
    # Get the image data - prioritize downloading from URL
    image_bytes = None
    
    # Try to get image from URL first (most reliable)
    if image_url:
        try:
            import requests
            logger.info(f"Downloading image from URL for analysis: {image_url}")
            response = requests.get(image_url)
            if response.status_code == 200:
                image_bytes = response.content
                logger.info(f"Successfully downloaded image from URL ({len(image_bytes)} bytes)")
            else:
                logger.error(f"Failed to download image from URL: HTTP {response.status_code}")
        except Exception as dl_err:
            logger.error(f"Error downloading image from URL: {dl_err}")
    
    # If URL didn't work, try base64 data
    if not image_bytes and image_base64:
        try:
            # Clean the base64 string
            image_base64 = image_base64.strip()
            
            # Check if the length is valid for base64 (multiple of 4)
            padding_needed = len(image_base64) % 4
            if padding_needed > 0:
                padding_to_add = 4 - padding_needed
                image_base64 += '=' * padding_to_add
                
            # Filter invalid characters
            valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=')
            if not all(c in valid_chars for c in image_base64):
                image_base64 = ''.join(c for c in image_base64 if c in valid_chars)
                # Re-pad if needed after filtering
                padding_needed = len(image_base64) % 4
                if padding_needed > 0:
                    padding_to_add = 4 - padding_needed
                    image_base64 += '=' * padding_to_add
            
            # Decode the base64 data
            image_bytes = base64.b64decode(image_base64)
            logger.info(f"Successfully decoded base64 data ({len(image_bytes)} bytes)")
        except Exception as b64_err:
            logger.error(f"Error decoding base64 data: {b64_err}")
    
    # If we couldn't get image data from either source
    if not image_bytes:
        # Try to fetch image URL from database if we have the ID
        if image_id is not None and supabase and not image_url:
            try:
                logger.info(f"Attempting to retrieve image URL for ID {image_id}")
                response = supabase.table('images').select('image_url').eq('id', image_id).execute()
                
                if hasattr(response, 'data') and response.data and len(response.data) > 0:
                    fetched_url = response.data[0].get('image_url')
                    if fetched_url:
                        logger.info(f"Found image URL {fetched_url}, attempting to download")
                        # Try to download the image from the URL
                        import requests
                        img_response = requests.get(fetched_url)
                        if img_response.status_code == 200:
                            image_bytes = img_response.content
                            logger.info(f"Successfully downloaded image ({len(image_bytes)} bytes)")
                        else:
                            logger.error(f"Failed to download image: HTTP {img_response.status_code}")
            except Exception as fetch_err:
                logger.error(f"Failed to retrieve image: {fetch_err}")
    
    # If we still don't have image data, return an error
    if not image_bytes:
        st.error("Failed to retrieve image data for analysis.")
        return "Error: Could not obtain valid image data from any source. Please try uploading the image again."
    
    # Validate the image data
    try:
        # Try to open the image with PIL to validate it
        pil_img = Image.open(io.BytesIO(image_bytes))
        width, height = pil_img.size
        logger.info(f"Successfully validated image data: {width}x{height}, format: {pil_img.format}")
        
        # Convert the image to PNG format for consistency
        img_byte_arr = io.BytesIO()
        pil_img.save(img_byte_arr, format='PNG')
        image_bytes = img_byte_arr.getvalue()
    except Exception as img_err:
        logger.error(f"Invalid image data: {img_err}")
        return "Error: The image data is corrupted or in an unsupported format."

    # --- Prepare Prompt (Common for both models) --- 
    prompt_text = """Analyze this architectural/engineering plan image in detail. 

Please identify and explain:
1. The type of plan or drawing (floor plan, elevation, section, detail, structural, electrical, mechanical, etc.)
2. Key architectural or engineering elements visible
3. Any visible dimensions, measurements, or scale indicators
4. Material specifications or annotations
5. Any building codes or standards referenced
6. Structural components or systems shown
7. MEP (Mechanical, Electrical, Plumbing) elements if present
8. Any construction details or assembly instructions
9. Potential compliance issues or special requirements

Structure your analysis clearly and focus on technical accuracy. Provide as much specific detail as possible about the elements shown in the drawing.

Analysis:"""
    
    analysis_result = "Error: Analysis could not be completed."
    model_used_info = "Unknown"

    try:
        # Prepare the image in appropriate format for AI model
        if model_preference == 'local':
            # --- Ollama Logic --- 
            model_used_info = f"local Ollama ({ollama_model_name})"
            if not ollama_client:
                logger.error("Ollama client not available for image analysis.")
                return "Error: Ollama connection not available."
            
            # Convert image bytes to base64 for Ollama
            image_b64_for_api = base64.b64encode(image_bytes).decode('utf-8')
            
            target_model = "gemma3:4b" # Keep using the tag specified by the user
            logger.info(f"Sending image to {model_used_info} via GENERATE endpoint for analysis...")
            
            response = ollama_client.generate(
                model=target_model,
                prompt=prompt_text,
                images=[image_b64_for_api],
                stream=False
            )
            
            if isinstance(response, dict) and 'response' in response:
                 description = response['response'].strip()
                 analysis_result = description
            else:
                logger.error(f"Unexpected response format from Ollama generate endpoint: {response}")
                analysis_result = "Error: Received unexpected response format from analysis model."

        elif model_preference == 'api':
            # --- Google Gemini API Logic --- 
            model_used_info = "Google Gemini API"
            if not google_api_key:
                logger.warning("Google API Key not provided. Skipping API image analysis.")
                st.warning("Google API Key needed for analysis. Please provide it in the sidebar.")
                return "Error: Google API Key not provided."
            
            try:
                import google.generativeai as genai
                from google.generativeai.types import HarmCategory, HarmBlockThreshold
                
                genai.configure(api_key=google_api_key)
                # Use a model that supports images, like gemini-1.5-flash-latest or gemini-pro-vision
                gemini_model_name = 'gemini-2.5-flash-preview-05-20' 
                model = genai.GenerativeModel(gemini_model_name)
                model_used_info = f"Google Gemini API ({gemini_model_name})"
                
                logger.info(f"Sending image to {model_used_info} for analysis...")
                
                # Create PIL Image from bytes
                img_pil = Image.open(io.BytesIO(image_bytes))
                
                # Make the API call with prompt and image
                response = model.generate_content(
                    [prompt_text, img_pil], # Send prompt text and PIL image object
                    stream=False,
                    # Safety settings
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    }
                )
                
                if response.text:
                    analysis_result = response.text
                else:
                     logger.error(f"Gemini API returned empty response or error. Feedback: {response.prompt_feedback}")
                     analysis_result = f"Error: Analysis failed (Gemini). Feedback: {response.prompt_feedback}"

            except ImportError:
                 logger.error("google.generativeai library not installed. Cannot use Gemini API.")
                 st.error("Google AI library not found. Please install it (`pip install google-generativeai`).")
                 analysis_result = "Error: Google AI library not installed."
            except Exception as api_error:
                 logger.error(f"Error calling Google Gemini API for image analysis: {api_error}", exc_info=True)
                 st.error(f"Error connecting to Google Gemini API: {api_error}")
                 analysis_result = f"Error during Gemini analysis: {api_error}"
        else:
             logger.warning(f"Unknown model preference: {model_preference}. Cannot analyze image.")
             analysis_result = "Error: Invalid model preference selected."

        logger.info(f"Received image description from {model_used_info}. Length: {len(analysis_result)}")
        
        # --- Store Analysis in DB --- (Only if analysis seems successful)
        if image_id is not None and supabase and not analysis_result.startswith("Error:"):
            update_success = update_image_analysis(image_id, analysis_result)
            if update_success:
                logger.info(f"Successfully stored analysis for image ID: {image_id}")
            else:
                logger.warning(f"Failed to store analysis for image ID: {image_id}")
        elif analysis_result.startswith("Error:"):
             logger.warning(f"Skipping DB update for image ID {image_id} due to analysis error.")
        
        return analysis_result

    except Exception as e:
        # Catch-all for unexpected errors during the process
        logger.error(f"Generic error during image analysis (Model: {model_preference}, Image ID: {image_id}): {e}", exc_info=True)
        st.error(f"Unexpected error during image analysis: {e}")
        return f"Error: Unexpected issue during analysis - {e}"

# --- Function to Display Images and Analysis --- (Updated for URL display)
def display_images_and_analysis(images_to_display: List[Dict[str, Any]]):
    """Groups images by file and displays them (via URL) with analysis options."""
    if not images_to_display:
        st.info("No images found for this project/file in the database.")
        return

    logger.info(f"Displaying {len(images_to_display)} images")
    
    # Add a button to fix all images in this project if needed
    project_id = images_to_display[0].get('project_id') if images_to_display else None
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(f"Images ({len(images_to_display)})")
    with col2:
        if project_id and st.button("🔄 Fix Corrupted Images", help="Attempts to fix any corrupted base64 data for images in this project"):
            with st.spinner("Fixing corrupted images..."):
                fixed_count = fix_corrupted_base64_images(project_id=project_id)
                if fixed_count > 0:
                    st.success(f"Fixed {fixed_count} image(s). Refresh the page to see the results.")
                else:
                    st.info("No images needed fixing or the operation couldn't complete.")
    
    images_by_file = {}
    for img in images_to_display:
        file_name = img.get('file_name', 'Unknown File')
        if file_name not in images_by_file:
            images_by_file[file_name] = []
        images_by_file[file_name].append(img)

    for file_name, images in images_by_file.items():
        with st.expander(f"Images from: {file_name}", expanded=True):
            if not images:
                st.info("No images found in this specific file.") 
                continue

            cols = st.columns(3) # Adjust column count if needed
            for idx, img_data in enumerate(images):
                with cols[idx % len(cols)]:
                    image_id = img_data['id']
                    analysis_key = f"analysis_{image_id}" # Key for storing analysis in session state

                    try:
                        # --- Display Image using URL --- 
                        image_url = img_data.get('image_url')
                        if image_url:
                            try:
                                st.image(image_url, caption=f"Page {img_data.get('page_num', 'N/A')}", use_container_width=True)
                                logger.debug(f"Successfully displayed image ID {image_id} from URL: {image_url}")
                            except Exception as display_err:
                                st.error(f"Failed to display image from URL: {str(display_err)[:100]}")
                                logger.error(f"Error displaying image ID {image_id} from URL {image_url}: {display_err}", exc_info=True)
                        else:
                            st.error("Image URL missing")
                            logger.warning(f"No image URL found for image ID {image_id}")
                        
                        # --- Display Metadata --- (Remains the same)
                        st.markdown(f"**Image Info:**")
                        st.caption(f"Source File: {img_data.get('file_name', 'N/A')}\nPage: {img_data.get('page_num', 'N/A')}\nDB ID: {image_id}")
                        st.markdown("---") # Separator
                        
                        # --- Analysis Section --- 
                        current_analysis = img_data.get('analysis')
                        if st.session_state.get(analysis_key):
                            current_analysis = st.session_state[analysis_key]

                        if current_analysis:
                            st.markdown("**Analysis:**")
                            st.markdown(current_analysis)

                        # Analysis button (Uses image_data_b64)
                        button_key = f"analyze_img_{image_id}"
                        if st.button("Analyze Image", key=button_key):
                             # Get both the image URL and base64 for analysis
                            image_url = img_data.get('image_url')
                            image_b64_string = img_data.get('image_data_b64') 
                            
                            if not image_url and not image_b64_string:
                                st.error("Cannot analyze: Both image URL and base64 data are missing.")
                                logger.error(f"Analysis button clicked for ID {image_id}, but both image_url and image_data_b64 are missing.")
                            else:
                                with st.spinner("Analyzing image..."):
                                    try:
                                        # Call the unified analysis function with both URL and base64
                                        description = analyze_image(
                                            image_id=image_id,
                                            image_base64=image_b64_string,
                                            image_url=image_url
                                        )
                                        # Store result in session state to update display immediately
                                        st.session_state[analysis_key] = description
                                        # Rerun to show the updated analysis immediately
                                        st.rerun()
                                    except Exception as analysis_err:
                                        st.error(f"Analysis failed: {str(analysis_err)[:100]}")
                                        logger.error(f"Analysis failed for image {image_id}: {analysis_err}", exc_info=True)
                        elif not current_analysis:
                            st.caption("Click 'Analyze Image' to generate analysis.")

                    except Exception as e:
                        st.error(f"Unexpected error displaying info for image ID {image_id}: {e}")
                        logger.error(f"Unexpected error processing image metadata/analysis UI for ID {image_id}: {e}", exc_info=True)

# --- Direct Image Extraction Function (No Database) ---
def extract_images_direct(pdf_file: io.BytesIO) -> List[Tuple[Image.Image, int]]:
    """
    Extract images directly from PDF file without storing in database.
    Similar to the user's original working method.
    
    Args:
        pdf_file: BytesIO object containing PDF data
        
    Returns:
        List of tuples containing (PIL Image, page number)
    """
    images = []
    try:
        pdf_file.seek(0)
        pdf_content = pdf_file.read()
        
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        logger.info(f"Direct extraction: Processing PDF with {pdf_document.page_count} pages")
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            image_list = page.get_images(full=True)
            
            if not image_list:
                continue
                
            logger.info(f"Found {len(image_list)} potential images on page {page_num + 1}")
            
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                if xref == 0:
                    continue
                    
                try:
                    base_image = pdf_document.extract_image(xref)
                    if not base_image or not base_image["image"]:
                        continue
                        
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))
                    images.append((image, page_num))
                    logger.info(f"Successfully extracted image {img_index+1} from page {page_num + 1}")
                except Exception as e:
                    logger.error(f"Error extracting image {img_index} on page {page_num + 1}: {e}")
                    continue
        
        pdf_document.close()
        return images
    except Exception as e:
        logger.error(f"Error in direct PDF image extraction: {e}", exc_info=True)
        return []

# --- Modified render_image_tab --- 
def render_image_tab(uploaded_files: List[io.BytesIO]):
    st.subheader("Extracted Images")
    
    app_mode = st.session_state.get('app_mode', 'Upload')
    project_id = st.session_state.get('current_project_id', None)
    logger.info(f"[DEBUG] render_image_tab: app_mode='{app_mode}', project_id={project_id}") # DEBUG

    # --- Initialize Session State for Analysis Display ---
    if 'image_analysis_state_init' not in st.session_state:
        st.session_state['image_analysis_state_init'] = True

    # --- Mode Handling ---
    if not project_id:
        if app_mode == 'Load':
            st.info("No project loaded. Please select a project from the sidebar.")
        else:
             st.warning("Project context not found. Please ensure files are uploaded correctly.")
        return

    if app_mode == 'Load':
        # Try database method first
        st.info(f"Showing images for Project ID: {project_id}")
        with st.spinner("Loading images from database..."):
            logger.info(f"[DEBUG] Calling get_images_from_db for Load mode (Project ID: {project_id})") # DEBUG
            db_images = get_images_from_db(project_id)
            logger.info(f"[DEBUG] get_images_from_db returned {len(db_images)} images for Load mode.") # DEBUG
            
        # Check if we actually got any valid images
        display_success = False
        if db_images:
            try:
                display_images_and_analysis(db_images)
                display_success = True
            except Exception as e:
                logger.error(f"Error displaying database images: {e}", exc_info=True)
                st.error("Failed to display images from database. Will try direct extraction instead.")
                display_success = False
        
        if not display_success:
            st.warning("Failed to display images from database. Please try uploading the files again.")
        return

    # --- Upload Mode Logic ---
    # This part runs only if app_mode is 'Upload' and project_id exists
    if not uploaded_files:
        st.info("Upload PDF files in the 'Upload' section to extract images.")
        # Try to show existing images
        st.markdown("---")
        st.info(f"Showing existing images for Project ID: {project_id}")
        try:
            with st.spinner("Loading existing images..."):
                logger.info(f"[DEBUG] Calling get_images_from_db for Upload mode (no files uploaded) (Project ID: {project_id})") # DEBUG
                db_images = get_images_from_db(project_id)
                logger.info(f"[DEBUG] get_images_from_db returned {len(db_images)} images for Upload mode (no files)." ) # DEBUG
            display_images_and_analysis(db_images)
        except Exception as e:
            logger.error(f"Error displaying database images: {e}", exc_info=True)
            st.error("Failed to display images from database.")
        return

    # Process files using BOTH the database method AND direct display
    # First try database method (for persistence and analysis features)
    files_processed_this_run = 0
    with st.status("Processing uploaded files...", expanded=True) as status:
        for pdf_file in uploaded_files:
            file_name = getattr(pdf_file, 'name', 'Unknown File')
            st.write(f"Checking: {file_name}")
            logger.info(f"[DEBUG] Checking processing status for file '{file_name}' (Project ID: {project_id})" ) # DEBUG
            if not check_images_processed(project_id, file_name):
                st.write(f"Extracting images from {file_name}...")
                logger.info(f"[DEBUG] Calling process_pdf_images for '{file_name}' (Project ID: {project_id})" ) # DEBUG
                success = process_pdf_images(pdf_file, project_id)
                if success:
                    files_processed_this_run += 1
                    st.write(f"-> Successfully extracted images from {file_name}.")
                else:
                    st.write(f"-> No new images extracted or extraction failed for {file_name}.")
            else:
                st.write(f"-> Images already processed for {file_name}.")
        
        if files_processed_this_run > 0:
            status.update(label=f"Processed and stored images from {files_processed_this_run} new file(s).", state="complete")
        else:
             status.update(label="No new files required processing.", state="complete")

    # Try to display from database first
    st.markdown("---")
    st.info(f"Displaying images for Project ID: {project_id}")
    
    display_from_db_success = True
    try:
        with st.spinner("Loading images from database..."):
            logger.info(f"[DEBUG] Calling get_images_from_db for final display (Project ID: {project_id})") # DEBUG
            all_db_images = get_images_from_db(project_id)
            logger.info(f"[DEBUG] get_images_from_db returned {len(all_db_images)} images for final display.") # DEBUG
            
        if all_db_images:
            display_images_and_analysis(all_db_images)
        else:
            display_from_db_success = False
    except Exception as e:
        logger.error(f"Error displaying database images: {e}", exc_info=True)
        display_from_db_success = False
    
    # If database display fails, fall back to direct display method
    if not display_from_db_success:
        st.warning("Failed to display images from database. Using direct extraction method instead.")
        
        with st.spinner("Extracting images directly from PDFs..."):
            for pdf_file in uploaded_files:
                file_name = getattr(pdf_file, 'name', 'Unknown File')
                st.subheader(f"Images from {file_name}")
                
                # Use our direct extraction method
                direct_images = extract_images_direct(pdf_file)
                
                if not direct_images:
                    st.info(f"No images found in {file_name}")
                    continue
                    
                # Display images in columns
                cols = st.columns(3)
                for idx, (image, page_num) in enumerate(direct_images):
                    with cols[idx % 3]:
                        st.image(image, caption=f"Page {page_num + 1}", use_container_width=True)
                        
                        # Add analyze button, but inform it requires database storage
                        if st.button(f"Analyze (Save to DB first)", key=f"direct_analyze_{file_name}_{idx}"):
                            st.info("Direct images must be saved to database before analysis. Please try the database storage method.")

# --- Function to Fix Corrupted Base64 Data in Database ---
def fix_corrupted_base64_images(project_id: Optional[int] = None, image_id: Optional[int] = None):
    """
    Attempts to fix corrupted base64 data for images in the database.
    Can target a specific image or all images in a project.
    
    Args:
        project_id: Optional project ID to limit scope
        image_id: Optional specific image ID to fix
        
    Returns:
        Number of images fixed
    """
    if not supabase:
        logger.error("Supabase client not available. Cannot fix images.")
        st.error("Database connection not available")
        return 0
        
    try:
        # Build the query to fetch images with URL and ID
        query = supabase.table('images').select('id, image_url')
        
        if image_id:
            query = query.eq('id', image_id)
        elif project_id:
            query = query.eq('project_id', project_id)
            
        response = query.execute()
        
        if not hasattr(response, 'data') or not response.data:
            logger.warning(f"No images found to fix for project {project_id}")
            return 0
            
        fixed_count = 0
        for img in response.data:
            img_id = img.get('id')
            img_url = img.get('image_url')
            
            if not img_url:
                logger.warning(f"No URL found for image ID {img_id}, skipping")
                continue
                
            logger.info(f"Attempting to fix base64 data for image ID {img_id} using URL {img_url}")
            
            try:
                # Download image from URL
                import requests
                img_response = requests.get(img_url)
                
                if img_response.status_code != 200:
                    logger.error(f"Failed to download image from URL {img_url}: HTTP {img_response.status_code}")
                    continue
                    
                # Get image bytes
                img_bytes = img_response.content
                
                # Convert to a standard format
                pil_img = Image.open(io.BytesIO(img_bytes))
                img_byte_arr = io.BytesIO()
                pil_img.save(img_byte_arr, format='PNG')
                clean_img_bytes = img_byte_arr.getvalue()
                
                # Re-encode to base64
                fixed_base64 = base64.b64encode(clean_img_bytes).decode('utf-8')
                
                # Update the database record
                update_response = supabase.table('images').update({
                    'image_data': fixed_base64
                }).eq('id', img_id).execute()
                
                if hasattr(update_response, 'data') and update_response.data:
                    logger.info(f"Successfully fixed base64 data for image ID {img_id}")
                    fixed_count += 1
                else:
                    logger.error(f"Failed to update database for image ID {img_id}")
                    
            except Exception as e:
                logger.error(f"Error fixing base64 data for image ID {img_id}: {e}")
                continue
                
        logger.info(f"Fixed base64 data for {fixed_count}/{len(response.data)} images")
        return fixed_count
        
    except Exception as e:
        logger.error(f"Error in fix_corrupted_base64_images: {e}")
        return 0