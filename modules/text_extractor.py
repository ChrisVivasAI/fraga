import pytesseract
import cv2
import numpy as np
from PIL import Image
import fitz
import streamlit as st
import io
import os
import re
import logging
import hashlib
import time
from typing import List, Dict, Any, Tuple, Optional

from .text_utils import (
    preprocess_image,
    clean_ocr_text,
    preprocess_technical_text,
    chunk_text_with_context,
)
from .document_summary_utils import generate_document_summary

# Import shared resources
from .shared_resources import ( # Use relative import within the package
    get_supabase_client,
    get_sentence_transformer_model,
    get_initialization_errors,
    get_ollama_client,
    get_ollama_model_name
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get resources from shared module
supabase = get_supabase_client()
model = get_sentence_transformer_model()
ollama_client = get_ollama_client()
ollama_model_name = get_ollama_model_name()
initialization_errors = get_initialization_errors()

# Set Tesseract path for Windows
if os.name == 'nt':
    # Verify the path exists before setting it
    tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    else:
        logger.warning(f"Tesseract executable not found at {tesseract_path}. OCR might fail.")
        # Optionally provide instructions to the user via Streamlit
        # st.warning("Tesseract not found. Please install Tesseract-OCR and ensure it's in your PATH or update the path in text_extractor.py")

# --- File Identification Helpers ---
def generate_file_hash(file_bytes: bytes) -> str:
    """Generate a hash to uniquely identify a file by its contents"""
    return hashlib.md5(file_bytes).hexdigest()

def get_file_identifier(pdf_file: io.BytesIO) -> str:
    """Get a unique identifier for a file based on name and content hash"""
    file_name = getattr(pdf_file, 'name', 'Unknown')
    
    # Get current position
    current_pos = pdf_file.tell()
    
    # Go to beginning and read content for hashing
    pdf_file.seek(0)
    file_content = pdf_file.read(8192)  # Read first 8KB for hashing
    file_hash = generate_file_hash(file_content)
    
    # Restore position
    pdf_file.seek(current_pos)
    
    return f"{file_name}_{file_hash[:10]}"

# --- Init Session State for Text Extraction ---
def init_extraction_state():
    """Initialize session state variables for text extraction tracking"""
    if "extraction_state_initialized" not in st.session_state:
        st.session_state.extraction_state_initialized = True
        st.session_state.processed_files = {}  # Dict to track processed files
        st.session_state.file_extraction_results = {}  # Store extraction results
        logger.info("Initialized extraction session state")


# --- Cached Text Enhancement Function ---
@st.cache_data(ttl=3600, show_spinner=False)
def _cached_enhance_text(text_to_enhance: str) -> str:
    """Cached version of text enhancement using either local Ollama or Google Gemini API."""
    
    model_preference = st.session_state.get('model_preference', 'local') # Default to local
    google_api_key = st.session_state.get('google_api_key', None)

    if not text_to_enhance or len(text_to_enhance.strip()) < 20: # Don't enhance very short texts
        return text_to_enhance

    # Enhanced detailed prompt (remains the same for both models)
    prompt = f"""You are an expert assistant specialized in enhancing and correcting text from architectural and engineering plans specifically for building permit applications and structural engineering documentation.

Task: Improve the OCR-extracted text below to enhance readability and accuracy while preserving all technical information. 

Focus on the following:
1. Fix formatting and structure of technical specifications, building codes, and material requirements
2. Standardize measurement formats (e.g., "1'-6"" instead of "1 ft 6 in")
3. Correct technical terminology and specialized engineering vocabulary
4. Ensure building code references are properly formatted (e.g., "ASTM C90" not "ASTM C 90")
5. Preserve all numerical values, dimensions, and technical specifications exactly
6. Format lists, bullet points, and numbered specifications consistently
7. Maintain paragraph breaks and section structure
8. Preserve all compliance statements and regulatory references

IMPORTANT: Do NOT add explanatory text, commentary, or alter the meaning. Return ONLY the enhanced version of the text.

Text to enhance:

{text_to_enhance}"""

    enhanced_text = text_to_enhance # Default to original text

    try:
        if model_preference == 'local':
            # --- Ollama Logic --- 
            if not ollama_client:
                logger.warning("Ollama client not initialized. Skipping local text enhancement.")
                return text_to_enhance
                
            logger.info(f"Sending text (length: {len(text_to_enhance)}) to local Ollama model {ollama_model_name} for enhancement...")
            
            # Chunking logic for Ollama (remains the same)
            if len(text_to_enhance.split()) > 1000:
                chunks = chunk_text_with_context(text_to_enhance)
                enhanced_chunks = []
                for i, chunk in enumerate(chunks):
                    logger.info(f"Ollama: Processing chunk {i+1}/{len(chunks)} ({len(chunk.split())} words)...")
                    chunk_prompt = prompt.replace(text_to_enhance, chunk)
                    response = ollama_client.chat(
                        model="gemma3:4b", 
                        messages=[{'role': 'user', 'content': chunk_prompt}],
                        stream=False
                    )
                    enhanced_chunk = response['message']['content'].strip()
                    enhanced_chunks.append(enhanced_chunk)
                enhanced_text = "\n\n".join(enhanced_chunks)
            else:
                # Process shorter text as a single chunk with Ollama
                response = ollama_client.chat(
                    model="gemma3:4b", 
                    messages=[{'role': 'user', 'content': prompt}],
                    stream=False
                )
                enhanced_text = response['message']['content'].strip()
            
            logger.info(f"Received enhanced text (length: {len(enhanced_text)}) from Ollama.")

        elif model_preference == 'api':
            # --- Google Gemini API Logic --- 
            if not google_api_key:
                logger.warning("Google API Key not provided. Skipping API text enhancement.")
                st.warning("Google API Key needed for enhancement. Please provide it in the sidebar.")
                return text_to_enhance
            
            try:
                import google.generativeai as genai
                genai.configure(api_key=google_api_key)
                gemini_model_name = 'gemini-2.5-flash-preview-05-20' # Or choose another appropriate model
                model = genai.GenerativeModel(gemini_model_name)
                
                logger.info(f"Sending text (length: {len(text_to_enhance)}) to Google Gemini model {gemini_model_name} for enhancement...")
                
                # Chunking logic for Gemini (similar structure)
                if len(text_to_enhance.split()) > 1000: # Adjust threshold if needed for API
                    chunks = chunk_text_with_context(text_to_enhance)
                    enhanced_chunks = []
                    for i, chunk in enumerate(chunks):
                        logger.info(f"Gemini: Processing chunk {i+1}/{len(chunks)} ({len(chunk.split())} words)...")
                        chunk_prompt = prompt.replace(text_to_enhance, chunk)
                        # Note: Gemini API might handle longer prompts directly, adjust if needed
                        response = model.generate_content(chunk_prompt)
                        # TODO: Add more robust error checking for Gemini response (e.g., response.prompt_feedback)
                        enhanced_chunks.append(response.text)
                    enhanced_text = "\n\n".join(enhanced_chunks)
                else:
                    # Process shorter text as a single call with Gemini
                    response = model.generate_content(prompt)
                    # TODO: Add more robust error checking for Gemini response
                    enhanced_text = response.text
                    
                logger.info(f"Received enhanced text (length: {len(enhanced_text)}) from Gemini.")
                
            except ImportError:
                 logger.error("google.generativeai library not installed. Cannot use Gemini API.")
                 st.error("Google AI library not found. Please install it (`pip install google-generativeai`).")
                 return text_to_enhance # Fallback to original text
            except Exception as api_error:
                 logger.error(f"Error calling Google Gemini API: {api_error}", exc_info=True)
                 st.error(f"Error connecting to Google Gemini API: {api_error}")
                 return text_to_enhance # Fallback to original text
        else:
             logger.warning(f"Unknown model preference: {model_preference}. Defaulting to original text.")
             return text_to_enhance

        # Sanity check (remains the same)
        if not enhanced_text or len(enhanced_text) < len(text_to_enhance) * 0.5:
             logger.warning("Enhancement resulted in empty or significantly shorter text. Falling back to original.")
             return text_to_enhance
             
        return enhanced_text
        
    except Exception as e:
        logger.error(f"Generic error during text enhancement ({model_preference}): {e}", exc_info=True)
        return text_to_enhance # Fallback in case of unexpected errors

# --- Text Enhancement Wrapper Function --- 
def enhance_text(text_to_enhance: str) -> str:
    """Uses the configured model (Ollama or Gemini) to enhance extracted OCR text."""
    
    # Check session state for preference (needed for warnings/guidance)
    model_preference = st.session_state.get('model_preference', 'local')
    google_api_key = st.session_state.get('google_api_key', None)

    if model_preference == 'api' and not google_api_key:
        st.warning("API mode selected, but no Google API Key found. Skipping enhancement.")
        return text_to_enhance
        
    if not text_to_enhance or len(text_to_enhance.strip()) < 20:  # Don't enhance very short texts
        return text_to_enhance
    
    # First apply technical preprocessing (remains the same)
    preprocessed_text = preprocess_technical_text(text_to_enhance)
    
    # Use cached version for the actual enhancement (which now handles model choice)
    return _cached_enhance_text(preprocessed_text)

# --- Check if file has been processed ---
def is_file_processed(file_id: str, project_id: int) -> bool:
    """Check if a file has already been processed for the current project"""
    if file_id and project_id:
        processed_key = f"{file_id}_{project_id}"
        return processed_key in st.session_state.processed_files
    return False

# --- Store file processing result ---
def mark_file_as_processed(file_id: str, project_id: int, extracted_text: str) -> None:
    """Mark a file as processed and store its extracted text"""
    if file_id and project_id:
        processed_key = f"{file_id}_{project_id}"
        st.session_state.processed_files[processed_key] = True
        st.session_state.file_extraction_results[processed_key] = extracted_text
        logger.info(f"Marked file {file_id} as processed for project {project_id}")

# --- Get stored extraction result ---
def get_stored_extraction_result(file_id: str, project_id: int) -> Optional[str]:
    """Get previously stored extraction result for a file"""
    if file_id and project_id:
        processed_key = f"{file_id}_{project_id}"
        return st.session_state.file_extraction_results.get(processed_key)
    return None

# --- Supabase Project Creation ---
def _create_supabase_project(project_name: str = "Default Project") -> int | None:
    """Creates a new project entry in Supabase and returns its ID."""
    if not supabase:
        st.error("Supabase client not available. Cannot create project.")
        logger.error("Supabase client not initialized, cannot create project.")
        return None
    try:
        logger.info(f"Creating Supabase project with name: {project_name}")
        response = supabase.table('projects').insert({"name": project_name}).execute()
        
        if hasattr(response, 'data') and response.data and len(response.data) > 0:
            project_id = response.data[0]['id']
            logger.info(f"Successfully created project with ID: {project_id}")
            # Store in session state IMMEDIATELY upon successful creation
            st.session_state.current_project_id = project_id 
            return project_id
        else:
            error_info = getattr(response, 'error', 'Unknown error')
            st.error(f"Failed to create project in Supabase: {error_info}")
            logger.error(f"Failed to create Supabase project. Response: {response}")
            # Ensure session state is clear if creation fails
            if 'current_project_id' in st.session_state: 
                del st.session_state['current_project_id']
            return None
    except Exception as e:
        st.error(f"Error creating Supabase project: {e}")
        logger.error(f"Exception during Supabase project creation: {e}", exc_info=True)
        # Ensure session state is clear if creation fails
        if 'current_project_id' in st.session_state: 
            del st.session_state['current_project_id']
        return None

# Cache embedding storage
@st.cache_data(ttl=1800, show_spinner=False)
def _cached_store_embeddings(project_id: int, chunks_key: str, chunks: List[str], file_name: str) -> bool:
    """Cached version of embedding storage"""
    if not project_id:
        logger.error("_cached_store_embeddings called without project_id.")
        return False
    if not file_name:
        logger.error("_cached_store_embeddings called without file_name.")
        return False
    if not supabase or not model or not chunks:
        return False

    try:
        logger.info(f"Generating embeddings for {len(chunks)} chunks (Project ID: {project_id}, File: {file_name})...")
        embeddings = model.encode(chunks, show_progress_bar=True)
        logger.info(f"Generated {len(embeddings)} embeddings for Project ID: {project_id}, File: {file_name}.")

        data_to_insert = [
            {
                "content": chunk,
                "embedding": embedding.tolist(),
                "project_id": project_id,
                "file_name": file_name
            }
            for chunk, embedding in zip(chunks, embeddings)
        ]

        logger.info(f"Storing {len(data_to_insert)} chunks and embeddings in Supabase (Project ID: {project_id}, File: {file_name})...")
        response = supabase.table('documents').insert(data_to_insert).execute()

        if hasattr(response, 'data') and response.data:
             logger.info(f"Successfully inserted {len(response.data)} items into Supabase for Project ID: {project_id}, File: {file_name}.")
             return True
        elif hasattr(response, 'error') and response.error:
             logger.error(f"Supabase insertion error (Project ID: {project_id}, File: {file_name}): {response.error}")
             return False
        else:
             logger.warning(f"Unexpected Supabase response (Project ID: {project_id}, File: {file_name}): {response}")
             return False

    except Exception as e:
        logger.error(f"Error storing embeddings (Project ID: {project_id}, File: {file_name}): {e}", exc_info=True)
        return False

# --- Modified store_embeddings --- 
def store_embeddings(project_id: int, text_chunks: List[str], file_name: str) -> bool:
    """Generates embeddings and stores them in Supabase, associated with a project_id and file_name."""
    if not project_id:
        st.error("Project ID is missing. Cannot store embeddings.")
        logger.error("store_embeddings called without project_id.")
        return False
    if not file_name:
        st.error("File name is missing. Cannot store embeddings.")
        logger.error("store_embeddings called without file_name.")
        return False
    if not supabase or not model or not text_chunks:
        # Logging handled internally by the check
        return False

    try:
        st.info(f"Generating embeddings for {len(text_chunks)} chunks from summary (Project ID: {project_id}, File: {file_name})...")
        
        # Create a chunks key for caching (based on summary chunks)
        chunks_key = hashlib.md5("".join(text_chunks[:5]).encode()).hexdigest()
        
        # Call cached version, passing file_name
        success = _cached_store_embeddings(project_id, chunks_key, text_chunks, file_name)
        
        if success:
            st.success(f"Successfully stored {len(text_chunks)} summary chunks and embeddings for {file_name}!")
        else:
            st.error(f"Failed to store summary embeddings in the database for {file_name}.")
            
        return success

    except Exception as e:
        logger.error(f"Error in store_embeddings wrapper (Project ID: {project_id}, File: {file_name}): {e}", exc_info=True)
        st.error(f"Error storing summary embeddings for {file_name}: {e}")
        return False

# --- Cached PDF extraction function ---
@st.cache_data(ttl=3600, show_spinner=False)
def _cached_extract_text_from_pdf(
    file_hash: str,
    pdf_content: bytes,
    use_ocr: bool = True,
    enhance_resolution: bool = True
) -> List[str]:
    """
    Cached function to extract text from PDF content.
    Returns a list of text from each page.
    """
    extracted_pages = []
    
    try:
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        num_pages = pdf_document.page_count
        
        for page_num in range(num_pages):
            page = pdf_document[page_num]
            embedded_text = page.get_text().strip()
            page_text = ""

            if len(embedded_text) > 50 and not use_ocr:
                page_text = clean_ocr_text(embedded_text)
            else:
                # OCR process
                try:
                    pix = page.get_pixmap(alpha=False, dpi=300)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    opencv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    processed_img = preprocess_image(opencv_img, enhance_resolution)
                    custom_config = r'--oem 3 --psm 6'
                    ocr_text = pytesseract.image_to_string(processed_img, config=custom_config, lang='eng')
                    ocr_text = clean_ocr_text(ocr_text)
                    if len(ocr_text) < 20 and len(embedded_text) > len(ocr_text):
                        page_text = clean_ocr_text(embedded_text)
                    else:
                        page_text = ocr_text
                except Exception as ocr_error:
                    logger.error(f"OCR Error (File hash: {file_hash}, Page: {page_num + 1}): {ocr_error}")
                    if embedded_text:
                        page_text = clean_ocr_text(embedded_text)
                    else:
                        page_text = ""
            
            if page_text:
                extracted_pages.append(page_text)

        pdf_document.close()
        return extracted_pages
        
    except Exception as e:
        logger.error(f"Error in _cached_extract_text_from_pdf (File hash: {file_hash}): {e}", exc_info=True)
        return []

# Helper utilities for PDF processing
def _read_pdf_content_and_hash(pdf_file: io.BytesIO) -> Tuple[bytes, str]:
    """Return PDF bytes and a hash for caching."""
    pdf_file.seek(0)
    content = pdf_file.read()
    pdf_file.seek(0)
    return content, generate_file_hash(content)


def _enhance_text_if_requested(preprocessed_text: str, enhance: bool, file_name: str, status_container, file_id: str) -> str:
    """Optionally enhance text using the chosen model."""
    full_text = preprocessed_text
    if enhance and preprocessed_text:
        model_used = st.session_state.get('model_preference', 'local')
        status_container.info(f"File: {file_name} - Enhancing text with {model_used.upper()}...")
        enhanced_text = enhance_text(preprocessed_text)
        with status_container.expander(f"Show {model_used.upper()} Enhancement Comparison", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.text_area("Before Enhancement", preprocessed_text, height=200, key=f"before_{file_id}")
            with col2:
                st.text_area("After Enhancement", enhanced_text, height=200, key=f"after_{file_id}")
        if enhanced_text != preprocessed_text:
            logger.info(f"File: {file_name} - Text enhanced by {model_used.upper()}. New length: {len(enhanced_text)} chars.")
            full_text = enhanced_text
        else:
            logger.info(f"File: {file_name} - Text enhancement by {model_used.upper()} did not change the text or failed.")
    elif enhance:
        logger.warning(f"File: {file_name} - Skipping enhancement as no initial text was extracted.")
    return full_text


def _generate_summary_and_embeddings(full_text: str, file_name: str, project_id: int, status_container, process_embeddings: bool) -> bool:
    """Generate summary and, if requested, store embeddings."""
    status_container.info(f"Generating document summary for {file_name}...")
    document_summary = generate_document_summary(full_text, file_name)
    with status_container.expander("Document Summary", expanded=True):
        st.markdown(document_summary)

    embedding_success = False
    if process_embeddings and document_summary and not document_summary.startswith("Summary generation") and supabase and model:
        status_container.info(f"File: {file_name} - Chunking SUMMARY...")
        summary_chunks = chunk_text_with_context(document_summary, chunk_size=250, overlap=50)
        if summary_chunks:
            embedding_success = store_embeddings(project_id, summary_chunks, file_name)
        else:
            status_container.warning(f"File: {file_name} - Summary was too short to chunk.")
            embedding_success = False
    elif not process_embeddings:
        logger.info(f"File: {file_name} - Embedding skipped by settings.")
    elif not document_summary or document_summary.startswith("Summary generation"):
        status_container.warning(f"File: {file_name} - No valid summary generated, skipping embedding.")
    else:
        status_container.warning(f"File: {file_name} - Supabase/Model not ready, skipping embedding.")
    return embedding_success

# --- Modified extract_text_from_single_pdf function ---
def _extract_text_from_single_pdf(
    pdf_file: io.BytesIO, 
    project_id: int, 
    use_ocr=True, 
    enhance_resolution=True, 
    process_embeddings=True,
    enhance_with_gemma=False
) -> Tuple[str, bool]:
    """
    Extracts text from a SINGLE PDF, optionally enhances with Gemma, chunks it,
    generates embeddings, and stores them in Supabase under the given project_id.
    Returns the (potentially enhanced) extracted text and a boolean indicating embedding success.
    """
    full_extracted_text = ""
    embedding_success = False
    file_name = getattr(pdf_file, 'name', 'Unknown File')
    status_container = st.container()

    # Initialize session state for extraction tracking
    init_extraction_state()
    
    # Generate file identifier
    file_id = get_file_identifier(pdf_file)
    
    # Check if this file has already been processed for this project
    if is_file_processed(file_id, project_id):
        logger.info(f"File {file_name} already processed for project {project_id}, using cached result")
        status_container.success(f"Using cached extraction for {file_name}")
        
        # Get stored result
        stored_text = get_stored_extraction_result(file_id, project_id)
        if stored_text:
            return stored_text, True
    
    # If not processed or no stored result, continue with extraction
    try:
        status_container.info(f"Processing file: {file_name}...")
        pdf_content, file_hash = _read_pdf_content_and_hash(pdf_file)
        
        # Use cached extraction
        extracted_pages = _cached_extract_text_from_pdf(
            file_hash=file_hash,
            pdf_content=pdf_content,
            use_ocr=use_ocr,
            enhance_resolution=enhance_resolution
        )
        
        # Join pages and apply basic processing
        initial_extracted_text = "\n\n".join(extracted_pages) if extracted_pages else ""
        
        if not initial_extracted_text:
            status_container.warning(f"No text extracted from {file_name}")
            return "", False
        
        preprocessed_text = preprocess_technical_text(initial_extracted_text)
        full_extracted_text = preprocessed_text
        
        logger.info(f"File: {file_name} - Initial extracted text length: {len(initial_extracted_text)} chars.")
        logger.info(f"File: {file_name} - After basic preprocessing: {len(preprocessed_text)} chars.")

        full_extracted_text = _enhance_text_if_requested(preprocessed_text, enhance_with_gemma, file_name, status_container, file_id)
        
        embedding_success = _generate_summary_and_embeddings(
            full_extracted_text,
            file_name,
            project_id,
            status_container,
            process_embeddings,
        )
        
        # Mark file as processed and store result (storing full text for potential future use, not embedding)
        mark_file_as_processed(file_id, project_id, full_extracted_text)

    except fitz.fitz.FileDataError:
        status_container.error(f"File: {file_name} - Invalid or corrupted PDF file.")
    except pytesseract.TesseractNotFoundError:
         status_container.error("Tesseract not installed or not found. Please check installation.")
         logger.error("Tesseract executable not found.")
    except Exception as e:
        status_container.error(f"File: {file_name} - Error during processing: {str(e)}")
        logger.error(f"Error processing PDF (File: {file_name}): {str(e)}", exc_info=True)

    return full_extracted_text, embedding_success

# --- Modified render_text_tab --- Step 8 (Part 2)
def render_text_tab(uploaded_files: List[io.BytesIO]) -> str:
    """Processes a list of uploaded PDF files (if any), creates project/embeddings,
       or displays info for a loaded project."""
    
    # Initialize session state
    init_extraction_state()
    
    # Determine mode based on session state (set in app.py)
    app_mode = st.session_state.get('app_mode', 'Upload')
    project_id = st.session_state.get('current_project_id', None)
    
    combined_text = ""
    all_texts = []

    col1, col2 = st.columns([3, 1])

    with col2:
        # Only show settings in Upload mode
        if app_mode == 'Upload':
            st.write("Processing Settings")
            force_ocr = st.checkbox("Force OCR", value=False, key="force_ocr_checkbox",
                                   help="Use OCR even if embedded text is found")
            enhance_res = st.checkbox("Enhance Resolution", value=True, key="enhance_res_checkbox",
                                    help="Increase image resolution for better OCR")
            process_embeddings = st.checkbox("Process & Store Embeddings", value=True, key="process_embed_checkbox",
                                    help="Generate embeddings and store in Supabase")
            enhance_with_gemma_cb = st.checkbox("Enhance Text with AI Model", value=True, key="enhance_gemma_checkbox",
                                              help="Use the selected AI model (Ollama/Gemini) to clean/enhance OCR text (can be slow).")
        else:
            st.write("Project Info")
            if project_id:
                st.info(f"Currently loaded Project ID: {project_id}")
            else:
                st.warning("No project selected.")

    with col1:
        # --- Upload Mode Logic ---
        if app_mode == 'Upload':
            st.subheader("Text Extraction & Processing")
            if not uploaded_files:
                st.info("Upload files using the sidebar widget to begin processing.")
                return ""
                
            # Project creation logic
            if project_id is None:
                 st.info("No active project found, creating a new one...")
                 project_name = f"Project - {uploaded_files[0].name}" if uploaded_files else "Default Project"
                 created_id = _create_supabase_project(project_name)
                 if created_id is None:
                     st.error("Failed to create a project. Cannot proceed.")
                     return ""
                 else:
                     project_id = created_id
                     st.success(f"Created and using new project (ID: {project_id}).")
            # Ensure we definitely have a project ID 
            if project_id is None: return "" # Abort if still None

            overall_progress = st.progress(0)
            files_processed_count = 0
            total_files = len(uploaded_files)
            all_texts = []

            st.write(f"Processing {total_files} file(s) for Project ID: {project_id}")

            for pdf_file in uploaded_files:
                file_name = getattr(pdf_file, 'name', f'File {files_processed_count+1}')
                with st.spinner(f"Processing {file_name}..."):
                    extracted_text, _ = _extract_text_from_single_pdf(
                        pdf_file,
                        project_id=project_id, # Pass the confirmed project ID
                        use_ocr=force_ocr,
                        enhance_resolution=enhance_res,
                        process_embeddings=process_embeddings,
                        enhance_with_gemma=enhance_with_gemma_cb
                    )
                    if extracted_text:
                        all_texts.append(extracted_text)
                
                files_processed_count += 1
                overall_progress.progress(files_processed_count / total_files)

            overall_progress.empty() # Clear progress bar
            combined_text = "\n\n--- End of File ---\n\n".join(all_texts)
            st.subheader("Combined Extracted Text (Current Upload)")
            st.text_area("Combined Text Preview", combined_text, height=400)
            logger.info(f"Processing complete for Project ID: {project_id}. Combined text length: {len(combined_text)}")

        # --- Load Mode Logic ---
        elif app_mode == 'Load':
            st.subheader("Project Information")
            if project_id:
                 st.success(f"Loaded Project ID: {project_id}")
                 st.write("You can now use the Q&A tab to chat with the documents associated with this project.")
                 # We are not displaying extracted text in load mode for now.
                 combined_text = "" # Explicitly set to empty
            else:
                 st.warning("No project is currently loaded. Please select one from the sidebar.")
                 combined_text = ""
            
    # Return combined text (relevant for Upload mode, empty for Load mode)
    return combined_text 
