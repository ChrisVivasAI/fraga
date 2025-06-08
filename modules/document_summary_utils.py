import logging
import streamlit as st
from .shared_resources import (
    get_ollama_client,
    get_ollama_model_name,
)

logger = logging.getLogger(__name__)
ollama_client = get_ollama_client()
ollama_model_name = get_ollama_model_name()

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_generate_document_summary(text_to_summarize: str, document_name: str) -> str:
    """Generate a detailed summary using the configured AI model."""
    if not text_to_summarize or len(text_to_summarize.strip()) < 100:
        return "Insufficient text to generate a meaningful summary."

    model_preference = st.session_state.get('model_preference', 'local')
    google_api_key = st.session_state.get('google_api_key', None)

    prompt = f"""You are an expert technical analyst tasked with creating a DETAILED INFORMATION EXTRACTION from an architectural or engineering document named "{document_name}".

Your goal is to process the document section by section and extract the most critical information.

Instructions:
1.  First, try to identify the main sections of the document (e.g., Introduction, Specifications, Load Calculations, Material Requirements, Compliance Statements, Conclusion, Appendices, etc.).
2.  For EACH identified section, provide a concise summary of that section's purpose AND extract the most important technical details, data, specifications, measurements, code references, and key findings presented within that section.
3.  Structure your output clearly, perhaps using headings for each section you identify.
4.  Be comprehensive. The goal is NOT a brief overview, but a detailed extraction of core information that would be useful for answering specific technical questions about the document later.
5.  If the document is short or does not have clearly defined sections, then provide a detailed extraction of all key information found.
6.  Preserve all numerical values, dimensions, units, and technical terminology accurately.
7.  Focus on factual information extraction. Avoid interpretation or adding information not present in the text.

Document text:
{text_to_summarize}

DETAILED INFORMATION EXTRACTION:"""

    try:
        if model_preference == 'local':
            if not ollama_client:
                logger.warning("Ollama client not initialized. Cannot generate document summary.")
                return "Summary generation unavailable: Ollama client not initialized."

            if len(text_to_summarize) > 12000:
                beginning = text_to_summarize[:6000]
                end = text_to_summarize[-6000:]
                current_prompt = prompt.replace(text_to_summarize, beginning + "\n\n[...middle content omitted...]\n\n" + end)
            else:
                current_prompt = prompt

            response = ollama_client.chat(
                model="gemma3:4b",
                messages=[{'role': 'user', 'content': current_prompt}],
                stream=False,
            )
            summary = response['message']['content'].strip()
            logger.info(f"Generated detailed information extraction with Ollama for '{document_name}'")
            return summary
        elif model_preference == 'api':
            if not google_api_key:
                logger.warning("Google API Key not provided. Cannot generate document summary.")
                return "Summary generation unavailable: Google API Key not provided."
            try:
                import google.generativeai as genai
                genai.configure(api_key=google_api_key)
                gemini_model_name = 'gemini-1.5-flash-latest'
                model = genai.GenerativeModel(gemini_model_name)
                logger.info(f"Generating document summary for '{document_name}' with Gemini API...")
                if len(text_to_summarize) > 12000:
                    beginning = text_to_summarize[:6000]
                    end = text_to_summarize[-6000:]
                    current_prompt = prompt.replace(text_to_summarize, beginning + "\n\n[...middle content omitted...]\n\n" + end)
                else:
                    current_prompt = prompt
                response = model.generate_content(current_prompt)
                summary = response.text.strip()
                logger.info(f"Generated detailed information extraction with Gemini API for '{document_name}'")
                return summary
            except ImportError:
                logger.error("google.generativeai library not installed. Cannot use Gemini API.")
                return "Summary generation unavailable: Google AI library not installed."
            except Exception as api_error:
                logger.error(f"Error calling Google Gemini API for document summary: {api_error}", exc_info=True)
                return f"Summary generation failed: {api_error}"
        else:
            logger.warning(f"Unknown model preference: {model_preference}. Cannot generate document summary.")
            return "Summary generation unavailable: Unknown model preference."
    except Exception as e:
        logger.error(f"Error generating document summary: {e}", exc_info=True)
        return f"Summary generation failed: {e}"


def generate_document_summary(text: str, document_name: str) -> str:
    """Public wrapper around the cached summary generator."""
    if not text or len(text.strip()) < 100:
        return "Insufficient text to generate a meaningful summary."
    return _cached_generate_document_summary(text, document_name)
