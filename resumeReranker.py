import re
import nltk
import requests
from typing import Dict, List, Union
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import io
import PyPDF2
import pytesseract
from pdf2image import convert_from_bytes
import logging

# Set path to Tesseract OCR (only required for Windows users)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Adjust the path as needed

class WarningLogger:
    def __init__(self):
        self.warnings = []

    def write(self, message):
        if message.strip():  # Only store non-empty messages
            self.warnings.append(message)

class ResumeUrlMatcher:
    def __init__(self):
        # Download NLTK resources if not already available
        try:
            stopwords.words('english')
        except LookupError:
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
        
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text: str) -> List[str]:
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        
        return tokens

    def extract_text_from_image(self, pdf_url: str) -> str:
        """
        Function to extract text from a PDF containing scanned images using OCR.
        :param pdf_url: URL of the PDF to process
        :return: Extracted text from the PDF
        """
        try:
            # Fetch PDF content
            response = requests.get(pdf_url, timeout=10)
            response.raise_for_status()

            # Convert PDF to images using convert_from_bytes (since the PDF content is in memory)
            images = convert_from_bytes(response.content)

            # Apply OCR to each image and extract text
            text = ""
            for img in images:
                text += pytesseract.image_to_string(img) + "\n"

            return text
        except Exception as e:
            print(f"Error during OCR processing for {pdf_url}: {e}")
            return ""

    def extract_pdf_text(self, pdf_url: str) -> str:
        """
        Extracts text from a PDF file. If the PDF contains images, it attempts OCR.
        :param pdf_url: URL of the PDF to extract text from
        :return: Extracted text
        """
        try:
            # Fetch PDF content
            response = requests.get(pdf_url, timeout=10)
            response.raise_for_status()
            
            # Create warning logger
            warning_logger = WarningLogger()
            logging.getLogger('PyPDF2').handlers = []
            logging.getLogger('PyPDF2').addHandler(logging.StreamHandler(warning_logger))
            
            # Create PDF reader
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Extract text from all pages
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            # Check if any warnings were logged
            if warning_logger.warnings:
                print(f"PDF warnings detected for {pdf_url}, using OCR...")
                return self.extract_text_from_image(pdf_url)
            
            # If no text was extracted, try OCR
            if not text.strip():
                print(f"No text found in {pdf_url}, using OCR...")
                return self.extract_text_from_image(pdf_url)
            
            return text
            
        except Exception as e:
            print(f"Error extracting text from {pdf_url}: {e}")
            return ""

    def fetch_and_process_resume(self, resume_id: Union[str, int], resume_url: str, job_description_tokens: List[str]) -> tuple:
        # print(f"Processing resume ID: {resume_id}")
        try:
            resume_text = self.extract_pdf_text(resume_url)
            
            if not resume_text:
                return (resume_id, 0)
            
            resume_tokens = self.preprocess_text(resume_text)
            
            bm25 = BM25Okapi([resume_tokens])
            score = bm25.get_scores(job_description_tokens)[0]
            
            return (resume_id, score)
        
        except Exception as e:
            print(f"Error processing resume {resume_id}: {e}")
            return (resume_id, 0)

    def match_resume_urls(self, resume_urls: Dict[Union[str, int], str], job_description: str, top_n: int = None) -> List[Union[str, int]]:
        job_description_tokens = self.preprocess_text(job_description)
        
        results = []
        for resume_id, resume_url in resume_urls.items():
            resume_id, score = self.fetch_and_process_resume(resume_id, resume_url, job_description_tokens)
            results.append((resume_id, score))
        
        # Sort results by score in descending order
        ranked_resumes = sorted(results, key=lambda x: x[1], reverse=True)
        
        # Return all resumes if top_n is None, otherwise return top N resume IDs
        if top_n is None:
            return [resume_id for resume_id, score in ranked_resumes]
        return [resume_id for resume_id, score in ranked_resumes[:top_n]]
    
# Create matcher instance
matcher = ResumeUrlMatcher()

resume_urls = {
    "281": "https://nicheby.s3.amazonaws.com/files/CV/1e88f7ae1d1e4ae08b2a92152efd4825.pdf",
    "304": "https://nicheby.s3.amazonaws.com/files/CV/857b81bcee1948efa1224ac5773e2c31.pdf",
    "1": "https://nicheby.s3.amazonaws.com/files/CV/29ec0fe9bfa24cf5b76dde8fd53637e2.pdf",
    "2": "https://nicheby.s3.amazonaws.com/files/CV/9c1410ac5c4d4b5398fd512402a8d007.pdf",
    "3": "https://nicheby.s3.amazonaws.com/files/CV/e358968549a742b7b859ac86e29b1b02.pdf",
}

# Job description
job_description = """
    We're seeking a Python developer with strong experience in machine learning, 
    data analysis, and cloud computing. Ideal candidates should have expertise 
    in pandas, scikit-learn, and AWS cloud services.
    """

# Match and rank resumes
top_resume_ids = matcher.match_resume_urls(resume_urls, job_description)

print("Top Matching Resume IDs:", top_resume_ids)