import re
import nltk
import requests
import concurrent.futures
from typing import Dict, List, Union
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import io
import PyPDF2

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

    def extract_pdf_text(self, pdf_url: str) -> str:

        try:
            # Fetch PDF content
            response = requests.get(pdf_url, timeout=10)
            response.raise_for_status()
            
            # Create PDF reader
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Extract text from all pages
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text
        except Exception as e:
            print(f"Error : {pdf_url}: {e}")
            return ""

    def fetch_and_process_resume(self, resume_id: Union[str, int], resume_url: str, job_description_tokens: List[str]) -> tuple:
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

    def match_resume_urls(self, resume_urls: Dict[Union[str, int], str], job_description: str, top_n: int = 5, max_workers: int = 10) -> List[Union[str, int]]:
        
        job_description_tokens = self.preprocess_text(job_description)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.fetch_and_process_resume, 
                    resume_id, 
                    resume_url, 
                    job_description_tokens
                ): resume_id 
                for resume_id, resume_url in resume_urls.items()
            }
            
            # Collect results
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    resume_id, score = future.result()
                    results.append((resume_id, score))
                except Exception as e:
                    print("{e}")
        
        # Sort results by score in descending order
        ranked_resumes = sorted(results, key=lambda x: x[1], reverse=True)
        
        # Return top N resume IDs
        return [resume_id for resume_id, score in ranked_resumes[:top_n]]
    
# Create matcher instance
matcher = ResumeUrlMatcher()

resume_urls = {
        1: "https://nicheby.s3.amazonaws.com/files/CV/29ec0fe9bfa24cf5b76dde8fd53637e2.pdf",
        2: "https://nicheby.s3.amazonaws.com/files/CV/9c1410ac5c4d4b5398fd512402a8d007.pdf"
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
