import cv2
import numpy as np
import matplotlib.pyplot as plt
import PyPDF2
import pdf2image
import pytesseract
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Adjust the path as needed

@dataclass
class TextRegion:
    start: int
    end: int
    width: int
    is_empty: bool
    word_count: int
    text: str = ""
    relative_width: float = 1.0

def convert_pdf_to_image(pdf_path: str, dpi: int = 300) -> Tuple[np.ndarray, List[Dict]]:
    """
    Convert PDF to image using pdf2image.
    Returns both the image and extracted text with positions.
    """
    # Convert PDF to image
    images = pdf2image.convert_from_path(pdf_path, dpi=dpi)
    image = np.array(images[0])  # Get first page
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    return image

def process_region_with_ocr(image: np.ndarray, region: TextRegion) -> Tuple[int, str]:
    """
    Process a single region with Tesseract OCR.
    Returns word count and extracted text.
    """
    # Extract the region from the image
    region_slice = image[region.start:region.end+1, :]
    
    # Configure Tesseract to output word bounding boxes and text
    custom_config = r'--oem 3 --psm 6'
    
    # Get OCR result
    ocr_text = pytesseract.image_to_string(region_slice, config=custom_config).strip()
    
    # Count words (split by whitespace and filter out empty strings)
    words = [word for word in ocr_text.split() if word.strip()]
    word_count = len(words)
    
    return word_count, ocr_text

def find_regions_with_words(binary_image: np.ndarray) -> List[TextRegion]:
    """
    Find regions and use OCR to count words and extract text.
    """
    empty_rows = np.all(binary_image == 255, axis=1)
    regions = []
    start = 0
    current_type = empty_rows[0]
    
    for i, is_empty in enumerate(empty_rows[1:], 1):
        if is_empty != current_type:
            width = i - start
            
            region = TextRegion(
                start=start,
                end=i - 1,
                width=width,
                is_empty=current_type,
                word_count=0,
                text=""
            )
            
            # If not empty, process with OCR
            if not current_type:
                word_count, extracted_text = process_region_with_ocr(binary_image, region)
                region.word_count = word_count
                region.text = extracted_text
            
            regions.append(region)
            start = i
            current_type = is_empty
    
    # Add final region
    width = len(empty_rows) - start
    final_region = TextRegion(
        start=start,
        end=len(empty_rows) - 1,
        width=width,
        is_empty=current_type,
        word_count=0,
        text=""
    )
    
    if not current_type:
        word_count, extracted_text = process_region_with_ocr(binary_image, final_region)
        final_region.word_count = word_count
        final_region.text = extracted_text
    
    regions.append(final_region)
    
    # Calculate relative widths
    min_width = min(region.width for region in regions if region.width > 0)
    for region in regions:
        region.relative_width = round(region.width / min_width, 1)
    
    return regions

def draw_regions_with_words(image: np.ndarray, regions: List[TextRegion]) -> np.ndarray:
    """
    Enhanced visualization including word counts and extracted text preview.
    """
    color_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    viz_image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    SPACE_COLOR = (200, 240, 200)
    TEXT_COLOR = (240, 200, 200)
    
    for region in regions:
        start = region.start
        end = region.end
        
        # Fill region background
        color = SPACE_COLOR if region.is_empty else TEXT_COLOR
        viz_image[start:end+1, :] = color
        
        # Draw border lines
        cv2.line(viz_image, (0, start), (width, start), (100, 100, 100), 1)
        cv2.line(viz_image, (0, end), (width, end), (100, 100, 100), 1)
        
        # Add measurements, word count, and text preview
        mid_y = (start + end) // 2
        region_type = 'Space' if region.is_empty else 'Text'
        text = f"{region_type} ({region.relative_width}x)"
        if not region.is_empty:
            text += f" - {region.word_count} words"
            # Add text preview if available
            if region.text:
                preview = region.text[:50] + "..." if len(region.text) > 50 else region.text
                text += f"\n'{preview}'"
        
        # Draw text with proper line breaks
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        text_lines = text.split('\n')
        
        for idx, line in enumerate(text_lines):
            text_size = cv2.getTextSize(line, font, font_scale, 1)[0]
            text_x = 10
            text_y = mid_y + 5 + (idx * 20)
            
            # Draw text background
            cv2.rectangle(viz_image,
                         (text_x - 2, text_y - text_size[1] - 2),
                         (text_x + text_size[0] + 2, text_y + 2),
                         (255, 255, 255),
                         -1)
            
            # Draw text
            cv2.putText(viz_image, line, (text_x, text_y),
                        font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
    
    combined_image = np.hstack((color_image, viz_image))
    return combined_image

def analyze_resume(input_path: str) -> None:
    """
    Main function to analyze a resume from either PDF or image input.
    """
    # Handle PDF or image input
    if input_path.lower().endswith('.pdf'):
        binary_image = convert_pdf_to_image(input_path)
        _, binary_image = cv2.threshold(binary_image, 200, 255, cv2.THRESH_BINARY)
    else:
        binary_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        _, binary_image = cv2.threshold(binary_image, 200, 255, cv2.THRESH_BINARY)
    
    # Find regions and analyze with OCR
    regions = find_regions_with_words(binary_image)
    
    # Print analysis
    print("\nRegion Analysis:")
    for i, region in enumerate(regions):
        region_type = "Space" if region.is_empty else "Text"
        print(f"Region {i+1}: {region_type}")
        print(f"  - Lines: {region.width}")
        print(f"  - Relative width: {region.relative_width}x")
        if not region.is_empty:
            print(f"  - Words: {region.word_count}")
            print(f"  - Text: {region.text[:100]}...")
        print(f"  - Position: {region.start} to {region.end}")
        print()
    
    # Create visualization
    combined_image = draw_regions_with_words(binary_image, regions)
    
    # Display results
    plt.figure(figsize=(20, 10))
    plt.title("Resume Analysis with OCR")
    plt.imshow(combined_image)
    plt.axis('off')
    plt.show()
    
    # Save results
    output_path = Path(input_path).with_name("0c5d3800f82c400a9c16681aed1d9cf6.jpg")
    cv2.imwrite(str(output_path), 
                cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))
    print(f"\nProcessed image saved as '{output_path}'")

if __name__ == "__main__":
    input_path = "Keshav's_CV.pdf"  # or .png/.jpg
    analyze_resume(input_path)