import os
import ollama
import PyPDF2
from pdf2image import convert_from_path
import pytesseract

# File paths (update these paths as needed)
INPUT_PDF = r"C:\Users\karti\OneDrive\Desktop\projects\software\desiAssessment\text-quiz\input.txt"
QUESTIONS_PDF = r"C:\Users\karti\OneDrive\Desktop\projects\software\desiAssessment\text-quiz\quiz_questions.txt"
ANSWERS_PDF = r"C:\Users\karti\OneDrive\Desktop\projects\software\desiAssessment\text-quiz\quiz_answers.txt"

# If using pdf2image on Windows, set the Poppler path:
POPPLER_PATH = r"C:\path\to\poppler-xx\bin"  # Update with your actual Poppler bin path

def extract_text_from_pdf(pdf_path):
    """Extracts selectable text from a PDF using PyPDF2."""
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print("Error reading PDF text:", e)
        return ""

def extract_text_from_images(pdf_path):
    """Converts PDF pages to images and extracts text using OCR (pytesseract)."""
    ocr_text = ""
    try:
        # Convert PDF pages to images
        images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
        for idx, image in enumerate(images):
            # Use OCR to extract text from each image
            page_text = pytesseract.image_to_string(image)
            ocr_text += f"--- Page {idx+1} OCR ---\n{page_text}\n"
        return ocr_text.strip()
    except Exception as e:
        print("Error during OCR extraction:", e)
        return ""

def generate_quiz(text):
    """
    Sends the combined text to the Ollama Gemma model via `ollama.chat()`
    with instructions to generate a multiple-choice quiz.
    
    The expected output format:
      [Quiz Questions...]
      Answers: [Corresponding Answers...]
      
    Returns a tuple of (questions, answers).
    """
    prompt = (
        "Create a multiple-choice quiz based on the following content. "
        "Format the output as follows:\n"
        "1. List the quiz questions first.\n"
        "2. Then, include a section labeled 'Answers:' with the corresponding answers.\n\n"
        f"Content:\n{text}"
    )
    
    try:
        response = ollama.chat(
            model="gemma3",  # Ensure that this model is available in your Ollama environment
            messages=[{"role": "user", "content": prompt}]
        )
        output = response["message"]["content"].strip()
        if not output:
            raise ValueError("No response received from the model.")
        
        # Split the output into questions and answers using the 'Answers:' marker
        parts = output.split("Answers:", 1)
        if len(parts) < 2:
            raise ValueError("Invalid format received. Expecting an 'Answers:' section.")
        
        questions = parts[0].strip()
        answers = "Answers:" + parts[1].strip()
        return questions, answers
    except Exception as e:
        print("Error during quiz generation:", e)
        return None, None

def write_text_file(file_path, content):
    """Writes text content to a file."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Output saved to: {file_path}")
    except Exception as e:
        print("Error writing file:", e)

def main():
    # Verify that the input PDF exists
    if not os.path.exists(INPUT_PDF):
        print(f"Error: Input PDF file '{INPUT_PDF}' not found.")
        return

    print("Extracting native text from PDF...")
    native_text = extract_text_from_pdf(INPUT_PDF)
    
    print("Extracting OCR text from PDF images...")
    ocr_text = extract_text_from_images(INPUT_PDF)
    
    # Combine the native text and OCR text
    combined_text = native_text + "\n\n" + ocr_text
    if not combined_text.strip():
        print("No text extracted from the PDF.")
        return

    print("Generating quiz (questions and answers) from combined text...")
    questions, answers = generate_quiz(combined_text)
    if questions and answers:
        write_text_file(QUESTIONS_FILE, questions)
        write_text_file(ANSWERS_FILE, answers)
        print("Quiz generation complete!")
    else:
        print("Failed to generate quiz.")

if __name__ == "__main__":
    main()
