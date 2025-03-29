import os
import PyPDF2
import ollama

# File paths (update these paths as needed)
INPUT_PDF = r"path-to\input.pdf"
QUESTIONS_FILE = r"path-to\quiz_questions.txt"
ANSWERS_FILE = r"path-to\quiz_answers.txt"

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
        print("Error reading PDF:", e)
        return ""

def generate_quiz(text):
    """
    Sends the extracted text to the Ollama model using `ollama.chat()`
    with instructions to generate a multiple-choice quiz that includes both questions
    and answers. The expected output should have a clear separator "Answers:" on its own line.
    """
    prompt = (
        "Create a multiple-choice quiz based on the following content. "
        "Your output should strictly follow this format:\n\n"
        "Questions:\n"
        "1. [First question]\n"
        "2. [Second question]\n"
        "...\n\n"
        "Answers:\n"
        "1. [Answer to first question]\n"
        "2. [Answer to second question]\n"
        "...\n\n"
        "Content:\n" + text
    )
    
    try:
        response = ollama.chat(
            model="gemma3",  # Ensure that this model is available in your Ollama environment
            messages=[{"role": "user", "content": prompt}]
        )
        output = response["message"]["content"].strip()
        print("Raw model output:\n", output)  # Debug print
        
        # Check if the output contains the expected separator
        if "Answers:" not in output:
            raise ValueError("Invalid format received. Expected an 'Answers:' section.")
        
        # Split the output into questions and answers based on the 'Answers:' marker.
        parts = output.split("Answers:", 1)
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
    if not os.path.exists(INPUT_PDF):
        print(f"Error: Input PDF '{INPUT_PDF}' not found.")
        return

    print("Extracting text from PDF...")
    pdf_text = extract_text_from_pdf(INPUT_PDF)
    if not pdf_text:
        print("No text extracted from the PDF.")
        return

    print("Generating quiz (questions and answers) from extracted text...")
    questions, answers = generate_quiz(pdf_text)
    if questions and answers:
        write_text_file(QUESTIONS_FILE, questions)
        write_text_file(ANSWERS_FILE, answers)
        print("Quiz generation complete!")
    else:
        print("Failed to generate quiz.")

if __name__ == "__main__":
    main()
