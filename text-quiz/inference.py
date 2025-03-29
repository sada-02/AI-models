import os
import ollama  # Ensure the 'ollama' package is installed

# File paths (update these paths as needed)
INPUT_FILE = r"path-to\input.txt"
QUESTIONS_FILE = r"path-to\quiz_questions.txt"
ANSWERS_FILE = r"path-to\quiz_answers.txt"

def list_project_directory(directory):
    """Lists files in the specified directory for debugging."""
    try:
        files = os.listdir(directory)
        print(f"Files in '{directory}':")
        for f in files:
            print(f" - {f}")
    except Exception as e:
        print(f"Error listing directory '{directory}': {e}")

def read_text_file(file_path):
    """Reads and returns the text from a text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def write_text_file(file_path, content):
    """Writes text content to a text file."""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

def generate_quiz(text):
    """
    Sends the input text to the Ollama Gemma model using `ollama.chat()`
    and returns quiz questions and answers.
    """
    prompt = (
        "Create a multiple-choice quiz based on the following content. "
        "Format the output as follows:\n"
        "1. Provide the quiz questions first.\n"
        "2. Then, include a section labeled 'Answers:' with the correct answers.\n\n"
        f"Content:\n{text}"
    )
    
    try:
        # Using Ollama's chat function
        response = ollama.chat(
            model="gemma3",  # Make sure "gemma" is available in Ollama
            messages=[{"role": "user", "content": prompt}]
        )
        
        output = response["message"]["content"].strip()
        if not output:
            raise ValueError("No response received from Gemma.")

        # Split the output into questions and answers based on the 'Answers:' marker
        parts = output.split("Answers:", 1)
        if len(parts) < 2:
            raise ValueError("Invalid format received. Expecting an 'Answers:' section.")

        questions = parts[0].strip()
        answers = "Answers:" + parts[1].strip()

        return questions, answers

    except Exception as e:
        print("Error during quiz generation:", e)
        return None, None

def main():
    # Debug: list files in the project directory to verify file presence
    project_directory = r"path-to"
    list_project_directory(project_directory)

    # Check if the input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file '{INPUT_FILE}' not found.")
        return

    # Read input topics from the text file
    text = read_text_file(INPUT_FILE)
    if not text:
        print("Input file is empty.")
        return

    # Generate quiz questions and answers using Gemma via Ollama
    questions, answers = generate_quiz(text)
    if questions and answers:
        # Write the generated quiz questions and answers to text files
        write_text_file(QUESTIONS_FILE, questions)
        write_text_file(ANSWERS_FILE, answers)
        print("Quiz generation successful!")
        print(f"Questions saved to: {QUESTIONS_FILE}")
        print(f"Answers saved to: {ANSWERS_FILE}")
    else:
        print("Failed to generate quiz.")

if __name__ == "__main__":
    main()
