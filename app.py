import os
import google.generativeai as genai
from flask import Flask, render_template, request, session, jsonify
from flask_session import Session
from werkzeug.utils import secure_filename
from PIL import Image
import pytesseract
import markdown
from datetime import datetime
import PyPDF2
import pandas as pd
import time

# Flask app configuration
app = Flask(__name__)
app.secret_key = "secret_key_for_sessions"  # Change this in production
app.config["SESSION_TYPE"] = "filesystem"
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "webp", "heic", "heif",
                                    "aiff", "aac", "ogg", "flac", "mp3", "wav", "webm"
                                    "mp4", "mpeg", "3gpp", "mov", "avi", "x-flv", "mpg", "wmv",
                                    "txt", "pdf", "docx", "xlsx", "csv"}
Session(app)

# Ensure the upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Configure the Gemini model
api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# Set the Tesseract path explicitly if needed
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"


def allowed_file(filename):
    """Check if a file is allowed based on its extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


def process_file(file_path, file_type):
    if file_type == "txt":
        with open(file_path, "r") as file:
            return file.read()

    elif file_type == "pdf":
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        return text

    elif file_type in {"xls", "csv", "xlsx"}:
        try:
            df = pd.read_excel(file_path)
            return df.to_string()
        except:
            try:
                df = pd.read_csv(file_path)
                return df.to_string()
            except:
                return "Error reading Excel/CSV file."

    return "Unsupported file type."
def process_image(file_path):
    """Extract content or metadata from the uploaded image."""
    try:
        text = pytesseract.image_to_string(Image.open(file_path))
        return text or "No text found in the image."
    except Exception as e:
        return jsonify({"error": f"Error: {e}"})
def process_audio(file_path, prompt):
    try:
        myfile = genai.upload_file(file_path)
        print("Uploading Audio.....")
        result = model.generate_content([myfile,prompt])
        print(result.text)
        return result.text
    except Exception as e:
        return jsonify({"error": f"Error: {e}"})


def process_video(file_path, prompt):
    """Upload video file to Gemini and generate response."""
    try:
        print("Uploading file...")
        uploaded_file = genai.upload_file(path=file_path)
        print(f"Completed upload: {uploaded_file.uri}")

        while uploaded_file.state.name == "PROCESSING":
            print(".", end="")
            time.sleep(10)
            uploaded_file = genai.get_file(uploaded_file.name)
        print(uploaded_file.name)
        print(uploaded_file.uri)
        print(uploaded_file.state)

        if uploaded_file.state.name == "FAILED":
            raise ValueError("File processing failed.")

        print("Making LLM inference request...")
        response = model.generate_content([uploaded_file, prompt], request_options={"timeout": 600})
        print(response.text)
        return response.text
    except Exception as e:
        return jsonify({"error": f"Error: {e}"})


@app.route("/")
def index():
    if "history" not in session:
        session["history"] = []
    return render_template("index.html", history=session["history"])


@app.route("/predict", methods=["POST"])
def predict():
    history = session.get("history", [])
    uploaded_file = request.files.get("file")
    prompt = request.form.get("prompt", "").strip()  # Ensure prompt is not None and trim whitespace

    # Initialize variables
    file_content = ""
    file_name = ""
    response_text = ""

    try:
        # Check if a file is uploaded
        if uploaded_file and allowed_file(uploaded_file.filename):

            file_name = secure_filename(uploaded_file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file_name)
            uploaded_file.save(file_path)
            file_type = file_name.rsplit(".", 1)[1].lower()


            if file_type in {"mp4", "mpeg", "3gpp", "mov", "avi", "x-flv", "mpg", "wmv"}:
                # Process video with Gemini
                response_text = process_video(file_path, prompt)
            elif file_type in {"aiff", "aac", "ogg", "flac", "mp3", "wav", "webm"}:
                # Process audio with Gemini
                response_text = process_audio(file_path, prompt)
            elif file_type in {"png", "jpg", "jpeg", "webp", "heic", "heif"}:
                # Process image with Gemini
                file_content = process_image(file_path)
                combined_input = f"Uploaded file: {file_name}\n{file_content}\n\nUser Input: {prompt}".strip()
                response_text = model.generate_content(combined_input).text
            elif file_type in {"txt", "pdf", "docx", "xlsx", "csv"}:
                # Process standard documents
                file_content = process_file(file_path, file_type)
                combined_input = f"Uploaded file: {file_name}\n{file_content}\n\nUser Input: {prompt}".strip()
                response_text = model.generate_content(combined_input).text
            else:
                return jsonify({"error": "Unsupported file type."}), 400
        elif prompt:
            # Only a prompt is provided
            response_text = model.generate_content(prompt).text
        else:
            return jsonify({"error": "No valid file uploaded or prompt provided."}), 400

        # Format response
        output_html = markdown.markdown(response_text)
        now = datetime.now()
        formatted_datetime = now.strftime("%Y-%m-%d %H:%M:%S")
        history.append({
            "prompt": f"Uploaded file: {file_name if file_name else 'None'}\n{prompt}",
            "response_raw": response_text,
            "response_html": output_html,
            "created_at": formatted_datetime
        })
        session["history"] = history

        return jsonify({"prompt": prompt, "response_html": output_html})

    except Exception as e:
        return jsonify({"error": f"Error: {e}"}), 500



@app.route("/view-history/<int:index>", methods=["GET"])
def view_history(index):
    history = session.get("history", [])
    if 0 <= index < len(history):
        return jsonify(history[index])
    return jsonify({"error": "Invalid index."}), 400


@app.route("/edit-history/<int:index>", methods=["POST"])
def edit_history(index):
    history = session.get("history", [])
    if 0 <= index < len(history):
        data = request.json
        new_prompt = data.get("prompt")

        if new_prompt:
            try:
                response_text = model.generate_content(new_prompt).text
                output_html = markdown.markdown(response_text)
                now = datetime.now()
                formatted_datetime = now.strftime("%Y-%m-%d %H:%M:%S")

                history[index] = {
                    "prompt": new_prompt,
                    "response_raw": response_text,
                    "response_html": output_html,
                    "created_at": formatted_datetime
                }
                session["history"] = history
                return jsonify({"success": True, "prompt": new_prompt, "response_html": output_html})
            except Exception as e:
                return jsonify({"error": f"Error while regenerating response: {e}"})
    return jsonify({"error": "Invalid index or missing prompt."}), 400


@app.route("/delete-history/<int:index>", methods=["POST"])
def delete_history(index):
    history = session.get("history", [])
    if 0 <= index < len(history):
        history.pop(index)
        session["history"] = history
        return jsonify({"success": True})
    return jsonify({"error": "Invalid index."}), 400


if __name__ == "__main__":
    app.run()
