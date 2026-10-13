
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge
from google import genai
from google.genai import types

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}


@app.route("/")
def home():
    return render_template("index.html")


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(error):
    return jsonify({
        "error": "Image too large. The maximum upload size is 10 MB."
    }), 413


@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question", "").strip()
    section = request.form.get("section", "Math")
    image_file = request.files.get("image")

    if not question and not (
        image_file and image_file.filename
    ):
        return jsonify({
            "error": "Please enter a question or upload an image."
        }), 400

    if section not in ("Math", "Reading & Writing"):
        return jsonify({"error": "Invalid SAT section."}), 400

    image_part = None

    if image_file and image_file.filename:
        if image_file.mimetype not in ALLOWED_IMAGE_TYPES:
            return jsonify({
                "error": "Unsupported image type. Use JPG, PNG, or WebP."
            }), 400

        image_bytes = image_file.read()

        if not image_bytes:
            return jsonify({"error": "The uploaded image is empty."}), 400

        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type=image_file.mimetype,
        )

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return jsonify({
            "error": "Gemini API key missing. Check your .env file."
        }), 503

    prompt = f"""
You are SAT Tutor AI, a friendly and accurate SAT tutor.

SAT section: {section}
Student's written question or instructions: {question or "Please solve the question shown in the image."}

If an image is provided:
- Carefully read the question, equations, diagrams, and answer choices.
- Use the image as the source of the question.
- If text or a diagram is unclear, say exactly what you cannot read.
- Never invent missing numbers, labels, or answer choices.

Instructions:
- Explain the solution step by step in student-friendly language.
- For Math, show calculations and explain why each step works.
- For Reading & Writing, explain grammar, vocabulary, evidence, and reasoning.
- For multiple-choice questions, identify the correct option and explain why.
- Briefly explain why other options are incorrect when useful.
- If information is missing, ask the student for clarification.
- Never invent information.

Formatting:
- Use plain text with clear headings and numbered steps.
- Avoid unnecessary Markdown symbols.
- Write fractions as 1/2 and multiplication as ×.
"""

    try:
        client = genai.Client(api_key=api_key)

        contents = [prompt]

        if image_part is not None:
            contents.append(image_part)

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=contents,
        )

        answer = response.text

        if not answer:
            return jsonify({
                "error": "The AI returned an empty answer. Please retry."
            }), 502

        return jsonify({"answer": answer})

    except Exception:
        app.logger.exception("Gemini request failed")
        return jsonify({
            "error": (
                "The AI request failed. Check your API key, "
                "internet connection, model availability, and API quota."
            )
        }), 502


if __name__ == "__main__":
    app.run(debug=True)
