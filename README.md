# RAG FastAPI Project

## How to Run

1. Open the project folder in VS Code.

2. Open Terminal and activate the virtual environment:
   `venv\Scripts\activate`

3. Install the required packages:
   `pip install -r requirements.txt`

4. Open a new terminal and start Ollama:
   `ollama run llama3.2`

5. Open another terminal and start FastAPI:
   `uvicorn app.main:app --reload`

6. Open Swagger in the browser:
   `http://127.0.0.1:8000/docs`

7. Use `/upload` to upload a PDF.

8. After uploading the PDF, use `/ask` to ask questions.

9. Use `/health` to check whether the application is running.