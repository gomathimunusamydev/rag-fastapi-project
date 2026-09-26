from fastapi import FastAPI, UploadFile, File, HTTPException

from app.ingestion import (
    extract_text_from_pdf,
    split_text_into_chunks
)

from app.embeddings import create_embeddings

from app.rag_pipeline import (
    add_chunks_to_vectorstore,
    retrieve_relevant_chunks,
    generate_answer
)

from app.schemas import AskRequest


app = FastAPI()


@app.get("/")
def home():

    return {
        "message": "RAG FastAPI Project is running!"
    }


@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...)
):

    # Check PDF file
    if not file.filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    # Save uploaded PDF
    file_path = f"data/{file.filename}"

    with open(file_path, "wb") as f:

        f.write(await file.read())


    # Step 1: Extract text from PDF
    text = extract_text_from_pdf(file_path)

    if not text.strip():

        raise HTTPException(
            status_code=400,
            detail="No text found in PDF."
        )


    # Step 2: Split text into chunks
    chunks = split_text_into_chunks(text)


    # Step 3: Create embeddings for chunks
    embeddings = create_embeddings(chunks)


    # Step 4: Store embeddings + chunks in FAISS
    add_chunks_to_vectorstore(
        chunks,
        embeddings
    )


    return {

        "filename": file.filename,

        "total_characters": len(text),

        "total_chunks": len(chunks),

        "chunks": chunks

    }


@app.post("/ask")
def ask_question(request: AskRequest):

    question = request.question


    # Step 1: Retrieve relevant chunks
    relevant_chunks = retrieve_relevant_chunks(
        question,
        top_k=3
    )


    # If no relevant information is found
    if not relevant_chunks:

        return {
            "question": question,
            "answer": "No relevant information found. Please upload a PDF first.",
            "source_chunks": []
        }


    # Step 2: Combine retrieved chunks into context
    context = "\n\n".join(relevant_chunks)


    # Step 3: Generate answer using Ollama Llama 3.2
    answer = generate_answer(
        question,
        context
    )


    # Step 4: Return final response
    return {

        "question": question,

        "answer": answer,

        "source_chunks": relevant_chunks

    }
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "RAG FastAPI service is running"
    }