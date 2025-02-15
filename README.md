# RAG-Chatbot

This is a Flask-based chat application that provides functionalities such as managing chat sessions, handling PDF uploads, processing documents, selecting models for generating responses, and retrieving chat history.

## Features

- Create and manage chat sessions
- Handle PDF uploads and process them
- Manage model selection and response generation
- Retrieve chat history
- Debugging endpoints for system state and document information

## Setup

### Prerequisites

- Python 3.8 or higher
- Conda (Anaconda or Miniconda)

### Setting Up the Environment

1. **Create a Conda environment:**

    ```sh
    conda create -n flask_env python=3.8
    ```

2. **Activate the environment:**

    ```sh
    conda activate flask_env
    ```

3. **Navigate to the project directory and install dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

    If you don't have a `requirements.txt` file, you can manually install Flask and other dependencies:

    ```sh
    pip install Flask
    pip install werkzeug
    ```

4. **Save the environment configuration:**

    ```sh
    conda env export > environment.yml
    ```

### Running the Application

1. **Navigate to the project directory:**

    ```sh
    cd /path/to/project
    ```

2. **Run the Flask application:**

    ```sh
    python app.py
    ```

3. **Open your web browser and go to:**

    ```sh
    http://127.0.0.1:5000/
    ```

## API Endpoints

### Home

- **GET /**

    Renders the home page and creates a new chat session if none exists.

### New Chat

- **POST /new-chat**

    Starts a new chat session and returns the chat ID.

### Chat

- **POST /chat**

    Processes a chat request, generates a response, updates the chat history, and returns the response, chat ID, and generation time.

### Select Model

- **POST /select-model**

    Selects a model for the current session.

### Chat History

- **GET /chat-history**

    Retrieves the chat history for the current session.

### Debug System State

- **GET /debug/system-state**

    Retrieves the current system state for debugging purposes.

### Debug Document Info

- **GET /debug/document-info**

    Retrieves detailed information about processed documents.

### Test Document Retrieval

- **POST /debug/test-retrieval**

    Tests document retrieval based on a query.

### Upload PDF

- **POST /upload-pdf**

    Uploads a new PDF, processes it, and updates the vector store.

## Directory Structure

```
/path/to/project
│
├── src/
│   ├── __init__.py
│   ├── document_processor.py
│   ├── model_config.py
│   ├── model_handler.py
│
├── data/
│   ├── chat_history/
│   ├── pdfs/
│   └── vector_store/
│
├── models/
│
├── templates/
│   ├── index.html
│
├── app.py
├── requirements.txt
└── environment.yml
```
