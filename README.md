# RAG-Chatbot

This is a Flask-based chat application that provides functionalities such as managing chat sessions, handling PDF uploads, processing documents, selecting models for generating responses, and retrieving chat history.

## Features

- Create and manage chat sessions
- Handle PDF uploads and process them
- Manage model selection and response generation
- Retrieve chat history
- Debugging endpoints for system state and document information

## Demo

[![Watch the video](https://img.youtube.com/vi/1AWcTMlDe_4/0.jpg)](https://www.youtube.com/watch?v=1AWcTMlDe_4)

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

## Note to the Viewer

To use the RAG-Chatbot application, please follow these steps:

1. **Download the Model:**
   Download the required model from the provided Google Drive link. Make sure to save the downloaded model file in the `models` folder of the project.

2. **Rename and Set Up the Model:**
   After downloading the GGUF model, rename it to match one of the default names from the following list to ensure compatibility with the application:
   - `mistral-7b`
   - `llama-2`
   - `neural-chat`
   - `mistral-nemo-instruct-2407-gguf`
   - `model`

   For example, if you download a model named `my-gguf-model`, rename it to `mistral-7b` (or another model name from the list). Save it in the `models/` directory.

3. **(OR Alternatively)Update the DEFAULT_CONFIG (if needed):**
   The `DEFAULT_CONFIG` dictionary in the project has predefined model names, such as `mistral-7b`, `llama-2`, and others. When you rename the model to one of these default names, it will automatically match the configuration settings for that model.

   You can also update the configuration settings directly in the `model_config.py` file to suit your specific model.

   Here's an example of the `DEFAULT_CONFIG` dictionary:

   ```python
   DEFAULT_CONFIG = {
       'mistral-7b': {
           'n_ctx': 4096,
           'n_threads': os.cpu_count(),
           'n_gpu_layers': 0,
           'template': "System: You are a helpful assistant.\n\nUser: {query}\n\nAssistant:"
       },
       'llama-2': {
           'n_ctx': 4096,
           'n_threads': os.cpu_count(),
           'n_gpu_layers': 0,
           'template': "<s>[INST] {query} [/INST]"
       },
       'neural-chat': {
           'n_ctx': 4096,
           'n_threads': os.cpu_count(),
           'n_gpu_layers': 0,
           'template': "### System: You are a helpful assistant.\n\n### User: {query}\n\n### Assistant:"
       },
       'mistral-nemo-instruct-2407-gguf': {
           'n_ctx': 4096,
           'n_threads': os.cpu_count(),
           'n_gpu_layers': 0,
           'template': "System: You are a helpful assistant.\n\nUser: {query}\n\nAssistant:"
       },
       'model': {
           'n_ctx': 4096,
           'n_threads': os.cpu_count(),
           'n_gpu_layers': 0,
           'template': "System: You are a helpful assistant.\n\nUser: {query}\n\nAssistant:"
       }
   }
   ```

   **Alternatively**, you can update the configurations directly in the `model_config.py` file.

4. **Set Up the Environment:**
   Follow the steps in the "Setting Up the Environment" section of this README file to create and activate a Conda environment, and install the necessary dependencies.

5. **Run the Application:**
   Once the environment is set up, run the `app.py` file to start the Flask application:

    ```sh
    python app.py
    ```

6. **Chat with Your Documents:**
   Open your web browser and go to `http://127.0.0.1:5000/`. You can upload your documents via the GUI and start chatting with the chatbot.

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

## License

This project is licensed under the MIT License.
