# Excel RAG Chatbot

An AI chatbot to interact with Excel data using **OpenAI**, **LangChain**, and **Streamlit**.

## Features

- **Excel File Upload**: Upload and index Excel sheets.
- **Hybrid Retrieval**: Combines vector search and SQL queries.
- **Streamlit Interface**: User-friendly web interface.
- **Secure API Key**: Managed via `.env` file.

## Tech Stack

- **Python**, **Streamlit**, **LangChain**, **OpenAI**, **FAISS**,  **dotenv**.

## Installation

1. Clone the repository:
```
git clone https://github.com/JyotiJha01/excel-rag-chatbot.git

cd excel-rag-chatbot
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Add your OpenAI API key in a .env file:
```
OPENAI_API_KEY=your_openai_api_key
```

4. Run the app:
```
streamlit run app.py
```

