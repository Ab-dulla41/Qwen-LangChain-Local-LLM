# Qwen-LangChain-Local-LLM 🤖

A lightweight AI chatbot powered by **Qwen 2.5 - 0.5B Instruct**, **LangChain**, and **Streamlit**.

This project demonstrates how to run an open-source Large Language Model (LLM) using Hugging Face Transformers, integrate it with LangChain, and build an interactive chatbot interface using Streamlit.

## 🚀 Features

- **Qwen 2.5 - 0.5B Instruct:** Uses an open-source language model for generating responses.
- **LangChain Integration:** Connects the language model with a reusable prompt chain.
- **Interactive Chat Interface:** Provides a user-friendly chatbot using Streamlit.
- **Conversation History:** Maintains conversation context during the session.
- **GPU Acceleration:** Uses GPU acceleration when available.
- **Custom Dark UI:** Includes a modern dark-themed interface.
- **Clear Conversation:** Allows users to reset the current chat.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Qwen 2.5 - 0.5B | Language model |
| Hugging Face Transformers | Model loading and inference |
| LangChain | LLM integration and prompt management |
| Streamlit | Web-based chatbot interface |
| PyTorch | Deep learning framework |
| Google Colab | Cloud execution with GPU support |

## 🖥️ Application Screenshots

### Main Interface

![Main Interface](screenshots/main_interface.png)

### Conversation Example

![Conversation Example](screenshots/conversation.png)

## 🏗️ Project Architecture

```text
             User
               |
               v
        Streamlit Interface
               |
               v
           LangChain
               |
               v
          Prompt Chain
               |
               v
       Qwen 2.5 - 0.5B
               |
               v
       Generated Response
               |
               v
          User Interface
```

## 📂 Project Structure

```text
Qwen-LangChain-Local-LLM/
│
├── screenshots/
│   ├── main_interface.png
│   └── conversation.png
│
├── notebooks/
│   └── Qwen_LangChain.ipynb
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/vaais251/Qwen-LangChain-Local-LLM.git
```

### 2. Navigate to the Project

```bash
cd Qwen-LangChain-Local-LLM
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app.py
```

**Note:** The application is currently developed and tested in Google Colab with GPU acceleration (NVIDIA Tesla T4). Running it locally may require adjustments to the model cache path and device configuration in `app.py`.

## 🧠 Model Information

- **Model:** Qwen 2.5 - 0.5B Instruct
- **Provider:** Qwen
- **Model Source:** [Hugging Face - Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- **Integration:** Hugging Face Transformers and LangChain

## 🔮 Future Improvements

- Retrieval-Augmented Generation (RAG)
- PDF document upload and question answering
- Vector database integration
- Improved conversation management
- Deployment on a cloud hosting platform

## 👨‍💻 Author

**Abdullah Satti**

Bachelor of Science in Artificial Intelligence (BSAI)

Air University Islamabad

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

