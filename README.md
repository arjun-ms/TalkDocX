# TalkDocX

TalkDocX is a Streamlit app that allows users to upload a PDF, ask questions based on the content of the PDF, and get responses using Google Gemini (AI). Additionally, users can interact with the app using voice commands and get the answers in both text and speech.

## Demo Videos

### TalkDocX Voice Input Demo
Watch the demo video below to see how the voice input functionality works in TalkDocX:

[![Watch the demo](https://img.youtube.com/vi/Z5X2vOjKTUk/0.jpg)](https://youtu.be/Z5X2vOjKTUk)

### TalkDocX Text Input Demo
Here's a video demonstrating the text input functionality in TalkDocX:

[![Watch the demo](https://img.youtube.com/vi/HYAvr3GgPyc/0.jpg)](https://youtu.be/HYAvr3GgPyc)

## Prerequisites

Before running the application, make sure you have the following installed:

- **Python 3.8+**: Ensure you have Python 3.8 or later installed on your system.
- **Git**: To clone the repository, you will need Git installed on your machine.

## Setting Up Environment Variables

1. Create a `.env` file in the project root directory.
2. Add the following line to the `.env` file with your **Google Gemini API key**:

    ```env
    GEMINI_API_KEY=your_google_gemini_api_key_here
    ```

You can obtain the **Google Gemini API key** by following the [official documentation](https://ai.google.dev/gemini-api/docs/api-key).

## Running the Application

### 1. Clone the Repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/arjun-ms/TalkDocX.git
cd TalkDocX
```

### 2. Create and Activate a Virtual Environment

Create a virtual environment using the following command:

```bash
python -m venv venv
```

Next, activate the virtual environment:

- **On Windows**:
  ```bash
  venv\Scripts\activate
  ```

- **On macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies

Once the virtual environment is activated, install the necessary dependencies by running the following command:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes the following necessary libraries:

```txt
streamlit
pdfplumber
langchain
langchain-google-genai
faiss-cpu
python-dotenv
speechrecognition
pyttsx3
threading
```

### 4. Run the Streamlit App

Now, run the Streamlit app using the command:

```bash
streamlit run ui.py
```

This will start the app, and it will be accessible in your browser at `http://localhost:8501`.

## Using the Application

1. **Upload a PDF**:
   - Click the "Upload your PDF file" button to upload a PDF document. The app will extract text from the PDF and make it available for querying.

2. **Ask Questions**:
   - You can either type your questions in the **"Text Input"** box or use the **"Use Voice Command"** button to ask a question via speech.
   - The app will generate an answer based on the content of the uploaded PDF using Google Gemini.

3. **Voice Interaction**:
   - To use voice commands, click on **"Use Voice Command"** and speak your question. The app will convert your voice to text and retrieve the answer from the PDF.
   - The answer will be displayed in text form and also read out loud via the text-to-speech engine.

4. **Get Audio Response**:
   - Once the answer is generated, it will be converted to speech and played back to you.

## Troubleshooting

- **Error: "run loop already started"**:
    If you encounter this error related to `pyttsx3`, it may be due to a conflict with Streamlit's event loop. To resolve this, ensure the speech synthesis is processed in a separate thread, as described in the modified code.

- **Microphone Not Working**:
    If voice recognition is not working, ensure your microphone is properly connected and accessible by the `SpeechRecognition` library. Also, check that your system's microphone is configured correctly.

- **Voice Command Not Recognized**:
    If the app is not recognizing your speech properly, try speaking more clearly and ensure there is minimal background noise.
