# Sidekick Assistant

Sidekick is an AI assistant application that can help with browsing the web, searching for information, and answering questions.

## Project Structure

```
demo/
├── .env                  # Environment variables
├── __init__.py           # Makes the directory a Python package
├── app.py                # UI implementation
├── requirements.txt      # Python dependencies
├── run.py                # Main entry point
├── sidekick.py           # Core Sidekick implementation
└── sidekick_tools.py     # Tools for Sidekick (browser, search, etc.)
```

## Setup

1. Make sure you have Python 3.8+ installed
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Configure your API keys in the `.env` file:
   - GPT_API_KEY: Required for the OpenAI LLM
   - PUSHOVER_TOKEN and PUSHOVER_USER: Optional for push notifications
   - SERPER_API_KEY: Optional for web search

## Running the Application

To start the Sidekick Assistant, run:

```bash
python run.py
```

This will launch a Gradio web interface in your browser where you can interact with the assistant.

## Features

- Web browsing with Playwright
- Web search using Google Serper API
- Push notifications via Pushover
- Python code execution
- Wikipedia searches
- File management tools

## Notes

- The browser will launch in non-headless mode, so you'll be able to see what the assistant is doing
- The application uses LangChain and LangGraph for the AI workflow
