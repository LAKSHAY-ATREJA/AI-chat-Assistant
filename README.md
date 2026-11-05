# AI Chat Assistant

**[▶ Live Demo — Open App](https://ai-chat-assistant-elyt2rzcwcddgypk6zfslh.streamlit.app)**

A conversational AI assistant built with Streamlit and LangChain, powered by Groq's
hosted inference API. The application provides a polished chat interface with
persistent conversation memory, five purpose-built personas, and structured error
handling. The same LangChain pipeline is also exposed as a standalone command-line
demo that can be used to verify the setup without a browser. **[Launch Live Demo](https://ai-chat-assistant-elyt2rzcwcddgypk6zfslh.streamlit.app)**

## What it does

Each user message is sent to a Groq-hosted model through LangChain Core with
the selected persona's system prompt and up to 10 recent chat messages.
Switching persona or clearing the chat resets the conversation context.

Five personas ship out of the box:

- General Assistant: broad-purpose helper, clear and concise answers.
- Python Tutor: explains concepts with working code snippets.
- Writing Coach: rewrites and critiques prose with constructive feedback.
- Research Assistant: structured, fact-based explanations.
- Career Advisor: practical guidance on tech career decisions.

The sidebar shows live message counts and lets you export the full conversation as a
plain-text file at any point.


## Architecture

```
User input
    -> Persona system prompt + last 10 chat messages
    -> LangChain Core / ChatGroq (openai/gpt-oss-20b by default)
    -> Groq API
    -> Response displayed in Streamlit chat UI
    -> Successful response added to session history
```

The conversation history is held in Streamlit session state. Clearing the chat
or switching persona starts with empty message history. Failed API requests are
not recorded as assistant replies.


## Tech stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM provider | Groq (hosted inference) |
| Model | GPT-OSS 20B on Groq (configurable via GROQ_MODEL) |
| Orchestration | LangChain Core / ChatGroq |
| Memory | Last 10 messages held in Streamlit session state |
| Configuration | python-dotenv |


## Requirements

- Python 3.9 or later
- A free Groq API key from https://console.groq.com (no credit card required)


## Installation

Clone the repository and install dependencies into a virtual environment:

```bash
git clone https://github.com/LAKSHAY-ATREJA/AI-chat-Assistant.git
cd AI-chat-Assistant

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```


## Environment variables

Copy .env.example to .env and fill in your Groq key:

```bash
cp .env.example .env
```

Edit .env so it reads:

```
GROQ_API_KEY=gsk_your_actual_key_here
```

For Streamlit Community Cloud, configure GROQ_API_KEY through the app's
Secrets settings; do not commit your key to the repository. The app reads it
from the environment at startup.

| Variable | Required | Description |
|---|---|---|
| GROQ_API_KEY | Yes | API key from console.groq.com. Keys are prefixed with gsk_. |
| GROQ_MODEL | No | Groq model ID; defaults to openai/gpt-oss-20b. |

To obtain a key: create an account at https://console.groq.com, navigate to API
Keys, and click "Create API Key". Copy the key immediately as it is shown only once.


## Running the application locally

```bash
streamlit run app.py
```

Streamlit opens a browser tab at http://localhost:8501. If the tab does not open
automatically, copy the URL from the terminal output.

Set GROQ_API_KEY in your environment or local .env file before launching.
Choose a persona, then type in the chat input at the bottom of the screen.


## Running the demo script

demo.py runs three short scripted conversations from the terminal without needing a
browser. It exercises the identical LangChain pipeline and is useful for verifying
that the API key works and all dependencies are installed correctly before launching
the full UI.

```bash
python demo.py
```

Example output (actual model responses will vary):

```
========================================================================
  AI Chat Assistant -- Demo
  Model: Groq openai/gpt-oss-20b | Recent context: 10 messages
========================================================================

Persona: General Assistant
System: You are a helpful, friendly, and knowledgeable AI assistant. Answ...
------------------------------------------------------------------------

User:
    What is machine learning in one sentence?

Assistant:
    Machine learning is a branch of artificial intelligence where
    systems learn from data to improve their performance on a task
    without being explicitly programmed for every case.

User:
    Can you give me a real-world example of that?

Assistant:
    A spam filter is a classic example. The model is trained on millions
    of labelled emails and learns patterns that distinguish spam from
    legitimate mail. Once deployed, it applies those patterns to new
    emails it has never seen before.

------------------------------------------------------------------------

Persona: Python Tutor
System: You are an expert Python programming tutor. Explain concepts cle...
------------------------------------------------------------------------

User:
    How do I read a CSV file in Python?

Assistant:
    The standard approach uses the csv module from the standard library,
    or the pandas library for more analytical work.

    Using csv:

        import csv
        with open('data.csv', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                print(row)

    Using pandas:

        import pandas as pd
        df = pd.read_csv('data.csv')
        print(df.head())

------------------------------------------------------------------------

Demo complete -- 5 turn(s) run, 0 error(s).
```


## Deployment

### Render (recommended free tier)

The repository includes a render.yaml file. To deploy:

1. Push this repository to GitHub.
2. Log in to https://render.com and click New > Web Service.
3. Connect the GitHub repository. Render will detect render.yaml automatically.
4. In the Environment section, add a secret environment variable:
   Key: GROQ_API_KEY  Value: your actual key.
5. Click Create Web Service. Render builds and deploys the app automatically.

The start command in render.yaml passes --server.headless true, which
prevents Streamlit from attempting to open a browser on the server.


### Heroku / Railway

A Procfile is included for platforms that use that convention:

```
web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
```

Set GROQ_API_KEY as a config variable (Heroku) or environment variable
(Railway) through the platform dashboard.


## Error handling

The chat handler catches all exceptions from the Groq API and maps them
to user-readable messages:

- HTTP 401 or invalid_api_key: prompts the user to check their key.
- HTTP 429 or rate_limit: advises waiting before retrying.
- HTTP 503 or unavailable: reports a temporary service outage.
- Unsupported model IDs: suggests selecting a currently supported Groq model.
- Other errors: shows a brief message without exposing raw API details.

Errors are shown inline but are not saved as assistant responses, so the user
can retry without polluting the conversation memory.


## License

MIT
