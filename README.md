# Data Science ML Agent

![License](https://img.shields.io/badge/license-MIT-blue)

## Overview

Data Science ML Agent is an AI-powered platform that combines a FastAPI backend with a Streamlit frontend to let users perform common data science workflows using plain English commands.

The project supports:

- Loading CSV and Parquet datasets
- Uploading dataset files via API
- Setting target columns and describing data
- Training classification and regression models
- Optimizing models with Optuna
- Saving and downloading the best model and predictions
- Tracking experiment history

## Why it is useful

This repository is useful for developers and data scientists who want a conversational interface for fast prototyping and model exploration. It makes it easy to interact with datasets, train models, and inspect results without writing boilerplate code.

## Getting started

### Prerequisites

- Python 3.11+ (or compatible Python 3.x)
- `pip` package manager
- NVIDIA-compatible API key in environment variable `NVIDIA_API_KEY`

### Install dependencies

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### Configure the environment

Create a `.env` file in the repository root or export `NVIDIA_API_KEY` in your shell:

```bash
export NVIDIA_API_KEY="your_api_key_here"
```

The application also reads `app/config.yaml` for defaults such as `base_url`, `app_name`, and `model_name`.

### Run the backend

Start the FastAPI server from the repository root:

```bash
fastapi dev app/main.py
```

This exposes the API on `http://127.0.0.1:8000` and provides interactive docs at `http://127.0.0.1:8000/docs`.

### Run the frontend

In a separate terminal, launch the Streamlit UI:

```bash
streamlit run frontend/app.py
```

The UI connects to the backend and allows chat-style commands, file upload, and downloads.

## Usage examples

Use natural language commands in the Streamlit chat interface or send them to the `/chat` API endpoint.

Example commands:

```text
load_dataset("data/train.csv", "target")
set_target("target")
describe_data()
preview_data(5)
train_classification()
train_regression()
optimize_logistic(20)
optimize_forest_regressor(30)
show_best_model("accuracy")
show_history(5)
predict("data/test.csv", "predictions.csv")
help()
```

### API endpoints

The backend exposes the following operations:

- `POST /chat` — send a chat command and receive an AI-paraphrased response
- `POST /upload` — upload dataset files
- `GET /history` — get recent experiment history
- `POST /clear` — clear conversation history
- `GET /download/model` — download the best trained model
- `GET /download/predictions` — download generated prediction results

## Project structure

- `app/main.py` — FastAPI backend entrypoint
- `app/agents/chat_agent.py` — conversational ML agent and tool registry
- `app/services/llm.py` — LLM client configuration and API integration
- `app/agents/tools/` — data loading, model training, optimization utilities
- `frontend/app.py` — Streamlit user interface
- `requirements.txt` — Python dependencies
- `data/` — example or sample dataset files

## Help and support

- Use the built-in API docs at `http://127.0.0.1:8000/docs`
- Open an issue in the repository for bugs or feature requests
- Review `LICENSE` for licensing details

## Contributing

Contributions are welcome via issues and pull requests. Keep changes focused on features, bug fixes, and documentation improvements.

## License

This project is released under the MIT License. See `LICENSE` for details.
