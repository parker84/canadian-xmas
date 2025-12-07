# canadian-xmas

Repo to create a 🇨🇦 Canadian Christmas 🎄 shopping experience.

## Setup with uv

### Install uv
```sh
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with Homebrew
brew install uv
```

### Setup environment
```sh
# Create virtual environment and install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

## Running locally
```sh
# Set environment variables
export OPENAI_API_KEY=your_openai_api_key
export PUREMD_API_KEY=your_puremd_api_key
export COHERE_API_KEY=your_cohere_api_key

export LOG_LEVEL=DEBUG

# Run the app
uv run streamlit run main.py
```

## Quick Commands
```sh
# Add a new dependency
uv add package_name

# Run without activating venv
uv run streamlit run main.py

# Update dependencies
uv sync --upgrade
```