#!/usr/bin/env bash
# exit on error
set -o errexit

pip install --upgrade pip
pip install -r requirements.txt

# Initialize database (you might need to adjust this)
python -c "
import asyncio
from your_bot_file import init_db, engine
asyncio.run(init_db())
"
