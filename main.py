#!/usr/bin/env python3


import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import asyncio
import os
from dotenv import load_dotenv
from pathlib import Path

# Get the folder this file is in:
this_file_folder = os.path.dirname(os.path.realpath(__file__))
load_dotenv(Path(this_file_folder) / ".env")

from src.slackbot import slack_bot

async def start():
    await slack_bot.start()

if __name__ == "__main__":
    asyncio.run(start())
