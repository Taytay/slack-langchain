#!/usr/bin/env python3

print("here in main.py")
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import asyncio
import os
from dotenv import load_dotenv
from pathlib import Path

print("After imports")

# Get the folder this file is in:
this_file_folder = os.path.dirname(os.path.realpath(__file__))

print("About to load .env")
load_dotenv(Path(this_file_folder) / ".env")

print("Done")

print("About to load slackbot")
from src.slackbot import slack_bot
print("Done")

async def start():
    await slack_bot.start()

print("About to load the main function: " + __name__)
if __name__ == "__main__":
    print("About to start...")
    asyncio.run(start())
