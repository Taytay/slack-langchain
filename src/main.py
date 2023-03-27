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
# Get the parent folder of this file's folder:
parent_folder = os.path.dirname(this_file_folder)

print("About to load .env")
load_dotenv(Path(parent_folder) / ".env")

print("Done")

print("About to load slackbot")
from slackbot import slack_bot
print("Done")

async def start():
    await slack_bot.start()

print("About to load the main function: " + __name__)
if __name__ == "__main__":
    print("About to start...")
    asyncio.run(start())
