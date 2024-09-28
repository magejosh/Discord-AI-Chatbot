import os
import asyncio
import logging
from discord.ext import commands
from dotenv import load_dotenv
from bot_utilities.config_loader import config
from bot_utilities.discord_util import get_discord_token, check_token
from controllers.bot_controller import BotController

load_dotenv()

# Set up the Discord bot
intents = discord.Intents.all()
bot = commands.Bot(command_prefix="/", intents=intents, heartbeat_timeout=60)
TOKEN = os.getenv('DISCORD_TOKEN')  # Loads Discord bot token from environment variables

if TOKEN is None:
    TOKEN = get_discord_token()
else:
    print("\033[33mLooks like the environment variables exists...\033[0m")
    token_status = asyncio.run(check_token(TOKEN))
    if token_status is not None:
        TOKEN = get_discord_token()
        if TOKEN is None:
            print("Failed to retrieve a valid Discord token.")
            logging.error("Failed to retrieve a valid Discord token.")
            sys.exit(1)  # Exit if no valid token is found

# Initialize BotController
bot_controller = BotController(bot)

if __name__ == "__main__":
    bot.run(TOKEN)