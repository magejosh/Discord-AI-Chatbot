import logging
import discord
from discord.ext import commands
from bot_utilities.config_loader import load_current_language, load_instructions
from bot_utilities.ai_utils import generate_response, sentiment_analysis, vibe_check
from bot_utilities.discord_util import is_nsfw_channel

class EventController:
    def __init__(self, bot):
        self.bot = bot
        self.current_language = load_current_language()
        self.instruction = load_instructions()

    async def on_ready(self):
        logging.info(f"Logged in as {self.bot.user.name}")
        print(f"Logged in as {self.bot.user.name}")

    async def on_message(self, message):
        if message.author == self.bot.user:
            return

        if message.content.startswith('/'):
            await self.bot.process_commands(message)
            return

        if message.guild is None and not allow_dm:
            return

        if message.guild and not is_nsfw_channel(message.channel):
            return

        # Process the message
        response = await generate_response(
            instructions=self.instruction,
            search=None,
            history=[],
            user_message=message.content
        )
        await message.channel.send(response)

    async def on_message_delete(self, message):
        logging.info(f"Message deleted: {message.content}")
