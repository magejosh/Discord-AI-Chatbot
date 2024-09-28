import logging
import discord
from discord.ext import commands
from bot_utilities.config_loader import config, load_current_language, load_instructions
from bot_utilities.ai_utils import fetch_chat_models
from controllers.command_controller import CommandController
from controllers.event_controller import EventController

class BotController:
    def __init__(self, bot):
        self.bot = bot
        self.command_controller = CommandController(bot)
        self.event_controller = EventController(bot)
        self.setup()

    def setup(self):
        @self.bot.event
        async def on_ready():
            await self.event_controller.on_ready()

        @self.bot.event
        async def on_message(message):
            await self.event_controller.on_message(message)

        @self.bot.event
        async def on_message_delete(message):
            await self.event_controller.on_message_delete(message)

        # Register commands
        self.command_controller.register_commands()
