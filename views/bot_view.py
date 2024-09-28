import discord
from discord.ext import commands

class BotView:
    def __init__(self, bot):
        self.bot = bot

    async def send_embed(self, ctx, title, description, color=0x03a64b):
        embed = discord.Embed(title=title, description=description, color=color)
        await ctx.send(embed=embed)
