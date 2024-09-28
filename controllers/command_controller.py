import discord
from discord.ext import commands
from discord import app_commands
from bot_utilities.config_loader import load_current_language
from bot_utilities.ai_utils import generate_gpt4_response, split_response, sdxl_image_gen, poly_image_gen, dall_e_gen, generate_image_prodia
from bot_utilities.discord_util import is_nsfw_channel

class CommandController:
    def __init__(self, bot):
        self.bot = bot
        self.current_language = load_current_language()

    def register_commands(self):
        @self.bot.hybrid_command(name="pfp", description=self.current_language["pfp"])
        @commands.is_owner()
        async def pfp(ctx, attachment: discord.Attachment):
            await ctx.defer()
            if not attachment.content_type.startswith('image/'):
                await ctx.send("Please upload an image file.")
                return
            await ctx.send(self.current_language['pfp_change_msg_2'])
            await self.bot.user.edit(avatar=await attachment.read())

        @self.bot.hybrid_command(name="ping", description=self.current_language["ping"])
        async def ping(ctx):
            latency = self.bot.latency * 1000
            await ctx.send(f"{self.current_language['ping_msg']}{latency:.2f} ms")

        @self.bot.hybrid_command(name="changeusr", description=self.current_language["changeusr"])
        @commands.is_owner()
        async def changeusr(ctx, new_username):
            await ctx.defer()
            taken_usernames = [user.name.lower() for user in ctx.guild.members]
            if new_username.lower() in taken_usernames:
                message = f"{self.current_language['changeusr_msg_2_part_1']}{new_username}{self.current_language['changeusr_msg_2_part_2']}"
            else:
                try:
                    await self.bot.user.edit(username=new_username)
                    message = f"{self.current_language['changeusr_msg_3']}'{new_username}'"
                except discord.errors.HTTPException as e:
                    message = "".join(e.text.split(":")[1:])
            sent_message = await ctx.send(message)
            await asyncio.sleep(3)
            await sent_message.delete()

        @self.bot.hybrid_command(name="toggledm", description=self.current_language["toggledm"])
        @commands.has_permissions(administrator=True)
        async def toggledm(ctx):
            global allow_dm
            allow_dm = not allow_dm
            await ctx.send(f"DMs are now {'on' if allow_dm else 'off'}", delete_after=3)

        @self.bot.hybrid_command(name="toggleactive", description=self.current_language["toggleactive"])
        @app_commands.choices(persona=[
            app_commands.Choice(name=persona.capitalize(), value=persona)
            for persona in instruction
        ])
        @commands.has_permissions(administrator=True)
        async def toggleactive(ctx, persona: app_commands.Choice[str] = instruction[instruc_config]):
            channel_id = f"{ctx.channel.id}"
            if channel_id in active_channels:
                del active_channels[channel_id]
                with open("channels.json", "w", encoding='utf-8') as f:
                    json.dump(active_channels, f, indent=4)
                await ctx.send(f"{ctx.channel.mention} {self.current_language['toggleactive_msg_1']}", delete_after=3)
            else:
                if persona.value:
                    active_channels[channel_id] = persona.value
                else:
                    active_channels[channel_id] = persona
                with open("channels.json", "w", encoding='utf-8') as f:
                    json.dump(active_channels, f, indent=4)
                await ctx.send(f"{ctx.channel.mention} {self.current_language['toggleactive_msg_2']}", delete_after=3)

        @self.bot.hybrid_command(name="clear", description=self.current_language["bonk"])
        async def clear(ctx):
            key = f"{ctx.author.id}-{ctx.channel.id}"
            try:
                message_history[key].clear()
            except Exception as e:
                await ctx.send("⚠️ There is no message history to be cleared", delete_after=2)
                return
            await ctx.send(f"Message history has been cleared", delete_after=4)

        @self.bot.hybrid_command(name="stop_voice_response", description="Stop the bot from speaking in the voice channel")
        async def stop_voice_response(ctx):
            voice_client = discord.utils.get(self.bot.voice_clients, guild=ctx.guild)
            if voice_client and voice_client.is_connected():
                await voice_client.disconnect()
                await ctx.send("Stopped voice response and left the voice channel.", delete_after=5)
            else:
                await ctx.send("The bot is not in a voice channel.", delete_after=5)

        @self.bot.hybrid_command(name="imagine", description="Command to imagine an image")
        @app_commands.choices(sampler=[
            app_commands.Choice(name='📏 Euler (Recommended)', value='Euler'),
            app_commands.Choice(name='📏 Euler a', value='Euler a'),
            app_commands.Choice(name='📐 Heun', value='Heun'),
            app_commands.Choice(name='💥 DPM++ 2M Karras', value='DPM++ 2M Karras'),
            app_commands.Choice(name='💥 DPM++ SDE Karras', value='DPM++ SDE Karras'),
            app_commands.Choice(name='🔍 DDIM', value='DDIM')
        ])
        @app_commands.choices(model=[
            app_commands.Choice(name='🙂 SDXL (The best of the best)', value='sdxl'),
            app_commands.Choice(name='🌈 Elldreth vivid mix (Landscapes, Stylized characters, nsfw)', value='ELLDRETHVIVIDMIX'),
            app_commands.Choice(name='💪 Deliberate v2 (All-in-one, nsfw)', value='DELIBERATE'),
            app_commands.Choice(name='🔮 Dreamshaper', value='DREAMSHAPER_6'),
            app_commands.Choice(name='🎼 Lyriel', value='LYRIEL_V16'),
            app_commands.Choice(name='💥 Anything diffusion (Good for anime)', value='ANYTHING_V4'),
            app_commands.Choice(name='🌅 Midjourney', value='midjourney'),
            app_commands.Choice(name='🌅 Openjourney (Midjourney alternative)', value='OPENJOURNEY'),
            app_commands.Choice(name='🏞️ Realistic (Lifelike pictures)', value='REALISTICVS_V20'),
            app_commands.Choice(name='👨‍🎨 Portrait', value='PORTRAIT'),
            app_commands.Choice(name='🌟 Rev animated (Illustration, Anime)', value='REV_ANIMATED'),
            app_commands.Choice(name='🤖 Analog', value='ANALOG'),
            app_commands.Choice(name='🌌 AbyssOrangeMix', value='ABYSSORANGEMIX'),
            app_commands.Choice(name='🌌 Dreamlike v1', value='DREAMLIKE_V1'),
            app_commands.Choice(name='🌌 Dreamlike v2', value='DREAMLIKE_V2'),
            app_commands.Choice(name='🌌 Dreamshaper 5', value='DREAMSHAPER_5'),
            app_commands.Choice(name='🌌 MechaMix', value='MECHAMIX'),
            app_commands.Choice(name='🌌 MeinaMix', value='MEINAMIX'),
            app_commands.Choice(name='🌌 Stable Diffusion v14', value='SD_V14'),
            app_commands.Choice(name='🌌 Stable Diffusion v15', value='SD_V15'),
            app_commands.Choice(name="🌌 Shonin's Beautiful People", value='SBP'),
            app_commands.Choice(name="🌌 TheAlly's Mix II", value='THEALLYSMIX'),
            app_commands.Choice(name='🌌 Timeless', value='TIMELESS')
        ])
        @app_commands.describe(
            prompt="Write an amazing prompt for an image",
            model="Model to generate image",
            sampler="Sampler for denoising",
            negative="Specify what you do NOT want the model to include",
            num_images="Specify the number of images (Seed incremented)",
        )
        async def imagine(ctx, prompt: str, model: app_commands.Choice[str], sampler: app_commands.Choice[str], negative: str = None, num_images: int = 1, seed: int = None):
            try:
                deferred = False  # To track if the interaction has been deferred
                if is_nsfw_channel(ctx.channel) and any(word in prompt for word in blacklisted_words):
                    await ctx.send(f"⚠️ NSFW images can only be posted in age-restricted channels", delete_after=30)
                    return
                if seed is None:
                    seed = random.randint(10000, 99999)
                if negative is None:
                    negative = ', '.join(str(word) for word in image_negatives)
                if not is_nsfw_channel(ctx.channel):
                    negative += ', ' + ', '.join(str(word) for word in blacklisted_words)
                if not deferred:
                    await ctx.defer()
                    deferred = True
                if model.value == 'sdxl':
                    await imagine_sdxl(ctx, prompt, size=app_commands.Choice(name="Large", value="1024x1024"), num_images=num_images)
                    return
                model_uid = Model[model.value].value[0]
                if num_images > 10:
                    num_images = 10  # Limit number of images
                tasks = []
                async with aiohttp.ClientSession() as session:
                    while len(tasks) < num_images:
                        task = asyncio.ensure_future(generate_image_prodia(prompt, model_uid, sampler.value, seed + (len(tasks) - 1), negative))
                        tasks.append(task)
                    generated_images = await asyncio.gather(*tasks)
                if not generated_images or all(img is None for img in generated_images):
                    raise Exception("Failed to generate images.")
                files = []
                for index, image in enumerate(generated_images):
                    img_file = discord.File(image, filename=f"image_{seed + index}.png", description=prompt)
                    files.append(img_file)
                embed = discord.Embed(color=discord.Color.random())
                embed.title = f'🎨 {prompt}'
                embed.add_field(name='🤖 Model', value=f'🤖 {model.value}', inline=True)
                embed.add_field(name='🧬 Sampler', value=f'🧬 {sampler.value}', inline=True)
                embed.add_field(name='🌱 Seed', value=f'🌱 {str(seed)}', inline=True)
                await ctx.send(embed=embed, files=files)
            except Exception as e:
                if deferred:
                    await ctx.followup.send(f"An error occurred while generating the image: {str(e)}")
                else:
                    await ctx.send(f"An error occurred while generating the image: {str(e)}")
                logging.error(f"Error in imagine command: {str(e)}")

        @self.bot.hybrid_command(name="imagine-sdxl", description="Create images using SDXL")
        @commands.guild_only()
        @app_commands.choices(size=[
            app_commands.Choice(name='🔳 Small', value='256x256'),
            app_commands.Choice(name='🔳 Medium', value='512x512'),
            app_commands.Choice(name='🔳 Large', value='1024x1024')
        ])
        @app_commands.describe(
            prompt="Write an amazing prompt for an image",
            size="Choose the size of the image"
        )
        async def imagine_sdxl(ctx, prompt, size: app_commands.Choice[str], num_images : int = 1):
            await ctx.defer()
            size = size.value
            if num_images > 10:
                num_images = 10
            imagefileobjs = await sdxl_image_gen(prompt, size, num_images)
            for imagefileobj in imagefileobjs:
                file = discord.File(imagefileobj, filename="image.png", spoiler=True, description=prompt)
                sent_message =  await ctx.send(file=file)
                reactions = ["⬆️", "⬇️"]
                for reaction in reactions:
                    await sent_message.add_reaction(reaction)

        @self.bot.hybrid_command(name="imagine-pollinations", description="Bring your imagination into reality with pollinations.ai!")
        @commands.describe(images="Choose the number of images.")
        @commands.describe(prompt="Provide a description of your imagination to turn into images.")
        async def imagine_poly(ctx, *, prompt: str, images: int = 4):
            await ctx.defer(ephemeral=True)
            images = min(images, 18)
            tasks = []
            async with aiohttp.ClientSession() as session:
                while len(tasks) < images:
                    task = asyncio.ensure_future(poly_image_gen(session, prompt))
                    tasks.append(task)
                generated_images = await asyncio.gather(*tasks)
            files = []
            for index, image in enumerate(generated_images):
                file = discord.File(image, filename=f"image_{index+1}.png")
                files.append(file)
            await ctx.send(files=files, ephemeral=True)

        @self.bot.hybrid_command(name="gif", description=self.current_language["nekos"])
        @app_commands.choices(category=[
            app_commands.Choice(name=category.capitalize(), value=category)
            for category in ['baka', 'bite', 'blush', 'bored', 'cry', 'cuddle', 'dance', 'facepalm', 'feed', 'handhold', 'happy', 'highfive', 'hug', 'kick', 'kiss', 'laugh', 'nod', 'nom', 'nope', 'pat', 'poke', 'pout', 'punch', 'shoot', 'shrug']
        ])
        async def gif(ctx, category: app_commands.Choice[str]):
            base_url = "https://nekos.best/api/v2/"
            url = base_url + category.value
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        await ctx.channel.send("Failed to fetch the image.")
                        return
                    json_data = await response.json()
                    results = json_data.get("results")
                    if not results:
                        await ctx.channel.send("No image found.")
                        return
                    image_url = results[0].get("url")
                    embed = discord.Embed(colour=0x141414)
                    embed.set_image(url=image_url)
                    await ctx.send(embed=embed)

        @self.bot.hybrid_command(name="askgpt4", description="Ask gpt4 a question")
        async def ask(ctx, prompt: str):
            await ctx.defer()
            response = await generate_gpt4_response(prompt=prompt)
            for chunk in split_response(response):
                await ctx.send(chunk, allowed_mentions=discord.AllowedMentions.none(), suppress_embeds=True)

        self.bot.remove_command("help")
        @self.bot.hybrid_command(name="help", description=self.current_language["help"])
        async def help(ctx):
            embed = discord.Embed(title="Bot Commands", color=0x03a64b)
            embed.set_thumbnail(url=self.bot.user.avatar.url)
            command_tree = self.bot.commands
            for command in command_tree:
                if command.hidden:
                    continue
                command_description = command.description or "No description available"
                embed.add_field(name=command.name, value=command_description, inline=False)
            embed.set_footer(text=f"{self.current_language['help_footer']}")
            embed.add_field(name="Need Support?", value="For further assistance or support, run `/support` command.", inline=False)

            await ctx.send(embed=embed)