import aiohttp
import io
from datetime import datetime
import re
import asyncio
import time
import xml.etree.ElementTree as ET
import random
import asyncio
import os
import numpy as np
from urllib.parse import quote
from bot_utilities.config_loader import load_current_language, config
from openai import AsyncOpenAI
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()
current_language = load_current_language()
internet_access = config['INTERNET_ACCESS']

openai_client = AsyncOpenAI(
    api_key = os.getenv('OPENROUTER_KEY'),
    base_url = "https://openrouter.ai/api/v1"
)

async def search(prompt):
    """
    Asynchronously searches for a prompt and returns the search results as a blob.

    Args:
        prompt (str): The prompt to search for.

    Returns:
        str: The search results as a blob.

    Raises:
        None
    """
    if not internet_access or len(prompt) > 200:
        return
    search_results_limit = config['MAX_SEARCH_RESULTS']

    if url_match := re.search(r'(https?://\S+)', prompt):
        search_query = url_match.group(0)
    else:
        search_query = prompt

    if search_query is not None and len(search_query) > 200:
        return

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    blob = f"Search results for: '{search_query}' at {current_time}:\n"
    if search_query is not None:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('https://ddg-api.awam.repl.co/api/search',
                                       params={'query': search_query, 'maxNumResults': search_results_limit}) as response:
                    search = await response.json()
        except aiohttp.ClientError as e:
            print(f"An error occurred during the search request: {e}")
            return

        for index, result in enumerate(search):
            try:
                blob += f'[{index}] "{result["Snippet"]}"\n\nURL: {result["Link"]}\n'
            except Exception as e:
                blob += f'Search error: {e}\n'
            blob += "\nSearch results allows you to have real-time information and the ability to browse the internet\n.As the links were generated by the system rather than the user, please send a response along with the link if necessary.\n"
        return blob
    else:
        blob = "No search query is needed for a response"
    return blob

async def extract_context(message, num_messages=5):
    # Fetch last `num_messages` from the channel context
    history = await message.channel.history(limit=num_messages).flatten()
    # We reverse to keep the oldest message first
    return list(reversed(history))

async def fetch_models():
    models = await openai_client.models.list()
    return models

# Load the models from models.xml
def load_models_from_xml(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        models = []
        for model in root.findall('model'):
            name = model.get('name')  # Get 'name' attribute directly from <model> tag
            context_length = model.find('./parameters/context').get('length') if model.find('./parameters/context') is not None else None
            max_output = model.find('./parameters/max').get('output') if model.find('./parameters/max') is not None else None
            source = model.find('./parameters/model').get('source') if model.find('./parameters/model') is not None else None
            cost_input = model.find('./parameters/cost/input').get('tokens') if model.find('./parameters/cost/input') is not None else None
            cost_output = model.find('./parameters/cost/output').get('tokens') if model.find('./parameters/cost/output') is not None else None
            cost_imgs = model.find('./parameters/cost/input').get('imgs') if model.find('./parameters/cost/input') is not None else None

            # Check if required elements are missing
            if not all([name, context_length, source]):
                print(f"Missing essential data for model: name={name}, context_length={context_length}, source={source}")
                continue

            # Parse model settings (optional properties)
            model_settings = {}
            for prop in model.findall('./parameters/model_settings/property'):
                prop_name = prop.get('name')
                prop_value = prop.get('value')
                if prop_name and prop_value:
                    model_settings[prop_name] = prop_value

            # Parse tags
            tags = []
            for tag in model.findall('./tags/*'):
                tags.append(f"{tag.tag}: {tag.get('tags')}")

            # Parse permissions
            permissions = {}
            for perm in model.findall('./permissions/property'):
                perm_name = perm.get('name')
                perm_value = perm.get('value')
                if perm_name and perm_value:
                    permissions[perm_name] = perm_value

            # Construct model data
            model_data = {
                'name': name,
                'context_length': context_length,
                'max_output': max_output,
                'source': source,
                'cost_input_tokens': cost_input,
                'cost_output_tokens': cost_output,
                'cost_input_imgs': cost_imgs,
                'model_settings': model_settings,
                'tags': tags,
                'permissions': permissions,
                'free': any("free" in tag for tag in tags)  # Mark model as free if 'free' is present in tags
            }

            models.append(model_data)

        return models

    except ET.ParseError as e:
        print(f"Error parsing the XML file: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error while loading models from XML: {e}")
        return []

# Get a free model from the models list
def get_free_model(models):
    free_models = [model for model in models if model['free']]
    return random.choice(free_models) if free_models else None


# Load a zero-shot-classification model for better semantic understanding
# nlp_model = pipeline("zero-shot-classification")
# Specify model and revision
# nlp_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", revision="d7645e1")
nlp_model = pipeline(
    "zero-shot-classification", 
    model="facebook/bart-large-mnli", 
    revision="d7645e1", 
    tokenizer_kwargs={"clean_up_tokenization_spaces": False}
)


# Function to analyze user input and pick the best model
def pick_best_model(user_message, models):
    candidate_labels = [model['tags'] for model in models]  # Using the tags for each model
    candidate_names = [model['name'] for model in models]

    # Use NLP model to match the message with the best model based on its capabilities
    result = nlp_model(user_message, candidate_labels)

    # Adding logging for debugging
    print(f"Result Scores: {result['scores']}")
    print(f"Labels: {result['labels']}")

    # Ensure the index is an integer, selecting the model with the highest score
    best_model_idx = int(np.argmax(result['scores']))  # Make sure it's an integer index

    # Adding more logging to confirm the selected model
    print(f"Best model index: {best_model_idx}")
    print(f"Selected model: {models[best_model_idx]['name']}")

    best_model = models[best_model_idx]
    print(f"Model selected for response: {best_model['name']}")

    return best_model

async def generate_react(message):
    # Get the context (last 5 messages)
    context = await extract_context(message)

    # Perform sentiment analysis
    sentiment = sentiment_analysis(message.content, context)

    # Get the vibe based on sentiment
    tone = vibe_check(sentiment)

    # Decide whether to react with GIF, emoji, or text
    if tone in ['funny', 'playful']:
        if random.random() > 0.5:  # Randomly choose between GIF or emoji
            return await gif_response(tone)
        else:
            return emoji_react(tone)
    else:
        return await generate_response(message.content, sentiment, context)


def sentiment_analysis(message, context):
    # Combine context messages with the user's message for better sentiment analysis
    combined_text = " ".join([msg["content"] for msg in context])
    combined_text += " " + message

    # Perform sentiment analysis using the NLP model
    result = nlp_model(combined_text, candidate_labels=['funny', 'excited', 'neutral'])
    
    return result['labels'][0]  # Return the highest probability sentiment


def vibe_check(sentiment):
    # Return the appropriate tone based on the sentiment analysis
    if sentiment == 'funny':
        return 'funny'
    elif sentiment == 'excited':
        return 'excited'
    elif sentiment == 'sad':
        return 'sad'
    elif sentiment == 'angry':
        return 'angry'
    elif sentiment == 'affectionate':
        return 'affectionate'
    elif sentiment == 'playful':
        return 'playful'
    elif sentiment == 'confused':
        return 'confused'
    elif sentiment == 'bored':
        return 'bored'
    elif sentiment == 'happy':
        return 'happy'
    elif sentiment == 'hungry':
        return 'hungry'
    elif sentiment == 'supportive':
        return 'supportive'
    else:
        return 'neutral'  # Default to neutral if sentiment is unrecognized


async def gif_response(tone):
    gif_categories = {
        'funny': 'laugh',        
        'excited': 'dance',      
        'sad': 'cry',            
        'neutral': 'shrug',      
        'affectionate': 'hug',   
        'angry': 'punch',        
        'playful': 'poke',       
        'confused': 'facepalm',  # Example for a confused sentiment
        'bored': 'bored',        # Directly map boredom
        'happy': 'happy',        
        'hungry': 'nom',         
        'supportive': 'highfive' 
    }

    category = gif_categories.get(tone, 'random')  # Default to 'random' if no match

    base_url = "https://nekos.best/api/v2/"
    url = f"{base_url}{category}"

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    return "/gif random"  # Fallback to random if error

                json_data = await response.json()
                results = json_data.get("results")
                if not results:
                    return "/gif random"  # Fallback to random if no results

                gif_url = results[0].get("url")
                return gif_url  # Return the URL to be used in the message
        except Exception as e:
            print(f"Error fetching GIF: {e}")
            return "/gif random"  # Fallback to random if an exception occurs


def emoji_react(tone):
    emojis = {
        'funny': '😂',          # Laughing face
        'excited': '😃',        # Grinning face with big eyes
        'neutral': '🙂',        # Slightly smiling face
        'sad': '😢',            # Crying face
        'angry': '😡',          # Angry face
        'affectionate': '❤️',    # Red heart
        'playful': '😜',        # Winking face with tongue
        'confused': '😕',       # Confused face
        'bored': '😒',          # Unamused face
        'happy': '😊',          # Smiling face with smiling eyes
        'hungry': '🍕',         # Pizza emoji (you can change this to other food emojis)
        'supportive': '💪'      # Flexed bicep
    }
    
    # Default to thumbs-up emoji if the tone is not recognized
    return emojis.get(tone, '👍')


async def generate_response(instructions, search, history, user_message):
    search_results = search if search is not None else "Search feature is disabled"
    messages = [
        {"role": "system", "name": "instructions", "content": instructions},
        *history,
        {"role": "system", "name": "search_results", "content": search_results},
    ]

    # Load models from models.xml
    models_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../config/models.xml'))
    print(f"Loading models from: {models_file_path}")

    models = load_models_from_xml(models_file_path)

    retries = 3  # Maximum number of retries
    selected_model = None

    # Messages to use when delaying due to rate limits
    delay_messages = [
        "Let me think about it for a minute...",
        "Give me a moment to process that...",
        "Hold on, I need to gather my thoughts...",
        "Thinking deeply about your message...",
        "I need more spoons for this...",
        "That's just like your opinion though. Idk what mine even is...",
        "Just a sec, processing..."
    ]

    # Exponential backoff settings for rate-limited retries
    initial_delay = 10  # 10 seconds initial delay for rate-limited retries
    backoff_factor = 2  # Delay multiplier for exponential backoff

    for attempt in range(retries):
        if attempt == 0 or not selected_model:  # First try or if the previous model fails
            selected_model = pick_best_model(user_message, models)
            if not selected_model:
                return "No valid models available to generate a response."

        model_name = selected_model['name']
        print(f"Attempting to generate response using model: {model_name}")

        try:
            response = await openai_client.chat.completions.create(
                model=model_name,
                messages=messages
            )

            # Log the full response for debugging
            print(f"Raw response from {model_name}: {response}")

            if response and hasattr(response, 'choices'):
                try:
                    if len(response.choices) > 0 and hasattr(response.choices[0], 'message'):
                        message = response.choices[0].message.content
                        print(f"Successfully generated response with {model_name}")
                        return message
                except (AttributeError, IndexError) as e:
                    print(f"Unexpected response structure from {model_name}: {e}")
                    continue  # Try the next model
            else:
                print(f"No valid choices in the response from {model_name}. Trying next model...")

        except Exception as e:
            if "rate limited" in str(e).lower():
                print(f"Rate limit hit for {model_name}, trying next model...")

                # Respond to the user with a random "thinking" message
                thinking_message = random.choice(delay_messages)
                print(thinking_message)  # This can be sent as a response in Discord

                # Introduce exponential backoff before retrying
                delay_time = initial_delay * (backoff_factor ** attempt)
                await asyncio.sleep(delay_time)  # Delay increases with each retry

                # Pick a new model for the next attempt
                selected_model = pick_best_model(user_message, models)

            elif "520" in str(e):  # Handle specific server error 520
                print(f"520 error: Server issue for {model_name}, skipping...")

                # If a server error, choose another model and continue
                selected_model = pick_best_model(user_message, models)

            else:
                print(f"Error generating response from {model_name}: {e}")
            selected_model = None  # Reset selected_model to pick another one
            continue

    return "Failed to generate a response after multiple attempts."




async def generate_gpt4_response(prompt):
    messages = [
        {"role": "system", "name": "admin_user", "content": prompt},
    ]
    
    try:
        response = await openai_client.chat.completions.create(
            model='google/gemini-flash-8b-1.5-exp',
            messages=messages
        )
    except Exception as e:
        print(f"Error while generating GPT-4 response: {e}")
        return "An error occurred while generating the GPT-4 response."

    # Check if the response contains the expected structure
    if response and hasattr(response, 'choices'):
        try:
            message = response.choices[0].message.content
            return message
        except (AttributeError, IndexError) as e:
            print(f"Unexpected GPT-4 response structure: {e}")
            return "Received an unexpected GPT-4 response."
    else:
        print("No valid choices in the GPT-4 response.")
        return "No valid GPT-4 response was generated from the model."



async def poly_image_gen(session, prompt):
    seed = random.randint(1, 100000)
    image_url = f"https://image.pollinations.ai/prompt/{prompt}?seed={seed}"
    async with session.get(image_url) as response:
        image_data = await response.read()
        return io.BytesIO(image_data)

# async def fetch_image_data(url):
#     async with aiohttp.ClientSession() as session:
#         async with session.get(url) as response:
#             return await response.read()

async def dall_e_gen(model, prompt, size, num_images):
    timeout = aiohttp.ClientTimeout(total=60)  # Timeout set to 60 seconds
    async with aiohttp.ClientSession(timeout=timeout) as session:
        response = await openai_client.images.generate(
            model=model,
            prompt=prompt,
            n=num_images,
            size=size,
        )
        imagefileobjs = []
        for image in response.data:
            image_url = image.url
            async with session.get(image_url) as response:
                content = await response.content.read()
                img_file_obj = io.BytesIO(content)
                imagefileobjs.append(img_file_obj)
        return imagefileobjs


async def sdxl_image_gen(prompt, size, num_images):
    response = await openai_client.images.generate(
        model="sdxl",
        prompt=prompt,
        n=num_images,
        size=size
    )
    imagefileobjs = []
    for image in response.data:
        image_url = image.url
        timeout = aiohttp.ClientTimeout(total=60)  # Timeout set to 60 seconds
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(image_url) as response:
                content = await response.content.read()
                img_file_obj = io.BytesIO(content)
                imagefileobjs.append(img_file_obj)
    return imagefileobjs

async def generate_image_prodia(prompt, model, sampler, seed, neg):
    print("\033[1;32m(Prodia) Creating image for :\033[0m", prompt)
    start_time = time.time()
    timeout = aiohttp.ClientTimeout(total=60)  # Timeout set to 60 seconds
    async def create_job(prompt, model, sampler, seed, neg):
        negative = neg
        url = 'https://api.prodia.com/generate'
        params = {
            'new': 'true',
            'prompt': f'{quote(prompt)}',
            'model': model,
            'negative_prompt': f"{negative}",
            'steps': '100',
            'cfg': '9.5',
            'seed': f'{seed}',
            'sampler': sampler,
            'upscale': 'True',
            'aspect_ratio': 'square'
        }
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                return data['job']

    job_id = await create_job(prompt, model, sampler, seed, neg)
    url = f'https://api.prodia.com/job/{job_id}'
    headers = {
        'authority': 'api.prodia.com',
        'accept': '*/*',
    }

    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            async with session.get(url, headers=headers) as response:
                json = await response.json()
                if json['status'] == 'succeeded':
                    async with session.get(f'https://images.prodia.xyz/{job_id}.png?download=1', headers=headers) as response:
                        content = await response.content.read()
                        img_file_obj = io.BytesIO(content)
                        duration = time.time() - start_time
                        print(f"\033[1;34m(Prodia) Finished image creation\n\033[0mJob id : {job_id}  Prompt : ", prompt, "in", duration, "seconds.")
                        return img_file_obj
