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
import logging
import sys

# Set up logging
log_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'logs.txt'))

class UnicodeStreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream)
        if stream is None:
            stream = sys.stdout
        if hasattr(stream, 'encoding') and stream.encoding.lower() != 'utf-8':
            self.setStream(open(stream.fileno(), mode='w', encoding='utf-8', buffering=1))

logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to see all messages
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8'),
        UnicodeStreamHandler(sys.stdout)
    ]
)

# Add this in your initialization section, before running the bot
logging.info("Logging system initialized.")

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

def log_model_issue(issue, model_name, field_name):
    """
    Log an issue with a specific field in the model.
    """
    logging.error(f"Model '{model_name}' missing '{field_name}' field. Issue: {issue}")

# Load the models from models.xml
def load_models_from_xml(file_path):
    try:
        # Parse the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()

        models = []
        # Loop through each model in the XML
        for model_element in root.findall('.//model'):
            # Get model name
            name = model_element.get('name')
            if not name:
                logging.error("Model is missing 'name' attribute")
                continue  # Skip this model if the name is missing

            # Get parameters
            context_element = model_element.find('./parameters/context')
            context_length = context_element.get('length') if context_element is not None else None
            max_output_element = model_element.find('./parameters/max')
            max_output = max_output_element.get('output') if max_output_element is not None else None
            source_element = model_element.find('./parameters/model')
            source = source_element.get('source') if source_element is not None else None
            cost_input_element = model_element.find('./parameters/cost/input')
            cost_input = cost_input_element.get('tokens') if cost_input_element is not None else None
            cost_output_element = model_element.find('./parameters/cost/output')
            cost_output = cost_output_element.get('tokens') if cost_output_element is not None else None

            # Skip this model if essential data is missing
            if not all([context_length, max_output, source, cost_input, cost_output]):
                logging.error(f"Skipping model '{name}' due to missing essential data.")
                continue

            # Parse model settings (optional properties)
            model_settings = {}
            for prop in model_element.findall('./parameters/model_settings/property'):
                prop_name = prop.get('name')
                prop_value = prop.get('value')
                if prop_name and prop_value:
                    model_settings[prop_name] = prop_value

            # Parse tags and detect embedding models
            tags = []
            is_embedding_model = False
            for tag_element in model_element.findall('./tags/tag'):
                tag_value = tag_element.get('value')
                if tag_value:
                    tags.append(tag_value)
                    if 'embedding' in tag_value.lower():
                        is_embedding_model = True

            # Parse permissions
            permissions = {}
            for perm in model_element.findall('./permissions/property'):
                perm_name = perm.get('name')
                perm_value = perm.get('value')
                if perm_name and perm_value:
                    permissions[perm_name] = perm_value

            # Construct the model data
            model_data = {
                'name': name,
                'context_length': context_length,
                'max_output': max_output,
                'source': source,
                'cost_input_tokens': cost_input,
                'cost_output_tokens': cost_output,
                'model_settings': model_settings,
                'tags': tags,
                'permissions': permissions,
                'is_embedding_model': is_embedding_model,
                'free': any("free" in tag.lower() for tag in tags),  # Mark model as free if 'free' is present in tags
                'provider': source_element.get('provider') if source_element is not None else 'openrouter'
            }

            # Log the successful parsing of the model
            logging.info(f"Successfully parsed model '{name}'")

            # Append the parsed model data
            models.append(model_data)

        return models

    except ET.ParseError as e:
        logging.error(f"Error parsing the XML file: {e}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error while loading models from XML: {e}")
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

async def semantic_analysis(user_input):
    """
    Analyzes the user's input using the zero-shot classification model to determine the appropriate tags.

    Args:
        user_input (str): The message to be analyzed.

    Returns:
        list: A list of relevant tags that are suited for the model to process the input.
    """
    candidate_labels = ['informative', 'funny', 'angry', 'supportive', 'neutral', 'helpful']

    try:
        # Use the zero-shot classification model with multi_label=True
        result = nlp_model(user_input, candidate_labels, multi_label=True)
        
        # Log the labels and their corresponding scores
        labels_scores = list(zip(result['labels'], result['scores']))
        logging.info(f"Semantic analysis for '{user_input}': {labels_scores}")
        
        # You can set a threshold to consider which labels are relevant
        threshold = 0.5  # Adjust based on your requirements
        relevant_labels = [label for label, score in labels_scores if score >= threshold]
        
        return relevant_labels  # Return the list of relevant labels
    except Exception as e:
        logging.error(f"Error in semantic analysis: {e}")
        return ['neutral']  # Fallback to neutral if there's an error


# Function to analyze user input and pick the best model
def pick_best_model(user_message, models):
    """
    This function analyzes user input and picks the best model based on the semantic analysis.
    It adds more randomness to break ties between similar models and ensures provider alternation.
    """
    # Filter out embedding models
    non_embedding_models = [model for model in models if not model.get('is_embedding_model', False)]
    
    # Check if there are non-embedding models available
    if not non_embedding_models:
        logging.error("No available models to generate a response.")
        return None

    # Flatten the candidate labels (tags)
    candidate_labels = [' '.join(model['tags']) for model in non_embedding_models]

    # Perform zero-shot classification with multi_label=True
    try:
        result = nlp_model(user_message, candidate_labels, multi_label=True)
    except Exception as e:
        logging.error(f"Error in zero-shot classification: {e}")
        return random.choice(non_embedding_models)  # Fallback

    # Log the scores and labels for debugging
    logging.info(f"Result Scores: {result['scores']}")
    logging.info(f"Labels: {result['labels']}")

    # Introduce randomness among top 5 candidates
    top_n_candidates = 5
    sorted_indices = sorted(range(len(result['scores'])), key=lambda i: result['scores'][i], reverse=True)
    top_indices = sorted_indices[:top_n_candidates]
    top_models = [non_embedding_models[i] for i in top_indices]

    # Shuffle the top models to introduce randomness
    random.shuffle(top_models)

    # Log the randomized selection process
    logging.info(f"Shuffled top models: {[model['name'] for model in top_models]}")

    # Select the first model after shuffling
    best_model = top_models[0]
    logging.info(f"Selected model for response: {best_model['name']}")

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


async def sentiment_analysis(message, context, candidate_labels=None):
    """
    Analyzes the sentiment of the combined context and the current user message.

    Args:
        message (str): The current message from the user.
        context (list): A list of previous messages for context.
        candidate_labels (list, optional): List of sentiment categories to classify the message.

    Returns:
        list: A list of labels indicating the sentiment of the message.
    """
    # Default candidate labels for sentiment analysis
    if candidate_labels is None:
        candidate_labels = ['funny', 'excited', 'neutral', 'angry', 'supportive', 'sad']

    # Combine the context messages with the user's current message
    combined_text = " ".join([msg["content"] for msg in context]) + " " + message

    try:
        # Limit the text length to avoid performance issues
        combined_text = combined_text[:1000]  # Adjust as needed

        # Use the zero-shot classification model with multi_label=True
        result = await asyncio.wait_for(
            asyncio.to_thread(nlp_model, combined_text, candidate_labels=candidate_labels, multi_label=True),
            timeout=5
        )

        # Log the labels and their corresponding scores
        labels_scores = list(zip(result['labels'], result['scores']))
        logging.info(f"Sentiment analysis result: {labels_scores}")

        # Apply a threshold to determine significant sentiments
        threshold = 0.5  # Adjust based on your requirements
        significant_sentiments = [label for label, score in labels_scores if score >= threshold]

        return significant_sentiments  # Return the list of significant sentiments

    except asyncio.TimeoutError:
        logging.warning("Sentiment analysis timed out.")
        return ['neutral']  # Default to 'neutral' if a timeout occurs

    except Exception as e:
        logging.error(f"Error during sentiment analysis: {e}")
        return ['neutral']  # Fallback to neutral if an error occurs



def vibe_check(sentiments):
    """
    Determines the appropriate tone(s) based on the sentiments analysis.

    Args:
        sentiments (list): A list of sentiment labels.

    Returns:
        list: A list of tones corresponding to the sentiments.
    """
    sentiment_to_tone = {
        'funny': 'funny',
        'excited': 'excited',
        'sad': 'sad',
        'angry': 'angry',
        'affectionate': 'affectionate',
        'playful': 'playful',
        'confused': 'confused',
        'bored': 'bored',
        'happy': 'happy',
        'hungry': 'hungry',
        'supportive': 'supportive',
        'neutral': 'neutral'
    }

    tones = [sentiment_to_tone.get(sentiment, 'neutral') for sentiment in sentiments]
    # Remove duplicates while preserving order
    tones = list(dict.fromkeys(tones))
    return tones


def get_weighted_probs(tone):
    # Adjust these probabilities to make text responses more frequent
    weightings = {
        'funny': (0.7, 0.4, 0.3),       # High chance for text, moderate for GIF, low for emoji
        'excited': (0.6, 0.4, 0.5),     # Higher chance for text, balanced GIF/emoji
        'sad': (0.8, 0.3, 0.2),         # High chance for text, low for GIF and emoji
        'angry': (0.8, 0.13, 0.81),       # High chance for text, low for GIF and emoji
        'affectionate': (0.7, 0.5, 0.6),# High chance for text, moderate for GIF and emoji
        'playful': (0.6, 0.5, 0.4),     # Higher chance for text, moderate for GIF, low for emoji
        'confused': (0.17, 0.33, 0.63),    # High chance for text, low for GIF and emoji
        'bored': (0.81, 0.3, 0.5),       # High chance for text, low for GIF and emoji
        'happy': (0.6, 0.5, 0.4),       # Moderate chance for text, balanced GIF/emoji
        'hungry': (0.2, 0.4, 0.7),      # Higher chance for text, moderate for GIF, high for emoji
        'supportive': (0.81, 0.42, 0.69),  # High chance for text, moderate for GIF and emoji
        'neutral': (0.81, 0.13, 0.23)      # Default: high chance for text, low for GIF and emoji
    }

    # Return default if no tone is found
    return weightings.get(tone, (0.8, 0.3, 0.3))

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

# Function to analyze user input and pick the best model
def pick_model_based_on_tone(tone, models):
    # Match the tone with relevant model tags
    tag_mapping = {
        'funny': ['humor', 'dialogue', 'lighthearted', 'casual'],
        'excited': ['enthusiastic', 'high-energy', 'engaging'],
        'sad': ['empathetic', 'supportive'],
        'angry': ['argumentative', 'direct', 'persuasive'],
        'affectionate': ['empathetic', 'supportive', 'caring'],
        'playful': ['fun', 'interactive', 'lighthearted'],
        'confused': ['clarifying', 'instructive'],
        'bored': ['engaging', 'diverting', 'lighthearted'],
        'happy': ['positive', 'friendly', 'enthusiastic'],
        'hungry': ['food-related', 'recommendations', 'fun'],
        'supportive': ['empathetic', 'encouraging', 'uplifting'],
        'neutral': ['general', 'informative', 'neutral']
    }

    candidate_labels = tag_mapping.get(tone, ['general', 'informative'])  # Default to neutral/informative tags if unrecognized

    # Filter models based on matching tags
    matching_models = []
    for model in models:
        model_tags = model['tags']
        matching_score = len(set(model_tags).intersection(candidate_labels))
        if matching_score > 0:
            matching_models.append((model, matching_score))  # Append the model along with the match score

    # If no exact match, fallback to a general-purpose model
    if not matching_models:
        print(f"No matching model tags for tone '{tone}', using default fallback.")
        return random.choice(models)  # Fallback to any model if no match

    # Sort models by matching score
    matching_models.sort(key=lambda x: x[1], reverse=True)

    # Return the model with the highest matching score
    return matching_models[0][0]  # Return the model with the best score

def pick_models_ordered_by_tone(tone, models):
    # Sort models based on how closely their tags match the current tone
    def model_match_score(model, tone):
        tags = model.get('tags', [])
        # Exclude embedding models from this selection process
        if model.get('is_embedding_model'):
            return 0
        # Higher score if the model has tags closely related to the tone
        return sum(1 for tag in tags if tone in tag.lower())

    # Sort models by match score (descending)
    sorted_models = sorted(models, key=lambda model: model_match_score(model, tone), reverse=True)

    logging.info(f"Model order for tone '{tone}': {[model['name'] for model in sorted_models]}")

    return sorted_models

def pick_models_ordered_by_tones(tones, models):
    """
    Sorts models based on how closely their tags match the provided tones.

    Args:
        tones (list): A list of tones.
        models (list): A list of available models.

    Returns:
        list: A list of models ordered by their match score.
    """
    def model_match_score(model, tones):
        tags = model.get('tags', [])
        # Exclude embedding models from this selection process
        if model.get('is_embedding_model', False):
            return 0
        # Calculate match score based on overlapping tags and tones
        score = sum(1 for tag in tags for tone in tones if tone.lower() in tag.lower())
        return score

    # Filter out embedding models before sorting
    filtered_models = [model for model in models if not model.get('is_embedding_model', False)]

    # Sort models by match score (descending)
    sorted_models = sorted(filtered_models, key=lambda model: model_match_score(model, tones), reverse=True)

    logging.info(f"Model order for tones '{tones}': {[model['name'] for model in sorted_models]}")

    return sorted_models


async def generate_response(instructions, search, history, user_message, last_provider=None):
    """
    Generate a response based on the user's message, context, and the available models.
    
    Args:
        instructions (str): System instructions for the chatbot.
        search (str): The result of an internet search or search function.
        history (list): A list of previous messages in the conversation context.
        user_message (str): The current user message.
        last_provider (str): The name of the provider used in the previous attempt (to alternate providers).
    
    Returns:
        str: The generated response or an error message.
    """
    search_results = search if search is not None else "Search feature is disabled"
    
    # Ensure history is properly formatted for the message array
    history_messages = [{"role": "user", "content": msg["content"]} for msg in history]
    
    messages = [
        {"role": "system", "content": instructions},
        *history_messages,
        {"role": "system", "content": search_results},
        {"role": "user", "content": user_message}
    ]

    logging.info(f"Formatted messages being sent: {messages}")

    # Load models from models.xml
    models_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../config/models.xml'))
    logging.info(f"Loading models from: {models_file_path}")

    models = load_models_from_xml(models_file_path)
    logging.info(f"Available models: {[model['name'] for model in models]}")

    # Filter out embedding models
    models = [model for model in models if not model.get('is_embedding_model', False)]
    if not models:
        logging.error("No non-embedding models available to generate a response.")
        return "I'm unable to generate a response at this time."

    retries = 2  # Number of retries per model
    used_models = set()  # Track models we've already tried
    delay_time = 2  # 2-second delay between switching models

    logging.info(f"User message: {user_message}")
    logging.info(f"Search results: {search_results}")

    # Sentiment analysis and tone selection
    logging.info("Performing sentiment analysis...")
    sentiments = await sentiment_analysis(user_message, history)  # Now returns a list
    logging.info(f"Sentiment analysis results: {sentiments}")
    
    tones = vibe_check(sentiments)  # Now returns a list of tones
    logging.info(f"Tones after vibe check: {tones}")

    # Pick a list of models ordered by tone relevance
    model_order = pick_models_ordered_by_tones(tones, models)
    logging.info(f"Model order based on tones: {[model['name'] for model in model_order]}")

    if not model_order:
        logging.error("No models available after ordering by tone.")
        return "I'm unable to select a suitable model to generate a response."

    for attempt in range(retries):
        logging.info(f"Attempt #{attempt + 1}")

        # Filter models by provider, alternating providers each attempt
        for selected_model in model_order:
            if selected_model['name'] in used_models:
                continue  # Skip models we've already tried
            provider = selected_model.get('provider', 'openrouter')
            if provider != last_provider:
                break
        else:
            # If all remaining models are from the same provider, fallback to trying any model
            logging.warning(f"All remaining models are from provider {last_provider}, retrying with the same provider.")
            selected_model = model_order[0]

        used_models.add(selected_model['name'])  # Mark this model as used
        model_name = selected_model['name']
        last_provider = selected_model.get('provider', 'openrouter')

        # Switch API key and base_url according to the provider
        if last_provider == 'nagaAI':
            api_key = os.getenv('NAGA_GPT_KEY')
            base_url = 'https://api.naga.ac/v1'
        else:
            api_key = os.getenv('OPENROUTER_KEY')
            base_url = 'https://openrouter.ai/api/v1'

        if not api_key:
            logging.error(f"API key for provider {last_provider} is not set.")
            continue

        openai_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        logging.info(f"Attempting to generate response using model: {model_name} from {last_provider}")

        try:
            start_time = time.time()  # Record the start time for this model
            response = await openai_client.chat.completions.create(
                model=model_name,
                messages=messages
            )
            response_time = time.time() - start_time  # Calculate response time

            # Log the full response for debugging
            logging.debug(f"Raw response from {model_name}: {response}")
            logging.info(f"Response time for {model_name}: {response_time:.2f} seconds")

            if response and hasattr(response, 'choices'):
                try:
                    if len(response.choices) > 0 and hasattr(response.choices[0], 'message'):
                        message = response.choices[0].message.content
                        logging.info(f"Successfully generated response with {model_name}")
                        logging.debug(f"Generated message: {message}")
                        return message
                except (AttributeError, IndexError) as e:
                    logging.warning(f"Unexpected response structure from {model_name}: {e}")
                    continue  # Try the next model
            else:
                logging.warning(f"No valid choices in the response from {model_name}. Trying next model...")

        except Exception as e:
            logging.error(f"Error generating response from {model_name}: {e}")
            await asyncio.sleep(delay_time)  # Apply a 2-second delay before switching to the next model
            continue  # Continue to the next attempt with a new model

    # **Fallback to `pick_best_model` if all other models fail**
    logging.error("Failed to generate a response after multiple attempts. Falling back to `pick_best_model`.")

    fallback_model = pick_best_model(user_message, models)
    if fallback_model:
        logging.info(f"Fallback to `pick_best_model` selected model: {fallback_model['name']}")

        # Switch API key and base_url according to the provider
        last_provider = fallback_model.get('provider', 'openrouter')
        if last_provider == 'nagaAI':
            api_key = os.getenv('NAGA_GPT_KEY')
            base_url = 'https://api.naga.ac/v1'
        else:
            api_key = os.getenv('OPENROUTER_KEY')
            base_url = 'https://openrouter.ai/api/v1'

        if not api_key:
            logging.error(f"API key for provider {last_provider} is not set.")
            return "I'm unable to generate a response at this time."

        openai_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        # Retry the fallback model once
        try:
            response = await openai_client.chat.completions.create(
                model=fallback_model['name'],
                messages=messages
            )
            if response and hasattr(response, 'choices'):
                try:
                    if len(response.choices) > 0 and hasattr(response.choices[0], 'message'):
                        message = response.choices[0].message.content
                        logging.info(f"Successfully generated response with fallback model: {fallback_model['name']}")
                        return message
                except (AttributeError, IndexError) as e:
                    logging.warning(f"Unexpected response structure from fallback model: {e}")
        except Exception as e:
            logging.error(f"Error generating response from fallback model: {e}")

    return "Failed to generate a response after multiple attempts, including fallback."



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
