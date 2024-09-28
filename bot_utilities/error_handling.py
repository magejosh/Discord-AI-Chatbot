import logging

def log_model_issue(issue, model_name, field_name):
    """
    Log an issue with a specific field in the model.
    """
    logging.error(f"Model '{model_name}' missing '{field_name}' field. Issue: {issue}")

def handle_response_error(response, model_name):
    """
    Handle errors in the response from the model.
    """
    if response and hasattr(response, 'choices'):
        try:
            if len(response.choices) > 0 and hasattr(response.choices[0], 'message'):
                message = response.choices[0].message.content
                logging.info(f"Successfully generated response with {model_name}")
                logging.debug(f"Generated message: {message}")
                return message
        except (AttributeError, IndexError) as e:
            logging.warning(f"Unexpected response structure from {model_name}: {e}")
    else:
        logging.warning(f"No valid choices in the response from {model_name}. Trying next model...")
    return None