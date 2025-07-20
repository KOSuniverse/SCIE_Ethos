# --- utils/retry.py ---

import time

def openai_with_retry(call_fn, max_retries=3, delay=2):
    """
    Retry logic for OpenAI API calls with exponential backoff.
    
    Args:
        call_fn: Lambda function containing the OpenAI API call
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (doubles each attempt)
    
    Returns:
        Result of the API call
    
    Raises:
        Exception: If all retries are exhausted
    """
    for attempt in range(max_retries):
        try:
            return call_fn()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay * (2 ** attempt))  # Exponential backoff
            else:
                raise e
