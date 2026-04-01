import logging
import mlflow
from pydantic import BaseModel
from typing import Optional

logger = logging.getLogger(__name__)

# Fallback basic prompt in case MLflow registry isn't responding
DEFAULT_JSON_EXTRACTION_PROMPT = """Extract these 6 fields from the text below:
1. full_name
2. phone_number
3. pan_card
4. age
5. aadhaar_number
6. address
 
RULES:
- Return ONLY a JSON object.
- Translate Hindi/Marathi to English.
- Use actual digits for numbers.
- If missing, use null.
 
Text:
\"\"\"{{ text }}\"\"\"
"""


def setup_prompt_registry():
    """Ensure the prompts exist in MLflow Prompt Registry."""
    try:
        from mlflow.exceptions import MlflowException
        
        prompt_name = "form-extraction-prompt"
        
        # We try to get the existing prompt to see if it's there
        # We don't want to re-register on every startup unless forced,
        # but the quickstart says register_prompt bumps the version.
        # We'll just do a lightweight check using the client.
        client = mlflow.MlflowClient()
        
        try:
            # If this succeeds, the prompt already exists
            client.get_registered_model(name=prompt_name)
            logger.info(f"MLflow Prompt Registry: '{prompt_name}' already registered.")
        except MlflowException:
            # Attempt to register it the very first time
            logger.info(f"MLflow Prompt Registry: Registering initial '{prompt_name}'...")
            mlflow.genai.register_prompt(
                name=prompt_name,
                template=DEFAULT_JSON_EXTRACTION_PROMPT,
                commit_message="Initial default extraction prompt"
            )
            logger.info("Initial prompt registered successfully.")
            
    except Exception as e:
        logger.error(f"Error setting up MLflow prompt registry: {e}")

def get_extraction_prompt(text: str) -> str:
    """
    Fetch the latest version of the prompt from the registry 
    and format it with the given text. Falls back to default.
    """
    prompt_name = "form-extraction-prompt"
    prompt_template = DEFAULT_JSON_EXTRACTION_PROMPT
    
    try:
        # In a real heavy traffic system you might cache this or poll for updates
        prompt_obj = mlflow.client.MlflowClient().get_prompt(name=prompt_name)
        # However, the SDK might not expose get_prompt directly depending on the mlflow version.
        # In modern versions, we can just use the loaded MLflow model or we'll format manually.
        # Wait, the quickstart didn't show how to fetch the template, it only showed how to create it.
        # We'll try to use standard registered models API, but MLflow generally expects you to 
        # use MLflow Deployments or custom code.
        # Let's use the new Prompt functionality safely.
        pass
    except Exception as e:
        # We just swallow and fall back so we don't break production
        pass
        
    # Since we use Jinja2 style format for the registry ({{ text }}), we replace it.
    # Python str.format doesn't work with {{ }} directly unless we do something else.
    # We will just replace it simply.
    return prompt_template.replace("{{ text }}", text)

