import logging
import json
from typing import Any, Dict
import os

logger = logging.getLogger(__name__)

def evaluate_trace_realtime_sync(transcript: str, entities_json: str):
    """
    Sync wrapper to run in BackgroundTasks. Evaluates the extracted entities.
    """
    logger.info(f"Started Real-time background evaluation for transcript: '{transcript[:30]}...'")
    try:
        data = json.loads(entities_json)
        # We perform a lightweight validation check as an "eval metric"
        total_fields = 6
        filled = sum(1 for v in data.values() if v and str(v).strip())
        accuracy_score = filled / total_fields if total_fields else 0
        
        # If we had a running mlflow run, we could do:
        # mlflow.log_metric("realtime_extraction_accuracy", accuracy_score)
        
        logger.info(f"Real-time Eval Completed. Accuracy Score (fields filled): {accuracy_score:.2f}")
    except Exception as e:
        logger.error(f"Error in real-time evaluation: {e}")



def evaluate_trace_single_field_sync(transcript: str, field_name: str, extracted_value: str):
    """
    Sync wrapper to evaluate a single extracted field in the background.
    """
    logger.info(f"Started single field eval for '{field_name}'")
    try:
        score = 1.0 if extracted_value and str(extracted_value).strip() else 0.0
        logger.info(f"Single Field Eval Completed. Score for {field_name}: {score:.2f}")
    except Exception as e:
        logger.error(f"Error in single field eval: {e}")
