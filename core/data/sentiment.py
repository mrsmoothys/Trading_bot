import requests
from typing import Dict, Any, Optional
from loguru import logger
from datetime import datetime

def fetch_fear_and_greed_index() -> Dict[str, Any]:
    """
    Fetch the latest Crypto Fear & Greed Index.
    Source: https://api.alternative.me/fng/
    
    Returns:
        Dict containing:
        - value (int): 0-100
        - classification (str): e.g., "Extreme Fear", "Greed"
        - timestamp (str): ISO format date
    """
    try:
        response = requests.get("https://api.alternative.me/fng/", timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data and "data" in data and len(data["data"]) > 0:
            item = data["data"][0]
            result = {
                "value": int(item.get("value", 50)),
                "classification": item.get("value_classification", "Unknown"),
                "timestamp": datetime.now().isoformat()
            }
            logger.info(f"Fetched Fear & Greed Index: {result['value']} ({result['classification']})")
            return result
            
        logger.warning("Fear & Greed API returned unexpected format")
        return {"value": 50, "classification": "Neutral (Fallback)", "timestamp": datetime.now().isoformat()}
        
    except Exception as e:
        logger.error(f"Error fetching Fear & Greed Index: {e}")
        return {"value": 50, "classification": "Neutral (Error)", "timestamp": datetime.now().isoformat()}
