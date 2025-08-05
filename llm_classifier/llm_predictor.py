import os
import json
import pandas as pd
from typing import Dict, List, Tuple, Optional
import google.generativeai as genai
from dataclasses import dataclass
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingInsights:
    """Container for ML model training insights"""
    spam_keywords: List[str]
    ham_keywords: List[str]
    spam_patterns: List[str]
    ham_patterns: List[str]
    feature_importance: Dict[str, float]
    model_accuracy: float
    total_samples: int
    spam_count: int
    ham_count: int

class LLMEmailClassifier:
    """
    LLM-based email classifier that leverages ML training insights
    for quick and accurate spam detection.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        """
        Initialize the LLM classifier.
        
        Args:
            api_key: Google API key (if None, will try to get from environment)
            model: Gemini model to use
        """
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable or pass api_key parameter.")
        
        genai.configure(api_key=self.api_key)
        self.training_insights = None
        self.system_prompt = self._create_system_prompt()
    
    def load_ml_insights(self, insights_file: str = "ml_insights.json"):
        """
        Load ML training insights from file (optional).
        
        Args:
            insights_file: Path to the insights JSON file
        """
        try:
            with open(insights_file, 'r') as f:
                data = json.load(f)
                self.training_insights = TrainingInsights(**data)
                logger.info(f"Loaded ML insights: {self.training_insights.total_samples} samples")
        except FileNotFoundError:
            logger.info("No ML insights file found. LLM will work with default prompts.")
            self.training_insights = None
    
    def _create_system_prompt(self) -> str:
        """system prompt for the LLM."""
        base_prompt = """You are an expert email spam classifier. Your task is to analyze email content and classify it as either SPAM or HAM (legitimate).

Key indicators of SPAM:
- Urgent calls to action (e.g., "URGENT", "ACT NOW", "LIMITED TIME")
- Promises of money or wealth (e.g., "make money", "earn $", "fortune", "guaranteed income")
- Work-from-home opportunities (e.g., "work from home", "home-based jobs")
- Suspicious or shortened URLs
- Random strings of characters or gibberish
- Excessive use of capital letters or punctuation
- Poor grammar, spelling mistakes, or unprofessional language
- Requests for personal, financial, or sensitive information
- Unrealistic claims or promises (e.g., “you’ve won a million dollars”)

Key indicators of HAM (legitimate):
- Professional tone and formatting
- Clearly identified sender (e.g., from a known company or colleague)
- Relevant to the recipient’s interests, work, or subscriptions
- Correct grammar, spelling, and sentence structure
- Emails from trusted or verifiable domains
- Specific and realistic content without exaggerated claims

Your task:
Analyze the provided email content and return a JSON response with the following fields:

1. classification: "SPAM" or "HAM"
2. confidence: A percentage value from 0 to 100 (e.g., 97.5) indicating your confidence in the classification
3. reasons: A list of key reasons supporting your classification
4. suspicious_elements: A list of words, phrases, or patterns that appear suspicious (if any)
5. explanation: A brief explanation summarizing your reasoning

IMPORTANT: You must respond with ONLY valid JSON. Do not include any other text before or after the JSON.

Example response format:
{
    "classification": "SPAM",
    "confidence": 95.5,
    "reasons": ["Contains urgent call to action", "Promises unrealistic money"],
    "suspicious_elements": ["URGENT", "make money"],
    "explanation": "This email exhibits multiple spam indicators including urgent language and unrealistic promises."
}"""

        if self.training_insights:
            insights_prompt = f"""

TRAINING DATA INSIGHTS:
- Model trained on {self.training_insights.total_samples} emails ({self.training_insights.spam_count} spam, {self.training_insights.ham_count} ham)
- Model accuracy: {self.training_insights.model_accuracy:.2%}
- Top spam keywords: {', '.join(self.training_insights.spam_keywords[:10])}
- Top ham keywords: {', '.join(self.training_insights.ham_keywords[:10])}
- Common spam patterns: {', '.join(self.training_insights.spam_patterns[:5])}

Use these insights to improve your classification accuracy."""
            return base_prompt + insights_prompt
        
        return base_prompt
    
    async def classify_email(self, email_text: str) -> Dict:
        """
        Classify an email using the LLM.
        
        Args:
            email_text: The email text to classify
            
        Returns:
            Dictionary with classification results
        """
        try:
            # Clean the email text (remove headers, etc.)
            cleaned_text = self._clean_email_text(email_text)
            
            # Create the user prompt
            user_prompt = f"""Please classify this email:

{cleaned_text}

Remember to respond with valid JSON format."""

            # Make the API call
            model = genai.GenerativeModel(self.model)
            response = await model.generate_content_async(
                f"{self.system_prompt}\n\n{user_prompt}",
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Low temperature for consistent results
                    max_output_tokens=500
                )
            )
            
            # Parse the response
            result_text = response.text.strip()
            
            # Debug: Log the raw response
            logger.info(f"Raw LLM response: {result_text}")
            
            # Try to extract JSON from the response
            json_text = self._extract_json_from_response(result_text)
            
            # Try to parse as JSON
            try:
                result = json.loads(json_text)
                
                # Handle confidence as percentage (0-100) or convert from level
                confidence_value = result.get("confidence", 50)
                
                # If confidence is a string (old format), convert it
                if isinstance(confidence_value, str):
                    confidence_percentage = self._convert_confidence_to_percentage(confidence_value)
                    confidence_level = confidence_value.upper()
                else:
                    # New format: confidence is already a percentage (0-100)
                    confidence_percentage = confidence_value / 100.0  # Convert to 0-1 range
                    confidence_level = self._convert_percentage_to_level(confidence_percentage)
                
                return {
                    "classification": result.get("classification", "UNKNOWN"),
                    "confidence": confidence_percentage,
                    "confidence_level": confidence_level,
                    "reasons": result.get("reasons", []),
                    "suspicious_elements": result.get("suspicious_elements", []),
                    "explanation": result.get("explanation", ""),
                    "llm_response": result_text
                }
            except json.JSONDecodeError as e:
                # Log the JSON parsing error
                logger.warning(f"JSON parsing failed: {e}")
                logger.warning(f"Response text: {result_text}")
                # Fallback parsing if JSON is malformed
                return self._parse_fallback_response(result_text)
                
        except Exception as e:
            logger.error(f"Error classifying email: {e}")
            return {
                "classification": "ERROR",
                "confidence": "LOW",
                "reasons": [f"Classification failed: {str(e)}"],
                "suspicious_elements": [],
                "explanation": "Error occurred during classification",
                "llm_response": ""
            }
    
    def _clean_email_text(self, email_text: str) -> str:
        """Clean email text by removing headers and formatting."""
        # Split by first empty line to remove headers
        parts = email_text.split('\n\n', 1)
        if len(parts) > 1:
            body = parts[1]
        else:
            body = parts[0]
        
        # Remove excessive whitespace
        body = ' '.join(body.split())
        
        return body.strip()
    
    def _convert_confidence_to_percentage(self, confidence_level: str) -> float:
        """Convert confidence level to percentage."""
        confidence_level = confidence_level.upper()
        
        if confidence_level == "HIGH":
            return 0.95  # 95%
        elif confidence_level == "MEDIUM":
            return 0.75  # 75%
        elif confidence_level == "LOW":
            return 0.50  # 50%
        else:
            return 0.50  # Default to 50%
    
    def _convert_percentage_to_level(self, confidence_percentage: float) -> str:
        """Convert percentage confidence to level."""
        if confidence_percentage >= 0.85:
            return "HIGH"
        elif confidence_percentage >= 0.65:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _extract_json_from_response(self, response_text: str) -> str:
        """Extract JSON from LLM response that might contain extra text."""
        # Try to find JSON object in the response
        import re
        
        # Look for JSON object pattern
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, response_text, re.DOTALL)
        
        if matches:
            # Return the longest match (most likely to be complete JSON)
            return max(matches, key=len)
        
        # If no JSON found, return the original text
        return response_text
    
    def _parse_fallback_response(self, response_text: str) -> Dict:
        """Parse response when JSON parsing fails."""
        response_text = response_text.lower()
        
        # Simple keyword-based parsing
        if "spam" in response_text:
            classification = "SPAM"
        elif "ham" in response_text:
            classification = "HAM"
        else:
            classification = "UNKNOWN"
        
        # Determine confidence
        if "high" in response_text:
            confidence_level = "HIGH"
            confidence_percentage = 0.85
        elif "medium" in response_text:
            confidence_level = "MEDIUM"
            confidence_percentage = 0.70
        else:
            confidence_level = "LOW"
            confidence_percentage = 0.50
        
        return {
            "classification": classification,
            "confidence": confidence_percentage,
            "confidence_level": confidence_level,
            "reasons": ["Fallback parsing used"],
            "suspicious_elements": [],
            "explanation": "Used fallback parsing due to malformed JSON",
            "llm_response": response_text
        }
    
    async def batch_classify(self, emails: List[str]) -> List[Dict]:
        """
        Classify multiple emails.
        
        Args:
            emails: List of email texts
            
        Returns:
            List of classification results
        """
        results = []
        for i, email in enumerate(emails):
            logger.info(f"Classifying email {i+1}/{len(emails)}")
            result = await self.classify_email(email)
            result["email_id"] = i
            results.append(result)
        
        return results 