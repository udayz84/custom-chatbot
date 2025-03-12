import os
from typing import Optional
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

class GroqAPI:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.client = Groq(api_key=self.api_key)
        self.model = "mixtral-8x7b-32768"  # Using Mixtral model for better performance
        
    def ask_question(self, query: str, context: Optional[str] = None) -> str:
        """
        Ask a question to Groq API and get the response
        
        Args:
            query (str): The question or query to ask
            context (Optional[str]): Additional context to help answer the question
            
        Returns:
            str: The response from Groq
        """
        try:
            # Prepare the prompt with context if provided
            prompt = f"Context: {context}\n\nQuestion: {query}" if context else query
            
            # Make API call
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant. Provide clear, concise, and accurate answers."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=0.5,
                max_tokens=1000,
                top_p=1,
                stream=False
            )
            
            # Extract and return the response
            return chat_completion.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error getting response from Groq: {str(e)}"
