import requests
from typing import Dict, List, Optional
from crewai.tools import BaseTool
from dotenv import load_dotenv
import os
from pydantic import Field

load_dotenv()



class BraveSearchTool(BaseTool):
    """
    A tool for searching the web using the Brave Search API.
    """
    name: str = Field(default="Brave Search")
    description: str = Field(default="Search the web using the Brave Search API")
    api_key: str = Field(default_factory=lambda: os.getenv("BRAVE_SEARCH_API_KEY"))
    base_url: str = Field(default="https://api.search.brave.com/res/v1/web/search")
    
    def _run(self, query: str) -> Dict:
        """
        Run the Brave Search API with the given query.
        
        Args:
            query (str): The search query
            
        Returns:
            Dict: The search results
        """
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key
        }
        
        params = {
            "q": query,
            "count": 5,
            "result_filter": "web"
        }
        
        response = requests.get(
            self.base_url,
            headers=headers,
            params=params
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Brave Search API error: {response.status_code} - {response.text}")
            
    async def _arun(self, query: str) -> Dict:
        """
        Async version of _run.
        """
        return self._run(query) 


#test the tool
if __name__ == "__main__":
    tool = BraveSearchTool()
    print(tool.run("What is the capital of France?"))