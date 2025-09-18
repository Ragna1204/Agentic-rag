"""
A wrapper for performing web searches.

This module provides a tool to query a web search engine (e.g., Google, Bing, Tavily)
to find information not present in the local document corpus. It's essential for
answering questions about recent events or topics beyond the knowledge base.
"""
import requests
from typing import List, Dict, Any
from ..utils.logging import get_logger

logger = get_logger(__name__)

class WebSearch:
    """
    A tool for performing web searches using a search API.
    """
    def __init__(self, api_key: str, api_url: str = "https://api.tavily.com/search"):
        """
        Initializes the WebSearch tool.

        Args:
            api_key (str): The API key for the search service.
            api_url (str): The endpoint URL for the search API.
                           Defaults to Tavily AI's API.
        """
        # TODO: Add support for other search providers like Google Custom Search or Bing Search.
        # This would involve abstracting the request/response logic.
        if not api_key or api_key == "YOUR_SEARCH_API_KEY":
            logger.warning("Web search API key is not configured. The tool will not work.")
            self.api_key = None
        else:
            self.api_key = api_key
        self.api_url = api_url
        logger.info("Web Search tool initialized.")

    def execute(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """
        Executes a web search for the given query.

        Args:
            query (str): The search query.
            max_results (int): The maximum number of search results to return.

        Returns:
            List[Dict[str, Any]]: A list of search results, where each result is a
                                  dictionary with keys like 'title', 'url', and 'content'.
                                  Returns an empty list if the API key is not set.
        """
        if not self.api_key:
            logger.error("Cannot perform web search: API key is missing.")
            return [{"error": "API key not configured."}]

        logger.debug(f"Performing web search for query: '{query}'")
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "include_answer": False,
            "max_results": max_results
        }
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(self.api_url, json=payload, headers=headers)
            response.raise_for_status()
            results = response.json().get("results", [])
            logger.info(f"Web search returned {len(results)} results.")
            # TODO: The content from web search might be long. Implement a summarization
            # or chunking strategy before returning it.
            return results
        except requests.exceptions.RequestException as e:
            logger.error(f"Error during web search request: {e}")
            return [{"error": f"Failed to connect to search API: {e}"}]
        except Exception as e:
            logger.error(f"An unexpected error occurred during web search: {e}")
            return [{"error": f"An unexpected error occurred: {e}"}]

# Example usage:
# if __name__ == '__main__':
#     import os
#     # Assumes TAVILY_API_KEY is set as an environment variable
#     search_tool = WebSearch(api_key=os.environ.get("TAVILY_API_KEY"))
#     results = search_tool.execute("latest advancements in quantum error correction")
#     for res in results:
#         print(f"Title: {res.get('title')}\nURL: {res.get('url')}\n")
