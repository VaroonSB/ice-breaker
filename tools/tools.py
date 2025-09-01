from langchain_tavily import TavilySearch

def get_profile_url_tavily(name: str):
    """Searches for Linkedin or twitter Profile Page."""
    search = TavilySearch()
    res = search.run(f"{name}")
    return res