import os
from typing import Literal, TypedDict, List, Optional
from langchain_core.tools import tool

from dotenv import load_dotenv
from tavily import TavilyClient


load_dotenv(dotenv_path="/Users/yhep/DRC/KEK/.env")

TAVILY_API_KEY: str | None = os.getenv("TAVILY_API_KEY")
assert TAVILY_API_KEY, "Please set the `TAVILY_API_KEY` environment variable."

client = TavilyClient(api_key=TAVILY_API_KEY)


class WebSearchResult(TypedDict):
    title: str
    url: str
    content: str
    score: float


@tool
def search_web_tool(
    query: str,
    search_topic: Literal["general", "news", "finance"] = "general",
    search_deptch: Literal["basic", "advanced"] = "basic",
    max_results: int = 1,
    time_range: Literal["day", "week", "month", "year", None] = None,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
) -> str:
    """
    Search the web using Tavily API. Use this tool whenever you need to search the internet and find results matching the user's query.
    This is particularly useful for finding up-to-date information, news, or specific content from the web.

    Args:
        query (str): The search query.
        search_topic (str, optional): The topic of the search. Can be "general", "news", or "finance". Defaults to "general".
        search_deptch (str, optional): The depth of the search. Can be "basic" or "advanced". Defaults to "basic".
        max_results (int, optional): The maximum number of results to return. Defaults to 1.
        time_range (str, optional): The time range for the search. Can be "day", "week", "month", or "year". Defaults to None.
        include_domains (list[str], optional): A list of domains to include in the search. Defaults to None.
        exclude_domains (list[str], optional): A list of domains to exclude from the search. Defaults to None.

    Returns:
        str: A comprehensive answer based on the web search results.
    """
    try:
        response = client.search(
            query=query,
            topic=search_topic,
            depth=search_deptch,
            max_results=max_results,
            time_range=time_range,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
        )
    except Exception as error:
        return f"Error: {str(error)}"
    
    if not response.get("results"):
        return f"No search results found for '{query}'"
    
    # Collect search results content
    search_content = ""
    for i, search_result in enumerate(response["results"], 1):
        search_content += f"Result {i} - {search_result['title']}\n"
        search_content += f"URL: {search_result['url']}\n"
        search_content += f"Content: {search_result['content']}\n"
        search_content += f"Score: {search_result['score']:.3f}\n\n"
    
    # Generate answer based on search results
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from llm_provider import get_llm
        
        llm = get_llm()
        
        prompt = f"""Based on the following web search results, provide a comprehensive and helpful answer to the user's question.

Question: {query}

Web Search Results:
{search_content}

Instructions:
- Use the information from the search results to answer the question
- Provide a clear, informative response
- If the search results don't fully answer the question, mention what information is available
- Be concise but comprehensive
- Cite relevant details from the search results

Answer:"""

        response = llm.invoke([{"role": "user", "content": prompt}])
        return response.content.strip()
        
    except Exception as e:
        print(f"Error generating answer with LLM: {e}")
        # Fallback to formatted search results
        result = f"Based on web search for '{query}':\n\n"
        for i, search_result in enumerate(response["results"], 1):
            result += f"{i}. {search_result['title']}\n"
            result += f"   {search_result['content'][:200]}...\n"
            result += f"   Source: {search_result['url']}\n\n"
        return result


if __name__ == "__main__":
    # MCP 서버로 실행할 때는 원래 코드 사용
    from mcp.server.fastmcp import FastMCP
    from mcp.types import CallToolResult, TextContent
    
    mcp = FastMCP("web_seach")
    
    @mcp.tool()
    def search_web_tool_mcp(
        query: str,
        search_topic: Literal["general", "news", "finance"] = "general",
        search_deptch: Literal["basic", "advanced"] = "basic",
        max_results: int = 1,
        time_range: Literal["day", "week", "month", "year", None] = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
    ) -> List[WebSearchResult] | CallToolResult:
        try:
            response = client.search(
                query=query,
                topic=search_topic,
                depth=search_deptch,
                max_results=max_results,
                time_range=time_range,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
            )
        except Exception as error:
            return CallToolResult(
                isError=True,
                content=[
                    TextContent(
                        type="text",
                        text=f"Error: {str(error)}",
                    )
                ]
            )
        search_results: List[WebSearchResult] = []
        for result in response["results"]:
            search_results.append({
                "title": result["title"],
                "url": result["url"],
                "content": result["content"],
                "score": result["score"],
            })
        return search_results
    
    mcp.run(transport="stdio")
