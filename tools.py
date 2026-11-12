"""Query engines, router, and the extra web/research tools."""

from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.tools.arxiv import ArxivToolSpec
from llama_index.tools.brave_search import BraveSearchToolSpec

from config import BRAVE_API_KEY


def build_report_tool(summary_index, vector_index) -> QueryEngineTool:
    """One tool that routes between summary and vector search over the PDF."""
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )
    vector_query_engine = vector_index.as_query_engine()

    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        description="Useful for summarization questions related to the State of AI paper.",
    )
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description="Useful for retrieving specific context from the State of AI paper.",
    )

    router = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[summary_tool, vector_tool],
        verbose=True,
    )

    return QueryEngineTool.from_defaults(
        query_engine=router,
        name="state_of_ai_report_assistant",
        description="Answers questions based on the McKinsey 2025 State of AI report.",
        return_direct=False,
    )


def build_all_tools(summary_index, vector_index) -> list:
    """Report tool + ArXiv + (optional) Brave Search."""
    tools = [build_report_tool(summary_index, vector_index)]
    tools.extend(ArxivToolSpec().to_tool_list())

    if BRAVE_API_KEY:
        tools.extend(BraveSearchToolSpec(api_key=BRAVE_API_KEY).to_tool_list())
    else:
        print("BRAVE_API_KEY not set: web search tool disabled.")
    return tools
