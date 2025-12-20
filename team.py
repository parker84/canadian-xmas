import streamlit as st
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.cohere import Cohere
from textwrap import dedent
from agno.db.postgres import PostgresDb
from decouple import config
from tools import fetch_urls, search_web_multi
import os
import coloredlogs, logging
import psycopg
import cohere
from typing import List

# Create a logger object.
logger = logging.getLogger(__name__)
coloredlogs.install(level=os.getenv("LOG_LEVEL", "INFO"), logger=logger)

# ------------constants
DEBUG_MODE = os.getenv("LOG_LEVEL", "INFO").upper() == "DEBUG"

# LLM Provider Configuration - easily switchable
AGENT_LLM_PROVIDER = "openai"
AGENT_MODEL_ID = "gpt-5-nano" # $0.050 / 1M input tokens
# AGENT_LLM_PROVIDER = "cohere"
# AGENT_MODEL_ID = "command-r7b-12-2024" # $0.0375 / 1M input tokens
# AGENT_MODEL_ID = "command-a-03-2025" # $2.50 / 1M input tokens

# Knowledge Base Configuration
EMBEDDING_MODEL = "embed-v4.0"
EMBEDDING_DIMENSIONS = 1536

ADDITIONAL_CONTEXT = dedent("""
    Your outputs will be in markdown format so when using $ for money you need to escape it with a backslash.
    Focus on helping Canadian businesses, artists, creators, and the Canadian economy.
    Spell using Canadian proper grammar (ex: "favor" -> "favour").
""")
MAX_TOOL_CALLS = 3
NUM_HISTORY_RUNS = 3

# TODO: it's still searching the web before checking the knowledge base sometimes
# TODO: get the chats / messages storing properly again
# TODO: get the memory working again

# TODO: make sure this new instructions works well
def get_product_finding_instructions(
        search_knowledge_base: bool = True,
        search_web: bool = True,
    ) -> str:
    """Get the instructions for the product finding agent"""

    if search_knowledge_base and search_web:
        tool_instructions = dedent("""
            ALWAYS USE THE KNOWLEDGE BASE TOOL FIRST.

            Here are the steps you need to follow:
            
            First: Search the knowledge base for similar queries (using search_knowledge_base_sync - do this once).
            
            If you find what you need from the knowledge base -> then stop and return the results.

            If you don't find what you need from the knowledge base -> then continue to the steps below:
            - Search the web for information (using search_web_multi - do this once)
            - Fetch the contents of the urls (using fetch_urls - do this once)
            - Return the results in a table format 
            
            That's it, DO NOT REPEAT ANY STEPS AND STOP AFTER THE FIRST STEP IF YOU FIND SOMETHING FROM THE KNOWLEDGE BASE.
                            
            When searching the web use search queries like:
            - "Made in Canada <insert product name>"
            - "Canadian owned <insert product name> companies"
            - "Top Canadian <insert product name> brands"
            But don't just assume every result is a canadian company or product, you need to check the sources and pull out the relevant information from the sources to make sure it's a canadian company or product.
        """)

    elif search_knowledge_base and not search_web:
        tool_instructions = dedent("""
            Search the knowledge base for similar queries (ALWAYS DO THIS FIRST).
            - search_knowledge_base_sync: search the knowledge base for similar queries
        """)
    
    elif not search_knowledge_base and search_web:
        tool_instructions = dedent("""
            Here are the steps you need to follow:
            - Step 1: Search the web for information (using search_web_multi - do this once)
            - Step 2: Fetch the contents of the urls (using fetch_urls - do this once)
            - Step 3: Return the results in a table format 
            
            That's it, DO NOT REPEAT ANY STEPS.
                            
            When searching the web use search queries like:
            - "Made in Canada <insert product name>"
            - "Canadian owned <insert product name> companies"
            - "Top Canadian <insert product name> brands"
            But don't just assume every result is a canadian company or product, you need to check the sources and pull out the relevant information from the sources to make sure it's a canadian company or product.
        """)

    product_finding_instructions = dedent(f"""
        Find and recommend the best Canadian products - that are from Canadian owned and operated businesses.
        Don't forget to include classic / iconic and well known Canadian brands (when applicable) like: Roots, Lululemon, Canada Goose, Aritzia, Joe Fresh, Red Canoe, Province of Canada, Mejuri, Duer, etc.
        Find 5-10 options ranked by your evaluation of which are the best (prioritize made in canada options where possible).

        Ensure for each product you check whether it's made in canada or not.
        This information should be in the product page or the search results.
        If it's not then assume it's not made in canada.

        {tool_instructions}

        Only return products / brands that are either:
        A) Made in Canada or 
        B) From Canadian owned and operated businesses
        For any other products / brands -> don't recommend them
                        
        Format your response into a table with the following columns:
        - Product Name
        - Product Description
        - Product Link (make sure the link actually works -> don't make it up)
        - Product Price
        - Product Features
        - Canadian Owner / Made

        If there's made in canada options -> rank these at the top of the table.

        You don't need to return much else other than the table.
        At the end ask the user a meaningful follow up question to keep the conversation going.
    """)

    return product_finding_instructions


# ------------database / storage / setup
db_url = f"postgresql+psycopg://{config('POSTGRES_USER')}:{config('POSTGRES_PASSWORD')}@{config('POSTGRES_HOST')}/{config('POSTGRES_DB')}"

team_storage = PostgresDb(
    db_url=db_url
)

# TODO:
# 1. verify the memory still works
# 2. understand the routing
# 3. verify the web search is working

# ------------knowledge base
async def generate_embedding(text: str) -> List[float]:
    """Generate embeddings for text using Cohere"""
    cohere_client = cohere.AsyncClientV2(api_key=config("COHERE_API_KEY"))
    response = await cohere_client.embed(
        texts=[text],
        model=EMBEDDING_MODEL,
        input_type="search_query",  # For searching
        embedding_types=["float"],
        output_dimension=int(EMBEDDING_DIMENSIONS),
    )
    return response.embeddings.float_[0]

# TODO: add re-ranking

async def search_knowledge_base(query: str, limit: int = 3) -> str:
    """Search the knowledge base for similar past queries and responses.
    
    Args:
        query: The user's query to search for
        limit: Number of similar results to return (default: 3)
        
    Returns:
        A formatted string with relevant past experiences
    """
    try:
        # Generate embedding for the query
        # query = "Looking for Christmas gifts in Canada: eco-friendly, handmade or tech-forward options under $180 for my spouse, mom, younger sister, college roommate, coworker, and nephew; prefer Canadian brands with solid reviews and sustainable materials, practical gadgets that simplify daily life, and I’d love help tracking upcoming sales."
        embedding = await generate_embedding(query)
        embedding_str = '[' + ','.join(str(x) for x in embedding) + ']'
        
        # Connect to database
        conn_string = f"host={config('POSTGRES_HOST')} dbname={config('POSTGRES_DB')} user={config('POSTGRES_USER')} password={config('POSTGRES_PASSWORD')}"
        conn = await psycopg.AsyncConnection.connect(conn_string)
        
        async with conn.cursor() as cur:
            # Search for similar queries using vector similarity
            sql_query = """ 
                with similarity_scores as (
                    SELECT 
                        gq.query,
                        gq.context,
                        1 - (gq.embedding <=> %s::vector) AS similarity,
                        sr.response
                    FROM gift_queries gq
                    join snowman_responses sr on sr.query_id = gq.id
                )

                select * from similarity_scores order by similarity desc limit %s
            """

            await cur.execute(sql_query, (embedding_str, limit))
            results = await cur.fetchall()

            logger.info(
                f"""🔍 The top result details:
- Search Query: {query}
- Similarity Score: {results[0][2]}
- Knowledge Base Query: {results[0][0]}
- Knowledge Base Response: {results[0][3]}
                """
            )
        
        await conn.close()
        
        if not results:
            return "No relevant past experiences found in knowledge base."
        
        # Format results
        formatted_results = ["Here are similar past queries and responses from the knowledge base:\n"]
        for i, (past_query, context, similarity, response) in enumerate(results, 1):
            formatted_results.append(f"{i}. **Past Query:** {past_query}")
            formatted_results.append(f"   **Context:** {context}")
            formatted_results.append(f"   **Response:** {response[:300]}...")  # Truncate long responses
            formatted_results.append(f"   **Similarity:** {similarity:.2%}\n")
        
        return "\n".join(formatted_results)
        
    except Exception as e:
        logger.error(f"Error searching knowledge base: {e}")
        return f"Error accessing knowledge base: {str(e)}"

def search_knowledge_base_sync(query: str, limit: int = 3) -> str:
    """Synchronous wrapper for search_knowledge_base to use as an agent tool"""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(search_knowledge_base(query, limit))

def get_llm_model():
    """Get the configured LLM model based on provider"""
    if AGENT_LLM_PROVIDER == "openai":
        return OpenAIChat(id=AGENT_MODEL_ID)
    elif AGENT_LLM_PROVIDER == "cohere":
        return Cohere(id=AGENT_MODEL_ID)
    else:
        raise ValueError(f"Unsupported LLM provider: {AGENT_LLM_PROVIDER}")

@st.cache_resource
def get_agent_team(search_knowledge_base: bool = True, search_web: bool = True):
    logger.info(f"🤖 Initializing agent with {AGENT_LLM_PROVIDER}/{AGENT_MODEL_ID}")
    
    product_finder_agent = Agent(
        name="Product Finder Agent",
        role="Find and recommend products",
        model=get_llm_model(),
        tools=[
            search_knowledge_base_sync,
            search_web_multi,
            fetch_urls,
        ],
        instructions=get_product_finding_instructions(search_knowledge_base, search_web),
        additional_context=ADDITIONAL_CONTEXT,
        debug_mode=DEBUG_MODE,
        markdown=True,
        add_datetime_to_context=True,
        tool_call_limit=MAX_TOOL_CALLS,
        # ----------memory----------
        # adding previous 5 questions and answers to the prompt
        # read more here: https://docs.agno.com/memory/introduction
        db=team_storage,
        add_history_to_context=True,
        num_history_runs=NUM_HISTORY_RUNS,
    )

    # brand_finder_agent = Agent(
    #     name="Brand Finder Agent",
    #     role="Find and recommend brands",
    #     # model=Cohere(id="command-a-03-2025"),
    #     model=OpenAIChat(id=AGENT_MODEL_ID),
    #     tool_call_limit=MAX_TOOL_CALLS,
    #     tools=[
    #         search_web_multi,
    #         fetch_urls,
    #     ],
    #     instructions=[
    #         "Find and recommend the best and most iconic Canadian brands",
    #         "Include brand information and links",
    #         "Always include sources (and link out to them)",
    #         "But don't just include the sources, pull out the relevant information from the sources",
    #         "Always include the brand name, description, and link",
    #         "Bias towards Canadian brands that are Canadian made",
    #         "Bias towards Canadian brands that are Canadian designed",
    #         "Bias towards Canadian brands that are Canadian owned and operated",
    #         "Include a table at the bottom comparing all the brands",
    #         "At a minimum include price, rating, features, link and Canadian owner / made as columns in the table",
    #         "You should batch all search and fetch operations to minimize tool calls.",
    #         "In general you shouldn't be making more than {MAX_TOOL_CALLS} tool calls per request.",
    #         "You shouldn't take longer than 10 seconds to complete your task.",
    #     ],
    #     debug_mode=DEBUG_MODE,
    #     markdown=True,
    #     additional_context=ADDITIONAL_CONTEXT,
    #     add_datetime_to_context=True,
    # )

    # gift_finder_agent = Agent(
    #     name="Gift Finder Agent",
    #     role="Find and recommend gifts",
    #     # model=Cohere(id="command-a-03-2025"),
    #     model=OpenAIChat(id=AGENT_MODEL_ID),
    #     tool_call_limit=MAX_TOOL_CALLS,
    #     tools=[
    #         search_web_multi,
    #         fetch_urls,
    #     ],
    #     instructions=[
    #         "Find and recommend the best Canadian gifts",
    #         "Try to make the gift very personalized by asking the user questions about the person you're recommending a gift for",
    #         "Do not recommend gifts without a link that actually works, and include the correct ratings and the volume of reviews",
    #         "Then use that information to recommend the best gift for them",
    #         "Include gift information and links",
    #         "Always include sources (and link out to them)",
    #         "But don't just include the sources, pull out the relevant information from the sources",
    #         "Always include the gift name, description, and link",
    #         "Always include the gift price",
    #         "Always include the gift rating",
    #         "Always include the gift reviews",
    #         "Always include the gift features",
    #         "Bias towards Canadian gifts that are Canadian made",
    #         "Bias towards Canadian gifts that are Canadian designed",
    #         "Bias towards Canadian gifts that are Canadian owned and operated",
    #         "You should batch all search and fetch operations to minimize tool calls.",
    #         "In general you shouldn't be making more than {MAX_TOOL_CALLS} tool calls per request.",
    #         "You shouldn't take longer than 10 seconds to complete your task.",
    #     ],
    #     debug_mode=DEBUG_MODE,
    #     additional_context=ADDITIONAL_CONTEXT,
    #     add_datetime_to_context=True,
    # )

    # agent_team = Team(
    #     name="Canadian AI",
    #     description="You're a Canadian AI assistant that can help users accomplish a multitude of tasks (ex: find a gift, find a product, find a service, find a movie, find a book, find a tv show, find a music artist, find a brand, etc.) but you are intentionally biased towards supporting Canadian businesses, artists, creators, and the Canadian economy.",
    #     members=[
    #         product_finder_agent,
    #         brand_finder_agent,
    #         gift_finder_agent,
    #     ],
    #     respond_directly=True,
    #     model=OpenAIChat(id=ROUTER_MODEL_ID), # this does better w yoga pants question
    #     instructions=dedent(
    #         """
    #         Answer the user's question to the best of your abilities.
    #         But generally bias towards supporting Canadian businesses, artists, creators, and the Canadian economy.

    #         Route EXACTLY ONE task to ONE agent (or don't route at all if you just need to ask clarifying questions).
    #         Do not involve multiple agents unless strictly required.

    #         Here's the agents you can route to:
    #         - if the user is asking for / about a product, use the product finder agent (to find 3-5 options)
    #         - if the user is asking for / about a brand, use the brand finder agent (to find 3-5 options)
    #         - if the user is asking for / about a gift, use the gift finder agent (to find 3-5 options)

    #         When routing to an agent, don't add commentary to the response, just route to the agent and let the agent respond.

    #         Ask questions to get a better understanding of the user's needs, but  not too many to annoy the user.
    #         Usually keep it to 1 follow up question max before trying to answer the user's question.
    #         """
    #     ),
    #     debug_mode=DEBUG_MODE,
    #     show_members_responses=True,
    #     markdown=True,
    #     additional_context=ADDITIONAL_CONTEXT,
    #     # ----------memory----------
    #     # adding previous 5 questions and answers to the prompt
    #     # read more here: https://docs.agno.com/memory/introduction
    #     db=team_storage,
    #     # enable_team_history=True,
    #     add_datetime_to_context=True,
    #     add_history_to_context=True,
    #     num_history_runs=5,
    #     num_history_messages=5
    # )
    return product_finder_agent

async def main():
    team = get_agent_team()
    print("☃️ Snowman CLI Agent is ready. Type 'exit' to quit.")
    while True:
        user_input = input("💁‍♀️ You: ")
        if user_input.strip().lower() == "exit":
            break
        response = await team.arun(user_input)
        print(f"☃️ Snowman: {response.content}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

# help me find a gift for my father
# he's 62, retired, loves travelling, into star wars, hockey (especially the leafs), and he's a bit of a nerd (likes star wars, star trek, space, etc.)


# I want to find some new music
# I like Rock recently have been into alanis morset, and love the tragically hip

# Looking for Christmas gifts in Canada: eco-friendly, handmade or tech-forward options under $180 for my spouse, mom, younger sister, college roommate, coworker, and nephew; prefer Canadian brands with solid reviews and sustainable materials, practical gadgets that simplify daily life, and I’d love help tracking upcoming sales.
# Help me find a Montreal-made, sustainably sourced tea gift set for my mom who loves tea, under $120, with thoughtful packaging.