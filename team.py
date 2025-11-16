import streamlit as st
from agno.agent import Agent
from agno.models.openai import OpenAIChat
# from agno.models.cohere import Cohere # TODO: fix this not working now
from textwrap import dedent
from agno.db.postgres import PostgresDb
from decouple import config
from agno.team.team import Team
from tools import fetch_url_contents, search_web, fetch_urls, search_web_multi
import os
import coloredlogs, logging

# Create a logger object.
logger = logging.getLogger(__name__)
coloredlogs.install(level=os.getenv("LOG_LEVEL", "INFO"), logger=logger)

# TODO: handle context windows getting too large

# ------------constants
DEBUG_MODE = os.getenv("DEBUG_MODE", "True").lower() == "true"
# MODEL_ID = "gpt-4.1-mini" # -> not good enough
# MODEL_ID = "gpt-4.1" # -> hitting TPM rate limit w pure md
AGENT_MODEL_ID = "gpt-5-mini"
ROUTER_MODEL_ID = "gpt-5-nano"
TEMPERATURE = 0.0
ADDITIONAL_CONTEXT = dedent("""
    Your outputs will be in markdown format so when using $ for money you need to escape it with a backslash.
    Focus on helping Canadian businesses, artists, creators, and the Canadian economy.
    Spell using Canadian proper grammar (ex: "favor" -> "favour").
""")
MAX_TOOL_CALLS = 3

# TODO: more search results with LLM reranking on top?
# TODO: switch over to cohere LLM

product_finding_instructions = dedent("""
    Find and recommend the best Canadian products - that are from Canadian owned and operated businesses.
    Don't forget to include classic / iconic and well known Canadian brands (when applicable) like: Roots, Lululemon, Canada Goose, Aritzia, Joe Fresh, Red Canoe, etc.
    Try to find 5-10 options ranked by ratings / your evaluation of the best options.

    Here's the tools you have to use:
    - fetch_url_contents: fetch the contents of a url
    - search_web: search the web for information
    - fetch_urls: fetch the contents of a list of urls
    - search_web_multi: search the web for information in parallel

    You should batch all search and fetch operations to minimize tool calls.
    In general you shouldn't be making more than {MAX_TOOL_CALLS} tool calls per request.
    You shouldn't take longer than 10 seconds to complete your task.
                    
    When searching the web use search queries like:
    - "Canadian owned <insert product name> companies"
    - "<insert product name> that are made in Canada"
    - "Top Canadian <insert product name> brands"
    But don't just assume every result is a canadian company or product, you need to check the sources and pull out the relevant information from the sources to make sure it's a canadian company or product.
    You may want to use the ReasoningTools to help you reason about the information you find.

    Only return products / brands that are either:
    A) Made in Canada or 
    B) From Canadian owned and operated businesses
    For any other products / brands -> don't recommend them
                    
    Format your response into a table with the following columns:
    - Product Name
    - Product Description
    - Product Link
    - Product Price
    - Product Features
    - Canadian Owner / Made

    You don't need to return much else other than the table.
    At the end ask the user a meaningful follow up question to keep the conversation going.
""")


# ------------database / storage / setup
db_url = f"postgresql+psycopg://{config('POSTGRES_USER')}:{config('POSTGRES_PASSWORD')}@{config('POSTGRES_HOST')}/{config('POSTGRES_DB')}"

logger.info("Setting up storage")
team_storage = PostgresDb(
    db_url=db_url
)
logger.info("Storage setup complete ✅")

# TODO:
# 1. verify the memory still works
# 2. understand the routing
# 3. verify the web search is working

@st.cache_resource
def get_agent_team():
    product_finder_agent = Agent(
        name="Product Finder Agent",
        role="Find and recommend products",
        # model=Cohere(id="command-a-03-2025"),
        # model=OpenAIChat(id="gpt-4.1"), # so much better than 4.1-mini for the umbrella question
        model=OpenAIChat(id=AGENT_MODEL_ID),
        tools=[
            fetch_url_contents, 
            search_web, 
            fetch_urls,
            search_web_multi,
            # ReasoningTools()
        ],
        instructions=product_finding_instructions,
        additional_context=ADDITIONAL_CONTEXT,
        debug_mode=DEBUG_MODE,
        markdown=True,
        add_datetime_to_context=True,
        tool_call_limit=MAX_TOOL_CALLS,
    )

    brand_finder_agent = Agent(
        name="Brand Finder Agent",
        role="Find and recommend brands",
        # model=Cohere(id="command-a-03-2025"),
        model=OpenAIChat(id=AGENT_MODEL_ID),
        tool_call_limit=MAX_TOOL_CALLS,
        tools=[
            fetch_url_contents, 
            search_web, 
            fetch_urls,
            search_web_multi,
            # ReasoningTools()
        ],
        instructions=[
            "Find and recommend the best and most iconic Canadian brands",
            "Include brand information and links",
            "Always include sources (and link out to them)",
            "But don't just include the sources, pull out the relevant information from the sources",
            "Always include the brand name, description, and link",
            "Bias towards Canadian brands that are Canadian made",
            "Bias towards Canadian brands that are Canadian designed",
            "Bias towards Canadian brands that are Canadian owned and operated",
            "Include a table at the bottom comparing all the brands",
            "At a minimum include price, rating, features, link and Canadian owner / made as columns in the table",
            "You should batch all search and fetch operations to minimize tool calls.",
            "In general you shouldn't be making more than {MAX_TOOL_CALLS} tool calls per request.",
            "You shouldn't take longer than 10 seconds to complete your task.",
        ],
        debug_mode=DEBUG_MODE,
        markdown=True,
        additional_context=ADDITIONAL_CONTEXT,
        add_datetime_to_context=True,
    )

    gift_finder_agent = Agent(
        name="Gift Finder Agent",
        role="Find and recommend gifts",
        # model=Cohere(id="command-a-03-2025"),
        model=OpenAIChat(id=AGENT_MODEL_ID),
        tool_call_limit=MAX_TOOL_CALLS,
        tools=[
            fetch_url_contents, 
            search_web, 
            fetch_urls,
            search_web_multi,
            # ReasoningTools()
        ],
        instructions=[
            "Find and recommend the best Canadian gifts",
            "Try to make the gift very personalized by asking the user questions about the person you're recommending a gift for",
            "Do not recommend gifts without a link that actually works, and include the correct ratings and the volume of reviews",
            "Then use that information to recommend the best gift for them",
            "Include gift information and links",
            "Always include sources (and link out to them)",
            "But don't just include the sources, pull out the relevant information from the sources",
            "Always include the gift name, description, and link",
            "Always include the gift price",
            "Always include the gift rating",
            "Always include the gift reviews",
            "Always include the gift features",
            "Bias towards Canadian gifts that are Canadian made",
            "Bias towards Canadian gifts that are Canadian designed",
            "Bias towards Canadian gifts that are Canadian owned and operated",
            "You should batch all search and fetch operations to minimize tool calls.",
            "In general you shouldn't be making more than {MAX_TOOL_CALLS} tool calls per request.",
            "You shouldn't take longer than 10 seconds to complete your task.",
        ],
        debug_mode=DEBUG_MODE,
        additional_context=ADDITIONAL_CONTEXT,
        add_datetime_to_context=True,
    )

    agent_team = Team(
        name="Canadian AI",
        # description="A team of AI agents that can help you accomplish a number of things that are biased towards supporting Canadian businesses, artists, creators, and the Canadian economy.",
        description="You're a Canadian AI assistant that can help users accomplish a multitude of tasks (ex: find a gift, find a product, find a service, find a movie, find a book, find a tv show, find a music artist, find a brand, etc.) but you are intentionally biased towards supporting Canadian businesses, artists, creators, and the Canadian economy.",
        members=[
            product_finder_agent,
            brand_finder_agent,
            gift_finder_agent,
        ],
        respond_directly=True,
        # tools=[
        #     fetch_url_contents, 
        #     search_web, 
        #     # ReasoningTools(),
        # ],
        # model=Cohere(id="command-a-03-2025"),
        model=OpenAIChat(id=ROUTER_MODEL_ID), # this does better w yoga pants question
        instructions=dedent(
            """
            Answer the user's question to the best of your abilities.
            But generally bias towards supporting Canadian businesses, artists, creators, and the Canadian economy.

            Route EXACTLY ONE task to ONE agent (or don't route at all if you just need to ask clarifying questions).
            Do not involve multiple agents unless strictly required.

            Here's the agents you can route to:
            - if the user is asking for / about a product, use the product finder agent.
            - if the user is asking for / about a brand, use the brand finder agent.
            - if the user is asking for / about a gift, use the gift finder agent.

            When routing to an agent, don't add commentary to the response, just route to the agent and let the agent respond.

            Ask questions to get a better understanding of the user's needs, but  not too many to annoy the user.
            Usually keep it to 1 follow up question max before trying to answer the user's question.
            """
        ),
        debug_mode=DEBUG_MODE,
        show_members_responses=True,
        markdown=True,
        additional_context=ADDITIONAL_CONTEXT,
        # ----------memory----------
        # adding previous 5 questions and answers to the prompt
        # read more here: https://docs.agno.com/memory/introduction
        db=team_storage,
        # enable_team_history=True,
        add_datetime_to_context=True,
        add_history_to_context=True,
        num_history_runs=5,
        num_history_messages=5
    )
    return agent_team

def main():
    team = get_agent_team()
    print("🤖 Agno CLI Agent is ready. Type 'exit' to quit.")
    while True:
        user_input = input("💁‍♀️ You: ")
        if user_input.strip().lower() == "exit":
            break
        response = team.run(user_input)
        print(f"🤖 Agno: {response.content}")

if __name__ == "__main__":
    main()

# help me find a gift for my father
# he's 62, retired, loves travelling, into star wars, hockey (especially the leafs), and he's a bit of a nerd (likes star wars, star trek, space, etc.)


# I want to find some new music
# I like Rock recently have been into alanis morset, and love the tragically hip