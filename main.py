import streamlit as st
from typing import AsyncIterator, AsyncGenerator
from agno.agent import RunOutput
from team import get_agent_team
from random import choice
import uuid
import asyncio
import coloredlogs, logging
import os
import time

logger = logging.getLogger(__name__)
coloredlogs.install(level=os.getenv("LOG_LEVEL", "INFO"), logger=logger)

# ALLOWED_EMAILS = set(config('ALLOWED_EMAILS').split(','))
SHOW_PROGRESS_STATUS = True  # Show detailed progress updates to user
AGENT_MODE = True # True if using a single agent, False if using the entre team

# User-friendly tool names
TOOL_DISPLAY_NAMES = {
    "search_web": "Searching the Web",
    "search_web_multi": "Searching the Web",
    "fetch_url_contents": "Reading Product Pages",
    "fetch_urls": "Reading Product Pages",
}

def get_thinking_message() -> str:
    messages = [
        "Wrapping up ideas... 🎁",
        "Shoveling through insights... ❄️",
        "Skating through the data... ⛸️",
        "Checking the North Pole archives... 🎅",
        "Hopping province to province like a snowflake... ❄️",
        "Sleigh-ing the search... 🛷",
        "Brewing hot cocoa and facts... 🍫",
        "Cutting fresh tracks through the web... 🎿",
        "Gliding across frozen data lakes... 🧊",
        "Lighting up the search like holiday lights... ✨",
        "Searching from coast to *frozen* coast... 🌊",
        "Consulting Canadian elves... 🧝‍♂️",
        "Scooping up frosty findings... 🥶",
        "Tuning into Santa’s signal... 🎅",
        "Crunching through snow-covered stats... 📊",
    ]

    return choice(messages)

# TODO: remove waitlist concept and just have the login screen

def login_screen():
    st.header("Welcome to Snowman ☃️")
    st.write("Please log in to continue.")
    if st.button("🔐 Log in with Google", type="primary"):
        st.login("google")
        st.stop()  # pause render until OAuth round-trip completes
    st.stop()      # keep showing login screen until user exists

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())  # Generate new UUID

# def add_floating_button(
#     colours: dict,
#     link: str = "https://buymeacoffee.com/brydon",
#     emoji: str = "☕️",
#     text: str = "Buy me a coffee",
#     position: dict = {"bottom": "40px", "right": "40px"},
# ):
#     """Add a floating button to the page with customizable properties."""
#     # Initialize session state for page refresh counter
#     if "page_refresh_count" not in st.session_state:
#         st.session_state.page_refresh_count = 0
    
#     # Increment the counter on each page load
#     st.session_state.page_refresh_count += 1
    
#     # Use st.markdown with a direct HTML anchor tag for the button
#     st.markdown(
#         f"""
#     <style>
#         .coffee-btn {{
#             position: fixed;
#             bottom: {position["bottom"]};
#             right: {position["right"]};
#             z-index: 100;
#             background: {colours["background"]};
#             border-radius: 8px;
#             box-shadow: 0 2px 8px rgba(0,0,0,0.10);
#             padding: 0;
#             opacity: 0.85;
#             transition: opacity 0.2s;
#             min-width: 0;
#         }}
#         .coffee-btn:hover {{
#             opacity: 1;
#         }}
#         .coffee-btn a {{
#             display: block;
#             padding: 10px 20px 10px 20px;
#             color: {colours["text"]};
#             text-decoration: none;
#             font-weight: normal;
#             font-size: 15px;
#             background: none;
#             border-radius: 8px;
#             transition: background 0.2s, color 0.2s;
#         }}
#         .coffee-btn a:hover {{
#             background: {colours["background_hover"]};
#             color: {colours["text_hover"]};
#         }}
#     </style>
#     <div class="coffee-btn" id="coffee-btn">
#         <a href="{link}" target="_blank" rel="noopener noreferrer">
#             {emoji if st.session_state.page_refresh_count > 1 else f"{text} {emoji}"}
#         </a>
#     </div>
#     """,
#         unsafe_allow_html=True,
#     )

# Set page config
st.set_page_config(
    page_title="Snowman",
    page_icon="☃️",
    initial_sidebar_state="collapsed",
)

with st.sidebar:
    st.link_button("❤️ Help us improve", "https://forms.gle/5dWaY279oFsfwhTw9")
    st.link_button("📧 Contact us", "mailto:parkerbrydon@gmail.com")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

async def parse_stream(stream: AsyncIterator[RunOutput]) -> AsyncGenerator[tuple[str, str], None]:
    last_event_time = time.time()
    last_event = "start"
    tool_start_time = None
    current_tool = None
    planning_start_time = time.time()  # Start timing from the beginning
    
    async for chunk in stream:
        logger.debug(f"{chunk.event if hasattr(chunk, 'event') else 'unknown'}")
        if hasattr(chunk, "event"):
            if AGENT_MODE:
                if chunk.event == 'RunContent' and chunk.content:
                    if last_event != "content":
                        # Complete any pending analyzing phase
                        if planning_start_time and last_event == "analyzing":
                            elapsed = time.time() - planning_start_time
                            logger.info(f"💭 LLM generating response (took {elapsed:.2f}s to process)")
                            yield ("status_complete", f"✅ ({int(round(elapsed))}s)")
                            planning_start_time = None
                        elif planning_start_time:
                            elapsed = time.time() - planning_start_time
                            logger.info(f"💭 LLM generating response (took {elapsed:.2f}s to process)")
                            yield ("status_complete", f"✅ ({int(round(elapsed))}s)")
                            planning_start_time = None
                        yield ("status_start", "💭 Generating response...")
                        last_event = "content"
                        last_event_time = time.time()
                    yield ("content", chunk.content)
                elif SHOW_PROGRESS_STATUS and chunk.event == "ToolCallStarted":
                    # Complete previous analyzing/thinking phase
                    if last_event in ["analyzing", "start"]:
                        elapsed = time.time() - last_event_time
                        logger.info(f"🧠 LLM planning took {elapsed:.2f}s")
                        yield ("status_complete", f"✅ ({int(round(elapsed))}s)")
                        planning_start_time = None
                    
                    logger.info(f"🔧 Calling {chunk.tool.tool_name}")
                    
                    # Make tool names more user-friendly with context from args
                    current_tool = chunk.tool.tool_name
                    tool_args = chunk.tool.tool_args if hasattr(chunk.tool, 'tool_args') else {}
                    
                    # Create descriptive message based on tool and args
                    if current_tool == "search_web_multi" and "queries" in tool_args:
                        queries = tool_args.get("queries", [])
                        if queries:
                            first_query = queries[0][:50] + "..." if len(queries[0]) > 50 else queries[0]
                            if len(queries) > 1:
                                tool_display = f"Searching for '{first_query}' and {len(queries)-1} more"
                            else:
                                tool_display = f"Searching for '{first_query}'"
                        else:
                            tool_display = "Searching the Web"
                    elif current_tool == "fetch_urls" and "urls" in tool_args:
                        urls = tool_args.get("urls", [])
                        count = len(urls)
                        tool_display = f"Reading {count} product page{'s' if count != 1 else ''}"
                    else:
                        tool_display = TOOL_DISPLAY_NAMES.get(current_tool, current_tool.replace("_", " ").title())
                    
                    tool_start_time = time.time()
                    last_event = "tool_call"
                    last_event_time = time.time()
                    yield ("status_start", f"🔍 {tool_display}...")
                elif chunk.event == "ToolCallCompleted":
                    if tool_start_time and current_tool:
                        elapsed = time.time() - tool_start_time
                        logger.info(f"✅ {current_tool} completed in {elapsed:.2f}s total")
                        yield ("status_complete", f"✅ ({int(round(elapsed))}s)")
                        last_event_time = time.time()
                        # Start analyzing phase immediately after tool completion
                        planning_start_time = time.time()
                        yield ("status_start", "🧠 Analyzing results...")
                        last_event = "analyzing"
            else:
                if chunk.event == 'TeamRunContent' and chunk.content:
                    if last_event != "content":
                        # Complete any pending planning
                        if planning_start_time:
                            elapsed = time.time() - planning_start_time
                            logger.info(f"💭 LLM generating response (took {elapsed:.2f}s to process)")
                            yield ("status_complete", f"✅ ({int(round(elapsed))}s)")
                            planning_start_time = None
                        yield ("status_start", "💭 Generating response...")
                        last_event = "content"
                        last_event_time = time.time()
                    yield ("content", chunk.content)
                elif SHOW_PROGRESS_STATUS and chunk.event == "ToolCallStarted":
                    # Complete previous analyzing/thinking phase
                    if last_event in ["analyzing", "start"]:
                        elapsed = time.time() - last_event_time
                        logger.info(f"🧠 LLM planning took {elapsed:.2f}s")
                        yield ("status_complete", f"✅ ({int(round(elapsed))}s)")
                        planning_start_time = None
                    
                    logger.info(f"🔧 Calling {chunk.tool.tool_name}")
                    
                    # Make tool names more user-friendly with context from args
                    current_tool = chunk.tool.tool_name
                    tool_args = chunk.tool.tool_args if hasattr(chunk.tool, 'tool_args') else {}
                    
                    # Create descriptive message based on tool and args
                    if current_tool == "search_web_multi" and "queries" in tool_args:
                        queries = tool_args.get("queries", [])
                        if queries:
                            first_query = queries[0][:50] + "..." if len(queries[0]) > 50 else queries[0]
                            if len(queries) > 1:
                                tool_display = f"Searching for '{first_query}' and {len(queries)-1} more"
                            else:
                                tool_display = f"Searching for '{first_query}'"
                        else:
                            tool_display = "Searching the Web"
                    elif current_tool == "fetch_urls" and "urls" in tool_args:
                        urls = tool_args.get("urls", [])
                        count = len(urls)
                        tool_display = f"Reading {count} product page{'s' if count != 1 else ''}"
                    else:
                        tool_display = TOOL_DISPLAY_NAMES.get(current_tool, current_tool.replace("_", " ").title())
                    
                    tool_start_time = time.time()
                    last_event = "tool_call"
                    last_event_time = time.time()
                    yield ("status_start", f"🔍 {tool_display}...")
                elif chunk.event == "ToolCallCompleted":
                    if tool_start_time and current_tool:
                        elapsed = time.time() - tool_start_time
                        logger.info(f"✅ {current_tool} completed in {elapsed:.2f}s total")
                        yield ("status_complete", f"✅ ({int(round(elapsed))}s)")
                        last_event_time = time.time()
                        # Start analyzing phase immediately after tool completion
                        planning_start_time = time.time()
                        yield ("status_start", "🧠 Analyzing results...")
                        last_event = "analyzing"
            

def show_waitlist(show_error: bool = True):
    """Display the waitlist signup form and message"""
    st.markdown("---")
    if show_error:
        st.warning("🔒 You don't have access to Snowman just yet.")
    st.write("Please join our waitlist to get access!")
    st.write("[Join the waitlist 📬](https://stan.store/brydon/p/canadian-ai-waitlist-)")
    st.markdown("---")

if not hasattr(st, 'user') or not hasattr(st.user, 'is_logged_in') or not st.user.is_logged_in:
    login_screen()
    # show_waitlist(show_error=False)
# elif st.user.email not in ALLOWED_EMAILS:
#     st.sidebar.button("🔒 Log out", on_click=st.logout, type="secondary")
#     show_waitlist(show_error=True)
else:
    # Show the main app interface
    # Sidebar content
    with st.sidebar:
        st.button("🔐 Log out", on_click=st.logout, type="secondary")
    
    # Main content
    st.title("Snowman ☃️")
    st.caption("Christmas ❄️ Shopping Assistant that is biased to support Canadian businesses 🍁")
    first_name = st.user.name.split(' ')[0] if hasattr(st, 'user') else 'Guest'
    intro_messages = [
        f"Welcome {first_name}, how can I make your holiday season better? 🎄",
        f"Hi {first_name}, what can I do to help you this holiday season? ❄️",
        f"I'm glad you're here {first_name}, how can I help you this holiday season? 🎄",
        f"What can I do to help you this holiday season {first_name}? ❄️",
    ]
    st.write(choice(intro_messages)) 


if hasattr(st.user, 'is_logged_in') and st.user.is_logged_in:

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🎄" if message["role"] == "assistant" else "❄️"):
            st.markdown(message["content"])

    @st.cache_data # not sure why but it breaks if we don't cache this
    def get_placeholder():
        return choice([
            "Help me find a Christmas gift for my father 🎁",
            "Looking for cozy Canadian-made slippers for my mom ❄️",
            "Help me find a new flannel for my husband 🍁",
            "Canadian made hockey stick for my son 🏒",
            "Find me some cozy Canadian Christmas pajamas for my kids 🎄",
            "I want to get my mom a new pair of snow boots ❄️",
            "Looking for a Canadian-made sweater for my wife ❤️",
            "Help me find a new pair of jeans for my daughter 👖",
            "My wife needs a new pair of yoga pants - can you help? 🧘‍♀️",
        ])

    # add_floating_button(
    #     link="https://buymeacoffee.com/brydon",
    #     emoji="☕️",
    #     text="Buy me a coffee",
    #     position={"bottom": "40px", "right": "40px"},
    #     colours={"background": "#f8f9fa", "text": "#666", "text_hover": "#222", "background_hover": "#f1f3f4"}
    # )

    async def run_agent():
        return agent_team.arun(
            prompt, 
            stream=True,
            stream_events=True,
            user_id=st.user.email, # stores memories for the user
            session_id=st.session_state.session_id, # stores the session history for each user
        )

    if prompt := st.chat_input(
            placeholder=get_placeholder()
        ):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user", avatar="❄️"):
            st.markdown(prompt)
        
        # Display assistant response
        with st.chat_message("assistant", avatar="🎄"):
            message_placeholder = st.empty()
            
            with st.spinner(get_thinking_message()):
                start_time = time.time()
                logger.info(f"🚀 Starting agent run for: {prompt[:50]}...")
                agent_team = get_agent_team()
                
                # Create and run the async processing
                async def process_stream():
                    response_parts = []
                    stream_start = time.time()
                    stream = await run_agent()
                    stream_ready_time = time.time() - stream_start
                    logger.info(f"⚡ Stream ready in {stream_ready_time:.2f}s")
                    
                    parsed_stream = parse_stream(stream)
                    
                    # Use separate placeholders for status and content
                    status_container = st.empty()
                    response_placeholder = st.empty()
                    current_response = ""
                    status_lines = ["🧠 Thinking..."]  # Start with initial thinking status
                    status_container.caption("\n\n".join(status_lines))
                    
                    async for content_type, content in parsed_stream:
                        if content_type == "status_start":
                            # Add a new status line (in progress)
                            status_lines.append(content)
                            # Display all status lines with double line breaks and faded color
                            status_container.caption("\n\n".join(status_lines))
                        elif content_type == "status_complete":
                            # Update the last in-progress line with completion info
                            if status_lines:
                                # Keep the "..." and add checkmark with timing
                                status_lines[-1] = f"{status_lines[-1]} {content}"
                            # Display all status lines with double line breaks and faded color
                            status_container.caption("\n\n".join(status_lines))
                        elif content_type == "content":
                            # Clear status when regular content arrives
                            if status_lines:
                                status_container.empty()
                                status_lines = []
                            
                            # Update regular content
                            response_parts.append(content)
                            current_response = "".join(response_parts)
                            response_placeholder.markdown(current_response)
                    
                    # Ensure status placeholder is cleared at the end
                    if status_lines:
                        status_container.empty()
                    
                    return current_response
                
                # Run the async process
                full_response = asyncio.run(process_stream())
                total_time = time.time() - start_time
                logger.info(f"✨ Total response time: {total_time:.2f}s")

            st.session_state.messages.append({"role": "assistant", "content": full_response})