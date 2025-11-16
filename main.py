import streamlit as st
from typing import AsyncIterator, AsyncGenerator
from agno.agent import RunOutput
from team import get_agent_team
from random import choice
import uuid
import asyncio
from decouple import config
import coloredlogs, logging
import os

logger = logging.getLogger(__name__)
coloredlogs.install(level=os.getenv("LOG_LEVEL", "INFO"), logger=logger)

ALLOWED_EMAILS = set(config('ALLOWED_EMAILS').split(','))
SHOW_TOOL_CALLS = True

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

def add_floating_button(
    colours: dict,
    link: str = "https://buymeacoffee.com/brydon",
    emoji: str = "☕️",
    text: str = "Buy me a coffee",
    position: dict = {"bottom": "40px", "right": "40px"},
):
    """Add a floating button to the page with customizable properties."""
    # Initialize session state for page refresh counter
    if "page_refresh_count" not in st.session_state:
        st.session_state.page_refresh_count = 0
    
    # Increment the counter on each page load
    st.session_state.page_refresh_count += 1
    
    # Use st.markdown with a direct HTML anchor tag for the button
    st.markdown(
        f"""
    <style>
        .coffee-btn {{
            position: fixed;
            bottom: {position["bottom"]};
            right: {position["right"]};
            z-index: 100;
            background: {colours["background"]};
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.10);
            padding: 0;
            opacity: 0.85;
            transition: opacity 0.2s;
            min-width: 0;
        }}
        .coffee-btn:hover {{
            opacity: 1;
        }}
        .coffee-btn a {{
            display: block;
            padding: 10px 20px 10px 20px;
            color: {colours["text"]};
            text-decoration: none;
            font-weight: normal;
            font-size: 15px;
            background: none;
            border-radius: 8px;
            transition: background 0.2s, color 0.2s;
        }}
        .coffee-btn a:hover {{
            background: {colours["background_hover"]};
            color: {colours["text_hover"]};
        }}
    </style>
    <div class="coffee-btn" id="coffee-btn">
        <a href="{link}" target="_blank" rel="noopener noreferrer">
            {emoji if st.session_state.page_refresh_count > 1 else f"{text} {emoji}"}
        </a>
    </div>
    """,
        unsafe_allow_html=True,
    )

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
    async for chunk in stream:
        if hasattr(chunk, "event"):
            if chunk.event == 'TeamRunContent' and chunk.content:
                yield ("content", chunk.content)
            elif SHOW_TOOL_CALLS and chunk.event == "ToolCallStarted":
                yield ("tool_call", f"🔧 {chunk.tool.tool_name} - {chunk.tool.tool_args}")  # More concise tool call display
            

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
    show_waitlist(show_error=False)
elif st.user.email not in ALLOWED_EMAILS:
    st.sidebar.button("🔒 Log out", on_click=st.logout, type="secondary")
    show_waitlist(show_error=True)
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


if hasattr(st.user, 'is_logged_in') and st.user.is_logged_in and st.user.email in ALLOWED_EMAILS:

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
                agent_team = get_agent_team()
                
                # Create and run the async processing
                async def process_stream():
                    response_parts = []
                    stream = await run_agent()
                    parsed_stream = parse_stream(stream)
                    
                    # Use separate placeholders for tool calls and content
                    tool_call_placeholder = st.empty()
                    response_placeholder = st.empty()
                    current_response = ""
                    current_tool_call = ""
                    
                    async for content_type, content in parsed_stream:
                        logger.debug(f"content_type: {content_type}, content: {content}")
                        if content_type == "tool_call":
                            # Show tool call as a temporary caption
                            current_tool_call = content
                            tool_call_placeholder.caption(current_tool_call)
                        elif content_type == "content":
                            # Clear tool call when regular content arrives
                            if current_tool_call:
                                tool_call_placeholder.empty()
                                current_tool_call = ""
                            
                            # Update regular content
                            response_parts.append(content)
                            current_response = "".join(response_parts)
                            response_placeholder.markdown(current_response)
                    
                    # Ensure tool call placeholder is cleared at the end
                    if current_tool_call:
                        tool_call_placeholder.empty()
                    
                    return current_response
                
                # Run the async process
                full_response = asyncio.run(process_stream())

            st.session_state.messages.append({"role": "assistant", "content": full_response})