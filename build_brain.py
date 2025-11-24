"""
Build Brain - Generate training data for Snowman AI

This script generates comprehensive training data by:
1. Creating diverse Christmas shopping personas
2. Generating gift requests for each persona
3. Getting Snowman responses for each request
4. Storing everything in pgvector for future retrieval
"""

import asyncio
import logging
import time
from typing import List, Dict
from datetime import datetime
import os
from dataclasses import dataclass
import json

import coloredlogs
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
from decouple import config
import psycopg
from psycopg.types.json import Json
from openai import AsyncOpenAI
import cohere

from team import get_agent_team

# Setup logging
logger = logging.getLogger(__name__)
coloredlogs.install(level=os.getenv("LOG_LEVEL", "INFO"), logger=logger)

# Configuration
# NUM_PERSONAS = 10
NUM_PERSONAS = 2
# QUERIES_PER_PERSONA = 10
QUERIES_PER_PERSONA = 2
MAX_CONCURRENT_PERSONAS = 10  # Parallel persona generation
MAX_CONCURRENT_QUERIES = 5    # Parallel query generation per persona
MAX_CONCURRENT_RESPONSES = 3  # Parallel Snowman responses

# LLM Configuration - easily switchable
LLM_PROVIDER = config("LLM_PROVIDER", default="openai") # openai, cohere, etc.
LLM_MODEL = config("LLM_MODEL", default="gpt-5-nano") # gpt-4o, gpt-5-mini, etc.
# LLM_PROVIDER = "cohere"
# LLM_MODEL = "command-a-03-2025" # $2.50 / 1M input tokens
# LLM_MODEL = 'command-r7b-12-2024' # $0.0375 / 1M input tokens
EMBEDDING_PROVIDER = "cohere"
EMBEDDING_MODEL = "embed-v4.0"
EMBEDDING_DIMENSIONS = 1536

# Database Configuration
DB_CONFIG = {
    "host": config("POSTGRES_HOST"),
    "dbname": config("POSTGRES_DB"),
    "user": config("POSTGRES_USER"),
    "password": config("POSTGRES_PASSWORD"),
}


@dataclass
class Persona:
    """Represents a Christmas shopping persona"""
    id: int
    name: str
    age: int
    description: str
    shopping_preferences: str
    budget_range: str
    gift_recipients: List[str]


@dataclass
class GiftRequest:
    """Represents a gift request from a persona"""
    persona_id: int
    query: str
    context: str


@dataclass
class SnowmanResponse:
    """Represents Snowman's response to a gift request"""
    persona_id: int
    query: str
    response: str
    response_time: float
    timestamp: datetime


class LLMProvider:
    """Abstraction layer for different LLM providers"""
    
    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini"):
        self.provider = provider
        self.model = model
        self.embedding_provider = EMBEDDING_PROVIDER
        self.embedding_model = EMBEDDING_MODEL
        
        if provider == "openai":
            self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif provider == "cohere":
            self.client = cohere.AsyncClientV2(api_key=config("COHERE_API_KEY"))
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        if self.embedding_provider == "openai":
            self.embedding_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.embedding_provider == "cohere":
            self.embedding_client = cohere.AsyncClientV2(api_key=config("COHERE_API_KEY"))
        else:
            raise ValueError(f"Unsupported embedding provider: {self.embedding_provider}")
    
    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate text using the configured LLM"""
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        if self.provider == "openai":
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            return response.choices[0].message.content
        
        elif self.provider == "cohere":
            response = await self.client.chat(
                model=self.model,
                messages=messages,
            )
            return response.message.content[0].text
        
        raise NotImplementedError(f"Provider {self.provider} not implemented")
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings for text"""
        if self.embedding_provider == "openai":
            response = await self.embedding_client.embeddings.create(
                model=self.embedding_model,
                input=text,
            )
            return response.data[0].embedding
        
        elif self.embedding_provider == "cohere":
            response = await self.embedding_client.embed(
                texts=[text],
                model=self.embedding_model,
                input_type="search_document",  # For indexing/storage
                embedding_types=["float"],
                output_dimension=int(EMBEDDING_DIMENSIONS),
            )
            return response.embeddings.float_[0]
        
        raise NotImplementedError(f"Provider {self.provider} not implemented")


class DatabaseManager:
    """Manages pgvector database operations"""
    
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.conn = None
    
    async def connect(self):
        """Connect to the database"""
        logger.info("🔌 Connecting to database...")
        start = time.time()
        
        conn_string = f"host={self.db_config['host']} dbname={self.db_config['dbname']} user={self.db_config['user']} password={self.db_config['password']}"
        self.conn = await psycopg.AsyncConnection.connect(conn_string)
        
        elapsed = time.time() - start
        logger.info(f"✅ Database connected in {elapsed:.2f}s")
    
    async def initialize_schema(self):
        """Create necessary tables if they don't exist"""
        logger.info("📋 Initializing database schema...")
        start = time.time()
        
        async with self.conn.cursor() as cur:
            # Enable pgvector extension
            await cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Create personas table
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS personas (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    age INTEGER,
                    description TEXT,
                    shopping_preferences TEXT,
                    budget_range TEXT,
                    gift_recipients JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Create queries table with embeddings
            await cur.execute(f"""
                CREATE TABLE IF NOT EXISTS gift_queries (
                    id SERIAL PRIMARY KEY,
                    persona_id INTEGER REFERENCES personas(id),
                    query TEXT NOT NULL,
                    context TEXT,
                    embedding vector({EMBEDDING_DIMENSIONS}),
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # TODO: why do we need a responses table?
            # Create responses table
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS snowman_responses (
                    id SERIAL PRIMARY KEY,
                    persona_id INTEGER REFERENCES personas(id),
                    query_id INTEGER REFERENCES gift_queries(id),
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    response_time FLOAT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # TODO: why do we need a vector index?
            # Create vector index for similarity search
            await cur.execute("""
                CREATE INDEX IF NOT EXISTS gift_queries_embedding_idx 
                ON gift_queries 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)
            
            await self.conn.commit()
        
        elapsed = time.time() - start
        logger.info(f"✅ Schema initialized in {elapsed:.2f}s")
    
    async def save_persona(self, persona: Persona) -> int:
        """Save a persona and return its ID"""
        async with self.conn.cursor() as cur:
            # Ensure all fields are proper types (handle unexpected LLM responses)
            name = str(persona.name) if not isinstance(persona.name, dict) else json.dumps(persona.name)
            age = int(persona.age) if not isinstance(persona.age, dict) else 0
            description = str(persona.description) if not isinstance(persona.description, dict) else json.dumps(persona.description)
            shopping_prefs = str(persona.shopping_preferences) if not isinstance(persona.shopping_preferences, dict) else json.dumps(persona.shopping_preferences)
            budget = str(persona.budget_range) if not isinstance(persona.budget_range, dict) else json.dumps(persona.budget_range)
            recipients = persona.gift_recipients if isinstance(persona.gift_recipients, list) else []
            
            logger.debug(f"Saving persona: {name}, all types validated")
            
            await cur.execute(
                """
                INSERT INTO personas (name, age, description, shopping_preferences, budget_range, gift_recipients)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (name, age, description, shopping_prefs, budget, Json(recipients)),
            )
            result = await cur.fetchone()
            await self.conn.commit()
            return result[0]
    
    async def save_query(self, query: GiftRequest, embedding: List[float]) -> int:
        """Save a gift query with its embedding"""
        async with self.conn.cursor() as cur:
            # Convert embedding list to pgvector format
            embedding_str = '[' + ','.join(str(x) for x in embedding) + ']'
            await cur.execute(
                """
                INSERT INTO gift_queries (persona_id, query, context, embedding)
                VALUES (%s, %s, %s, %s::vector)
                RETURNING id
                """,
                (query.persona_id, query.query, query.context, embedding_str),
            )
            result = await cur.fetchone()
            await self.conn.commit()
            return result[0]
    
    async def save_response(self, response: SnowmanResponse, query_id: int):
        """Save a Snowman response"""
        async with self.conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO snowman_responses (persona_id, query_id, query, response, response_time)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    response.persona_id,
                    query_id,
                    response.query,
                    response.response,
                    response.response_time,
                ),
            )
            await self.conn.commit()
    
    async def close(self):
        """Close database connection"""
        if self.conn:
            await self.conn.close()
            logger.info("🔌 Database connection closed")


class BrainBuilder:
    """Main class for building the Snowman brain"""
    
    def __init__(self):
        self.llm = LLMProvider(provider=LLM_PROVIDER, model=LLM_MODEL)
        self.db = DatabaseManager(DB_CONFIG)
        self.agent = None
    
    async def generate_persona(self, persona_id: int) -> Persona:
        """Generate a single Christmas shopping persona"""
        prompt = f"""Generate a realistic Christmas shopping persona (persona #{persona_id}). 
        This should be a diverse individual with unique characteristics.
        
        Return a JSON object with:
        - name: Full name (STRING)
        - age: Age (INTEGER between 20-75)
        - description: 2-3 sentence personality description (STRING)
        - shopping_preferences: Their shopping style and preferences (STRING)
        - budget_range: Their typical budget (STRING, e.g., "$50-$200 per gift")
        - gift_recipients: Array of 3-8 people they shop for (ARRAY of STRINGS, e.g., ["spouse", "mother", "best friend"])
        
        Make this person feel real and diverse. Vary demographics, income levels, family situations, etc.
        Return ONLY the JSON, no other text.
        """
        
        response = await self.llm.generate(
            prompt,
            system_prompt="You are an expert at creating realistic, diverse personas. Return ONLY valid JSON with no markdown formatting."
        )
        
        # Parse JSON response
        cleaned_response = response.strip().replace("```json", "").replace("```", "").strip()
        try:
            persona_data = json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse persona JSON: {cleaned_response[:200]}")
            raise
        
        # Ensure all fields are the correct type
        return Persona(
            id=persona_id,
            name=str(persona_data["name"]),
            age=int(persona_data["age"]),
            description=str(persona_data["description"]),
            shopping_preferences=str(persona_data["shopping_preferences"]),
            budget_range=str(persona_data["budget_range"]),
            gift_recipients=list(persona_data["gift_recipients"]) if isinstance(persona_data["gift_recipients"], list) else [str(persona_data["gift_recipients"])],
        )
    
    async def generate_query_for_persona(self, persona: Persona, query_num: int) -> GiftRequest:
        """Generate a gift request query for a persona"""
        prompt = f"""You are {persona.name}, a {persona.age}-year-old with these characteristics:
        {persona.description}
        
        Shopping preferences: {persona.shopping_preferences}
        Budget range: {persona.budget_range}
        You're shopping for: {', '.join(persona.gift_recipients)}
        
        Generate a realistic Christmas gift search query as if you're asking Snowman (a Canadian-focused shopping assistant).
        This is query #{query_num} from you. Make it natural and specific.
        
        Return only the query text, nothing else. Examples:
        - "I need a warm winter coat for my mom who loves hiking"
        - "Looking for a unique kitchen gadget for my foodie husband under $100"
        - "Help me find eco-friendly toys for my 5-year-old nephew"
        """
        
        query = await self.llm.generate(prompt)
        query = query.strip().strip('"').strip("'")
        
        return GiftRequest(
            persona_id=persona.id,
            query=query,
            context=f"Persona: {persona.name}, Budget: {persona.budget_range}",
        )
    
    async def get_snowman_response(self, query: GiftRequest) -> SnowmanResponse:
        """Get Snowman's response to a query"""
        if not self.agent:
            self.agent = get_agent_team()
        
        start = time.time()
        
        try:
            # Run Snowman agent (use arun for async tools)
            response = await self.agent.arun(
                query.query,
                stream=False,
                user_id=f"persona_{query.persona_id}",
                session_id=f"build_brain_{query.persona_id}",
            )
            
            response_text = response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            logger.error(f"❌ Error getting Snowman response: {e}")
            response_text = f"Error: {str(e)}"
        
        elapsed = time.time() - start
        
        return SnowmanResponse(
            persona_id=query.persona_id,
            query=query.query,
            response=response_text,
            response_time=elapsed,
            timestamp=datetime.now(),
        )
    
    async def step_1_generate_personas(self) -> List[Persona]:
        """Step 1: Generate all personas"""
        logger.info(f"🎭 Step 1: Generating {NUM_PERSONAS} personas...")
        start = time.time()
        
        # Generate personas in parallel
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_PERSONAS)
        
        async def generate_with_limit(persona_id: int) -> Persona:
            async with semaphore:
                return await self.generate_persona(persona_id)
        
        tasks = [generate_with_limit(i) for i in range(1, NUM_PERSONAS + 1)]
        personas = await tqdm_asyncio.gather(*tasks, desc="Generating personas")
        
        # Save to database
        logger.info("💾 Saving personas to database...")
        for persona in tqdm(personas, desc="Saving personas"):
            persona.id = await self.db.save_persona(persona)
        
        elapsed = time.time() - start
        logger.info(f"✅ Step 1 complete: {NUM_PERSONAS} personas generated in {elapsed:.2f}s ({elapsed/NUM_PERSONAS:.2f}s per persona)")
        
        return personas
    
    async def step_2_generate_queries(self, personas: List[Persona]) -> List[GiftRequest]:
        """Step 2: Generate queries for all personas"""
        logger.info(f"❓ Step 2: Generating {QUERIES_PER_PERSONA} queries per persona ({NUM_PERSONAS * QUERIES_PER_PERSONA} total)...")
        start = time.time()
        
        all_queries = []
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_QUERIES)
        
        async def generate_with_limit(persona: Persona, query_num: int) -> GiftRequest:
            async with semaphore:
                return await self.generate_query_for_persona(persona, query_num)
        
        for persona in tqdm(personas, desc="Processing personas"):
            tasks = [generate_with_limit(persona, i) for i in range(1, QUERIES_PER_PERSONA + 1)]
            queries = await asyncio.gather(*tasks)
            all_queries.extend(queries)
        
        elapsed = time.time() - start
        logger.info(f"✅ Step 2 complete: {len(all_queries)} queries generated in {elapsed:.2f}s ({elapsed/len(all_queries):.3f}s per query)")
        
        return all_queries
    
    async def step_3_get_responses(self, queries: List[GiftRequest]) -> List[SnowmanResponse]:
        """Step 3: Get Snowman responses for all queries"""
        logger.info(f"🤖 Step 3: Getting Snowman responses for {len(queries)} queries...")
        start = time.time()
        
        responses = []
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_RESPONSES)
        
        async def get_response_with_limit(query: GiftRequest) -> SnowmanResponse:
            async with semaphore:
                return await self.get_snowman_response(query)
        
        tasks = [get_response_with_limit(q) for q in queries]
        responses = await tqdm_asyncio.gather(*tasks, desc="Getting responses")
        
        elapsed = time.time() - start
        avg_time = sum(r.response_time for r in responses) / len(responses)
        logger.info(f"✅ Step 3 complete: {len(responses)} responses in {elapsed:.2f}s (avg {avg_time:.2f}s per response)")
        
        return responses
    
    async def step_4_save_to_vector_db(self, queries: List[GiftRequest], responses: List[SnowmanResponse]):
        """Step 4: Save everything to vector database with embeddings"""
        logger.info(f"💾 Step 4: Saving {len(queries)} queries and responses to vector database...")
        start = time.time()
        
        for query, response in tqdm(zip(queries, responses), total=len(queries), desc="Saving to DB"):
            # Generate embedding for the query
            embedding = await self.llm.generate_embedding(query.query)
            
            # Save query with embedding
            query_id = await self.db.save_query(query, embedding)
            
            # Save response
            await self.db.save_response(response, query_id)
        
        elapsed = time.time() - start
        logger.info(f"✅ Step 4 complete: Saved to database in {elapsed:.2f}s ({elapsed/len(queries):.3f}s per item)")
    
    async def build(self):
        """Run the complete brain building process"""
        total_start = time.time()
        logger.info("🧠 Starting Snowman Brain Build Process")
        logger.info(f"📊 Config: {NUM_PERSONAS} personas × {QUERIES_PER_PERSONA} queries = {NUM_PERSONAS * QUERIES_PER_PERSONA} total queries")
        logger.info(f"🤖 LLM: {LLM_PROVIDER}/{LLM_MODEL}")
        logger.info(f"🗄️  Database: {DB_CONFIG['host']}")
        
        try:
            # Connect to database
            await self.db.connect()
            await self.db.initialize_schema()
            
            # TODO: we're probably going to need to batch this more to not overload memory when we scale
            # ex: do any outer_loop batching up personas by 100 or something

            # Run all steps
            personas = await self.step_1_generate_personas()
            queries = await self.step_2_generate_queries(personas)
            responses = await self.step_3_get_responses(queries)
            await self.step_4_save_to_vector_db(queries, responses)
            
            # Final summary
            total_elapsed = time.time() - total_start
            logger.info("=" * 60)
            logger.info("🎉 Brain Build Complete!")
            logger.info(f"⏱️  Total time: {total_elapsed:.2f}s ({total_elapsed/60:.2f} minutes)")
            logger.info(f"👥 Personas: {len(personas)}")
            logger.info(f"❓ Queries: {len(queries)}")
            logger.info(f"💬 Responses: {len(responses)}")
            logger.info(f"📊 Average response time: {sum(r.response_time for r in responses)/len(responses):.2f}s")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"❌ Error during brain build: {e}", exc_info=True)
            raise
        finally:
            await self.db.close()


async def main():
    """Main entry point"""
    builder = BrainBuilder()
    await builder.build()


if __name__ == "__main__":
    asyncio.run(main())

