# Build Brain - Snowman Training Data Generator

## Overview

This script generates comprehensive training data for Snowman by:
1. 🎭 Creating 100 diverse Christmas shopping personas
2. ❓ Generating 100 gift requests per persona (10,000 total)
3. 🤖 Getting Snowman v1 responses for each request
4. 💾 Storing everything in pgvector for semantic search

## Requirements

### Environment Variables

Add these to your `.env` file:

```bash
# Database (pgvector in ca-central-1)
POSTGRES_HOST=your-postgres-host.ca-central-1.rds.amazonaws.com
POSTGRES_DB=snowman_brain
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password

# LLM Configuration
LLM_PROVIDER=openai  # or cohere (when implemented)
LLM_MODEL=gpt-4o-mini  # or gpt-4o, gpt-5-mini, etc.
OPENAI_API_KEY=your_openai_key

# Logging
LOG_LEVEL=INFO  # or DEBUG for more detail
```

### Database Setup

Make sure you have PostgreSQL with pgvector extension installed:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

The script will automatically create the necessary tables.

## Configuration

Edit these constants in `build_brain.py` to customize:

```python
NUM_PERSONAS = 100              # Number of personas to generate
QUERIES_PER_PERSONA = 100       # Queries per persona
MAX_CONCURRENT_PERSONAS = 10    # Parallel persona generation
MAX_CONCURRENT_QUERIES = 5      # Parallel query generation
MAX_CONCURRENT_RESPONSES = 3    # Parallel Snowman responses
```

## Usage

### Run the full pipeline:

```bash
python build_brain.py
```

### Expected Output:

```
🧠 Starting Snowman Brain Build Process
📊 Config: 100 personas × 100 queries = 10000 total queries
🤖 LLM: openai/gpt-4o-mini
🗄️  Database: your-host.ca-central-1.rds.amazonaws.com

🔌 Connecting to database...
✅ Database connected in 0.45s

📋 Initializing database schema...
✅ Schema initialized in 0.23s

🎭 Step 1: Generating 100 personas...
Generating personas: 100%|████████████| 100/100 [00:45<00:00,  2.22it/s]
💾 Saving personas to database...
Saving personas: 100%|████████████| 100/100 [00:02<00:00, 45.23it/s]
✅ Step 1 complete: 100 personas generated in 47.34s (0.47s per persona)

❓ Step 2: Generating 100 queries per persona (10000 total)...
Processing personas: 100%|████████████| 100/100 [08:23<00:00,  5.04s/it]
✅ Step 2 complete: 10000 queries generated in 503.21s (0.050s per query)

🤖 Step 3: Getting Snowman responses for 10000 queries...
Getting responses: 100%|████████████| 10000/10000 [2:45:23<00:00,  1.01it/s]
✅ Step 3 complete: 10000 responses in 9923.45s (avg 8.92s per response)

💾 Step 4: Saving 10000 queries and responses to vector database...
Saving to DB: 100%|████████████| 10000/10000 [05:34<00:00, 29.89it/s]
✅ Step 4 complete: Saved to database in 334.56s (0.033s per item)

============================================================
🎉 Brain Build Complete!
⏱️  Total time: 10808.56s (180.14 minutes)
👥 Personas: 100
❓ Queries: 10000
💬 Responses: 10000
📊 Average response time: 8.92s
============================================================
```

## Performance Tuning

### For Faster Generation:
- Increase `MAX_CONCURRENT_PERSONAS` (be mindful of API rate limits)
- Increase `MAX_CONCURRENT_QUERIES`
- Use a faster LLM model (e.g., `gpt-4o-mini` instead of `gpt-4o`)

### For Better Quality:
- Decrease concurrency for more thoughtful responses
- Use `gpt-4o` or `gpt-5-mini` for persona/query generation
- Set `LOG_LEVEL=DEBUG` to see detailed progress

## Database Schema

### Tables Created:

1. **personas** - Stores all generated personas
   - id, name, age, description, shopping_preferences, budget_range, gift_recipients

2. **gift_queries** - Stores all queries with embeddings
   - id, persona_id, query, context, embedding (vector)

3. **snowman_responses** - Stores Snowman's responses
   - id, persona_id, query_id, query, response, response_time

### Vector Search Example:

```python
# Find similar queries
SELECT query, response 
FROM gift_queries gq
JOIN snowman_responses sr ON gq.id = sr.query_id
ORDER BY embedding <=> '[your_query_embedding]'
LIMIT 10;
```

## Troubleshooting

### Database Connection Issues:
- Verify your `POSTGRES_*` environment variables
- Ensure pgvector extension is installed
- Check firewall rules for ca-central-1

### API Rate Limits:
- Reduce `MAX_CONCURRENT_*` values
- Add delays between batches if needed

### Memory Issues:
- Process in smaller batches
- Reduce `NUM_PERSONAS` or `QUERIES_PER_PERSONA`

## Next Steps

After building the brain, you can:
1. Query the vector database for similar past queries
2. Use responses for fine-tuning
3. Analyze persona patterns
4. Build a RAG system using the embeddings

