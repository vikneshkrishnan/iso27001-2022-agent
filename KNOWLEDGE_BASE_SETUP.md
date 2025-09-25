# Knowledge Base Setup Guide

This guide helps you set up the Pinecone vector database and knowledge base for the ISO 27001:2022 Agent.

## 🔧 Prerequisites

### 1. API Keys Required
You need to obtain API keys from the following services:

- **Pinecone**: Vector database for storing ISO knowledge
  - Sign up at: https://www.pinecone.io/
  - Get your API key from the dashboard

- **OpenAI**: For generating embeddings and LLM processing
  - Sign up at: https://platform.openai.com/
  - Get your API key from the API section

### 2. Environment Configuration
Add the following to your `.env` file:

```bash
# Pinecone Configuration
PINECONE_API_KEY=your-pinecone-api-key-here
PINECONE_ENVIRONMENT=us-east-1  # or your preferred region
PINECONE_INDEX_NAME=iso27001
PINECONE_DIMENSION=1024

# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key-here

# Optional: Anthropic (for additional LLM capabilities)
ANTHROPIC_API_KEY=your-anthropic-api-key-here
```

## 🚀 Setup Steps

### 1. Check Current Status
```bash
./kb_cli.py status
```

### 2. Initialize Knowledge Base
Once your API keys are configured:
```bash
./kb_cli.py init
```

This will:
- Create the Pinecone index if it doesn't exist
- Index all 93 ISO 27001:2022 Annex A controls
- Index all management clauses (4-10)
- Set up vector embeddings for semantic search

### 3. Validate Setup
```bash
./kb_cli.py validate
```

### 4. Check Final Status
```bash
./kb_cli.py status --json
```

## 📊 Expected Results

After successful initialization, you should see:
- **Total Documents**: ~200+ (controls + clauses + requirements)
- **Namespaces**: `knowledge`, `controls`, `clauses` populated
- **Health Status**: ✅ Healthy
- **Initialization Required**: No

## 🛠 Troubleshooting

### Common Issues

**"Pinecone client not available"**
- Check that your `PINECONE_API_KEY` is set in `.env`
- Verify the API key is valid in your Pinecone dashboard

**"OpenAI client not available for embeddings"**
- Check that your `OPENAI_API_KEY` is set in `.env`
- Verify you have credits/billing set up in OpenAI

**"Failed to initialize knowledge base"**
- Check your internet connection
- Verify API keys are correct
- Check the logs in the `logs/` directory for detailed errors

### Manual Cleanup
If you need to start over:
```bash
./kb_cli.py cleanup --yes
./kb_cli.py init
```

## 💡 Tips

1. **API Costs**: Initial indexing uses ~$2-5 in OpenAI credits
2. **Time**: Initial setup takes 5-10 minutes depending on network speed
3. **Storage**: Uses minimal Pinecone storage (~50MB for free tier)
4. **Updates**: Re-run `init` to refresh knowledge base with latest data

## 🔍 Verification

Once set up successfully:
- Web UI: `http://localhost:8000/api/v1/knowledge-base/status`
- CLI: `./kb_cli.py status`
- Test analysis with any ISO-related document

The knowledge base will dramatically improve analysis quality by enabling:
- Semantic similarity matching
- Historical pattern recognition
- Intelligent control selection
- Context-aware recommendations