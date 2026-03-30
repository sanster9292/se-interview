# AI Travel Companion

An intelligent travel assistant powered by LangGraph and OpenAI that provides real-time weather forecasts and travel information through web search capabilities. Features full observability with Phoenix tracing for monitoring LLM interactions and tool usage.

## Features

- **Weather Forecasting**: Get current weather conditions and 3-day forecasts for any location worldwide
- **Web Search**: Access up-to-date travel information, attractions, and general knowledge
- **Interactive Chat UI**: User-friendly web interface for conversing with the AI assistant
- **Full Observability**: Track and analyze all LLM calls, tool usage, and conversation flows with Phoenix

## Tools Added

### 1. Weather Forecast Tool (`get_weather_forecast`)
- **Purpose**: Retrieves current weather and 3-day forecasts for any location
- **API**: Uses Open-Meteo API (free, no authentication required)
- **Features**:
  - Geocoding support for location name resolution
  - Current weather conditions (temperature, feels-like, wind speed, weather description)
  - 3-day forecast with daily highs/lows and precipitation probability
  - WMO weather code mapping (27 different weather conditions)
  - Comprehensive error handling with timeout support
  - Returns structured data in JSON format

### 2. Web Search Tool (`duckduckgo_search`)
- **Purpose**: Searches the web for current information and general knowledge
- **Implementation**: DuckDuckGo search via LangChain Community
- **Features**:
  - No API key required
  - Access to current events and up-to-date information
  - Travel recommendations and tourist attractions
  - General knowledge queries

## Libraries and Technologies Used

### Core Framework
- **LangGraph** (`^0.2`): Agent orchestration and workflow management
- **LangChain** (`^0.3`): LLM framework and tool integration
- **LangChain OpenAI** (`^0.2`): OpenAI model integration
- **LangChain Community** (`^0.3`): Community tools including DuckDuckGo search

### Web Framework
- **FastAPI** (`>=0.115`): Modern, fast web framework for the API
- **Uvicorn** (`^0.32`): ASGI server for running the application
- **Pydantic**: Data validation and settings management

### Observability
- **Arize Phoenix** (`^13.19.2`): LLM observability platform
- **Arize Phoenix OTEL** (`^0.15.0`): OpenTelemetry integration for Phoenix
- **OpenInference LangChain** (`^0.1.61`): Automatic instrumentation for LangChain

### Utilities
- **Requests** (`^2.33.0`): HTTP library for API calls
- **Python Dotenv** (`^1.0`): Environment variable management
- **DDGS** (`^9.0`): DuckDuckGo search implementation

## Local Setup

### Prerequisites
- Python 3.11 or higher (3.11-3.13 supported)
- [Poetry](https://python-poetry.org/docs/#installation) for dependency management

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd se-interview
   ```

2. **Install dependencies**
   ```bash
   poetry install
   ```

3. **Set up environment variables**

   Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   PHOENIX_API_KEY=your_phoenix_api_key_here
   PHOENIX_COLLECTOR_ENDPOINT=https://app.phoenix.arize.com
   SERPER_API_KEY=optional_serper_key
   ```

   **Required:**
   - `OPENAI_API_KEY`: Get from https://platform.openai.com/api-keys

   **Optional:**
   - `PHOENIX_API_KEY`: For Phoenix Cloud (or use local Phoenix server)
   - `SERPER_API_KEY`: Alternative to DuckDuckGo (not required)

4. **Start the Phoenix server** (for local observability)
   ```bash
   poetry run python -m phoenix.server.main serve
   ```

   Phoenix UI will be available at: **http://localhost:6006**

5. **Start the application server**
   ```bash
   poetry run uvicorn api:app --reload --host 0.0.0.0 --port 8000
   ```

   API will be available at: **http://localhost:8000**

## How to Use the UI

### Accessing the Web Interface

1. Open your browser and navigate to: **http://localhost:8000**

2. You'll see the AI Travel Companion chat interface with:
   - **Chat container**: Displays conversation history
   - **Message input**: Multi-line text area for your queries (auto-expands)
   - **Example buttons**: Quick-start queries to try

### Using the Chat Interface

**Example Queries:**
- "What's the weather like in Paris?"
- "Will it rain in London today?"
- "What are the top tourist attractions in Rome?"
- "What's the weather in Barcelona and what are some things to do there?"

**Response Features:**
- **Formatted output**: Temperatures highlighted, bullet points, numbered lists
- **Real-time updates**: Loading indicator while processing
- **Auto-scroll**: Automatically scrolls to show latest messages

### API Endpoints

**Chat Endpoint:**
```bash
POST http://localhost:8000/chat
Content-Type: application/json

{
  "message": "What's the weather in Tokyo?"
}
```

**Health Check:**
```bash
GET http://localhost:8000/health
```

**Web UI:**
```bash
GET http://localhost:8000/
```

## How to Review the Phoenix UI

Phoenix provides comprehensive observability for your AI agent, allowing you to trace and analyze every interaction.

### Accessing Phoenix

**Local Phoenix Server:**
- URL: **http://localhost:6006**
- Automatically captures traces when the application runs

**Phoenix Cloud:**
- URL: **https://app.phoenix.arize.com**
- Requires API key configuration

### Running Test Queries

Generate sample traces for analysis:

```bash
poetry run python run_test_queries.py
```

This script runs 15 diverse queries covering:
- Weather queries (various cities)
- Web search queries (travel info, facts)
- Edge cases (invalid locations, ambiguous queries)
- User frustration scenarios

All traces will appear in Phoenix for analysis.

## User Frustration Analysis

Analyze conversation quality using GPT-4 as a judge to detect user frustration.

**Run the Dashboard:**
```bash
./venv/bin/python frustration_dashboard.py
```

Open http://localhost:8080 and click "Run Frustration Analysis" to:
- Test 15 queries against your agent
- Get GPT-4 analysis of each conversation
- See frustration rate with explanations
- View example frustrated scenarios

## Project Structure

```
se-interview/
├── agent.py                      # LangGraph agent definition
├── tools.py                      # Custom tools (weather forecast)
├── api.py                        # FastAPI application
├── static/
│   └── index.html               # Chat UI
├── run_test_queries.py          # Test query script
├── frustration_dashboard.py     # Frustration analysis UI
├── simple_frustration_analysis.py  # CLI frustration analysis
├── .env                         # Environment variables
├── pyproject.toml               # Poetry dependencies
└── README.md                    # This file
```

## Development

### Adding New Tools

1. Define the tool in `tools.py`:
   ```python
   from langchain_core.tools import tool

   @tool("tool_name")
   def your_tool(param: str) -> dict:
       """Tool description."""
       # Implementation
       return {"result": "data"}
   ```

2. Register in `agent.py`:
   ```python
   from tools import your_tool

   tools = [duckduckgo_search, get_weather_forecast, your_tool]
   ```

3. Update system prompt to describe when to use the tool

### Customizing the UI

- Edit `static/index.html` for UI changes
- Modify CSS in `<style>` section
- Update JavaScript for behavior changes

### Phoenix Configuration

**For Local Phoenix:**
- Endpoint: `http://localhost:6006/v1/traces`
- No authentication required

**For Phoenix Cloud:**
- Endpoint: `https://app.phoenix.arize.com/v1/traces`
- Requires `PHOENIX_API_KEY` in `.env`

## Troubleshooting

**Port Already in Use:**
```bash
# Kill processes on port 8000
lsof -i :8000
kill -9 <PID>
```

**Phoenix Not Showing Traces:**
- Verify Phoenix server is running on port 6006
- Check Phoenix endpoint in `api.py` matches your setup
- Ensure no 401 authentication errors in server logs

**Weather API Errors:**
- Check internet connectivity
- Verify location name is valid
- Review error message in response

## License

This project is for educational and demonstration purposes.

## Acknowledgments

- Weather data provided by [Open-Meteo](https://open-meteo.com/)
- Observability powered by [Arize Phoenix](https://phoenix.arize.com/)
- LLM framework by [LangChain](https://langchain.com/)
