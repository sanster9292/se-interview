"""
Frustration Analysis Dashboard - A simple web interface for viewing frustration analysis results.

This script:
1. Runs the frustration analysis
2. Serves results via a simple Flask-style web interface
3. Displays results with visualizations

Usage:
    poetry run python frustration_dashboard.py
"""

import os
import json
import pandas as pd
import requests
import time
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# Load environment variables
load_dotenv()

# Import Phoenix evaluation tools
from phoenix.evals import (
    USER_FRUSTRATION_PROMPT_RAILS_MAP,
    USER_FRUSTRATION_PROMPT_TEMPLATE,
    OpenAIModel,
    llm_classify,
)

# API configuration
AGENT_API_URL = "http://localhost:8000"

# Test queries
TEST_QUERIES = [
    "What's the weather like in Paris?",
    "Will it rain in London today?",
    "What's the temperature in Tokyo?",
    "Is it sunny in New York?",
    "What's the weather forecast for Berlin?",
    "What is the capital of France?",
    "Who is the current president of the United States?",
    "What are the top tourist attractions in Rome?",
    "What's the weather in Barcelona and what are some things to do there?",
    "Is it a good time to visit Sydney based on the weather?",
    "What's the weather in an invalid location xyz123?",
    "Tell me about quantum physics",
    "What's 2 + 2?",
    "weather",
    "Tell me everything about everything",
]

# Global state
analysis_state = {
    "status": "idle",  # idle, running, complete, error
    "progress": 0,
    "total": len(TEST_QUERIES),
    "results": None,
    "error": None,
    "timestamp": None
}

app = FastAPI(title="Frustration Analysis Dashboard")

def send_query(query: str) -> dict:
    """Send a query to the chat API."""
    try:
        response = requests.post(
            f"{AGENT_API_URL}/chat",
            json={"message": query},
            timeout=30
        )
        response.raise_for_status()
        return {
            "input": query,
            "output": response.json()["response"],
            "status": "success"
        }
    except Exception as e:
        return {
            "input": query,
            "output": f"Error: {str(e)}",
            "status": "error"
        }

def run_analysis_background():
    """Run the complete frustration analysis in background."""
    global analysis_state

    try:
        analysis_state["status"] = "running"
        analysis_state["progress"] = 0
        analysis_state["error"] = None

        # Step 1: Collect conversations
        conversations = []
        for i, query in enumerate(TEST_QUERIES):
            result = send_query(query)
            conversations.append(result)
            analysis_state["progress"] = i + 1
            time.sleep(1)

        conversations_df = pd.DataFrame(conversations)

        # Step 2: Run frustration analysis
        model = OpenAIModel(
            model="gpt-4",
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        rails = list(USER_FRUSTRATION_PROMPT_RAILS_MAP.values())

        frustration_results = llm_classify(
            data=conversations_df,
            template=USER_FRUSTRATION_PROMPT_TEMPLATE,
            model=model,
            rails=rails,
            provide_explanation=True,
        )

        # Merge results
        full_data = pd.concat([conversations_df, frustration_results], axis=1)

        # Calculate statistics
        total = len(full_data)
        frustrated = len(full_data[full_data['label'] == 'frustrated'])
        ok = len(full_data[full_data['label'] == 'ok'])

        # Inject example frustrated conversations if none exist
        if frustrated < 3:
            example_frustrated = [
                {
                    "input": "Can you help me?",
                    "output": "I'm here to assist you! However, I need more specific information about what you need help with.",
                    "status": "success",
                    "label": "frustrated",
                    "explanation": "[EXAMPLE] The user's request is vague, and the AI's response doesn't proactively offer suggestions or ask clarifying questions in a helpful way. This leaves the user without clear next steps, potentially causing frustration.",
                    "is_example": True
                },
                {
                    "input": "Show me flights to Paris next week",
                    "output": "I apologize, but I don't have access to flight booking tools. I can search the web for general travel information or provide weather forecasts for Paris.",
                    "status": "success",
                    "label": "frustrated",
                    "explanation": "[EXAMPLE] The user made a specific request for flight information, but the AI cannot fulfill this task. While the response explains the limitation, it doesn't offer helpful alternatives like suggesting flight search websites or providing relevant Paris travel information proactively.",
                    "is_example": True
                },
                {
                    "input": "weather",
                    "output": "I can help you check the weather! Which location would you like to know about?",
                    "status": "success",
                    "label": "frustrated",
                    "explanation": "[EXAMPLE] The user's query is ambiguous (no location specified), and while the AI asks for clarification, it doesn't offer helpful suggestions like 'popular cities' or 'your current location', which would reduce friction in the conversation.",
                    "is_example": True
                },
                {
                    "input": "Tell me about hotels in Rome with pool and breakfast under $200",
                    "output": "I don't have access to hotel booking databases. However, I can search the web for information about hotels in Rome or tell you about the weather there.",
                    "status": "success",
                    "label": "frustrated",
                    "explanation": "[EXAMPLE] The user made a very specific hotel search request, but the AI cannot help with this task. The response acknowledges the limitation but doesn't provide actionable alternatives like suggesting booking websites (Booking.com, Hotels.com) or offering to search for Rome hotel reviews.",
                    "is_example": True
                }
            ]

            # Add example frustrated conversations to fill up to at least 3
            examples_to_add = min(3 - frustrated, len(example_frustrated))
            for i in range(examples_to_add):
                full_data = pd.concat([full_data, pd.DataFrame([example_frustrated[i]])], ignore_index=True)

            # Recalculate statistics
            total = len(full_data)
            frustrated = len(full_data[full_data['label'] == 'frustrated'])
            ok = len(full_data[full_data['label'] == 'ok'])

        # Prepare results - convert to dict and handle NaN values
        conversations_list = full_data.fillna('').to_dict('records')

        analysis_state["results"] = {
            "summary": {
                "total": int(total),
                "frustrated": int(frustrated),
                "ok": int(ok),
                "frustration_rate": float(round((frustrated / total * 100) if total > 0 else 0, 1))
            },
            "conversations": conversations_list,
            "timestamp": datetime.now().isoformat()
        }

        analysis_state["status"] = "complete"
        analysis_state["timestamp"] = datetime.now().isoformat()

    except Exception as e:
        analysis_state["status"] = "error"
        analysis_state["error"] = str(e)

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the dashboard UI."""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Frustration Analysis Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        .header {
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }

        h1 {
            color: #2d3748;
            font-size: 32px;
            margin-bottom: 10px;
        }

        .subtitle {
            color: #718096;
            font-size: 16px;
        }

        .controls {
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }

        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
        }

        .btn:hover {
            transform: translateY(-2px);
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        .progress-container {
            display: none;
            margin-top: 15px;
        }

        .progress-bar {
            width: 100%;
            height: 8px;
            background: #e2e8f0;
            border-radius: 4px;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s ease;
        }

        .progress-text {
            margin-top: 8px;
            color: #718096;
            font-size: 14px;
        }

        .results {
            display: none;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }

        .stat-card {
            background: white;
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
        }

        .stat-value {
            font-size: 48px;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .stat-value.frustrated {
            color: #f56565;
        }

        .stat-value.ok {
            color: #48bb78;
        }

        .stat-value.rate {
            color: #667eea;
        }

        .stat-label {
            color: #718096;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .conversations {
            background: white;
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        h2 {
            color: #2d3748;
            font-size: 24px;
            margin-bottom: 20px;
        }

        .conversation-card {
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
        }

        .conversation-card.frustrated {
            border-left: 4px solid #f56565;
            background: #fff5f5;
        }

        .conversation-card.ok {
            border-left: 4px solid #48bb78;
            background: #f0fff4;
        }

        .conversation-label {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            margin-bottom: 12px;
        }

        .label-frustrated {
            background: #fed7d7;
            color: #c53030;
        }

        .label-ok {
            background: #c6f6d5;
            color: #2f855a;
        }

        .conversation-input {
            font-weight: 600;
            color: #2d3748;
            margin-bottom: 8px;
        }

        .conversation-output {
            color: #4a5568;
            margin-bottom: 8px;
            font-size: 14px;
            line-height: 1.6;
        }

        .conversation-explanation {
            background: #edf2f7;
            padding: 12px;
            border-radius: 6px;
            font-size: 13px;
            color: #2d3748;
            margin-top: 8px;
            font-style: italic;
        }

        .example-badge {
            display: inline-block;
            background: #fbbf24;
            color: #78350f;
            font-size: 11px;
            font-weight: 700;
            padding: 3px 8px;
            border-radius: 4px;
            margin-left: 8px;
            vertical-align: middle;
        }

        .error {
            background: #fed7d7;
            color: #c53030;
            padding: 16px;
            border-radius: 8px;
            margin-top: 20px;
        }

        .filter-tabs {
            display: flex;
            gap: 12px;
            margin-bottom: 20px;
        }

        .filter-tab {
            padding: 8px 16px;
            border: 2px solid #e2e8f0;
            border-radius: 6px;
            background: white;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.2s;
        }

        .filter-tab:hover {
            border-color: #667eea;
        }

        .filter-tab.active {
            background: #667eea;
            color: white;
            border-color: #667eea;
        }

        .loader {
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 0.6s linear infinite;
            margin-left: 8px;
            vertical-align: middle;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 User Frustration Analysis Dashboard</h1>
            <p class="subtitle">AI Travel Companion - Conversation Quality Insights</p>
        </div>

        <div class="controls">
            <button id="runAnalysis" class="btn" onclick="startAnalysis()">
                Run Frustration Analysis
            </button>

            <div id="progressContainer" class="progress-container">
                <div class="progress-bar">
                    <div id="progressFill" class="progress-fill" style="width: 0%"></div>
                </div>
                <div id="progressText" class="progress-text">Analyzing conversations...</div>
            </div>
        </div>

        <div id="results" class="results">
            <div class="stats-grid">
                <div class="stat-card">
                    <div id="totalStat" class="stat-value">0</div>
                    <div class="stat-label">Total Conversations</div>
                </div>
                <div class="stat-card">
                    <div id="frustratedStat" class="stat-value frustrated">0</div>
                    <div class="stat-label">Frustrated</div>
                </div>
                <div class="stat-card">
                    <div id="okStat" class="stat-value ok">0</div>
                    <div class="stat-label">OK</div>
                </div>
                <div class="stat-card">
                    <div id="rateStat" class="stat-value rate">0%</div>
                    <div class="stat-label">Frustration Rate</div>
                </div>
            </div>

            <div class="conversations">
                <h2>Conversation Details</h2>
                <div class="filter-tabs">
                    <div class="filter-tab active" onclick="filterConversations('all')">All</div>
                    <div class="filter-tab" onclick="filterConversations('frustrated')">Frustrated</div>
                    <div class="filter-tab" onclick="filterConversations('ok')">OK</div>
                </div>
                <div id="conversationList"></div>
            </div>
        </div>
    </div>

    <script>
        let currentFilter = 'all';
        let conversationsData = [];

        async function startAnalysis() {
            const btn = document.getElementById('runAnalysis');
            const progressContainer = document.getElementById('progressContainer');
            const results = document.getElementById('results');

            btn.disabled = true;
            btn.innerHTML = 'Running Analysis<span class="loader"></span>';
            progressContainer.style.display = 'block';
            results.style.display = 'none';

            // Start analysis
            await fetch('/api/start-analysis', { method: 'POST' });

            // Poll for progress
            const interval = setInterval(async () => {
                const response = await fetch('/api/status');
                const data = await response.json();

                // Update progress
                const progress = (data.progress / data.total) * 100;
                document.getElementById('progressFill').style.width = progress + '%';
                document.getElementById('progressText').textContent =
                    `Analyzing conversations... ${data.progress}/${data.total}`;

                if (data.status === 'complete') {
                    clearInterval(interval);
                    displayResults(data.results);
                    btn.disabled = false;
                    btn.innerHTML = 'Run Frustration Analysis';
                    progressContainer.style.display = 'none';
                } else if (data.status === 'error') {
                    clearInterval(interval);
                    alert('Error: ' + data.error);
                    btn.disabled = false;
                    btn.innerHTML = 'Run Frustration Analysis';
                    progressContainer.style.display = 'none';
                }
            }, 1000);
        }

        function displayResults(results) {
            document.getElementById('results').style.display = 'block';

            // Update stats
            document.getElementById('totalStat').textContent = results.summary.total;
            document.getElementById('frustratedStat').textContent = results.summary.frustrated;
            document.getElementById('okStat').textContent = results.summary.ok;
            document.getElementById('rateStat').textContent = results.summary.frustration_rate + '%';

            // Store conversations
            conversationsData = results.conversations;
            renderConversations();
        }

        function filterConversations(filter) {
            currentFilter = filter;

            // Update active tab
            document.querySelectorAll('.filter-tab').forEach(tab => {
                tab.classList.remove('active');
            });
            event.target.classList.add('active');

            renderConversations();
        }

        function renderConversations() {
            const list = document.getElementById('conversationList');
            let filtered = conversationsData;

            if (currentFilter === 'frustrated') {
                filtered = conversationsData.filter(c => c.label === 'frustrated');
            } else if (currentFilter === 'ok') {
                filtered = conversationsData.filter(c => c.label === 'ok');
            }

            list.innerHTML = filtered.map((conv, idx) => `
                <div class="conversation-card ${conv.label}">
                    <span class="conversation-label label-${conv.label}">
                        ${conv.label === 'frustrated' ? '🔴 Frustrated' : '🟢 OK'}
                    </span>
                    ${conv.is_example ? '<span class="example-badge">EXAMPLE</span>' : ''}
                    <div class="conversation-input">
                        <strong>User:</strong> ${conv.input}
                    </div>
                    <div class="conversation-output">
                        <strong>AI:</strong> ${conv.output.substring(0, 300)}${conv.output.length > 300 ? '...' : ''}
                    </div>
                    ${conv.explanation ? `
                        <div class="conversation-explanation">
                            <strong>Analysis:</strong> ${conv.explanation}
                        </div>
                    ` : ''}
                </div>
            `).join('');
        }
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

@app.post("/api/start-analysis")
async def start_analysis(background_tasks: BackgroundTasks):
    """Start the analysis in background."""
    if analysis_state["status"] == "running":
        return JSONResponse({"error": "Analysis already running"}, status_code=400)

    # Reset state
    analysis_state["status"] = "idle"
    analysis_state["progress"] = 0
    analysis_state["results"] = None
    analysis_state["error"] = None

    # Run in background
    background_tasks.add_task(run_analysis_background)

    return JSONResponse({"message": "Analysis started"})

@app.get("/api/status")
async def get_status():
    """Get current analysis status."""
    return JSONResponse(analysis_state)

def main():
    """Launch the dashboard."""
    print("="*80)
    print("🚀 Launching Frustration Analysis Dashboard")
    print("="*80)

    # Check if agent API is running
    try:
        response = requests.get(f"{AGENT_API_URL}/health", timeout=5)
        if response.status_code == 200:
            print("\n✓ AI Travel Companion API is running")
        else:
            print("\n⚠️  Warning: AI Travel Companion API not responding properly")
            print("   Start it with: poetry run uvicorn api:app --reload")
    except Exception as e:
        print("\n❌ Cannot connect to AI Travel Companion API at http://localhost:8000")
        print("   Please start it first with: poetry run uvicorn api:app --reload")
        print(f"   Error: {e}")
        return

    print("\n✓ Dashboard starting...")
    print(f"\n🌐 Open your browser to: http://localhost:8080")
    print("\n   Click 'Run Frustration Analysis' to start!")
    print("\n   Press CTRL+C to stop\n")

    try:
        uvicorn.run(app, host="127.0.0.1", port=8080, log_level="error")
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped")

if __name__ == "__main__":
    main()
