"""
Simple user frustration analysis for AI Travel Companion.

This script runs test queries and evaluates them for user frustration using
Arize Phoenix's LLM-as-a-Judge evaluator.

Usage:
    poetry run python simple_frustration_analysis.py
"""

import os
import pandas as pd
import requests
import time
from datetime import datetime
from dotenv import load_dotenv

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
BASE_URL = "http://localhost:8000"

# Test queries covering different scenarios
TEST_QUERIES = [
    # Weather queries
    "What's the weather like in Paris?",
    "Will it rain in London today?",
    "What's the temperature in Tokyo?",
    "Is it sunny in New York?",
    "What's the weather forecast for Berlin?",

    # General knowledge queries (web search)
    "What is the capital of France?",
    "Who is the current president of the United States?",
    "What are the top tourist attractions in Rome?",

    # Mixed queries
    "What's the weather in Barcelona and what are some things to do there?",
    "Is it a good time to visit Sydney based on the weather?",

    # Edge cases
    "What's the weather in an invalid location xyz123?",
    "Tell me about quantum physics",
    "What's 2 + 2?",

    # User frustration scenarios (intentionally confusing/unclear)
    "weather",  # Ambiguous - no location specified
    "Tell me everything about everything",  # Too broad
]

def send_query_and_record(query: str) -> dict:
    """Send a query to the chat API and record the conversation."""
    try:
        print(f"   Querying: {query[:50]}...")

        response = requests.post(
            f"{BASE_URL}/chat",
            json={"message": query},
            timeout=30
        )
        response.raise_for_status()

        result = response.json()

        return {
            "input": query,
            "output": result["response"],
            "status": "success"
        }
    except Exception as e:
        return {
            "input": query,
            "output": f"Error: {str(e)}",
            "status": "error"
        }

def collect_conversations():
    """Run all test queries and collect conversations."""
    print("🚀 Running test queries and collecting responses...\n")

    conversations = []

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"[{i}/{len(TEST_QUERIES)}]", end=" ")

        result = send_query_and_record(query)
        conversations.append(result)

        # Small delay between requests
        time.sleep(1)

    print(f"\n✓ Collected {len(conversations)} conversations\n")
    return pd.DataFrame(conversations)

def run_frustration_analysis(conversations_df):
    """Run user frustration analysis using Phoenix LLM-as-a-Judge evaluator."""
    print("🤖 Running user frustration analysis with GPT-4...")
    print("   (This may take a minute...)\n")

    # Initialize OpenAI model for evaluation
    model = OpenAIModel(
        model="gpt-4",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Extract output constraints from rails map
    rails = list(USER_FRUSTRATION_PROMPT_RAILS_MAP.values())

    try:
        # Run LLM classification
        frustration_results = llm_classify(
            dataframe=conversations_df,
            template=USER_FRUSTRATION_PROMPT_TEMPLATE,
            model=model,
            rails=rails,
            provide_explanation=True,  # Get reasoning for classifications
        )

        print("✓ Analysis complete!\n")
        return frustration_results

    except Exception as e:
        print(f"✗ Error during analysis: {e}\n")
        return None

def generate_detailed_report(conversations_df, frustration_results):
    """Generate a comprehensive report with frustration analysis insights."""

    print("\n" + "="*80)
    print("📊 USER FRUSTRATION ANALYSIS REPORT")
    print("="*80)

    # Merge results with conversation data
    if frustration_results is not None:
        full_data = pd.concat([conversations_df, frustration_results], axis=1)
    else:
        print("⚠️  No results to report")
        return

    # Overall statistics
    total = len(full_data)
    frustrated = len(full_data[full_data['label'] == 'frustrated'])
    ok = len(full_data[full_data['label'] == 'ok'])
    frustration_rate = (frustrated / total * 100) if total > 0 else 0

    print(f"\n📈 OVERALL METRICS:")
    print(f"  Total Conversations Analyzed: {total}")
    print(f"  🔴 Frustrated: {frustrated} ({frustration_rate:.1f}%)")
    print(f"  🟢 OK: {ok} ({100-frustration_rate:.1f}%)")

    # Frustrated conversations details
    if frustrated > 0:
        print(f"\n⚠️  FRUSTRATED CONVERSATIONS ({frustrated}):")
        print("-" * 80)

        frustrated_convos = full_data[full_data['label'] == 'frustrated']

        for idx, row in frustrated_convos.iterrows():
            print(f"\n🔴 Query #{idx + 1}:")
            print(f"   User Input: {row['input']}")
            print(f"   AI Response: {row['output'][:300]}...")

            if 'explanation' in row and pd.notna(row['explanation']):
                print(f"   📝 Frustration Reason: {row['explanation']}")

            print()

    # Sample of OK conversations
    if ok > 0:
        print(f"\n✅ SAMPLE OF OK CONVERSATIONS ({min(3, ok)} of {ok}):")
        print("-" * 80)

        ok_convos = full_data[full_data['label'] == 'ok'].head(3)

        for idx, row in ok_convos.iterrows():
            print(f"\n🟢 Query #{idx + 1}:")
            print(f"   User Input: {row['input']}")
            print(f"   AI Response: {row['output'][:200]}...")
            print()

    # Analysis by query type
    print("\n📊 ANALYSIS BY QUERY TYPE:")
    print("-" * 80)

    # Categorize queries
    weather_queries = full_data[full_data['input'].str.contains('weather|rain|sunny|temperature', case=False, na=False)]
    search_queries = full_data[~full_data['input'].str.contains('weather|rain|sunny|temperature', case=False, na=False)]

    if len(weather_queries) > 0:
        weather_frustrated = len(weather_queries[weather_queries['label'] == 'frustrated'])
        weather_rate = (weather_frustrated / len(weather_queries) * 100)
        print(f"  Weather Queries: {len(weather_queries)} total, {weather_frustrated} frustrated ({weather_rate:.1f}%)")

    if len(search_queries) > 0:
        search_frustrated = len(search_queries[search_queries['label'] == 'frustrated'])
        search_rate = (search_frustrated / len(search_queries) * 100)
        print(f"  Non-Weather Queries: {len(search_queries)} total, {search_frustrated} frustrated ({search_rate:.1f}%)")

    # Insights and recommendations
    print("\n💡 INSIGHTS & RECOMMENDATIONS:")
    print("-" * 80)

    if frustration_rate > 20:
        print("⚠️  HIGH FRUSTRATION RATE DETECTED (>20%)")
        print("   Recommendations:")
        print("   • Review frustrated conversations for common failure patterns")
        print("   • Improve tool selection logic in agent.py")
        print("   • Enhance error handling for edge cases")
        print("   • Consider adding clarifying questions for ambiguous inputs")
    elif frustration_rate > 10:
        print("⚡ MODERATE FRUSTRATION DETECTED (10-20%)")
        print("   Recommendations:")
        print("   • Monitor frustrated cases for recurring issues")
        print("   • Enhance system prompts for better tool usage")
        print("   • Add input validation for edge cases")
    else:
        print("✅ LOW FRUSTRATION RATE (<10%) - Excellent performance!")
        print("   Recommendations:")
        print("   • Continue monitoring for quality assurance")
        print("   • Use successful patterns as benchmarks")
        print("   • Consider expanding to more complex queries")

    # Error handling insights
    error_conversations = full_data[full_data['status'] == 'error']
    if len(error_conversations) > 0:
        print(f"\n⚠️  {len(error_conversations)} queries resulted in errors - investigate:")
        for idx, row in error_conversations.iterrows():
            print(f"   • {row['input']}")

    print("\n" + "="*80)

    # Save detailed results
    output_file = f"frustration_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    full_data.to_csv(output_file, index=False)
    print(f"\n💾 Detailed results saved to: {output_file}")

    return full_data

def main():
    """Main execution flow."""
    print("="*80)
    print("🔍 AI TRAVEL COMPANION - USER FRUSTRATION ANALYSIS")
    print("="*80)
    print("\nThis script will:")
    print("1. Run test queries against your local API")
    print("2. Analyze responses for user frustration using GPT-4")
    print("3. Generate a detailed report with insights\n")

    # Check if API is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ API is not responding. Start it with:")
            print("   poetry run uvicorn api:app --reload")
            return
    except:
        print("❌ Cannot connect to API at http://localhost:8000")
        print("   Start it with: poetry run uvicorn api:app --reload")
        return

    print("✓ API is running\n")

    # Step 1: Collect conversations
    conversations_df = collect_conversations()

    # Step 2: Run frustration analysis
    frustration_results = run_frustration_analysis(conversations_df)

    if frustration_results is None:
        print("❌ Analysis failed. Check your OPENAI_API_KEY in .env")
        return

    # Step 3: Generate report
    generate_detailed_report(conversations_df, frustration_results)

    print("\n✅ Analysis complete!")
    print("\n💡 View traces in Phoenix UI:")
    print("   http://localhost:6006 (if Phoenix server is running)")

if __name__ == "__main__":
    main()
