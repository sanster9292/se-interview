"""
Analyze user frustration in AI Travel Companion conversations using Arize Phoenix evaluators.

This script:
1. Fetches conversation traces from Phoenix
2. Runs user frustration analysis using LLM-as-a-Judge
3. Generates a detailed report with insights

Usage:
    poetry run python analyze_user_frustration.py
"""

import os
import pandas as pd
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

def fetch_traces_from_phoenix():
    """
    Fetch conversation traces from Phoenix using HTTP API.
    Returns a DataFrame with conversation data.
    """
    print("📥 Fetching traces from Phoenix...")

    try:
        import requests

        # Try local Phoenix first
        phoenix_endpoint = "http://localhost:6006"

        # Get spans from Phoenix GraphQL API
        query = """
        query {
            spans(first: 100) {
                edges {
                    node {
                        spanId
                        traceId
                        name
                        spanKind
                        startTime
                        latencyMs
                        attributes
                        input
                        output
                    }
                }
            }
        }
        """

        response = requests.post(
            f"{phoenix_endpoint}/graphql",
            json={"query": query},
            timeout=10
        )

        if response.status_code != 200:
            print(f"⚠️  Could not connect to Phoenix at {phoenix_endpoint}")
            print("   Make sure Phoenix server is running: poetry run python -m phoenix.server.main serve")
            return None

        data = response.json()
        spans = data.get('data', {}).get('spans', {}).get('edges', [])

        if not spans:
            print("⚠️  No traces found. Run test queries first:")
            print("   poetry run python run_test_queries.py")
            return None

        print(f"✓ Fetched {len(spans)} spans from Phoenix")
        return spans

    except Exception as e:
        print(f"✗ Error fetching traces: {e}")
        print("   Ensure Phoenix server is running on http://localhost:6006")
        return None

def prepare_conversation_data(spans):
    """
    Transform span data into conversation format for evaluation.
    Extracts input/output pairs from LLM spans.
    """
    print("\n🔄 Preparing conversation data...")

    conversations = []

    for edge in spans:
        span = edge['node']

        # Only process LLM spans
        if span.get('spanKind') != 'LLM':
            continue

        # Extract input and output
        input_text = span.get('input', {}).get('value', '') if span.get('input') else ''
        output_text = span.get('output', {}).get('value', '') if span.get('output') else ''

        # Skip if no input/output
        if not input_text or not output_text:
            continue

        conversations.append({
            'trace_id': span.get('traceId', ''),
            'span_id': span.get('spanId', ''),
            'timestamp': span.get('startTime', ''),
            'input': str(input_text),
            'output': str(output_text),
            'latency_ms': span.get('latencyMs', 0),
        })

    if not conversations:
        print("⚠️  No valid conversations found in spans")
        return None

    df = pd.DataFrame(conversations)
    print(f"✓ Prepared {len(df)} conversation turns")
    return df

def run_frustration_analysis(conversations_df):
    """
    Run user frustration analysis using Phoenix LLM-as-a-Judge evaluator.
    """
    print("\n🤖 Running user frustration analysis with GPT-4...\n")

    # Initialize OpenAI model for evaluation
    model = OpenAIModel(
        model="gpt-4",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Extract output constraints from rails map
    rails = list(USER_FRUSTRATION_PROMPT_RAILS_MAP.values())

    # Run LLM classification
    try:
        frustration_results = llm_classify(
            dataframe=conversations_df,
            template=USER_FRUSTRATION_PROMPT_TEMPLATE,
            model=model,
            rails=rails,
            provide_explanation=True,  # Get reasoning for classifications
        )

        print("✓ Analysis complete!")
        return frustration_results

    except Exception as e:
        print(f"✗ Error during analysis: {e}")
        return None

def generate_report(conversations_df, frustration_results):
    """
    Generate a detailed report with frustration analysis insights.
    """
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
    total_conversations = len(full_data)
    frustrated_count = len(full_data[full_data['label'] == 'frustrated'])
    ok_count = len(full_data[full_data['label'] == 'ok'])
    frustration_rate = (frustrated_count / total_conversations * 100) if total_conversations > 0 else 0

    print(f"\n📈 OVERALL METRICS:")
    print(f"  Total Conversations: {total_conversations}")
    print(f"  Frustrated: {frustrated_count} ({frustration_rate:.1f}%)")
    print(f"  OK: {ok_count} ({100-frustration_rate:.1f}%)")

    # Frustrated conversations details
    if frustrated_count > 0:
        print(f"\n⚠️  FRUSTRATED CONVERSATIONS ({frustrated_count}):")
        print("-" * 80)

        frustrated_convos = full_data[full_data['label'] == 'frustrated']
        for idx, row in frustrated_convos.iterrows():
            print(f"\n🔴 Conversation {idx + 1}:")
            print(f"   Timestamp: {row['timestamp']}")
            print(f"   Input: {row['input'][:200]}...")
            print(f"   Output: {row['output'][:200]}...")
            if 'explanation' in row and pd.notna(row['explanation']):
                print(f"   Reason: {row['explanation']}")
            print()

    # OK conversations sample
    if ok_count > 0:
        print(f"\n✅ SAMPLE OK CONVERSATIONS ({min(3, ok_count)} of {ok_count}):")
        print("-" * 80)

        ok_convos = full_data[full_data['label'] == 'ok'].head(3)
        for idx, row in ok_convos.iterrows():
            print(f"\n🟢 Conversation {idx + 1}:")
            print(f"   Input: {row['input'][:150]}...")
            print(f"   Output: {row['output'][:150]}...")
            print()

    # Performance insights
    print("\n💡 INSIGHTS & RECOMMENDATIONS:")
    print("-" * 80)

    if frustration_rate > 20:
        print("⚠️  HIGH FRUSTRATION RATE DETECTED")
        print("   - Review frustrated conversations for common patterns")
        print("   - Consider improving tool accuracy or response quality")
        print("   - Add better error handling for edge cases")
    elif frustration_rate > 10:
        print("⚡ MODERATE FRUSTRATION DETECTED")
        print("   - Monitor frustrated cases for improvement opportunities")
        print("   - Enhance system prompts for clarity")
    else:
        print("✅ LOW FRUSTRATION RATE - Good performance!")
        print("   - Continue monitoring for quality assurance")

    # Average latency
    avg_latency = full_data['latency_ms'].mean()
    print(f"\n⏱️  Average Response Latency: {avg_latency:.0f}ms")

    if avg_latency > 5000:
        print("   ⚠️  High latency may contribute to user frustration")

    print("\n" + "="*80)

    # Save detailed results to CSV
    output_file = f"frustration_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    full_data.to_csv(output_file, index=False)
    print(f"\n💾 Detailed results saved to: {output_file}")

def main():
    """
    Main execution flow for user frustration analysis.
    """
    print("🔍 AI Travel Companion - User Frustration Analysis")
    print("=" * 80)

    # Step 1: Fetch traces from Phoenix
    spans = fetch_traces_from_phoenix()
    if spans is None:
        print("\n❌ Failed to fetch traces. Exiting.")
        return

    # Step 2: Prepare conversation data
    conversations_df = prepare_conversation_data(spans)
    if conversations_df is None or conversations_df.empty:
        print("\n❌ No conversation data available. Exiting.")
        return

    # Step 3: Run frustration analysis
    frustration_results = run_frustration_analysis(conversations_df)
    if frustration_results is None:
        print("\n❌ Analysis failed. Exiting.")
        return

    # Step 4: Generate report
    generate_report(conversations_df, frustration_results)

    print("\n✅ Analysis complete! Check Phoenix UI for detailed trace exploration:")
    print("   Local: http://localhost:6006")
    print("   Cloud: https://app.phoenix.arize.com")

if __name__ == "__main__":
    main()
