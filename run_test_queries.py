"""
Run test queries against the AI Travel Companion API to generate Phoenix traces.
"""
import requests
import time

BASE_URL = "http://localhost:8000"

# Test queries covering different scenarios
test_queries = [
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

def send_query(query: str) -> dict:
    """Send a query to the chat API and return the response."""
    try:
        response = requests.post(
            f"{BASE_URL}/chat",
            json={"message": query},
            timeout=30
        )
        response.raise_for_status()
        return {
            "query": query,
            "response": response.json()["response"],
            "status": "success"
        }
    except Exception as e:
        return {
            "query": query,
            "error": str(e),
            "status": "error"
        }

def main():
    """Run all test queries and display results."""
    print("🚀 Running test queries against AI Travel Companion API\n")
    print(f"Total queries: {len(test_queries)}\n")
    print("=" * 80)

    results = []

    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}/{len(test_queries)}] Query: {query}")
        print("-" * 80)

        result = send_query(query)
        results.append(result)

        if result["status"] == "success":
            print(f"✓ Response: {result['response'][:200]}...")
        else:
            print(f"✗ Error: {result['error']}")

        # Small delay between requests
        time.sleep(1)

    print("\n" + "=" * 80)
    print("\n📊 Summary:")
    print(f"Successful queries: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"Failed queries: {sum(1 for r in results if r['status'] == 'error')}")
    print("\n✅ Test queries completed!")
    print("\n🔍 Check Phoenix UI at https://app.phoenix.arize.com to view traces")

if __name__ == "__main__":
    main()
