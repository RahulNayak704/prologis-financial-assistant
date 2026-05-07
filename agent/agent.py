"""
Gemini-powered agent with function calling. Routes natural-language questions
to one of three data-source tools (query_postgres, query_sec_edgar,
query_press_releases) and synthesizes a natural-language answer.

The agent pattern (tool declaration, function calling, multi-turn orchestration)
is identical to Vertex AI ADK; this implementation uses google-generativeai
(AI Studio) for free-tier development. Code is portable to Vertex AI by
swapping the client initialization.
"""
import json
import os
from typing import Optional

import google.generativeai as genai
from dotenv import load_dotenv

from agent.tools import query_postgres, query_sec_edgar, query_press_releases
from agent.bedrock import summarize_with_bedrock

load_dotenv()

# Configure Gemini client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in .env")
genai.configure(api_key=GEMINI_API_KEY)

# Map tool names → callables
TOOL_FUNCTIONS = {
    "query_postgres": query_postgres,
    "query_sec_edgar": query_sec_edgar,
    "query_press_releases": query_press_releases,
    "summarize_with_bedrock": summarize_with_bedrock,
}

SYSTEM_INSTRUCTION = """You are a Financial Assistant for Prologis, a real estate
investment trust focused on industrial logistics properties.

You have access to four tools:
1. query_postgres - look up properties + financials from a database
   (filter by metro_area, property_type, or min_revenue).
2. query_sec_edgar - look up Prologis financial metrics (revenue, net_income,
   operating_expenses, total_assets, total_liabilities) from SEC filings.
3. query_press_releases - search recent Prologis press releases by keywords
   or category (earnings, acquisition, expansion, sustainability).
4. summarize_with_bedrock - condense long text using AWS Bedrock (Claude Haiku).
   Use this for press release summaries when the user wants brief output.

Decide which tool(s) to call based on the user's question. Call multiple tools
if the question spans multiple sources. After receiving tool results, write a
clear natural-language answer with concrete numbers and facts. Use $ formatting
for dollar values. Keep answers to 2-4 sentences unless the user wants detail.
"""


def get_model():
    """Build a Gemini model configured with our tools."""
    return genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=SYSTEM_INSTRUCTION,
        tools=[
            query_postgres,
            query_sec_edgar,
            query_press_releases,
            summarize_with_bedrock,
        ],
    )


def run_agent(user_query: str, verbose: bool = False) -> dict:
    """
    Run the agent on a single user query. Handles multi-turn function calling.

    Returns:
        {
            "answer": "natural language answer",
            "tool_calls": [{"name": ..., "args": ..., "result": ...}, ...],
        }
    """
    model = get_model()
    chat = model.start_chat(enable_automatic_function_calling=False)

    tool_calls_log = []
    response = chat.send_message(user_query)

    # Loop: while model wants to call a tool, execute it and feed back
    max_turns = 6
    for turn in range(max_turns):
        # Check if any part of the response is a function call
        fn_calls = []
        for part in response.parts:
            if hasattr(part, "function_call") and part.function_call.name:
                fn_calls.append(part.function_call)

        if not fn_calls:
            break  # Plain text response — done

        # Execute each requested function
        function_responses = []
        for fc in fn_calls:
            name = fc.name
            args = {k: v for k, v in fc.args.items()} if fc.args else {}
            if verbose:
                print(f"[agent] calling {name}({args})")
            try:
                result = TOOL_FUNCTIONS[name](**args)
            except Exception as e:
                result = {"error": str(e)}
            tool_calls_log.append({"name": name, "args": args, "result": result})
            function_responses.append(
                genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=name,
                        response={"result": result},
                    )
                )
            )

        # Send all function responses back to the model
        response = chat.send_message(function_responses)

    # Pull final text out of the last response
    answer_parts = []
    for part in response.parts:
        if hasattr(part, "text") and part.text:
            answer_parts.append(part.text)
    answer = "\n".join(answer_parts).strip() or "(no response)"

    return {"answer": answer, "tool_calls": tool_calls_log}


if __name__ == "__main__":
    # Quick sanity tests
    queries = [
        "What was Prologis' net income last year?",
        "Show me industrial properties in Chicago with their revenue.",
        "Did Prologis announce any acquisitions recently?",
    ]
    for q in queries:
        print(f"\n{'=' * 70}\nQ: {q}")
        result = run_agent(q, verbose=True)
        print(f"\nA: {result['answer']}")
        print(f"\nTools called: {[c['name'] for c in result['tool_calls']]}")
