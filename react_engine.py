import json
import os
from google import genai
from google.genai import types
from pydantic import BaseModel, Field 
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()

#Function for fetching the stock price.
def get_stock_price(ticker: str) -> str:

    """Mock database for demo purposes."""
    mock_db = {
        "AAPL":220.50,
        "GOOGL": 180.25,
        "NVDA": 135.00
    }

    ticker_clean = ticker.upper().strip()
    if ticker_clean in mock_db:
        return json.dumps({"ticker":ticker_clean, "price_usd":mock_db[ticker_clean]})
    return json.dumps({"error": f"Ticker: {ticker_clean} not found in database"})

#Calculates the currency exchange
def calculate_fx(amount: float, from_currency: str, to_currency: str) -> str:
    """Computes basic currency exchange."""
    rates_to_usd = {
        "USD": 1.00,
        "GBP": 1.28,
        "EUR": 1.09
    }

    from_curr = from_currency.upper().strip()
    to_curr = to_currency.upper().strip()

    if from_curr not in rates_to_usd and to_curr not in rates_to_usd:
        return json.dumps({"error: unsupported_currency"})

    amount_in_usd = amount * rates_to_usd[from_curr]
    converted_amount = amount_in_usd / rates_to_usd[to_curr]

    return json.dumps({
        "original_amount": amount,
        "from": from_curr,
        "to": to_curr,
        "converted_value": round(converted_amount, 2)
    })

# Mapping dispatch table: String name -> Callable Python function
TOOL_REGISTRY = {
    "get_stock_price": get_stock_price,
    "calculate_fx": calculate_fx
}

#Expose Tool Declarations to Gemini
tools_definition = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="get_stock_price",
                description="Retrieves the current share price in USD for a given stock ticker.",
                parameters= types.Schema(
                    type="OBJECT",
                    properties = { "ticker" : types.Schema(type="STRING", description="Stock ticker symber (e.g. AAPL, NVDA)")},
                    required = ["ticker"]
                )
            ),
            types.FunctionDeclaration(
                name="calculate_fx",
                description="Converts currency amounts between USD, GBP, and EUR.",
                parameters= types.Schema(
                    type="OBJECT",
                    properties = {
                        "amount" : types.Schema(type="NUMBER", description="The numerical monetary number."),
                        "from_currency" : types.Schema(type="STRING", description="Origin currency code (USD, GBP, EUR)"),
                        "to_currency" : types.Schema(type="STRING" , description="Target currency code (USD, GBP, EUR)")
                    },
                    required = ["amount", "from_currency", "to_currency"]
                )
            )
        ]
    )
]


#The Guarded ReAct Orchestration Loop
def react_run_agent(user_prompt: str, max_turns: int = 5):
    print(f"\n[USER_GOAL]: {user_prompt}\n" + "-"*10)

    # Maintain conversation state
    contents = [
        types.Content(
            role = "user",
            parts = [types.Part.from_text(text=user_prompt)]
        )
    ]

    # Guard against runaway loops
    for turn in range(1, max_turns + 1):
        print(f"\n-- [TURN {turn}/{max_turns} invoking model --]")

        response = client.models.generate_content(
            model = "gemini-2.5-flash",
            contents = contents,
            config=types.GenerateContentConfig(
                tools = tools_definition,
                temperature = 0.0,
                system_instruction = "You are a financial analysis assistant. Solve tasks step-by-step using provided tools. Only conclude when you have the final calculation."
            )
        )

        # Check if the model requested function calls
        candidate = response.candidates[0]
        content = candidate.content
        contents.append(content) # Preserve turn history

        function_calls = response.function_calls

        # Branch A: The model concluded with a final answer
        if not function_calls:
            print("\n[+] AGENT REACHED TERMINATION (No further tool calls needed).")
            print(f"[FINAL ANSWER]:\n{response.text}")
            return response.text

        # Branch B: The model decided to execute one or more tools

        tool_response_parts = []

        for call in function_calls:
            tool_name = call.name
            tool_args = call.args

            print(f"[ACTION]: Calling '{tool_name}' with args: {tool_args}")

            if tool_name not in TOOL_REGISTRY:
                tool_result_str = json.dumps({"error": f"Unknown_tool: '{tool_name}'"})
            else:
                tool_func = TOOL_REGISTRY[tool_name]
                tool_result_str = tool_func(**tool_args)

            print(f"OBSERVATION: {tool_result_str}")

            tool_response_parts.append(
                types.Part.from_function_response(
                    name = tool_name,
                    response = {"result": tool_result_str} 
                )
            )

        contents.append(
            types.Content(
                role="user",
                parts = tool_response_parts
            )
        )

    print("\n[!] Loop Guard Triggered: Agent exceeded maximum turn threshold.")
    return None

if __name__ == "__main__":
    task = "I have 15 shares of AAPL. How much are they worth in GBP if 1 GBP is 1.28 USD?"
    react_run_agent(task)
        