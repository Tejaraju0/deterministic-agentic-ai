import json
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
from typing import Dict, Any, Literal

load_dotenv()

client = genai.Client()

# Strict Pydantic Schemas for Tool Inputs

class StockPriceInput(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol, e.g., AAPL, NVDA")

class CurrencyConversionInput(BaseModel):
    amount: float = Field(..., gt=0, description="Numerical amount to convert, must be positive")
    from_currency: Literal["USD", "GBP", "EUR"] = Field(..., description="Source 3-letter currency code")
    to_currency: Literal["USD", "GBP", "EUR"] = Field(..., description="Target 3-letter currency code")

# Tool Implementations with Defensive Boundaries
def get_stock_price(raw_args: Dict[str, Any]) -> str:
    """Validates input via Pydantic before accessing data."""
    try:
        validated = StockPriceInput.model_validate(raw_args)
    except ValidationError as e:
        return json.dumps({"status": "error", "message": f"Invalid arguments {e.error()}"})

    """Mock database for demo purposes."""
    mock_db = {
        "AAPL":220.50,
        "GOOGL": 180.25,
        "NVDA": 135.00
    }

    ticker_clean = validated.ticker.upper().strip()
    if ticker_clean in mock_db:
        return json.dumps({"status": "success", "ticker":ticker_clean, "price_usd":mock_db[ticker_clean]})
    return json.dumps({"status": "error", "message": f"Ticker '{ticker_clean}' not found in database"})

#Calculates the currency exchange
def calculate_fx(raw_args: Dict[str, Any]) -> str:
    try:
        validated = CurrencyConversionInput.model_validate(raw_args)
    except ValidationError as e:
        return json.dumps({"status": "error", "message": f"Validation failure {e.error()}"})
    
    """Computes basic currency exchange."""
    rates_to_usd = {
        "USD": 1.00,
        "GBP": 1.28,
        "EUR": 1.09
    }

    if validated.from_currency not in rates_to_usd and validated.to_currency not in rates_to_usd:
        return json.dumps({"error: unsupported_currency"})

    amount_in_usd = validated.amount * rates_to_usd[validated.from_currency]
    converted_amount = amount_in_usd / rates_to_usd[validated.to_currency]

    return json.dumps({
        "status": "success",
        "original_amount": validated.amount,
        "from": validated.from_currency,
        "to": validated.to_currency,
        "converted_value": round(converted_amount, 2)
    })

# Mapping dispatch table: String name -> Callable Python function
TOOL_DISPATCH = {
    "get_stock_price": get_stock_price,
    "calculate_fx": calculate_fx
}

#Expose Tool Declarations to Gemini
agent_tools = [
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

# Production Structured Termination Contract
class AgentFinalAnswer(BaseModel):
    summary: str = Field(description="Direct synthesis answering the user's objective.")
    calculation_breakdown: list[str] = Field(description="Step-by-step intermediate figures computed during execution.")
    confidence_level: Literal["HIGH", "MEDIUM", "LOW"] = Field(description="Confidence rating of the output.")

# Guarded Loop with Defensive Dispatch & Structured Termination
def run_guarded_agent(user_prompt: str, max_turns: int = 5) -> AgentFinalAnswer:
    print(f"\n[USER_GOAL]: {user_prompt}\n" + "="*20)


    # Maintain conversation state
    contents = [
        types.Content(
            role = "user",
            parts = [types.Part.from_text(text=user_prompt)]
        )
    ]

    # Guard against runaway loops
    for turn in range(1, max_turns + 1):
        print(f"\n>>> [TURN {turn}/{max_turns} invoking model....]")

        response = client.models.generate_content(
            model = "gemini-3.5-flash-lite",
            contents = contents,
            config=types.GenerateContentConfig(
                tools = agent_tools,
                temperature = 0.0,
                system_instruction = "You are a financial analysis assistant. Solve tasks step-by-step using provided tools. Only conclude when you have the final calculation."
            )
        )

        # Check if the model requested function calls
        candidate = response.candidates[0]
        contents.append(candidate.content) # Preserve turn history

        function_calls = response.function_calls

        # Branch A: The model concluded with a final answer
        if not function_calls:
            print("\n[+] Agent finished tool executions. Generating structured response...")

            synthesis_prompt = types.Content(
                role = "user",
                parts = [types.Part.from_text(
                    text = "Provide your final response conforming strictly to the requested JSON schema."
                )]
            )
            contents.append(synthesis_prompt)

            structured_response = client.models.generate_content(
                model = "gemini-3.5-flash-lite",
                contents = contents,
                config = types.GenerateContentConfig(
                    response_mime_type = "application/json",
                    response_schema = AgentFinalAnswer,
                    temperature = 0.0
                )
            )
            final_contract = AgentFinalAnswer.model_validate_json(structured_response.text)
            return final_contract

        # Branch B: The model decided to execute one or more tools

        tool_response_parts = []

        for call in function_calls:
            name = call.name
            args = dict(call.args)

            print(f"[ACTION DISPATCH]: '{name}' | ARGS: {args}")

            if name not in TOOL_DISPATCH:
                err_obs = json.dumps({"status": "error", "message": f"Unknown_tool: '{name}'"})
                tool_response_parts.append(types.Part.from_function_response(name=name, response = {"result": err_obs}))
                continue

            try:
                tool_output = TOOL_DISPATCH[name](args)
            except Exception as e:
                tool_output = json.dumps({"status": "error", "message": f"Fatal Execution Exception: {str(e)}"})

            print(f"[OBSERVATION]: {tool_output}")

            tool_response_parts.append(
                types.Part.from_function_response(
                    name = name,
                    response = {"result": tool_output} 
                )
            )

        contents.append(
            types.Content(
                role="user",
                parts = tool_response_parts
            )
        )

    raise TimeoutError("Agent exceeded max turn limit without resolution.")

if __name__ == "__main__":
    query = "I have 25 shares of NVDA. What is their total value in EUR?"
    result = run_guarded_agent(query)

    print("\n" + "="*20)
    print("PROD-GRADE AGENT RESULT (Validated Pydantic Instance):")
    print("="*20)
    print(f"Summary:     {result.summary}")
    print(f"Confidence:  {result.confidence_level}")
    print("Calculations:")
    for step in result.calculation_breakdown:
        print(f" - {step}")
        