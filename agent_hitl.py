import json, re
import os
from typing import Any, Literal, Dict
from google import genai
from google.genai import types
import psycopg
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

client = genai.Client()

def list_tables(raw_args: Dict[str, Any]) -> str:
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public';
                """)
                return json.dumps({"tables": r[0] for r in cur.fetchall()})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})

def describe_table(raw_args: Dict[str, Any]) -> str:
    table_name = raw_args.get("table_name", "").strip().lower()

    if not re.match(r"^[a-zA-Z0-9_]+$", table_name):
        return json.dumps({"status": "error", "message": "Invalid table name syntax."})

    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_name = %s AND table_schema = 'public';
                """, (table_name,))
                columns = [{"column": r[0], "type": r[1]} for r in cur.fetchall()]
                if not columns:
                    return json.dumps({"status": "error", "message": f"Table '{table_name}' does not exist."})
                return json.dumps({"table": table_name, "columns": columns})
    except Exception as e:
            return json.dumps({"status":"error", "message": str(e)})

def execute_readonly_sql(raw_args: Dict[str, Any]) -> str:
    query = raw_args.get("query", "").strip()
            
    forbidden = ["DROP", "DELETE", "TRUNCATE", "ALTER", "UPDATE", "INSERT", "GRANT"]
    tokens = re.findall(r"\b\w+\b", query.upper())
    for token in tokens:
        if token in forbidden:
            return json.dumps({"status": "security_violation", "message": f"Query rejected: Destructive keyword '{token}' is forbidden."})

    if not query.upper().startswith("SELECT"):
        return json.dumps({
            "status": "security_violation", 
            "message": "Only queries beginning with SELECT are permitted."
        })

    # 2. Execution with syntax error trapping
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                colnames = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
                sample_rows = rows[:25]
                return json.dumps({
                    "status": "success",
                    "columns": colnames,
                    "row_count": len(rows),
                    "rows": [[str(val) for val in r] for r in sample_rows]
                })
    except psycopg.Error as e:
        # Pass syntax error directly to LLM to trigger self-correction
        return json.dumps({"status": "sql_syntax_error", "message": str(e).strip()})

def flag_transaction_for_audit(raw_args: Dict[str, Any]) -> str:
    transaction_id = raw_args.get("transaction_id")
    reason = raw_args.get("reason", "No reason provided.")

    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT department, vendor, amount_usd, is_flagged_audit
                    FROM corporate_transactions
                    WHERE transaction_id = %s;
                """, (transaction_id, ))
                row = cur.fetchone()

                if not row:
                    return json.dumps({"status": "error", "message": f"Transaction #{transaction_id} not found."})

                dept, vendor, amount, is_flagged = row
                if is_flagged:
                    return json.dumps({"status": "error", "message": f"Transaction #{transaction_id} is already marked as flagged."})

                cur.execute("""
                    UPDATE corporate_transactions
                    SET is_flagged_audit = TRUE
                    WHERE transaction_id = %s;
                """, (transaction_id,))
                conn.commit()

                return json.dumps({
                    "status": "mutation_success",
                    "transaction_id": transaction_id,
                    "department": dept,
                    "vendor": vendor,
                    "amount_usd": str(amount),
                    "audit_flag": True,
                    "audit_reason": reason
                })
    except psycopg.Error as e:
        return json.dumps({"status": "db_error", "message": str(e)})

# Configuration: Identify which tools require human confirmation
SENSITIVE_TOOLS = {"flag_transaction_for_audit"}

TOOL_DISPATCH = {
    "list_tables": list_tables,
    "describe_table": describe_table,
    "execute_readonly_sql": execute_readonly_sql,
    "flag_transaction_for_audit": flag_transaction_for_audit
}

hitl_tools = [
    types.Tool(
        function_declarations = [
            types.FunctionDeclaration(
                name="list_tables",
                description="List all public tables available in the database.",
                parameters=types.Schema(type="OBJECT", properties={})
            ),
            types.FunctionDeclaration(
                name="describe_table",
                description="Fetch columns and types for a specific table to understand its schema.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "table_name": types.Schema(type="STRING", description="Target Table name.") 
                    },
                    required=["table_name"]
                )
            ),
            types.FunctionDeclaration(
                name="execute_readonly_sql",
                description="Runs a read-only PostgreSQL SELECT query and returns the rows as JSON.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "query": types.Schema(type="STRING", description="Valid SQL SELECT query")
                    },
                    required=["query"]
                )
            ),
            types.FunctionDeclaration(
                name="flag_transaction_for_audit",
                description="Marks a specific transaction ID as flagged for compliance auditing.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "transaction_id": types.Schema(type="INTEGER", description="The numeric ID of the transaction to flag"),
                        "reason": types.Schema(type="STRING", description="Justification explaining why the transaction is suspicious")
                    },
                    required=["transaction_id", "reason"]
                )
            )
        ]
    )
]

class HITLFinalSummary(BaseModel):
    user_request: str = Field(description="The original user request.")
    investigation_summary: str = Field(description="Detailed overview of what the agent found.")
    actions_taken: list[str] = Field(description="List of actions and modifications completed or declined.")
    status: Literal["COMPLETED", "PARTIAL", "ABORTED"] = Field(description="Overall execution status.")

def run_hitl_agent(user_prompt: str, max_turns: int = 8) -> HITLFinalSummary:
    print(f"\n[USER GOAL]: {user_prompt}\n" + "="*70)
    
    contents = [
        types.Content(
            role="user", 
            parts=[types.Part.from_text(text=user_prompt)]
        )
    ]
    
    for turn in range(1, max_turns + 1):
        print(f"\n>>> TURN {turn}/{max_turns}: Invoking Model...")
        
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=contents,
            config=types.GenerateContentConfig(
                tools=hitl_tools,
                temperature=0.0,
                system_instruction=(
                    "You are a Corporate Financial Compliance Agent. When asked to audit or flag "
                    "transactions, first inspect the relevant table and run read-only queries to identify "
                    "the exact transaction_id. Once identified, propose flagging the transaction using "
                    "the flag_transaction_for_audit tool with a clear reason."
                )
            )
        )
        
        candidate = response.candidates[0]
        contents.append(candidate.content)
        function_calls = response.function_calls
        
        if not function_calls:
            print("\n[+] Agent concluded. Synthesizing final structured summary...")
            contents.append(types.Content(
                role="user",
                parts=[types.Part.from_text(text="Synthesize your final summary conforming to the HITLFinalSummary schema.")]
            ))
            
            synthesis = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=HITLFinalSummary,
                    temperature=0.0
                )
            )
            return HITLFinalSummary.model_validate_json(synthesis.text)
            
        observation_parts = []
        for call in function_calls:
            name = call.name
            args = dict(call.args)
            print(f"[ACTION PROPOSED]: {name}({args})")
            
            # --- HUMAN IN THE LOOP INTERCEPT GUARD ---
            if name in SENSITIVE_TOOLS:
                print("\n" + "!"*30)
                print(f"[!] HITL INTERCEPT: Agent requested SENSITIVE MUTATION: '{name}'")
                print(f"[!] Action Payload: {json.dumps(args, indent=2)}")
                print("!"*30)
                
                # Interactive Operator Authorization Prompt
                decision = input("Authorise this action? (Type 'yes' or 'y' to approve, anything else to deny): ").strip().lower()
                
                if decision not in ("yes", "y"):
                    print("[X] ACTION REJECTED BY OPERATOR. Halting mutation.")
                    tool_output = json.dumps({
                        "status": "permission_denied",
                        "message": "The human operator explicitly DENIED authorization for this mutation."
                    })
                    observation_parts.append(types.Part.from_function_response(name=name, response={"result": tool_output}))
                    continue
                else:
                    print("[/] ACTION APPROVED BY OPERATOR. Proceeding with execution...")
            # ----------------------------------------
            
            if name in TOOL_DISPATCH:
                tool_output = TOOL_DISPATCH[name](args)
            else:
                tool_output = json.dumps({"status": "error", "message": f"Tool '{name}' not found."})
                
            print(f"[OBSERVATION]: {tool_output}")
            observation_parts.append(types.Part.from_function_response(name=name, response={"result": tool_output}))
            
        contents.append(types.Content(role="user", parts=observation_parts))
        
    raise TimeoutError("Agent exceeded turn limit.")

if __name__ == "__main__":
    test_prompt = (
        "Find the highest unflagged transaction in the Engineering department. "
        "If it is over $10,000, flag it for audit due to high infrastructure cost."
    )
    result = run_hitl_agent(test_prompt)
    
    print("\n" + "~"*30)
    print("FINAL AUDIT SUMMARY (Pydantic Validated):")
    print("~"*30)
    print(f"Status:       {result.status}")
    print(f"Summary:      {result.investigation_summary}")
    print("Actions:")
    for a in result.actions_taken:
        print(f" - {a}")