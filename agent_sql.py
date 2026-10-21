import json
import re, os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from typing import Dict, Any
import psycopg
from pydantic import BaseModel, Field

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
                tables = [r[0] for r in cur.fetchall()]
                return json.dumps({"tables": tables})
    except Exception as e:
        return json.dumps({"status":"error", "message": str(e)})

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

TOOL_DISPATCH = {
    "list_tables" : list_tables,
    "describe_table" : describe_table,
    "execute_readonly_sql" : execute_readonly_sql
}

db_tools = [
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
            )
        ]
    )
]

# 3. Pydantic Response Contract

class DatabaseAuditReport(BaseModel):
    query_objective: str = Field(description="The user's original objective.")
    executed_sql: str = Field(description="The exact SQL statement that retrieved the answer.")
    findings: str = Field(description="Concise analytical explanation of the query findings.")
    numeric_result: float = Field(description="Primary aggregated number or count discovered.")
    audit_flag_present: bool = Field(description="True if flagged/risky records were detected.")

# 4. The Agentic Loop

def run_sql_agent(user_prompt: str, max_turns: int = 6) -> DatabaseAuditReport:
    print(f"\n[USER GOAL]: {user_prompt}\n" + "~"*30)

    contents = [
        types.Content(
            role = "user",
            parts = [types.Part.from_text(text=user_prompt)]
        )
    ]

    for turn in range(1, max_turns + 1):
        print(f"\n>>> TURN {turn}/{max_turns}: Calling Model...")

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents = contents,
            config = types.GenerateContentConfig(
                tools = db_tools,
                temperature= 0.0,
                system_instruction=(
                    "You are an expert PostgreSQL database auditor. Always inspect schemas using "
                    "describe_table before writing queries. Use execute_readonly_sql to verify data. "
                    "If you receive an SQL error, examine the message and rewrite your query."
                )
            )
        )

        candidate = response.candidates[0]
        contents.append(candidate.content)
        function_calls = response.function_calls

        if not function_calls:
            print("\n[+] Agent reached conclusion. Generating structured audit contract...")
            contents.append(types.Content(
                role = "user",
                parts = [types.Part.from_text(
                    text="Return your final findings conforming strictly to the DatabaseAuditReport schema."
                )]
            ))

            synthesis = client.models.generate_content(
                model= "gemini-3.5-flash-lite",
                contents = contents,
                config = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema = DatabaseAuditReport,
                    temperature= 0.0
                )
            )

            return DatabaseAuditReport.model_validate_json(synthesis.text)

        # Execute dispatched tool calls
        observation_parts = []
        for call in function_calls:
            name = call.name
            args = dict(call.args)
            print(f"[ACTION]: {name}({args})")

            if name in TOOL_DISPATCH:
                tool_output = TOOL_DISPATCH[name](args)
            else:
                return json.dumps({"status": "error", "message": f"Tool '{name}' not found."})

            print(f"[OBSERVATION]: {tool_output}")
            observation_parts.append(
                types.Part.from_function_response(name=name, response={"result": tool_output})
            )

        contents.append(types.Content(role="user", parts=observation_parts))

    raise TimeoutError("Agent exceeded turn limit.")

if __name__ == "__main__":
    prompt = "What is the total expenditure in USD on flagged audit transactions for the Legal department?"
    report = run_sql_agent(prompt)

    print("\n" + "~"*30)
    print("STRUCTURED AUDIT REPORT (Pydantic Validated):")
    print("~"*30)
    print(f"Objective:      {report.query_objective}")
    print(f"Executed SQL:   {report.executed_sql}")
    print(f"Numeric Total:  ${report.numeric_result:,.2f}")
    print(f"Flags Present:  {report.audit_flag_present}")
    print(f"Findings:       {report.findings}")