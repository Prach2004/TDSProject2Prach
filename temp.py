import os
import re
import json
import base64
import tempfile
import subprocess
import logging
import sys
from io import BytesIO, StringIO
from typing import Dict, Any, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Optional: Pillow for image processing
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# LangChain-related imports (for AI agent functionality)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

# Load environment variables from .env file
load_dotenv()

# Configure logging to console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# FastAPI application setup
# -------------------------------------------------
app = FastAPI(title="TDS Data Analyst Agent")

# Enable CORS for all origins (⚠ in production, restrict this!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Serve the frontend HTML
# -------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve index.html if available; otherwise return a 404 HTML message."""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Frontend not found</h1><p>Please ensure index.html is in the same directory.</p>",
            status_code=404
        )

# Timeout for LLM operations
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", 180))

# -------------------------------------------------
# Tool: scrape_url_to_dataframe
# -------------------------------------------------
@tool
def scrape_url_to_dataframe(url: str) -> Dict[str, Any]:
    """
    Fetches data from a URL and returns it as a Pandas DataFrame.
    Supports CSV, Excel, Parquet, JSON, HTML tables, or plain text.
    Always returns a dict with 'status', 'data', and 'columns'.
    """
    print(f"Scraping URL: {url}")
    try:
        from bs4 import BeautifulSoup

        # Pretend to be a browser to avoid blocks
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/138.0.0.0 Safari/537.36",
            "Referer": "https://www.google.com/",
        }

        # Fetch the URL
        resp = requests.get(url, headers=headers, timeout=50)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "").lower()

        df = None

        # Detect content type and parse accordingly
        if "text/csv" in content_type or url.lower().endswith(".csv"):
            df = pd.read_csv(BytesIO(resp.content))
        elif any(url.lower().endswith(ext) for ext in (".xls", ".xlsx")) or "spreadsheetml" in content_type:
            df = pd.read_excel(BytesIO(resp.content))
        elif url.lower().endswith(".parquet"):
            df = pd.read_parquet(BytesIO(resp.content))
        elif "application/json" in content_type or url.lower().endswith(".json"):
            try:
                df = pd.json_normalize(resp.json())
            except Exception:
                df = pd.DataFrame([{"text": resp.text}])
        elif "text/html" in content_type:
            # Try extracting tables first
            try:
                tables = pd.read_html(StringIO(resp.text), flavor="bs4")
                if tables:
                    df = tables[0]
            except ValueError:
                pass
            # Fallback: extract plain text
            if df is None:
                soup = BeautifulSoup(resp.text, "html.parser")
                text = soup.get_text(separator="\n", strip=True)
                df = pd.DataFrame({"text": [text]})
        else:
            # Unknown type — store as text
            df = pd.DataFrame({"text": [resp.text]})

        # Clean column names
        df.columns = df.columns.map(str).str.replace(r'\[.*\]', '', regex=True).str.strip()

        return {"status": "success", "data": df.to_dict(orient="records"), "columns": df.columns.tolist()}

    except Exception as e:
        return {"status": "error", "message": str(e)}

# -------------------------------------------------
# Utility: clean LLM output JSON safely
# -------------------------------------------------
def clean_llm_output(output: str) -> Dict:
    """
    Attempts to extract a valid JSON object from LLM output, even if extra text is present.
    Returns dict or {'error': ...}.
    """
    try:
        if not output:
            return {"error": "Empty LLM output"}

        # Remove optional code block formatting from LLM
        s = re.sub(r"^```(?:json)?\s*", "", output.strip())
        s = re.sub(r"\s*```$", "", s)

        # Find outermost JSON braces
        first, last = s.find("{"), s.rfind("}")
        if first == -1 or last == -1 or last <= first:
            return {"error": "No JSON object found", "raw": s}

        candidate = s[first:last+1]
        try:
            return json.loads(candidate)
        except Exception as e:
            # Try progressively shorter substrings from the end
            for i in range(last, first, -1):
                try:
                    return json.loads(s[first:i+1])
                except Exception:
                    continue
            return {"error": f"JSON parsing failed: {str(e)}", "raw": candidate}
    except Exception as e:
        return {"error": str(e)}

# -------------------------------------------------
# Run Python code in a temp file (sandbox-like)
# -------------------------------------------------
def write_and_run_temp_python(code: str, injected_pickle: str = None, timeout: int = 60) -> Dict[str, Any]:
    """
    Creates a temporary Python file with:
    - Required imports
    - Optional injected DataFrame (via pickle)
    - Helper function for converting matplotlib plots to base64 under 100KB
    Executes the code and returns parsed JSON output or error details.
    """
    # Boilerplate imports for sandboxed execution
    preamble = [
        "import json, sys, gc",
        "import pandas as pd, numpy as np",
        "import matplotlib; matplotlib.use('Agg')",
        "import matplotlib.pyplot as plt",
        "from io import BytesIO",
        "import base64",
    ]
    if PIL_AVAILABLE:
        preamble.append("from PIL import Image")

    # Inject DataFrame from pickle if provided
    if injected_pickle:
        preamble.append(f"df = pd.read_pickle(r'''{injected_pickle}''')")
        preamble.append("data = df.to_dict(orient='records')")
    else:
        preamble.append("data = globals().get('data', {})")

    # Helper function for plot compression
    helper_func = r'''
def plot_to_base64(max_bytes=100000):
    """Save matplotlib plot to base64, reducing size if needed."""
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_bytes = buf.getvalue()
    if len(img_bytes) <= max_bytes:
        return base64.b64encode(img_bytes).decode('ascii')
    for dpi in [80, 60, 50, 40, 30]:
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
        buf.seek(0)
        b = buf.getvalue()
        if len(b) <= max_bytes:
            return base64.b64encode(b).decode('ascii')
    try:
        from PIL import Image
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=40)
        buf.seek(0)
        im = Image.open(buf)
        out_buf = BytesIO()
        im.save(out_buf, format='WEBP', quality=80, method=6)
        out_buf.seek(0)
        if len(out_buf.getvalue()) <= max_bytes:
            return base64.b64encode(out_buf.getvalue()).decode('ascii')
    except Exception:
        pass
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=20)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('ascii')
'''

    # Combine all script components
    script_lines = preamble + [helper_func, "\nresults = {}\n", code,
                               "\nprint(json.dumps({'status':'success','result':results}, default=str), flush=True)\n"]

    # Write to temp file
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8')
    tmp.write("\n".join(script_lines))
    tmp.flush()
    tmp_path = tmp.name
    tmp.close()

    try:
        completed = subprocess.run([sys.executable, tmp_path], capture_output=True, text=True, timeout=timeout)
        if completed.returncode != 0:
            return {"status": "error", "message": completed.stderr.strip() or completed.stdout.strip()}

        # Parse execution output as JSON
        out = completed.stdout.strip()
        try:
            return json.loads(out)
        except Exception as e:
            return {"status": "error", "message": f"Could not parse JSON: {str(e)}", "raw": out}
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Execution timed out"}
    finally:
        # Cleanup temp files
        try:
            os.unlink(tmp_path)
            if injected_pickle and os.path.exists(injected_pickle):
                os.unlink(injected_pickle)
        except Exception:
            pass

# -------------------------------------------------
# LLM Agent Setup
# -------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model=os.getenv("GOOGLE_MODEL", "gemini-2.5-pro"),
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Register available tools for the agent
tools = [scrape_url_to_dataframe]

# Prompt template to instruct the agent
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a full-stack autonomous data analyst agent...
    (rules omitted here for brevity)"""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create the LangChain agent with scraping capability
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

# Executor for running agent steps
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,
                               max_iterations=3, early_stopping_method="generate",
                               handle_parsing_errors=True, return_intermediate_steps=False)

# -------------------------------------------------
# API Endpoint: POST /api
# -------------------------------------------------
@app.post("/api")
async def analyze_data(request: Request):
    """
    Main endpoint to analyze uploaded datasets + question file using LLM agent.
    """
    try:
        form = await request.form()

        # Extract all uploaded files
        uploads = [v for _, v in form.multi_items() if isinstance(v, UploadFile)]
        if not uploads:
            raise HTTPException(400, "Upload at least one file (.txt questions file required).")

        # Identify question file (must be exactly one .txt)
        txt_files = [f for f in uploads if (f.filename or "").lower().endswith(".txt")]
        if len(txt_files) != 1:
            raise HTTPException(400, "Exactly one .txt questions file is required.")
        questions_file = txt_files[0]
        raw_questions = (await questions_file.read()).decode("utf-8")

        # Map question keys to expected types
        type_map = {}
        patterns = [
            re.compile(r"^\s*-\s*`([^`]+)`\s*:\s*([a-zA-Z ]+)", re.MULTILINE),
            re.compile(r"^\s*-\s*([a-zA-Z0-9_]+)\s*:\s*([a-zA-Z ]+)", re.MULTILINE),
            re.compile(r"`([^`]+)`\s*\((number|string|boolean|base64)[s]?\)", re.IGNORECASE),
        ]
        for pat in patterns:
            for key, type_hint in pat.findall(raw_questions):
                t = type_hint.strip().lower()
                if t in ("number", "float", "int"): t = "number"
                elif t in ("string", "str"): t = "string"
                elif "base64" in t: t = "base64"
                elif t in ("bool", "boolean"): t = "boolean"
                type_map[key.strip()] = t

        # Build note for LLM about expected answer types
        type_note = "\nNote: Expected types:\n" + "".join(f"- {k}: {t}\n" for k, t in type_map.items())

        # Attempt to load first valid dataset from remaining uploads
        data_candidates = [f for f in uploads if f is not questions_file]
        pickle_path, df_preview, dataset_uploaded = None, "", False
        for file in data_candidates:
            try:
                if file.filename.lower().endswith(".csv"):
                    df = pd.read_csv(BytesIO(await file.read()))
                else:
                    continue
            except Exception:
                continue
            dataset_uploaded = True
            tmp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
            tmp_pkl.close()
            df.to_pickle(tmp_pkl.name)
            pickle_path = tmp_pkl.name
            df_preview = f"\nDataset preview:\n{df.head().to_markdown(index=False)}\n"
            break

        # Rules for LLM depending on dataset availability
        if dataset_uploaded:
            llm_rules = "Rules:\n1) Use only provided dataset (df)...\n"
        else:
            llm_rules = "Rules:\n1) Use scrape_url_to_dataframe() if needed...\n"

        # ✅ Renamed variable here
        agent_prompt_input = (
            f"{llm_rules}\nQuestions:\n{raw_questions}\n{df_preview}\n{type_note}\n"
            "Respond with JSON object only."
        )

        # Run the agent in a thread pool with timeout
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as ex:
            future = ex.submit(run_agent_safely_unified, agent_prompt_input, pickle_path, type_map)
            try:
                result = future.result(timeout=LLM_TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError:
                raise HTTPException(408, "Processing timeout")

        if "error" in result:
            raise HTTPException(500, result["error"])
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("analyze_data failed")
        raise HTTPException(500, str(e))

# -------------------------------------------------
# Health check + favicon routes
# -------------------------------------------------
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Serve favicon.ico if available, otherwise return transparent PNG."""
    path = "favicon.ico"
    if os.path.exists(path):
        return FileResponse(path, media_type="image/x-icon")
    return Response(content=base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO3n+9QAAAAASUVORK5CYII="
    ), media_type="image/png")

@app.get("/api", include_in_schema=False)
async def analyze_get_info():
    """Simple server status check."""
    return JSONResponse({"ok": True, "message": "Server is running."})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
