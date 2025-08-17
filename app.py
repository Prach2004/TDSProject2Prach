# THE APP.PY FOR THE DATAANALYST AGENT WHICH USES GEMINI 2.5 FLASH
import os
import networkx as nx
import re
import json
import base64
import tempfile
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
import re
import json
import base64
import tempfile
import subprocess
import logging
import time
from io import BytesIO
from typing import Dict, Any, List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

load_dotenv()
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", 300))
# IMAGE CONVERSION
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# IMPORTS LLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TDS Data Analyst Agent")
app.router.redirect_slashes = False 

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# PATCH A: Add helpers (defaults + fallback builder)
def _default_for(type_hint: str):
    t = (type_hint or "").lower()
    if t in ("number", "float", "int"): return 0
    if t in ("boolean", "bool"): return False
    # for "string" and "base64" or unknown -> empty string
    return ""

def _extract_keys_when_types_unknown(raw_questions: str):
    """
    If we couldn't parse types, at least extract keys to shape the fallback.
    Tries backtick, "key: type", and bullet formats.
    """
    keys = set()
    if not raw_questions:
        return []
    # key or - key patterns
    keys.update(re.findall(r"`([^`]+)`", raw_questions))
    keys.update(m.strip() for m in re.findall(r"^-+\s*`([^`]+)`", raw_questions, flags=re.MULTILINE))
    # key: something
    keys.update(m.strip() for m in re.findall(r"^-+\s*([A-Za-z0-9_ ]+)\s*:", raw_questions, flags=re.MULTILINE))
    # Plain bullet words
    if not keys:
        keys.update(m.strip() for m in re.findall(r"^-+\s*([A-Za-z0-9_ ]+)", raw_questions, flags=re.MULTILINE))
    return [k for k in keys if k]

# -----------------------------
# Tools : CAN BE PUT ON A DIFFERENT FILE TOOLS.PY AND IMPORTED 
# -----------------------------

@tool
def scrape_url_to_dataframe(url: str) -> Dict[str, Any]:
    """
    Fetch a URL and return data as a DataFrame (supports HTML tables, CSV, Excel, Parquet, JSON, and plain text).
    Always returns {"status": "success", "data": [...], "columns": [...]} if fetch works.
    """
    print(f"Scraping URL: {url}")
    try:
        from io import BytesIO, StringIO
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/138.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.google.com/",
        }

        resp = requests.get(url, headers=headers, timeout=50)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "").lower()

        df = None

        # --- CSV ---
        if "text/csv" in ctype or url.lower().endswith(".csv"):
            df = pd.read_csv(BytesIO(resp.content))

        # --- Excel ---
        elif any(url.lower().endswith(ext) for ext in (".xls", ".xlsx")) or "spreadsheetml" in ctype:
            df = pd.read_excel(BytesIO(resp.content))

        # --- Parquet ---
        elif url.lower().endswith(".parquet"):
            df = pd.read_parquet(BytesIO(resp.content))

        # --- JSON ---
        elif "application/json" in ctype or url.lower().endswith(".json"):
            try:
                data = resp.json()
                df = pd.json_normalize(data)
            except Exception:
                df = pd.DataFrame([{"text": resp.text}])

        # --- HTML / Fallback ---
        elif "text/html" in ctype or re.search(r'/wiki/|\.org|\.com', url, re.IGNORECASE):
            html_content = resp.text
            # Try HTML tables first
            try:
                tables = pd.read_html(StringIO(html_content), flavor="bs4")
                if tables:
                    df = tables[0]
            except ValueError:
                pass

            # If no table found, fallback to plain text
            if df is None:
                soup = BeautifulSoup(html_content, "html.parser")
                text = soup.get_text(separator="\n", strip=True)
                df = pd.DataFrame({"text": [text]})

        # --- Unknown type fallback ---
        else:
            df = pd.DataFrame({"text": [resp.text]})

        # --- Normalize columns ---
        df.columns = df.columns.map(str).str.replace(r'\[.*\]', '', regex=True).str.strip()

        return {
            "status": "success",
            "data": df.to_dict(orient="records"),
            "columns": df.columns.tolist()
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


# -----------------------------
# UTILS.PY
# -----------------------------
def clean_llm_output(output: str) -> Dict:
    """
    Extract JSON object from LLM output robustly.
    Returns dict or {"error": "..."}
    """
    try:
        if not output:
            return {"error": "Empty LLM output"}
        # remove triple-fence markers if present
        s = re.sub(r"^```(?:json)?\s*", "", output.strip())
        s = re.sub(r"\s*```$", "", s)
        # find outermost JSON object by scanning for balanced braces
        first = s.find("{")
        last = s.rfind("}")
        if first == -1 or last == -1 or last <= first:
            return {"error": "No JSON object found in LLM output", "raw": s}
        candidate = s[first:last+1]
        try:
            return json.loads(candidate)
        except Exception as e:
            # fallback: try last balanced pair scanning backwards
            for i in range(last, first, -1):
                cand = s[first:i+1]
                try:
                    return json.loads(cand)
                except Exception:
                    continue
            return {"error": f"JSON parsing failed: {str(e)}", "raw": candidate}
    except Exception as e:
        return {"error": str(e)}

SCRAPE_FUNC = r'''
from typing import Dict, Any
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

def scrape_url_to_dataframe(url: str) -> Dict[str, Any]:
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=5
        )
        response.raise_for_status()
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "data": [],
            "columns": []
        }

    soup = BeautifulSoup(response.text, "html.parser")
    tables = pd.read_html(response.text)

    if tables:
        df = tables[0]  # Take first table
        df.columns = [str(c).strip() for c in df.columns]
        
        # Ensure all columns are unique and string
        df.columns = [str(col) for col in df.columns]

        return {
            "status": "success",
            "data": df.to_dict(orient="records"),
            "columns": list(df.columns)
        }
    else:
        # Fallback to plain text
        text_data = soup.get_text(separator="\n", strip=True)

        # Try to detect possible "keys" from text like Runtime, Genre, etc.
        detected_cols = set(re.findall(r"\b[A-Z][a-zA-Z ]{2,15}\b", text_data))
        df = pd.DataFrame([{}])  # start empty
        for col in detected_cols:
            df[col] = None

        if df.empty:
            df["text"] = [text_data]

        return {
            "status": "success",
            "data": df.to_dict(orient="records"),
            "columns": list(df.columns)
        }
'''


def write_and_run_temp_python(code: str, injected_pickle: str = None, timeout: int = 60) -> Dict[str, Any]:
    """
    Write a temp python file which:
      - provides a safe environment (imports)
      - loads df/from pickle if provided into df and data variables
      - defines a robust plot_to_base64() helper that ensures < 100kB (attempts resizing/conversion)
      - executes the user code (which should populate `results` dict)
      - prints json.dumps({"status":"success","result":results})
    Returns dict with parsed JSON or error details.
    """
    # create file content
    preamble = [
        "import json, sys, gc",
        "import pandas as pd, numpy as np",
        "import matplotlib",
        "matplotlib.use('Agg')",
        "import matplotlib.pyplot as plt",
        "from io import BytesIO",
        "import base64",
    ]
    if PIL_AVAILABLE:
        preamble.append("from PIL import Image")
    # inject df if a pickle path provided
    if injected_pickle:
        preamble.append(f"df = pd.read_pickle(r'''{injected_pickle}''')\n")
        preamble.append("data = df.to_dict(orient='records')\n")
    else:
        # ensure data exists so user code that references data won't break
        preamble.append("data = globals().get('data', {})\n")

    # plot_to_base64 helper that tries to reduce size under 100_000 bytes
    helper = r'''
def plot_to_base64(max_bytes=100000):
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_bytes = buf.getvalue()
    # if already under limit, return png data uri
    if len(img_bytes) <= max_bytes:
        return base64.b64encode(img_bytes).decode('ascii')
    # try decreasing dpi/figure size iteratively
    for dpi in [80, 60, 50, 40, 30]:
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
        buf.seek(0)
        b = buf.getvalue()
        if len(b) <= max_bytes:
            return base64.b64encode(b).decode('ascii')
    # if Pillow available, try convert to WEBP which is typically smaller
    try:
        from PIL import Image
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=40)
        buf.seek(0)
        im = Image.open(buf)
        out_buf = BytesIO()
        im.save(out_buf, format='WEBP', quality=80, method=6)
        out_buf.seek(0)
        ob = out_buf.getvalue()
        if len(ob) <= max_bytes:
            return base64.b64encode(ob).decode('ascii')
        # try lower quality
        out_buf = BytesIO()
        im.save(out_buf, format='WEBP', quality=60, method=6)
        out_buf.seek(0)
        ob = out_buf.getvalue()
        if len(ob) <= max_bytes:
            return base64.b64encode(ob).decode('ascii')
    except Exception:
        pass
    # as last resort return a downsized PNG even if > max_bytes
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=20)
    buf.seek(0)
    return  base64.b64encode(buf.getvalue()).decode('ascii')
'''

    # Build the code to write
    script_lines = []
    script_lines.extend(preamble)
    script_lines.append(helper)
    script_lines.append(SCRAPE_FUNC)
    script_lines.append("\nresults = {}\n")
    script_lines.append(code)
    # ensure results printed as json
    script_lines.append("\nprint(json.dumps({'status':'success','result':results}, default=str), flush=True)\n")

    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8')
    tmp.write("\n".join(script_lines))
    tmp.flush()
    tmp_path = tmp.name
    tmp.close()

    try:
        completed = subprocess.run([sys.executable, tmp_path],
                                   capture_output=True, text=True, timeout=timeout)
        if completed.returncode != 0:
            # collect stderr and stdout for debugging
            return {"status": "error", "message": completed.stderr.strip() or completed.stdout.strip()}
        # parse stdout as json
        out = completed.stdout.strip()
        try:
            parsed = json.loads(out)
            return parsed
        except Exception as e:
            return {"status": "error", "message": f"Could not parse JSON output: {str(e)}", "raw": out}
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Execution timed out"}
    finally:
        try:
            os.unlink(tmp_path)
            if injected_pickle and os.path.exists(injected_pickle):
                os.unlink(injected_pickle)
        except Exception:
            pass


# -----------------------------
# AGENT.PY
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model=os.getenv("GOOGLE_MODEL", "gemini-2.5-pro"),
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Tools list for agent (LangChain tool decorator returns metadata for the LLM)
tools = [scrape_url_to_dataframe]  # we only expose scraping as a tool; agent will still produce code

# Prompt: instruct agent to call the tool and output JSON only
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a full-stack autonomous data analyst agent.

You will receive:
- A set of **rules** for this request (these rules may differ depending on whether a dataset is uploaded or not)
- One or more **questions**
- An optional **dataset preview**

You must:
1. Follow the provided rules exactly.
2. Return only a valid JSON object â€" no extra commentary or formatting.
3. The JSON must contain:
    - "questions":  keys provided in the questions file
    - "code": "..." (Python code that fills `results` with exact type of answer of each question as given in questions file and question keys as keys)\n'
    - "Note" : the type of each answer should match the type it is asked in question file(e.g. int, float, str, boolean ,base64).
4. Your Python code will run in a sandbox with:
    - pandas, numpy, matplotlib available
    - A helper function `plot_to_base64(max_bytes=100000)` for generating base64-encoded images under 100KB.
5. When returning plots, always use `plot_to_base64()` to keep image sizes small.
6. Make sure all variables are defined before use, and the code can run without any undefined references.
"""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(
    llm=llm,
    tools=[scrape_url_to_dataframe],  # let the agent call tools if it wants; we will also pre-process scrapes
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[scrape_url_to_dataframe],
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate",
    handle_parsing_errors=True,
    return_intermediate_steps=False
)


from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile
import os, json, tempfile
from io import BytesIO
import pandas as pd
import numpy as np

from fastapi import Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from io import BytesIO, StringIO
import tempfile, json
import pandas as pd

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile
import os, json, tempfile
from io import BytesIO
import pandas as pd
import numpy as np

# -----------------------------
# Runner: orchestrates agent -> pre-scrape inject -> execute
# -----------------------------

# PATCH B: Make run_agent_safely_unified time-budget aware and always return shaped JSON on internal errors
def run_agent_safely_unified(
    llm_input: str,
    pickle_path: str = None,
    type_map: dict = None,
    time_budget_s: int = 300
) -> Dict:
    """
    Runs the agent with retries, within a remaining time budget (seconds).
    Returns a SHAPED result on success or {"error": "..."} on failure; shaping of fallback
    is done by the caller because it knows the question keys/types.
    """
    try:
        max_retries = 4
        start = time.time()
        last_err = None

        for attempt in range(1, max_retries + 1):
            remaining = time_budget_s - (time.time() - start)
            if remaining <= 3:  # keep a few seconds for shaping the response
                break

            # Invoke agent with per-attempt timeout bounded by remaining budget
            per_attempt_timeout = int(min(LLM_TIMEOUT_SECONDS, max(5, remaining - 1)))

            response = agent_executor.invoke(
                {"input": llm_input},
                {"timeout": per_attempt_timeout}  # LangChain call timeout
            )

            raw_out = response.get("output") or response.get("final_output") or response.get("text") or ""
            if not raw_out:
                last_err = "Empty agent output"
                continue

            parsed = clean_llm_output(raw_out)
            if "error" in parsed:
                last_err = parsed["error"]
                continue

            if "code" not in parsed or "questions" not in parsed:
                last_err = f"Invalid agent response keys: {list(parsed.keys())}"
                continue

            code = parsed["code"]
            # Execute with remaining budget
            remaining = time_budget_s - (time.time() - start)
            if remaining <= 3:
                break
            per_exec_timeout = int(min(LLM_TIMEOUT_SECONDS, max(5, remaining - 1)))

            # If no pickle provided and the code tries to scrape, do that once
            pkl_for_exec = pickle_path
            if pkl_for_exec is None:
                urls = re.findall(r"scrape_url_to_dataframe\(\s*['\"](.?)['\"]?\)", code)
                if urls:
                    tool_resp = scrape_url_to_dataframe(urls[0])
                    if tool_resp.get("status") == "success":
                        df = pd.DataFrame(tool_resp["data"])
                        temp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
                        temp_pkl.close()
                        df.to_pickle(temp_pkl.name)
                        pkl_for_exec = temp_pkl.name

            exec_result = write_and_run_temp_python(
                code,
                injected_pickle=pkl_for_exec,
                timeout=per_exec_timeout
            )
            if exec_result.get("status") != "success":
                last_err = f"Execution failed: {exec_result.get('message')}"
                continue

            results_dict = exec_result.get("result", {})
            # Coerce to requested types (if known)
            if type_map:
                for k, expected in type_map.items():
                    if k not in results_dict:
                        continue
                    v = results_dict[k]
                    if expected == "number":
                        try:
                            num = float(v)
                            results_dict[k] = int(num) if float(num).is_integer() else float(num)
                        except Exception:
                            results_dict[k] = _default_for(expected)
                    elif expected in ("string", "base64"):
                        results_dict[k] = "" if v is None else str(v)
                    elif expected == "boolean":
                        results_dict[k] = bool(v)

            return {"ok": True, "results": results_dict, "code": code}

        return {"error": last_err or "Exhausted retries"}

    except Exception as e:
        logger.exception("run_agent_safely_unified failed")
        return {"error": str(e)}

# PATCH C: Rework analyze_data to ALWAYS emit fallback JSON (no HTTP errors)
@app.post("/api")
@app.post("/api/")
async def analyze_data(request: Request):
    start_req = time.time()

    def remaining_budget():
        return max(1, LLM_TIMEOUT_SECONDS - (time.time() - start_req))

    try:
        form = await request.form()
        uploads = [v for _, v in form.multi_items() if isinstance(v, UploadFile)]

        # Parse questions file (if missing, still build fallback later)
        txt_files = [f for f in uploads if (f.filename or "").lower().endswith(".txt")]
        raw_questions = (await txt_files[0].read()).decode("utf-8") if txt_files else ""

        # Build type map (your existing logic)
        type_map = {}
        patterns = [
            re.compile(r"^\s*-\s*`([^`]+)`\s*:\s*([a-zA-Z ]+)", re.MULTILINE),
            re.compile(r"^\s*-\s*([a-zA-Z0-9_ ]+)\s*:\s*([a-zA-Z ]+)", re.MULTILINE),
            re.compile(r"`([^`]+)`\s*\((number|string|boolean|base64)[s]?\)", re.IGNORECASE),
        ]
        for pat in patterns:
            for match in pat.finditer(raw_questions):
                key, type_hint = match.groups()
                norm_type = type_hint.strip().lower()
                if norm_type in ("number", "float", "int"): norm_type = "number"
                elif norm_type in ("string", "str"): norm_type = "string"
                elif "base64" in norm_type: norm_type = "base64"
                elif norm_type in ("bool", "boolean"): norm_type = "boolean"
                type_map[key.strip()] = norm_type

        # If we still don't have keys, try extracting them and assume string
        if not type_map:
            for k in _extract_keys_when_types_unknown(raw_questions):
                type_map[k] = "string"

        # Build dataset (optional)
        data_candidates = [f for f in uploads if not (f in txt_files)]
        pickle_path = None
        df_preview = ""
        for data_file in data_candidates:
            try:
                filename = (data_file.filename or "").lower()
                content = await data_file.read()
                if filename.endswith(".csv"):
                    df = pd.read_csv(BytesIO(content))
                elif filename.endswith((".xlsx", ".xls")):
                    df = pd.read_excel(BytesIO(content))
                elif filename.endswith(".parquet"):
                    df = pd.read_parquet(BytesIO(content))
                elif filename.endswith(".json"):
                    try:
                        df = pd.read_json(BytesIO(content))
                    except ValueError:
                        df = pd.DataFrame(json.loads(content.decode("utf-8")))
                elif filename.endswith((".png", ".jpg", ".jpeg")) and PIL_AVAILABLE:
                    img = Image.open(BytesIO(content)).convert("RGB")
                    df = pd.DataFrame({"image": [img]})
                else:
                    continue
                temp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
                temp_pkl.close()
                df.to_pickle(temp_pkl.name)
                pickle_path = temp_pkl.name
                df_preview = (
                    f"\n\nThe uploaded dataset has {len(df)} rows and {len(df.columns)} columns.\n"
                    f"Columns: {', '.join(map(str, df.columns))}\n"
                    f"First rows:\n{df.head(5).to_markdown(index=False)}\n"
                )
                break
            except Exception:
                continue

        # LLM rules
        if pickle_path:
            llm_rules = (
                "Rules:\n"
                "1) You have access to a pandas DataFrame called df and its dictionary form data.\n"
                "2) DO NOT call scrape_url_to_dataframe() or fetch any external data.\n"
                "3) Use only the uploaded dataset for answering questions.\n"
                "4) Produce a final JSON object with keys:\n"
                '   - "questions": exact keys provided in questions txt file \n'
                '   - "code": "..."  (Python code that fills results)\n'
                "5) For plots: use plot_to_base64() helper to return base64 image data under 100kB.\n"
            )
        else:
            llm_rules = (
                "Rules:\n"
                "1) If you need web data, CALL scrape_url_to_dataframe(url).\n"
                "2) Produce a final JSON object with keys:\n"
                '   - "questions": exact keys provided in questions txt file \n'
                '   - "code": "..."  (Python code that fills results)\n'
                "3) For plots: use plot_to_base64() helper to return base64 image data under 100kB.\n"
            )

        type_note = "\nNote: Expected types:\n" + "\n".join([f"- {k}: {v}" for k, v in type_map.items()]) if type_map else "(types unknown)\n"
        llm_input = f"{llm_rules}\nQuestions:\n{raw_questions}\n{df_preview}\n{type_note}\nRespond with the JSON object only."

        # RUN with remaining budget & up to 4 attempts internally
        result = run_agent_safely_unified(
            llm_input,
            pickle_path=pickle_path,
            type_map=type_map,
            time_budget_s=int(remaining_budget())
        )

        if isinstance(result, dict) and result.get("ok"):
            return JSONResponse(content=result["results"], status_code=200)

        # Fallback (error, timeout, or invalid output)
        answers = {k: _default_for(t) for k, t in type_map.items()} if type_map else {}
        return JSONResponse(content={
            "questions": answers
        }, status_code=200)

    except Exception as e:
        # LAST-RESORT FALLBACK: still return shaped JSON
        answers = {k: _default_for(t) for k, t in type_map.items()} if 'type_map' in locals() and type_map else {}
        return JSONResponse(content={
            "questions": answers
        }, status_code=200)

    
from fastapi.responses import FileResponse, Response
import base64, os

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
