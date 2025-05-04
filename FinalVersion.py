import base64
import io
import time
import random
import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import webbrowser
import threading
import subprocess
import os
import chardet
import multiprocessing
import re
import requests
import dash
import platform

# ────────────────────────────────────────────────────────────────
# Selenium & ML deps
# ────────────────────────────────────────────────────────────────
from selenium import webdriver
from selenium.webdriver.common.by import By
#from selenium.webdriver.chrome.service import Service
#from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from rapidfuzz import fuzz
from transformers import pipeline
from collections import Counter
from dash import dcc, html, dash_table, Output, Input, State, no_update

# ────────────────────────────────────────────────────────────────
# Globals & Thread‑Pool
# ────────────────────────────────────────────────────────────────

uploaded_people: list = []
start_time = None
last_result_time = None
MAX_RETRIES = 2
retry_attempts = {}
final_table_ready = False
finalized_table_data = []

logger = logging.getLogger("LinkedInScraper")

# CPU‑aware thread count (max 8)
available_threads = os.cpu_count() or multiprocessing.cpu_count()
max_threads = min(8, max(4, int(available_threads * 0.75)))
executor = ThreadPoolExecutor(max_workers=max_threads)
pending_tasks = []
completed_results = []

scraping_active = False
start_scrape_time = None

action_lock = threading.Lock()

# ────────────────────────────────────────────────────────────────
# Logging — keep history (no truncation)
# ────────────────────────────────────────────────────────────────

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "searchlog.txt")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] :: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
    ]
)
logger.info("🔁 Logger started — appending to existing searchlog.txt")

# Cross‑platform helper to open the log – optional UI button can call this

def open_log_file():
    try:
        system = platform.system()
        if system == "Windows":
            os.startfile(log_file_path)
        elif system == "Darwin":  # macOS
            subprocess.run(["open", log_file_path])
        else:  # Linux & friends
            subprocess.run(["xdg-open", log_file_path])
    except Exception as e:
        logger.warning(f"⚠️ Failed to open log file: {e}")

# ────────────────────────────────────────────────────────────────
# ML Models (single NER pipeline)
# ────────────────────────────────────────────────────────────────

# Use lightweight model (only ~100MB RAM instead of 800MB)
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

# Replace large Roberta model with smaller BERT NER
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

def extract_ner_entities(text: str):
    """Return unique PER / ORG / LOC from a text blob."""
    ents = ner_pipeline(text)
    loc = {e["word"] for e in ents if e["entity_group"] == "LOC"}
    org = {e["word"] for e in ents if e["entity_group"] == "ORG"}
    per = {e["word"] for e in ents if e["entity_group"] == "PER"}
    return {"locations": list(loc), "organizations": list(org), "persons": list(per)}


def extract_best_title(raw_title: str) -> str:
    """Strip boiler‑plate and heuristically pick best job‑title fragment."""
    if not raw_title:
        return "Not Found"
    clean = raw_title
    for ph in ("United States", "Professional Profile", "Connections", "LinkedIn"):
        clean = clean.replace(ph, "").strip()
    parts = [p.strip() for p in clean.split("|") if p.strip()]
    if not parts:
        return "Not Found"
    ref = clean.lower()
    return max(parts, key=lambda p: fuzz.token_set_ratio(p, ref))


def is_best_match(full_name: str, title: str, cos_th=0.4, fuzz_th=0.75):
    if not (full_name and title):
        return False
    cos_score = cos_sim(
        sentence_model.encode(full_name, convert_to_tensor=True),
        sentence_model.encode(title, convert_to_tensor=True)
    ).item()
    fuzz_score = fuzz.token_set_ratio(full_name.lower(), title.lower()) / 100.0
    logger.info(f"[Similarity] Cosine: {cos_score:.2f} | Fuzzy: {fuzz_score:.2f}")
    return cos_score >= cos_th or fuzz_score >= fuzz_th


university_map = {
    "KU": "Kean University",
    "RUN": "Rutgers University - Newark",
    "RUNB": "Rutgers University - Newark",
    "WPU": "William Paterson University",
    "FDU": "Fairleigh Dickinson University",
    "MSU": "Montclair State University",
    "NJCU": "New Jersey City University",
    "BC": "Bloomfield College",
}

# ────────────────────────────────────────────────────────────────
# Chrome driver (headless)
# ────────────────────────────────────────────────────────────────
#DRIVER_PATH = ChromeDriverManager().install()

def create_driver():
    options = Options()
    options.binary_location = "/usr/bin/chromium"
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920x1080")
    options.add_argument("--user-agent=Mozilla/5.0")

    driver = webdriver.Chrome(
        executable_path="/usr/bin/chromedriver",
        options=options
    )
    driver.set_page_load_timeout(15)
    return driver

# ────────────────────────────────────────────────────────────────
# SerpAPI helpers — single call for profile + inline location & income
# ────────────────────────────────────────────────────────────────

CITY_KEYWORDS = [
    # big US cities – quick heuristic list
    "Chicago", "Boston", "Atlanta", "Seattle", "San Francisco", "Los Angeles",
    "San Diego", "Philadelphia", "Phoenix", "Houston", "Dallas", "Miami", "Las Vegas",
    "Austin", "Denver", "Minneapolis", "Orlando", "San Jose", "Portland", "New Orleans",
    "Tampa", "Charlotte"
]

US_STATE_ABBR = {
    "al": "Alabama", "ak": "Alaska", "az": "Arizona", "ar": "Arkansas", "ca": "California",
    "co": "Colorado", "ct": "Connecticut", "de": "Delaware", "fl": "Florida", "ga": "Georgia",
    "hi": "Hawaii", "id": "Idaho", "il": "Illinois", "in": "Indiana", "ia": "Iowa",
    "ks": "Kansas", "ky": "Kentucky", "la": "Louisiana", "me": "Maine", "md": "Maryland",
    "ma": "Massachusetts", "mi": "Michigan", "mn": "Minnesota", "ms": "Mississippi", "mo": "Missouri",
    "mt": "Montana", "ne": "Nebraska", "nv": "Nevada", "nh": "New Hampshire", "nj": "New Jersey",
    "nm": "New Mexico", "ny": "New York", "nc": "North Carolina", "nd": "North Dakota", "oh": "Ohio",
    "ok": "Oklahoma", "or": "Oregon", "pa": "Pennsylvania", "ri": "Rhode Island", "sc": "South Carolina",
    "sd": "South Dakota", "tn": "Tennessee", "tx": "Texas", "ut": "Utah", "vt": "Vermont",
    "va": "Virginia", "wa": "Washington", "wv": "West Virginia", "wi": "Wisconsin", "wy": "Wyoming"
}


# Main profile search — one SerpAPI call per person

def serpapi_search_linkedin_profile(person: dict):
    serp_key = os.getenv("SERPAPI_KEY")
    if not serp_key:
        logger.error("❌ SERPAPI_KEY env not set — abort SerpAPI search.")
        return None

    full_name = f"{person['First Name']} {person['Last Name']}".strip()
    university = university_map.get(person["University"], person["University"])
    query = f'"{full_name}" "{university}" site:linkedin.com'

    params = {"q": query, "api_key": serp_key, "engine": "google", "num": 10}
    try:
        logger.info(f"🔍 SerpAPI query: {query}")
        data = requests.get("https://serpapi.com/search", params=params, timeout=20).json()
        if "error" in data:
            logger.error(f"SerpAPI error: {data['error']}")
            return None

        best_result = None
        best_score = -1.0

        for res in data.get("organic_results", []):
            title = res.get("title", "")
            link = res.get("link", "")
            snippet = res.get("snippet", "")

            if not (full_name.lower() in title.lower() or full_name.lower() in snippet.lower()):
                continue  # quick filter

            if not is_best_match(full_name, title):
                continue

            # compute score only once we know it passes threshold
            cos_s = cos_sim(
                sentence_model.encode(full_name, convert_to_tensor=True),
                sentence_model.encode(title, convert_to_tensor=True)
            ).item()
            fz_s = fuzz.token_set_ratio(full_name.lower(), title.lower()) / 100
            score = max(cos_s, fz_s)
            if score <= best_score:
                continue

            # NER location (prefer snippet+title)
            ner = extract_ner_entities(f"{title}. {snippet}")
            loc = ner["locations"][0] if ner["locations"] else "Unknown"

            best_result = {
                "First Name": person["First Name"],
                "Last Name": person["Last Name"],
                "University": university,
                "Graduation Year": person.get("Graduation Year", "N/A"),
                "LinkedIn Title": extract_best_title(title),
                "LinkedIn URL": f'<a href="{link}" target="_blank">Open Profile</a>',
                "Score": f"{int(score*100)}%",
                "Location (Estimated)": loc,
            }
            best_score = score

        if best_result:
            # grab income once, based on chosen title/loc
            best_result["Income (Estimated)"] = "Unknown"
            logger.info(f"🏆 Best match {full_name}: {best_result['LinkedIn Title']} @ {best_score:.2f}")
            return best_result
        else:
            logger.warning(f"No SerpAPI match for {full_name}")
    except Exception as e:
        logger.error(f"SerpAPI failure {full_name}: {e}")

    return None

# ────────────────────────────────────────────────────────────────
# Bing / Selenium fallback (kept, but no extra SerpAPI calls)
# ────────────────────────────────────────────────────────────────

def search_person(person, cosine_threshold=0.4, fuzzy_threshold=0.75):
    result = serpapi_search_linkedin_profile(person)
    if result:
        return result

    # Fallback via Bing (rare)
    full_name = f"{person['First Name']} {person['Last Name']}".strip()
    query = f'"{full_name}" "{person["University"]}" site:linkedin.com'
    attempt = retry_attempts.get(full_name, 0)

    driver = None
    try:
        driver = create_driver()
        driver.get(f"https://www.bing.com/search?q={query}")
        time.sleep(random.uniform(2, 3))
        entries = driver.find_elements(By.CSS_SELECTOR, "li.b_algo")[:10]
        best = None
        best_score = -1.0
        for ent in entries:
            try:
                title = ent.find_element(By.TAG_NAME, "h2").text.strip()
                href = ent.find_element(By.TAG_NAME, "a").get_attribute("href")
                snippet = ent.find_element(By.CLASS_NAME, "b_caption").text.strip()
            except Exception:
                continue

            if not is_best_match(full_name, title, cos_th=cosine_threshold, fuzz_th=fuzzy_threshold):
                continue
            cos_s = cos_sim(sentence_model.encode(full_name, convert_to_tensor=True),
                            sentence_model.encode(title, convert_to_tensor=True)).item()
            fz_s = fuzz.token_set_ratio(full_name.lower(), title.lower()) / 100
            sc = max(cos_s, fz_s)
            if sc <= best_score:
                continue

            ner = extract_ner_entities(f"{title}. {snippet}")
            loc = ner["locations"][0] if ner["locations"] else "Unknown"
            best = {
                "First Name": person["First Name"],
                "Last Name": person["Last Name"],
                "University": person["University"],
                "Graduation Year": person.get("Graduation Year", "N/A"),
                "LinkedIn Title": extract_best_title(title),
                "LinkedIn URL": f'<a href="{href}" target="_blank">Open Profile</a>',
                "Score": f"{int(sc*100)}%",
                "Location (Estimated)": loc,
                "Income (Estimated)": "Unknown",
            }
            best_score = sc
        return best
    except Exception as e:
        logger.error(f"Selenium fallback failed for {full_name}: {e}")
        if attempt < MAX_RETRIES:
            retry_attempts[full_name] = attempt + 1
            return search_person(person)
    finally:
        if driver:
            driver.quit()
    return None

def finalize_income_estimates(results):
    salary_ranges = {
        # Engineering roles
        'engineer': (85000, 150000),
        'software': (110000, 180000),
        'developer': (95000, 160000),
        'architect': (130000, 190000),
        'data scientist': (120000, 180000),
        'devops': (115000, 170000),
        'product manager': (125000, 190000),
        
        # Business/Finance
        'analyst': (75000, 120000),
        'manager': (90000, 150000),
        'director': (140000, 220000),
        'executive': (180000, 300000),
        'ceo': (200000, 500000),
        'cfo': (180000, 350000),
        'cto': (160000, 300000),
        'vp': (150000, 280000),
        'finance': (90000, 160000),
        'accountant': (70000, 120000),
        
        # Marketing/Sales
        'marketing': (75000, 130000),
        'sales': (65000, 140000),
        'account manager': (80000, 130000),
        'customer': (60000, 100000),
        
        # Healthcare
        'doctor': (180000, 350000),
        'physician': (200000, 400000),
        'nurse': (75000, 120000),
        'healthcare': (80000, 150000),
        
        # Legal
        'attorney': (130000, 250000),
        'lawyer': (120000, 240000),
        'legal': (100000, 200000),
        
        # Education
        'professor': (80000, 150000),
        'teacher': (50000, 85000),
        'educator': (55000, 90000),
        
        # Other common roles
        'consultant': (90000, 170000),
        'advisor': (85000, 150000),
        'specialist': (70000, 120000),
        'researcher': (75000, 130000),
        'student': (0, 30000),
        'intern': (30000, 60000),
        'associate': (65000, 110000),
    }
    
    # Default range for unknown professions
    default_range = (60000, 100000)
    
    for result in results:
        # Skip if we already have an income estimate that isn't "Unknown"
        if result.get("Income (Estimated)") and result.get("Income (Estimated)") != "Unknown":
            continue
            
        title = result.get("LinkedIn Title", "").lower()
        
        if title == "Not Found" or not title:
            result["Income (Estimated)"] = "Unknown"
            continue
        
        # Find matching salary range
        matched_range = None
        for keyword, salary_range in salary_ranges.items():
            if keyword in title:
                matched_range = salary_range
                break
        
        # Use default if no match found
        if not matched_range:
            matched_range = default_range
            
        # Add some randomness within the range
        salary = random.randint(matched_range[0], matched_range[1])
        # Format with $ and commas
        result["Income (Estimated)"] = f"${salary:,}"
        
        # Add a status field for successful matches
        if result.get("LinkedIn URL"):
            result["Status"] = "✅ Match Found"
    
    logger.info(f"📊 Income estimates added for {len(results)} results")
    return results

# ────────────────────────────────────────────────────────────────
# Task orchestration utils (enqueue / done)
# ────────────────────────────────────────────────────────────────

def enqueue_tasks(people, cosine_threshold=0.4, fuzzy_threshold=0.75):
    global scraping_active, start_scrape_time, pending_tasks, total_tasks
    scraping_active = True
    start_scrape_time = time.time()
    pending_tasks.clear()
    completed_results.clear()
    total_tasks = len(people)

    logger.info(f"🚀 Starting search for {total_tasks} person(s)")

    for p in people:
        fut = executor.submit(search_person, p)
        fut.add_done_callback(task_done)
        pending_tasks.append(fut)


def task_done(fut):
    global scraping_active, last_result_time
    with action_lock:
        try:
            res = fut.result(timeout=60)
            completed_results.append(res or {
                "First Name": "N/A", "Last Name": "N/A", "University": "N/A",
                "Graduation Year": "N/A", "LinkedIn Title": "Not Found", "LinkedIn URL": "",
                "Score": "N/A", "Location (Estimated)": "Unknown", "Income (Estimated)": "Unknown",
            })
            last_result_time = time.time()
        except Exception as e:
            logger.error(f"Task failure: {e}")
            completed_results.append({
                "First Name": "N/A", "Last Name": "N/A", "University": "N/A",
                "Graduation Year": "N/A", "LinkedIn Title": "Error", "LinkedIn URL": "",
                "Score": "N/A", "Location (Estimated)": "Unknown", "Income (Estimated)": "Unknown",
            })
        if len(completed_results) == total_tasks:
            scraping_active = False
            logger.info("✅ All tasks finished")

# ────────────────────────────────────────────────────────────────
# Dash UI (identical to original, duplicate index_string removed)
# ────────────────────────────────────────────────────────────────

external_stylesheets = [{
    'href': 'https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap',
    'rel': 'stylesheet'
}, {
    'href': 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css',
    'rel': 'stylesheet'
}]

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=external_stylesheets)
server = app.server

# Track retry attempts for each person
retry_attempts = {}
MAX_RETRIES = 2

# ------------------------------------------------------------
# 🧠 Dash Callbacks — Upload, Search, Stop, Restart, Download
# ------------------------------------------------------------

app.layout = html.Div([
    
    # Header section with logo and title
    html.Div([
        html.Div([
            html.I(className='fab fa-linkedin', style={
                'fontSize': '48px', 
                'color': '#0072B2', 
                'marginRight': '15px'
            }),
            html.H1("LinkedIn Profile Finder", style={
                'color': '#0072B2',
                'fontWeight': '700',
                'fontSize': '40px',
                'display': 'inline-block',
                'verticalAlign': 'middle'
            })
        ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),
        
        # Subtitle
        html.Div("Find professional profiles with precision", style={
            'fontSize': '18px',
            'color': '#56565A',
            'marginBottom': '30px',
            'textAlign': 'center'
        }),
        
        # Info buttons container with enhanced styling
        html.Div([
            html.A(html.Span([
                html.I(className='fas fa-info-circle', style={'marginRight': '8px'}),
                "How this program works"
            ]),
                href="/assets/42f021a5-7b2b-40eb-99d8-42e07bc11f8d.pdf",

                target="_blank",
                className='floating-btn info-btn',
                style={
                    'fontSize': '16px',
                    'padding': '12px 20px',
                    'borderRadius': '12px',
                    'backgroundColor': '#0057B7',
                    'color': '#FFFFFF',
                    'border': 'none',
                    'boxShadow': '0 4px 12px rgba(0,87,183,0.2)',
                    'cursor': 'pointer',
                    'transition': 'all 0.3s ease',
                    'fontWeight': '500',
                    'textDecoration': 'none',
                    'display': 'flex',
                    'alignItems': 'center'
                }
            ),
            
            html.A(html.Span([
                html.I(className='fas fa-shield-alt', style={'marginRight': '8px'}),
                "Ethical Use Guidelines"
            ]),
                href="/assets/Ethical%20Use%20of%20Web%20Scraping%20Technologies.pdf",  # <--- Fixed
                target="_blank",
                className='floating-btn info-btn',
                style={
                    'fontSize': '16px',
                    'padding': '12px 20px',
                    'borderRadius': '12px',
                    'backgroundColor': '#0057B7',
                    'color': '#FFFFFF',
                    'border': 'none',
                    'boxShadow': '0 4px 12px rgba(0,87,183,0.2)',
                    'cursor': 'pointer',
                    'transition': 'all 0.3s ease',
                    'fontWeight': '500',
                    'textDecoration': 'none',
                    'display': 'flex',
                    'alignItems': 'center'
                }
            ),
        ], style={
            'display': 'flex',
            'justifyContent': 'center',
            'alignItems': 'center',
            'flexWrap': 'wrap',
            'gap': '20px'
        })
    ], style={
        'display': 'flex',
        'flexDirection': 'column',
        'alignItems': 'center',
        'marginBottom': '40px',
        'backgroundColor': 'rgba(255,255,255,0.9)',
        'borderRadius': '20px',
        'padding': '30px 20px',
        'boxShadow': '0 8px 20px rgba(0,0,0,0.05)'
    }),
    
    # Main control panel with improved card styling
    html.Div([
        html.Div([
            # Action buttons with improved structure
            html.Div([
                html.Div([
                    html.I(className='fas fa-file-csv', style={
                        'fontSize': '36px', 
                        'color': '#0a66c2', 
                        'marginBottom': '15px'
                    }),
                    dcc.Upload(
                        id='upload-data',
                        children=html.Button(html.Span([
                            html.I(className='fas fa-upload', style={'marginRight': '10px'}),
                            "Upload CSV"
                        ]), className='floating-btn upload-btn', style={
                            'fontSize': '18px',
                            'padding': '16px 36px',
                            'borderRadius': '16px',
                            'backgroundColor': '#ffffff',
                            'color': '#0a66c2',
                            'border': '2px solid #0a66c2',
                            'boxShadow': '0 6px 16px rgba(0,0,0,0.1)',
                            'cursor': 'pointer',
                            'transition': 'all 0.3s ease',
                            'fontWeight': '600',
                            'width': '100%'
                        }),
                        multiple=False
                    ),
                ], style={
                    'display': 'flex',
                    'flexDirection': 'column',
                    'alignItems': 'center',
                    'padding': '20px',
                    'borderRadius': '15px',
                    'backgroundColor': 'rgba(10, 102, 194, 0.05)',
                    'width': '200px'
                }),
                
                html.Div([
                    html.I(className='fas fa-sync-alt', style={
                        'fontSize': '36px', 
                        'color': '#56565A', 
                        'marginBottom': '15px'
                    }),
                    html.Button(html.Span([
                        html.I(className='fas fa-sync-alt', style={'marginRight': '10px'}),
                        "Restart"
                    ]), id="restart-button", className='floating-btn restart-btn', style={
                        'fontSize': '18px',
                        'padding': '16px 36px',
                        'borderRadius': '16px',
                        'backgroundColor': '#9467bd',
                        'color': '#ffffff',
                        'border': 'none',
                        'boxShadow': '0 6px 16px rgba(148, 103, 189, 0.3)',
                        'cursor': 'pointer',
                        'transition': 'all 0.3s ease',
                        'fontWeight': '600',
                        'width': '100%'
                    }),
                ], style={
                    'display': 'flex',
                    'flexDirection': 'column',
                    'alignItems': 'center',
                    'padding': '20px',
                    'borderRadius': '15px',
                    'backgroundColor': 'rgba(86, 86, 90, 0.05)',
                    'width': '200px'
                }),
                
                html.Div([
                    html.I(className='fas fa-download', style={
                        'fontSize': '36px', 
                        'color': '#009E73', 
                        'marginBottom': '15px'
                    }),
                    html.Button(html.Span([
                        html.I(className='fas fa-download', style={'marginRight': '10px'}),
                        "Download Results"
                    ]), id="download-button", className='floating-btn download-btn', style={
                        'fontSize': '18px',
                        'padding': '16px 36px',
                        'borderRadius': '16px',
                        'backgroundColor': '#009E73',
                        'color': '#ffffff',
                        'border': 'none',
                        'boxShadow': '0 6px 16px rgba(0,0,0,0.15)',
                        'cursor': 'pointer',
                        'transition': 'all 0.3s ease',
                        'fontWeight': '600',
                        'width': '100%'
                    }),
                ], style={
                    'display': 'flex',
                    'flexDirection': 'column',
                    'alignItems': 'center',
                    'padding': '20px',
                    'borderRadius': '15px',
                    'backgroundColor': 'rgba(0, 158, 115, 0.05)',
                    'width': '200px'
                }),
            ], style={
                'display': 'flex',
                'justifyContent': 'center',
                'gap': '30px',
                'marginBottom': '40px',
                'flexWrap': 'wrap'
            }),
            
            # Manual entry section
            html.Div([
                html.Div([
                    html.Label("Manual Entry (Optional)", style={
                        'fontWeight': '600',
                        'fontSize': '18px',
                        'color': '#0072B2',
                        'marginBottom': '15px',
                        'display': 'block'
                    }),
                    html.Div([
                        html.Div([
                            html.Label("First Name", style={
                                'fontWeight': '500',
                                'fontSize': '14px',
                                'marginBottom': '5px',
                                'color': '#555'
                            }),
                            dcc.Input(
                                id='first-name', 
                                placeholder='Enter first name', 
                                type='text', 
                                style={
                                    'width': '100%',
                                    'padding': '12px 15px',
                                    'borderRadius': '8px',
                                    'border': '1px solid #ddd',
                                    'fontSize': '16px',
                                    'transition': 'border-color 0.3s',
                                    'boxSizing': 'border-box'
                                }
                            )
                        ], style={'flex': '1', 'minWidth': '200px'}),
                        
                        html.Div([
                            html.Label("Last Name", style={
                                'fontWeight': '500',
                                'fontSize': '14px',
                                'marginBottom': '5px',
                                'color': '#555'
                            }),
                            dcc.Input(
                                id='last-name', 
                                placeholder='Enter last name', 
                                type='text', 
                                style={
                                    'width': '100%',
                                    'padding': '12px 15px',
                                    'borderRadius': '8px',
                                    'border': '1px solid #ddd',
                                    'fontSize': '16px',
                                    'transition': 'border-color 0.3s',
                                    'boxSizing': 'border-box'
                                }
                            )
                        ], style={'flex': '1', 'minWidth': '200px'}),
                    ], style={
                        'display': 'flex',
                        'gap': '15px',
                        'marginBottom': '15px',
                        'flexWrap': 'wrap'
                    }),
                    
                    html.Div([
                        html.Div([
                            html.Label("University", style={
                                'fontWeight': '500',
                                'fontSize': '14px',
                                'marginBottom': '5px',
                                'color': '#555'
                            }),
                            dcc.Input(
                                id='university', 
                                placeholder='Enter university name', 
                                type='text', 
                                style={
                                    'width': '100%',
                                    'padding': '12px 15px',
                                    'borderRadius': '8px',
                                    'border': '1px solid #ddd',
                                    'fontSize': '16px',
                                    'transition': 'border-color 0.3s',
                                    'boxSizing': 'border-box'
                                }
                            )
                        ], style={'flex': '2', 'minWidth': '300px'}),
                        
                        html.Div([
                            html.Label("Graduation Year", style={
                                'fontWeight': '500',
                                'fontSize': '14px',
                                'marginBottom': '5px',
                                'color': '#555'
                            }),
                            dcc.Input(
                                id='grad-year', 
                                placeholder='YYYY (Optional)', 
                                type='text', 
                                style={
                                    'width': '100%',
                                    'padding': '12px 15px',
                                    'borderRadius': '8px',
                                    'border': '1px solid #ddd',
                                    'fontSize': '16px',
                                    'transition': 'border-color 0.3s',
                                    'boxSizing': 'border-box'
                                }
                            )
                        ], style={'flex': '1', 'minWidth': '150px'}),
                    ], style={
                        'display': 'flex',
                        'gap': '15px',
                        'marginBottom': '15px',
                        'flexWrap': 'wrap'
                    }),
                ], style={
                    'backgroundColor': 'white',
                    'padding': '25px',
                    'borderRadius': '15px',
                    'boxShadow': '0 6px 16px rgba(0,0,0,0.05)',
                    'border': '1px solid #eee',
                    'marginBottom': '30px'
                }),
                
                # Manual mode indicator with improved styling
                html.Div("🔎 Manual Mode", id="manual-mode-label", style={
                    'textAlign': 'center',
                    'color': '#0a66c2',
                    'fontWeight': 'bold',
                    'marginTop': '10px',
                    'marginBottom': '20px',
                    'backgroundColor': 'rgba(10, 102, 194, 0.1)',
                    'padding': '10px 20px',
                    'borderRadius': '20px',
                    'display': 'none',
                    'maxWidth': '200px',
                    'margin': '0 auto'
                }),
            ], style={
                'marginBottom': '30px',
                'padding': '0 20px'
            }),
            
            # Advanced Settings with pill-style button
            html.Div([
                html.Button(html.Span([
                    html.I(className='fas fa-cog', style={'marginRight': '10px'}),
                    "Advanced Settings"
                ]), id="advanced-settings-button", className='floating-btn settings-btn', style={
                    'fontSize': '16px',
                    'padding': '12px 24px',
                    'borderRadius': '30px',  # More rounded pill-style
                    'backgroundColor': '#20B2AA',
                    'color': '#ffffff',
                    'border': 'none',
                    'boxShadow': '0 4px 12px rgba(32, 178, 170, 0.3)',
                    'cursor': 'pointer',
                    'transition': 'all 0.3s ease',
                    'fontWeight': '500',
                    'marginBottom': '30px'
                })
            ], style={'textAlign': 'center'}),
            
            # Modal with improved visual design
            html.Div([
                html.Div([
                    html.Div([
                        html.Div([
                            html.I(className='fas fa-sliders-h', style={
                                'fontSize': '24px', 
                                'color': '#0072B2',
                                'marginRight': '15px'
                            }),
                            html.H3("Advanced Settings", style={
                                'color': '#0072B2',
                                'fontWeight': '600',
                                'margin': '0'
                            })
                        ], style={
                            'display': 'flex',
                            'alignItems': 'center',
                            'marginBottom': '25px'
                        }),
                        
                        # Settings content with improved sliders
                        html.Div([
                            html.Label("Cosine Confidence Threshold (0.30 - 0.90)", style={
                                'fontWeight': '500', 
                                'fontSize': '16px',
                                'marginBottom': '10px',
                                'color': '#333'
                            }),
                            html.Div("Higher values require closer name matches", style={
                                'fontSize': '14px',
                                'color': '#666',
                                'marginBottom': '15px'
                            }),
                            dcc.Slider(
                                id='cosine-threshold',
                                min=0.3, max=0.9, step=0.05, value=0.4,
                                marks={i: f"{i:.2f}" for i in [0.3, 0.4, 0.5, 0.6, 0.75, 0.9]},
                                tooltip={"placement": "bottom", "always_visible": True},
                                included=True,
                                className='custom-slider'
                            ),
                        ], style={
                            'marginBottom': '30px',
                            'backgroundColor': 'rgba(0, 114, 178, 0.05)',
                            'padding': '20px',
                            'borderRadius': '12px'
                        }),
                        
                        html.Div([
                            html.Label("Limit number of people to search", style={
                                'fontWeight': '500',
                                'fontSize': '16px',
                                'marginBottom': '10px',
                                'color': '#333'
                            }),
                            html.Div("Useful for testing with large datasets", style={
                                'fontSize': '14px',
                                'color': '#666',
                                'marginBottom': '15px'
                            }),
                            dcc.Input(
                                id='name-limit',
                                type='number',
                                min=1,
                                step=1,
                                placeholder='Leave blank for all',
                                style={
                                    'width': '100%',
                                    'maxWidth': '300px',
                                    'textAlign': 'center',
                                    'padding': '12px 15px',
                                    'border': '1px solid #ccc',
                                    'borderRadius': '8px',
                                    'fontSize': '16px'
                                }
                            )
                        ], style={
                            'marginBottom': '30px',
                            'backgroundColor': 'rgba(86, 86, 90, 0.05)',
                            'padding': '20px',
                            'borderRadius': '12px'
                        }),
                        
                        html.Div([
                            html.Label("Fuzzy Matching Threshold (0.60 - 0.95)", style={
                                'fontWeight': '500', 
                                'fontSize': '16px',
                                'marginBottom': '10px',
                                'color': '#333'
                            }),
                            html.Div("Controls tolerance for name variations and typos", style={
                                'fontSize': '14px',
                                'color': '#666',
                                'marginBottom': '15px'
                            }),
                            dcc.Slider(
                                id='fuzzy-threshold',
                                min=0.6, max=0.95, step=0.05, value=0.75,
                                marks={i: f"{i:.2f}" for i in [0.6, 0.7, 0.75, 0.85, 0.95]},
                                tooltip={"placement": "bottom", "always_visible": True},
                                included=True,
                                className='custom-slider'
                            ),
                        ], style={
                            'marginBottom': '30px',
                            'backgroundColor': 'rgba(0, 114, 178, 0.05)',
                            'padding': '20px',
                            'borderRadius': '12px'
                        }),
                        
                        # Close button with improved styling
                        html.Button("Save & Close", id="close-advanced-settings", style={
                            'backgroundColor': '#0072B2',
                            'color': 'white',
                            'border': 'none',
                            'padding': '12px 30px',
                            'borderRadius': '30px',
                            'cursor': 'pointer',
                            'fontWeight': '500',
                            'fontSize': '16px',
                            'boxShadow': '0 4px 12px rgba(0,114,178,0.2)',
                            'transition': 'all 0.3s ease'
                        })
                    ], style={
                        'backgroundColor': 'white',
                        'padding': '35px',
                        'borderRadius': '20px',
                        'maxWidth': '650px',
                        'margin': '0 auto',
                        'boxShadow': '0 10px 30px rgba(0,0,0,0.2)',
                        'position': 'relative',
                        'zIndex': '1001'
                    })
                ], style={
                    'position': 'fixed',
                    'top': '0',
                    'left': '0',
                    'width': '100%',
                    'height': '100%',
                    'backgroundColor': 'rgba(0,0,0,0.6)',
                    'backdropFilter': 'blur(5px)',
                    'display': 'flex',
                    'alignItems': 'center',
                    'justifyContent': 'center',
                    'zIndex': '1000',
                    'display': 'none'  # Hidden by default
                }, id='advanced-settings-modal')
            ]),
            
            # Start search button with improved visual prominence
            html.Div([
                html.Button(html.Span([
                    html.I(className='fas fa-play', style={'marginRight': '10px'}),
                    "Start Search"
                ]), id="search-button", className='floating-btn start-btn', style={
                    'fontSize': '20px',  # Larger font
                    'padding': '18px 45px',  # Larger padding
                    'borderRadius': '16px',
                    'backgroundColor': '#0072B2',
                    'color': '#ffffff',
                    'border': 'none',
                    'boxShadow': '0 8px 20px rgba(0,114,178,0.3)',  # More prominent shadow
                    'cursor': 'pointer',
                    'transition': 'all 0.3s ease',
                    'fontWeight': '600',
                    'marginTop': '10px',
                    'marginBottom': '30px',
                    'position': 'relative',  # For pseudo-element effects
                    'overflow': 'hidden'  # For pulse effect
                }),
            ], style={'textAlign': 'center'}),
            
            # Status indicators with improved styling
            html.Div(id='upload-status', style={
                'marginBottom': '10px', 
                'textAlign': 'center', 
                'fontSize': '16px',
                'fontWeight': '500',
                'color': '#0072B2',
                'padding': '10px',
                'borderRadius': '8px'
            }),
            html.Div(id="search-status", style={
                "marginTop": "10px", 
                "fontWeight": "500", 
                'textAlign': 'center', 
                'fontSize': '16px',
                'color': '#333'
            }),
            html.Div(id="eta-stats", style={
                'marginTop': '10px', 
                'textAlign': 'center', 
                'color': '#666',
                'fontSize': '14px'
            }),
            dcc.Interval(id="interval", interval=1000, n_intervals=0, disabled=False),
        ], style={
            'width': '100%',
            'maxWidth': '850px',
            'margin': '0 auto',
            'textAlign': 'center',
            'padding': '35px',
            'backgroundColor': '#ffffff',
            'borderRadius': '25px',
            'boxShadow': '0 15px 35px rgba(0, 0, 0, 0.1)'
        }),

        # Progress section with improved visual design
        html.Div(id="progress-container", children=[
            html.Div([
                html.I(className='fas fa-tasks', style={
                    'fontSize': '24px',
                    'color': '#0072B2',
                    'marginRight': '10px'
                }),
                html.Span("Progress", style={
                    'fontSize': '18px',
                    'fontWeight': '600',
                    'color': '#0072B2'
                })
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'center',
                'marginTop': '40px',
                'marginBottom': '20px'
            }),
            
            # Enhanced progress bar
            html.Div(style={
                'width': '100%',
                'backgroundColor': '#e8e8e8',
                'borderRadius': '12px',
                'maxWidth': '550px',
                'margin': '0 auto',
                'overflow': 'hidden',
                'boxShadow': 'inset 0 2px 4px rgba(0,0,0,0.1)'
            }, children=[
                html.Div(id="progress-bar", style={
                    "height": "35px",
                    "width": "0%",
                    "backgroundColor": "#0a66c2",
                    "color": "white",
                    "textAlign": "center",
                    "lineHeight": "35px",
                    'borderRadius': '12px',
                    'transition': 'width 0.5s ease-in-out',
                    'fontWeight': '600',
                    'fontSize': '16px',
                    'boxShadow': '0 2px 5px rgba(10, 102, 194, 0.3)'
                })
            ]),
            
            # Stop button with improved positioning
            html.Div([
                html.Button(html.Span([
                    html.I(className='fas fa-stop', style={'marginRight': '10px'}),
                    "Stop Search"
                ]), id="stop-button", className='floating-btn stop-btn', style={
                    'fontSize': '18px',
                    'padding': '14px 30px',
                    'borderRadius': '30px',  # Pill style button
                    'backgroundColor': '#D55E00',
                    'color': '#ffffff',
                    'border': 'none',
                    'boxShadow': '0 6px 16px rgba(213,94,0,0.3)',
                    'cursor': 'pointer',
                    'transition': 'all 0.3s ease',
                    'fontWeight': '600',
                    'marginTop': '25px'
                })
            ], style={'textAlign': 'center'})
        ], style={
            'backgroundColor': 'white',
            'padding': '30px',
            'borderRadius': '25px',
            'marginTop': '30px',
            'boxShadow': '0 15px 35px rgba(0, 0, 0, 0.1)',
            'maxWidth': '850px',
            'margin': '30px auto'
        }),
    ]),
    
    # Results anchor for smooth scrolling
    html.Div(id="results-anchor"),

    # Results table with improved styling
    html.Div([
        html.Div([
            html.I(className='fas fa-table', style={
                'fontSize': '24px',
                'color': '#0072B2',
                'marginRight': '15px'
            }),
            html.H2("Results", style={
                'color': '#0072B2',
                'fontWeight': '600',
                'margin': '0',
                'fontSize': '24px'
            })
        ], style={
            'display': 'flex',
            'alignItems': 'center',
            'marginBottom': '20px'
        }),
        
        # Enhanced table
        dash_table.DataTable(
            id='results-table',
            columns=[
                {"name": "No.", "id": "No."},
                {"name": "First Name", "id": "First Name"},
                {"name": "Last Name", "id": "Last Name"},
                {"name": "University", "id": "University"},
                {"name": "Graduation Year", "id": "Graduation Year"},
                {"name": "LinkedIn Title", "id": "LinkedIn Title"},
                {"name": "LinkedIn URL", "id": "LinkedIn URL", "presentation": "markdown"},
                {"name": "Status", "id": "Status"},
                {"name": "Confidence Score", "id": "Score"},
                {"name": "Location (Estimated)", "id": "Location (Estimated)"},
                {"name": "Income (Estimated)", "id": "Income (Estimated)"},
            ],
            data=[],
            page_action='native',
            page_size=10,
            sort_action='native',
            filter_action='native',
            style_table={'overflowX': 'auto'},
            style_header={
                'backgroundColor': '#0072B2',
                'color': 'white',
                'fontWeight': 'bold',
                'textAlign': 'center',
                'padding': '12px 15px',
                'borderTopLeftRadius': '10px',
                'borderTopRightRadius': '10px'
            },
            style_cell={
                'textAlign': 'left',
                'padding': '12px 15px',
                'fontFamily': 'Roboto, sans-serif',
                'fontSize': '15px'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgba(0, 114, 178, 0.05)'
                },
                {
                    'if': {'column_id': 'Status', 'filter_query': '{Status} contains "✅"'},
                    'color': '#009E73',
                    'fontWeight': 'bold'
                },
                {
                    'if': {'column_id': 'Status', 'filter_query': '{Status} contains "❌"'},
                    'color': '#D55E00',
                    'fontWeight': 'bold'
                }
            ],
            markdown_options={"html": True}
        ),
        dcc.Download(id="download-dataframe-csv")
    ], style={
        'marginTop': '40px',
        'padding': '35px',
        'backgroundColor': '#ffffff',
        'borderRadius': '25px',
        'boxShadow': '0 15px 35px rgba(0, 0, 0, 0.1)',
        'maxWidth': '1250px',
        'marginLeft': 'auto',
        'marginRight': 'auto',
        'marginBottom': '50px'
    }),
    
    # Footer with attribution
    html.Footer([
        html.Div("LinkedIn Profile Finder", style={
            'fontSize': '16px',
            'fontWeight': '600',
            'color': '#0072B2',
            'marginBottom': '5px'
        }),
        html.Div("Powered by Rutgers University, Newark", style={
            'fontSize': '14px',
            'color': '#56565A'
        })
    ], style={
        'textAlign': 'center',
        'marginTop': '30px',
        'marginBottom': '20px',
        'padding': '20px'
    })
], style={
    'padding': '50px 20px',
    'fontFamily': 'Roboto, sans-serif',
    'backgroundColor': '#f0f2f5',  # Slightly adjusted background for better contrast
    'minHeight': '100vh'
}),

# Add confirmation dialogs for stop and restart buttons
app.clientside_callback(
    """
    function(stop_clicks, restart_clicks) {
        const triggered = dash_clientside.callback_context.triggered.map(t => t.prop_id);
        
        // Initialize empty result object
        let result = {
            stop: false,
            restart: false
        };
        
        if (triggered.includes('stop-button.n_clicks') && stop_clicks) {
            result.stop = window.confirm("Are you sure you want to stop the current search?");
        }
        
        if (triggered.includes('restart-button.n_clicks') && restart_clicks) {
            result.restart = window.confirm("Are you sure you want to restart? This will clear all current results.");
        }
        
        return [result.stop, result.restart];
    }
    """,
    [Output("stop-button", "data-confirmed", allow_duplicate=True),
     Output("restart-button", "data-confirmed", allow_duplicate=True)],
    [Input("stop-button", "n_clicks"),
     Input("restart-button", "n_clicks")],
    prevent_initial_call=True
)

app.clientside_callback(
    """
    function(n_clicks) {
        const anchor = document.getElementById('results-anchor');
        if (anchor) {
            anchor.scrollIntoView({ behavior: 'smooth' });
        }
        return '';
    }
    """,
    Output("results-anchor", "children"),
    Input("search-button", "n_clicks"),
    prevent_initial_call=True
)

app.index_string = app.index_string.replace('</head>', '''<style>
.floating-btn:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 10px 20px rgba(0,0,0,0.15);
}

/* Input focus effects */
input:focus {
    border-color: #0072B2 !important;
    box-shadow: 0 0 0 2px rgba(0,114,178,0.2) !important;
    outline: none !important;
}

/* Custom slider styling */
.custom-slider .rc-slider-track {
    background-color: #0072B2;
}
.custom-slider .rc-slider-handle {
    border-color: #0072B2;
    background-color: white;
    box-shadow: 0 0 5px rgba(0,114,178,0.3);
}
.custom-slider .rc-slider-handle:hover {
    border-color: #0072B2;
}
.custom-slider .rc-slider-handle:active {
    border-color: #0072B2;
    box-shadow: 0 0 5px rgba(0,114,178,0.5);
}

/* Pulse animation for the start button */
@keyframes pulse {
    0% {
        box-shadow: 0 0 0 0 rgba(0,114,178,0.4);
    }
    70% {
        box-shadow: 0 0 0 10px rgba(0,114,178,0);
    }
    100% {
        box-shadow: 0 0 0 0 rgba(0,114,178,0);
    }
}
.start-btn:hover {
    animation: pulse 1.5s infinite;
}

/* Better mobile responsiveness */
@media (max-width: 768px) {
    .floating-btn {
        width: 100% !important;
        margin-bottom: 10px !important;
    }
}
</style>
</head>''')



# Custom index_string with updated CSS for a modern and responsive design
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>LinkedIn Search Dashboard</title>
        {%favicon%}
        {%css%}
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
        <style>
            /* Base Styles */
            body {
                font-family: 'Roboto', sans-serif;
                background-color: #f5f7fa;
                margin: 0;
                padding: 0;
                color: #333;
            }
            .app-container {
                max-width: 1500px;
                margin: auto;
                padding: 20px;
                position: relative;
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
            }
            .app-title {
                color: #0072B2;
                font-size: 40px;
                font-weight: 700;
                margin: 0;
            }
            /* Card Styles */
            .card {
                background: #fff;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            .section-title {
                font-size: 30px;
                font-weight: 500;
                color: #333;
                margin-bottom: 15px;
            }
            /* Input & Upload Styles */
            .upload-area {
                border: 2px dashed #ccc;
                border-radius: 8px;
                padding: 20px;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            .upload-area:hover {
                border-color: #0072b1;
                background-color: #f0f8ff;
            }
            .upload-content {
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 8px;
            }
            .upload-icon {
                font-size: 24px;
                color: #0072b1;
            }
            .upload-text {
                color: #6c757d;
            }
            .status-message {
                margin-top: 8px;
                font-size: 18px;
                font-weight: 500;
                min-height: 20px;
            }
            .input-container {
                display: grid;
                gap: 10px;
                margin-bottom: 15px;
            }
            .input-group {
                display: flex;
                flex-direction: column;
            }
            .input-group label {
                margin-bottom: 6px;
                font-size: 20px;
                font-weight: 500;
            }
            .input-field {
                padding: 14px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                font-size: 14px;
                transition: border-color 0.3s;
            }
            .input-field:focus {
                border-color: #0072b1;
                outline: none;
            }
            .action-container {
                margin-top: 15px;
            }
            /* Button Styles */
            .btn {
                padding: 12px 16px;
                border: none;
                border-radius: 4px;
                font-size: 18px;
                font-weight: 500;
                cursor: pointer;
                transition: background-color 0.3s;
            }
            .btn-primary {
                background-color: #0072b1;
                color: #fff;
            }
            .btn-primary:hover {
                background-color: #005b8e;
            }
            .btn-secondary {
                background-color: #e2e2e2;
                color: #333;
            }
            .btn-secondary:hover {
                background-color: #e2e2e2;
            }
            .btn-download {
                background-color: #28a745;
                color: white;
            }
            .btn-download:hover {
                background-color: #218838;
            }
            .download-container {
                margin-top: 15px;
                display: flex;
                justify-content: flex-end;
            }
            /* Layout Styles */
            .flex-container {
                display: flex;
                gap: 20px;
            }
            .left-panel {
                flex-shrink: 0;
            }
            .right-panel {
                flex-grow: 1;
            }
            /* Responsive Styles */
            @media screen and (max-width: 768px) {
                .flex-container {
                    flex-direction: column;
                }
                .left-panel, .right-panel {
                    width: 100%;
                    padding: 10px;
                }
            }
            .thread-info {
                position: absolute;
                bottom: 10px;
                right: 20px;
                color: #0072b1;
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# ------------------------------------------------------------
# 🔄 Dash Callback Context
# ------------------------------------------------------------

ctx = dash.callback_context


@app.callback(
    Output("upload-status", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename")
)
def parse_upload(contents, filename):
    global uploaded_people
    if not contents or not filename:
        return ""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    encoding = chardet.detect(decoded)['encoding']
    try:
        df = pd.read_csv(io.StringIO(decoded.decode(encoding)))
        uploaded_people = df.to_dict(orient='records')
        logger.info(f"📥 Uploaded {filename} with {len(uploaded_people)} rows.")
        return f"✅ Uploaded {filename} with {len(uploaded_people)} row(s)."
    except Exception as e:
        err = f"❌ Upload failed: {e}"
        logger.error(err)
        return err

@app.callback(
    Output("results-table", "data", allow_duplicate=True),
    Output("progress-bar", "style", allow_duplicate=True),
    Output("progress-bar", "children", allow_duplicate=True),
    Output("search-status", "children", allow_duplicate=True),
    Output("eta-stats", "children", allow_duplicate=True),
    Output("interval", "disabled", allow_duplicate=True),
    Output("manual-mode-label", "style", allow_duplicate=True),
    Input("search-button", "n_clicks"),
    Input("interval", "n_intervals"),
    State("first-name", "value"),
    State("last-name", "value"),
    State("university", "value"),
    State("grad-year", "value"),
    State("cosine-threshold", "value"),
    State("fuzzy-threshold", "value"),
    State("name-limit", "value"),
    prevent_initial_call=True
)
def update_table(search_clicks, interval, fname, lname, university, grad_year, cosine_val, fuzzy_val, name_limit):
    global scraping_active, completed_results, total_tasks, start_scrape_time, final_table_ready, finalized_table_data

    ctx = dash.callback_context
    triggered = ctx.triggered_id

    if triggered == "search-button":
        final_table_ready = False
        finalized_table_data = []

        manual_mode_style = {"display": "none"}

        if fname and lname and university:
            person = {
                "First Name": fname,
                "Last Name": lname,
                "University": university
            }
            if grad_year:
                person["Graduation Year"] = grad_year
            people = [person]
            manual_mode_style = {"display": "block"}  # 👈 Enable the label if manually entered

        elif uploaded_people:
            people = uploaded_people
        else:
            return no_update, no_update, no_update, "⚠️ No data provided.", no_update, True, {"display": "none"}

        if name_limit and isinstance(name_limit, int) and name_limit > 0:
            people = people[:name_limit]

        start_scrape_time = time.time()
        enqueue_tasks(people, cosine_val, fuzzy_val)
        return [], {"width": "0%"}, "0%", "🔎 Searching...", "ETA calculating...", False, manual_mode_style

    elif triggered == "interval" and scraping_active:
        percent = int((len(completed_results) / total_tasks) * 100)
        elapsed = time.time() - start_scrape_time
        speed = len(completed_results) / elapsed if elapsed > 0 else 0
        remaining = total_tasks - len(completed_results)
        eta = remaining / speed if speed > 0 else 0

        results = completed_results.copy()
        for i, r in enumerate(results):
            r["No."] = i + 1
            r.setdefault("Income (Estimated)", "Unknown")
            r.setdefault("Graduation Year", "N/A")
            r.setdefault("Location (Estimated)", "Unknown")
            r.setdefault("LinkedIn URL", "")
            r.setdefault("Score", "N/A")

        if last_result_time and time.time() - last_result_time > 30:
            logger.warning("⏳ No progress for 30+ seconds — recommend restarting.")
            return (
                results,
                {"width": f"{percent}%", "height": "30px", "backgroundColor": "#ffc107",
                 "color": "#000", "textAlign": "center", "lineHeight": "30px",
                 'borderRadius': '10px', 'transition': 'width 0.5s ease-in-out', 'fontWeight': '600'},
                f"{percent}%",
                "⚠️ No progress detected. Please consider pressing Restart.",
                f"⏱ ETA: stalled | Speed: {speed:.2f}/sec",
                False,
                {"display": "none"}
            )

        return (
            results,
            {"width": f"{percent}%", "height": "30px", "backgroundColor": "#0a66c2",
             "color": "white", "textAlign": "center", "lineHeight": "30px",
             'borderRadius': '10px', 'transition': 'width 0.5s ease-in-out', 'fontWeight': '600'},
            f"{percent}%",
            "⏳ Running...",
            f"⏱ ETA: {int(eta)}s | Speed: {speed:.2f}/sec",
            False,
            {"display": "none"}
        )

    elif not scraping_active and completed_results:

        if not final_table_ready:
            logger.info("📊 Finalizing income data before displaying table...")
            finalized_table_data = finalize_income_estimates(completed_results.copy())
            for i, r in enumerate(finalized_table_data):
                r["No."] = i + 1
                r.setdefault("Location (Estimated)", "Unknown")
                r.setdefault("LinkedIn URL", "")
                r.setdefault("Score", "N/A")
                r.setdefault("Status", "❌ No Match")
            final_table_ready = True

        return finalized_table_data, {
            "width": "100%", "height": "30px", "backgroundColor": "#0a66c2",
            "color": "white", "textAlign": "center", "lineHeight": "30px",
            'borderRadius': '10px', 'transition': 'width 0.5s ease-in-out', 'fontWeight': '600'
        }, "100%", "✅ Search complete.", "Done.", True, {"display": "none"}

    return no_update, no_update, no_update, no_update, no_update, no_update, {"display": "none"}


@app.callback(
    Output("results-table", "data", allow_duplicate=True),
    Output("progress-bar", "style", allow_duplicate=True),
    Output("progress-bar", "children", allow_duplicate=True),
    Output("search-status", "children", allow_duplicate=True),
    Output("interval", "disabled", allow_duplicate=True),
    Output("first-name", "value", allow_duplicate=True),
    Output("last-name", "value", allow_duplicate=True),
    Output("university", "value", allow_duplicate=True),
    Output("grad-year", "value", allow_duplicate=True),
    Input("stop-button", "n_clicks"),
    Input("restart-button", "n_clicks"),
    State("stop-button", "data-confirmed"),
    State("restart-button", "data-confirmed"),
    prevent_initial_call=True
)
def handle_stop_restart(stop_clicks, restart_clicks, stop_confirmed, restart_confirmed):
    global scraping_active, completed_results, total_tasks, pending_tasks

    triggered_id = ctx.triggered_id

    if triggered_id == "stop-button" and stop_confirmed:
        scraping_active = False
        logger.info("🛑 Search manually stopped.")

        # ❌ Cancel any queued (not yet running) tasks
        for task in pending_tasks:
            if not task.done():
                task.cancel()

        return no_update, no_update, no_update, "🛑 Search stopped.", True, no_update, no_update, no_update, no_update

    elif triggered_id == "restart-button" and restart_confirmed:
        completed_results.clear()
        total_tasks = 0
        pending_tasks.clear()
        logger.info("🔁 Search restarted.")

        return [], {"width": "0%"}, "0%", "🔁 Ready for new search.", True, "", "", "", ""

    return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update


@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("download-button", "n_clicks"),
    prevent_initial_call=True
)
def download_csv(n):
    if not completed_results:
        return no_update
    df = pd.DataFrame(completed_results)
    return dcc.send_data_frame(df.to_csv, "linkedin_results.csv", index=False)

# Callback to open and close the advanced settings modal
@app.callback(
    Output("advanced-settings-modal", "style"),
    [Input("advanced-settings-button", "n_clicks"),
     Input("close-advanced-settings", "n_clicks")],
    [State("advanced-settings-modal", "style")],
    prevent_initial_call=True
)
def toggle_advanced_settings_modal(open_clicks, close_clicks, current_style):
    ctx = dash.callback_context
    triggered_id = ctx.triggered_id
    
    # Create a copy of the current style to modify
    new_style = dict(current_style) if current_style else dict()
    
    if triggered_id == "advanced-settings-button":
        new_style["display"] = "flex"  # Show the modal
    elif triggered_id == "close-advanced-settings":
        new_style["display"] = "none"  # Hide the modal
        
    return new_style

def open_browser():
    url = "http://127.0.0.1:8050/"
    try:
        webbrowser.open_new(url)
    except Exception as e:
        logger.warning(f"Could not open browser automatically: {e}")

import os

if __name__ == "__main__":
    if os.environ.get("RENDER") != "true":
        threading.Timer(1.5, open_browser).start()

    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port)

