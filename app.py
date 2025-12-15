from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import sqlite3, hashlib, datetime, re, json
from scraper import scrape_all
from scraper import GROQ_API_KEY

from urllib.parse import unquote, quote_plus, urlencode, urlparse
import html



# ✨ Chatbot Additions (ONLY ADDED — NOTHING MODIFIED)
import os
import openai


app = Flask(__name__)
app.secret_key = "super_secure_secret_key_123"

# No permanent cookies — everything stored in RAM only
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_REFRESH_EACH_REQUEST"] = False

latest_products = []
active_sessions = set()  # 🧠 Keeps track of valid sessions in RAM

# Set OpenAI Key if exists (doesn’t break if missing)
openai.api_key = os.getenv("OPENAI_API_KEY")


# ---------------- DATABASE ----------------
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL
                )''')
    conn.commit()
    conn.close()

init_db()


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# ---------------- AUTH ----------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = hash_password(request.form["password"])

        if not username or not password:
            flash("⚠️ Please fill all fields!", "danger")
            return render_template("register.html")

        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            flash("✅ Registration successful! Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("⚠️ Username already exists!", "danger")
        conn.close()

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    session.clear()
    if request.method == "POST":
        username = request.form["username"].strip()
        password = hash_password(request.form["password"])

        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = c.fetchone()
        conn.close()

        if user:
            session["user"] = username
            active_sessions.add(username)
            flash(f"🎉 Welcome, {username}!", "success")
            return redirect(url_for("home"))
        else:
            flash("❌ Invalid username or password", "danger")

    return render_template("login.html")


@app.route("/logout")
def logout():
    username = session.get("user")
    if username in active_sessions:
        active_sessions.remove(username)
    session.clear()
    flash("👋 Logged out successfully!", "info")
    return redirect(url_for("login"))


# ---------------- SECURITY ----------------
@app.before_request
def enforce_login():
    public_routes = ["login", "register", "static", "check_session"]
    if request.endpoint not in public_routes:
        username = session.get("user")
        if not username or username not in active_sessions:
            session.clear()
            return redirect(url_for("login"))


@app.route("/check_session")
def check_session():
    username = session.get("user")
    return jsonify({"active": username in active_sessions})


# ---------------- Helper functions ----------------

def safe_url(u: str) -> str:
    try:
        parsed = urlparse(u)
        if parsed.scheme in ("http", "https"):
            return u
    except:
        pass
    return None


STORE_SEARCH_TEMPLATES = {
    "Amazon": "https://www.amazon.in/s?k={q}",
    "Flipkart": "https://www.flipkart.com/search?q={q}",
    "Snapdeal": "https://www.snapdeal.com/search?keyword={q}",
    "VleBazaar": "https://www.google.com/search?q={q}+vlebazaar",
    "Default": "https://www.google.com/search?q={q}"
}


def build_store_search_url(store_name: str, product_name: str):
    tpl = STORE_SEARCH_TEMPLATES.get(store_name, STORE_SEARCH_TEMPLATES["Default"])
    return tpl.format(q=quote_plus(product_name))


def append_specs_to_url(url: str, specs: dict):
    try:
        parsed = urlparse(url)
        base = parsed.scheme + "://" + parsed.netloc + parsed.path
        existing_q = parsed.query
        specs_enc = json.dumps(specs, ensure_ascii=False)
        new_q = existing_q + "&specs=" + specs_enc if existing_q else "specs=" + specs_enc
        rebuilt = base + "?" + new_q
        if parsed.fragment:
            rebuilt += "#" + parsed.fragment
        return rebuilt
    except:
        sep = "&" if "?" in url else "?"
        return url + sep + urlencode({"specs": json.dumps(specs, ensure_ascii=False)})


def enhance_products_for_display(products):
    enhanced = []
    for p in products:
        product = dict(p)
        pname = product.get("name") or product.get("title") or "Unknown Product"

        price_raw = product.get("price", "")
        try:
            price_val = float(re.sub(r"[^\d.]", "", str(price_raw))) if price_raw else 0.0
        except:
            price_val = 0.0
        product["price_value"] = price_val

        specs = product.get("specs")
        if not isinstance(specs, dict):
            specs = {
                "Category": product.get("category", "Electronics"),
                "Functions": product.get("functions", "401"),
                "Display Type": product.get("display", "Two Line"),
                "Power Source": product.get("power", "Solar + Battery"),
                "Notes": product.get("notes", "")
            }
        product["specs"] = specs

        if product.get("stores") and isinstance(product["stores"], list):
            final_stores = []
            for s in product.get("stores"):
                s_name = s.get("name", "Store")
                s_url = s.get("url") or build_store_search_url(s_name, pname)
                s_url_safe = safe_url(s_url) or build_store_search_url(s_name, pname)
                s_url_with_specs = append_specs_to_url(s_url_safe, specs)
                s_price = s.get("price", product.get("price"))
                final_stores.append({
                    "name": s_name,
                    "price": s_price,
                    "url": s_url_with_specs,
                })
            product["stores"] = final_stores
        else:
            stores = []
            base_price = price_val
            store_order = ["Amazon", "VleBazaar", "Snapdeal", "Flipkart", "Local"]
            adjustments = [0, -48, -10, -30, +2]
            for i, sname in enumerate(store_order):
                if i == 0 and product.get("url"):
                    s_url = product["url"]
                else:
                    s_url = build_store_search_url(sname, pname)
                s_url_safe = safe_url(s_url) or build_store_search_url(sname, pname)
                if base_price:
                    try:
                        s_price = "{:.2f}".format(max(1.0, base_price + adjustments[i]))
                    except:
                        s_price = str(base_price)
                else:
                    s_price = ""
                stores.append({
                    "name": sname,
                    "price": s_price,
                    "url": append_specs_to_url(s_url_safe, specs),
                })
            product["stores"] = stores

        if not product.get("id"):
            product["id"] = str(abs(hash(pname + str(price_val))))[:12]

        enhanced.append(product)
    return enhanced



# ---------------- MAIN ROUTES ----------------

@app.route("/")
def home():
    return render_template("index.html", products=[], query="", filters={})


@app.route("/search")
def search():

    query = request.args.get("query", "").strip()
    brand_filter = request.args.get("brand", "").strip().lower()
    price_sort = request.args.get("priceSort", "")
    min_price = request.args.get("minPrice", "")
    max_price = request.args.get("maxPrice", "")

    if not query:
        return render_template("index.html", products=[], query="", filters={})

    global latest_products
    raw_products = scrape_all(query)
    # Clear chatbot local history on new search
    session["reset_chat"] = True

    latest_products = enhance_products_for_display(raw_products)

    def extract_price(p):
        price = re.sub(r"[^\d.]", "", str(p.get("price", p.get("price_value", "0"))))
        return float(price) if price else 0.0

    filtered_products = []
    for p in latest_products:
        price_value = extract_price(p)
        if min_price:
            try:
                if price_value < float(min_price):
                    continue
            except:
                pass
        if max_price:
            try:
                if price_value > float(max_price):
                    continue
            except:
                pass
        if brand_filter and brand_filter not in p.get("name", "").lower():
            continue
        filtered_products.append(p)

    if price_sort == "low-high":
        filtered_products.sort(key=lambda x: extract_price(x))
    elif price_sort == "high-low":
        filtered_products.sort(key=lambda x: extract_price(x), reverse=True)



    # **********************************************************************
    # 🔥🔥🔥 ADD THIS BLOCK (SPARKLINE + TREND + HISTORY + SAVINGS) 🔥🔥🔥
    # **********************************************************************

    import random

    for p in filtered_products:

        try:
            main_price = float(re.sub(r"[^\d.]", "", str(p.get("price", p.get("price_value", "0")))))
        except:
            main_price = 0.0

        history = []

        # Base price used to build the 7-day history
        base = main_price if main_price > 0 else random.uniform(100, 500)

        # Variation rules
        if base < 1000:
            min_step, max_step = 5, 10
        elif base <= 10000:
            min_step, max_step = 40, 50
        else:
            min_step, max_step = 80, 100

        for i in range(7):
            direction = -1 if random.random() < 0.5 else 1
            step = random.uniform(min_step, max_step)
            change = direction * step

            base = max(1, base + change)
            history.append(round(base, 2))

        p["history_prices"] = history

        if history[-1] > history[-2]:
            p["trend"] = "up"
        elif history[-1] < history[-2]:
            p["trend"] = "down"
        else:
            p["trend"] = "equal"

        prices = []
        for s in p.get("stores", []):
            try:
                pr = float(re.sub(r"[^\d.]", "", str(s.get("price", ""))))
                prices.append(pr)
            except:
                pass

        if prices:
            p["min_price"] = min(prices)
            p["max_price"] = max(prices)
            p["saving"] = round(p["max_price"] - p["min_price"], 2)
        else:
            p["min_price"] = main_price
            p["max_price"] = main_price
            p["saving"] = 0

    # **********************************************************************
    # 🔥🔥🔥 END OF ADDED BLOCK 🔥🔥🔥
    # **********************************************************************



    filters = {
        "brand": brand_filter,
        "priceSort": price_sort,
        "minPrice": min_price,
        "maxPrice": max_price,
    }

    return render_template("index.html", products=filtered_products, query=query, filters=filters)



@app.route("/product/<product_id>")
def product_page(product_id):
    global latest_products
    product = next((p for p in latest_products if str(p["id"]) == str(product_id)), None)
    if not product:
        return render_template("error.html", message="⚠️ Product not found. Please search again.")
    return render_template("product.html", product=product)



@app.route("/go")
def go_to_store():

    direct_url = request.args.get("url")
    product_id = request.args.get("product_id")
    store_name = request.args.get("store")

    if direct_url:
        safe = safe_url(unquote(direct_url))
        if not safe:
            return "Invalid store link", 400
        return redirect(safe)

    if product_id and store_name:
        global latest_products
        product = next((p for p in latest_products if str(p["id"]) == str(product_id)), None)
        if not product:
            return "Invalid product id", 400

        store = next((s for s in product.get("stores", []) if s.get("name", "").lower() == store_name.lower()), None)
        if not store:
            stores = product.get("stores", [])
            if not stores:
                return "Store not found for product", 404
            store = stores[0]

        dest = store.get("url") or store.get("link")
        if not dest:
            dest = build_store_search_url(store.get("name", "") + " " + product.get("name", ""))

        dest_safe = safe_url(dest)
        if not dest_safe:
            dest_safe = build_store_search_url(store.get("name", "") + " " + product.get("name", ""))

        return redirect(dest_safe)

    return "Invalid store link", 400

# -------------------------------------------------------------------
# 🔥 Product-aware Chatbot (Groq primary )
#    - Supports voice input (base64 WAV/ogg) via "audio" field
#    - Uses latest_products global to build PRODUCT_CONTEXT
#    - Detects lowest/highest product automatically
# -------------------------------------------------------------------
import base64
import traceback
from flask import request, jsonify

def _to_num_price(x):
    try:
        s = str(x or "")
        s = s.replace("₹", "").replace(",", "").strip()
        s = re.sub(r"[^\d.]", "", s)
        return float(s) if s else 0.0
    except:
        return 0.0

def _build_product_context_for_prompt(latest_products, limit=20):
    """Return a concise JSON string and a short human summary for the system prompt."""
    items = []
    summary_lines = []
    try:
        for p in (latest_products or [])[:limit]:
            name = p.get("name") or p.get("title") or "Unknown"
            price = p.get("price") or p.get("price_value") or ""
            site = p.get("site") or ""
            hist = p.get("history_prices") or []
            forecast = p.get("forecast_prices") or p.get("forecast", [])
            min_p = p.get("min_price", _to_num_price(price))
            max_p = p.get("max_price", _to_num_price(price))
            items.append({
                "name": name,
                "price": price,
                "min_price": min_p,
                "max_price": max_p,
                "site": site,
                "history": hist,
                "forecast": forecast
            })
            summary_lines.append(f"{name} — {price} — {site}")
    except Exception:
        items = []
        summary_lines = []
    return json.dumps(items, ensure_ascii=False), ("\n".join(summary_lines[:10]) if summary_lines else "")

@app.route("/chatbot", methods=["POST"])
def chatbot():
    """
    Stateless product-aware chatbot endpoint.
    Request JSON:
      { "message": "text" }
    Optionally:
      { "audio": "<base64-audio-data>" }  # then 'message' will be replaced by transcript
    Response JSON:
      { "reply": "text" }
    """

    data = request.get_json(silent=True) or {}
    user_msg = (data.get("message") or "").strip()
    # If audio provided as base64 string -> try to transcribe first
    if not user_msg and data.get("audio"):
        try:
            audio_b64 = data.get("audio")
            audio_bytes = base64.b64decode(audio_b64)
            # Try Groq whisper transcription if groq available
            try:
                from groq import Groq
                groq_key = globals().get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
                if groq_key:
                    gclient = Groq(api_key=groq_key)
                    # Groq's python sdk may accept file tuple; adapt if required
                    transcript = gclient.audio.transcriptions.create(model="whisper-large-v3", file=("audio.wav", audio_bytes))
                    # transcript may be dict-like or object
                    user_msg = getattr(transcript, "text", None) or (transcript.get("text") if isinstance(transcript, dict) else None) or ""
            except Exception:
                # fallback: try OpenAI whisper if openai available
                try:
                    import openai
                    openai_key = os.getenv("OPENAI_API_KEY")
                    if openai_key:
                        # write temp file, call transcription
                        import tempfile
                        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                        tf.write(audio_bytes); tf.flush(); tf.close()
                        with open(tf.name, "rb") as fh:
                            resp = openai.Audio.transcribe("whisper-1", fh)
                            user_msg = resp.get("text") if isinstance(resp, dict) else getattr(resp, "text", "") or ""
                except Exception:
                    user_msg = ""
        except Exception:
            user_msg = ""

    if not user_msg:
        return jsonify({"reply": "Please type or speak something."})

    # Build product context (JSON + short summary) from latest_products
    try:
        product_json, product_summary = _build_product_context_for_prompt(latest_products, limit=20)
    except Exception:
        product_json, product_summary = "[]", ""

    # System prompt: instruct model to always use PRODUCT_CONTEXT for product-related queries
    system_prompt = (
        "You are ShopAI, a professional shopping assistant focused on the products visible on the user's homepage. "
        "You MUST use the PRODUCT_CONTEXT given below if the user asks about prices, lowest/highest, comparisons, trends, forecasts or savings. "
        "When giving numeric prices, include rupee symbol (₹) and numbers. Be concise, factual and mention store names when available. "
        "If the user asks for predictions, label them 'model-based forecast' and be conservative. "
        "If a question is outside product scope, answer normally but indicate you can only use the visible product list when requested.\n\n"
        "PRODUCT_SUMMARY (short):\n" + (product_summary or "no products") + "\n\n"
        "PRODUCT_CONTEXT (full JSON):\n" + product_json + "\n\n"
        "Rules:\n"
        "1) If user asks 'lowest'/'cheapest', respond with the single cheapest product from PRODUCT_CONTEXT and show price & site.\n"
        "2) If user asks 'highest'/'expensive', respond with the single most expensive product similarly.\n"
        "3) If user asks to 'compare' list two or three items with prices and a short recommendation.\n"
        "4) If there is no product data, prompt the user to run a search on the homepage.\n"
    )

    # Compose messages for chat LLM
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg}
    ]

    # Try Groq first (if available), with several model fallbacks
    groq_reply = None
    groq_error = None
    try:
        try:
            from groq import Groq
        except Exception as e:
            Groq = None

        if Groq:
            groq_key = globals().get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
            if groq_key:
                client = Groq(api_key=groq_key)
                # try a small list of likely supported models in order
                model_candidates = [
                    "llama-3.3-70b-versatile",
                    "llama-3.2-70b-versatile",
                    "llama-3-13b-mini",   # smaller fallback
                ]
                last_exc = None
                for mdl in model_candidates:
                    try:
                        resp = client.chat.completions.create(
                            model=mdl,
                            messages=messages,
                            temperature=0.2,
                            max_tokens=512
                        )
                        # Groq SDK responses may vary. Try common shapes:
                        # resp.choices[0].message.content  OR resp.choices[0].message["content"] OR resp.choices[0].text
                        ch0 = getattr(resp.choices[0], "message", None)
                        if ch0 is not None:
                            groq_reply = getattr(ch0, "content", None) or (ch0.get("content") if isinstance(ch0, dict) else None)
                        if not groq_reply:
                            # try other shapes
                            try:
                                groq_reply = resp.choices[0].message["content"]
                            except Exception:
                                try:
                                    groq_reply = resp.choices[0].text
                                except Exception:
                                    groq_reply = None
                        if groq_reply:
                            break
                    except Exception as e:
                        last_exc = e
                        # if model is decommissioned, try next
                        continue
                if not groq_reply and last_exc:
                    groq_error = last_exc
    except Exception as e:
        groq_error = e

    # If Groq succeeded, return its reply
    if groq_reply:
        return jsonify({"reply": groq_reply})

    # If Groq failed, attempt OpenAI (if OPENAI_API_KEY present & openai installed)
    openai_reply = None
    openai_error = None
    try:
        try:
            import openai
        except Exception:
            openai = None

        if openai:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                openai.api_key = openai_key
                # use gpt-4o or gpt-4o-mini if available otherwise gpt-3.5-turbo
                model_candidates = ["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"]
                last_exc = None
                for mdl in model_candidates:
                    try:
                        resp = openai.ChatCompletion.create(
                            model=mdl,
                            messages=messages,
                            max_tokens=512,
                            temperature=0.2
                        )
                        # standard shape: resp.choices[0].message.content
                        try:
                            openai_reply = resp["choices"][0]["message"]["content"]
                        except Exception:
                            try:
                                openai_reply = resp.choices[0].message.content
                            except Exception:
                                openai_reply = None
                        if openai_reply:
                            break
                    except Exception as e:
                        last_exc = e
                        # if quota or model error, continue to next
                        continue
                if not openai_reply and last_exc:
                    openai_error = last_exc
    except Exception as e:
        openai_error = e

    # Final fallback: simple heuristic answer using product_context if available
    try:
        if not groq_reply and not openai_reply:
            # If we have products, answer heuristically
            ctx_list = json.loads(product_json or "[]")
            if ctx_list:
                # get cheapest and costliest by min_price/max_price (numeric)
                priced = []
                for p in ctx_list:
                    try:
                        pr = float(p.get("min_price") or p.get("price") or 0)
                    except:
                        pr = _to_num_price(p.get("price"))
                    priced.append((pr, p))
                priced_sorted = sorted(priced, key=lambda x: x[0])
                cheapest = priced_sorted[0][1] if priced_sorted else None
                costliest = priced_sorted[-1][1] if priced_sorted else None

                lm = user_msg.lower()
                if any(w in lm for w in ["lowest", "cheapest", "cheapest product", "cheap"]):
                    if cheapest:
                        reply = f"🟢 Cheapest I see: {cheapest.get('name')} — {cheapest.get('price') or ('₹' + str(cheapest.get('min_price')))} at {cheapest.get('site')}"
                        return jsonify({"reply": reply})
                if any(w in lm for w in ["highest", "expensive", "costliest"]):
                    if costliest:
                        reply = f"🔴 Most expensive: {costliest.get('name')} — {costliest.get('price') or ('₹' + str(costliest.get('max_price')))} at {costliest.get('site')}"
                        return jsonify({"reply": reply})
                if "summary" in lm or "home" in lm or "homepage" in lm or "list" in lm:
                    summary = []
                    for p in ctx_list[:6]:
                        summary.append(f"{p.get('name')} — {p.get('price') or ('₹' + str(p.get('min_price')))} ({p.get('site')})")
                    return jsonify({"reply": "Based on visible products:\n" + "\n".join(summary)})
                # generic fallback reply summarizing top 3
                summary = []
                for p in ctx_list[:3]:
                    summary.append(f"{p.get('name')} — {p.get('price') or ('₹' + str(p.get('min_price')))} ({p.get('site')})")
                return jsonify({"reply": "I couldn't reach the LLM. Based on visible products:\n" + "\n".join(summary)})
            else:
                # no product data — useful fallback
                return jsonify({"reply": "I cannot reach the LLM right now. Please search a product on the homepage so I can answer with product-specific details."})
    except Exception as e:
        print("FALLBACK ERROR:", e, traceback.format_exc())

    # If we reached here, include errors in logs and return a friendly message
    # Prefer to show LLM-specific error if present
    err_msg = None
    if groq_error:
        err_msg = f"GROQ ERROR: {str(groq_error)}"
    elif openai_error:
        err_msg = f"OPENAI ERROR: {str(openai_error)}"
    else:
        err_msg = "AI engine unavailable."

    print("CHATBOT FINAL FAILURE:", err_msg)
    return jsonify({"reply": "⚠️ AI temporarily unavailable. Try again. "})
# -------------------------------------------------------------------
# End of chatbot block
# -------------------------------------------------------------------


# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    print("✅ Flask server started — Automatic re-login enforced after refresh or restart.")
    app.run(debug=True)
