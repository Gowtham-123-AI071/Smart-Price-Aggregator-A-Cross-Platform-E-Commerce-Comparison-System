import re
import os
import requests
import numpy as np
import random
from datetime import datetime, timedelta

# 🔥 NEW: ML imports
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# ============================================
#  GROQ API KEY (Used by chatbot in app.py)
# ============================================
GROQ_API_KEY = "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"


# SerpApi API key (replace if needed)
API_KEY = "96eff0c4add42f004f82be864d87a93617bfcfec785f4665ef72a403e4f712be"


def _rupees_to_int(s: str) -> int:
    """Convert ₹6,999 → 6999"""
    if not s:
        return 0
    digits = re.sub(r"[^\d]", "", s)
    return int(digits) if digits else 0


def _merchant_search_url(site: str, name: str) -> str:
    """Return accurate merchant URLs based on platform name."""
    q = requests.utils.quote(name)
    site_l = (site or "").lower()

    if "flipkart" in site_l:
        return f"https://www.flipkart.com/search?q={q}"
    elif "croma" in site_l:
        return f"https://www.croma.com/search/?text={q}"
    elif "blinkit" in site_l:
        return f"https://blinkit.com/s/?q={q}"
    elif "instamart" in site_l:
        # Instamart runs under Swiggy’s domain
        return f"https://www.swiggy.com/instamart/search?query={q}"
    elif "zomato" in site_l:
        return f"https://www.zomato.com/india/search?q={q}"
    elif "myntra" in site_l:
        return f"https://www.myntra.com/{q.replace('%20', '-')}"
    elif "amazon" in site_l:
        return f"https://www.amazon.in/s?k={q}"
    elif "jiomart" in site_l or "jio" in site_l:
        return f"https://www.jiomart.com/search/{q.replace(' ', '%20')}"
    elif "boat" in site_l:
        return f"https://www.boat-lifestyle.com/search?q={q}"
    else:
        # Fallback to Google Shopping if unknown site
        return f"https://www.google.com/search?tbm=shop&q={q}"


def scrape_all(query: str):
    """Fetch product info via SerpAPI and generate price history + forecast."""
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_shopping",
        "q": query,
        "location": "India",
        "hl": "en",
        "gl": "in",
        "api_key": API_KEY
    }

    try:
        r = requests.get(url, params=params, timeout=25)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[ERROR] SerpApi fetch failed: {e}")
        return []

    products = []
    results = data.get("shopping_results") or []

    for item in results[:20]:
        name = item.get("title") or "Unknown Product"
        price_str = item.get("price") or ""
        price_num = _rupees_to_int(price_str)
        site = item.get("source") or "Unknown"
        image = item.get("thumbnail") or ""
        product_id = (item.get("product_id") or name.replace(" ", "_"))[:64]
        merchant_url = _merchant_search_url(site, name)

        # ---------- Generate Realistic 7-Day Price History ----------
        # History range: roughly [price - 100, price + 100]
        random.seed(hash(name) & 0xFFFFFFFF)

        if price_num > 0:
            center = price_num
            low_bound = max(1, center - 100)
            high_bound = center + 100

            # start near current price
            current = random.randint(low_bound, high_bound)
            history_prices = []

            for _ in range(7):
                # small random walk, but stay inside [low_bound, high_bound]
                step = random.randint(-30, 30)
                current = max(low_bound, min(high_bound, current + step))
                history_prices.append(int(current))
        else:
            history_prices = [0] * 7

        # ---------- Predict next 10 days using 5 ML models ----------
        # Models: Linear Regression, k-NN, SVR, Random Forest, Gradient Boosting
        forecast_prices = [0] * 10
        forecast_comparison = {}
        best_model_name = "Linear Regression"

        if price_num > 0 and any(history_prices):
            # Prepare training data: x = day index, y = price
            X = np.arange(len(history_prices)).reshape(-1, 1)
            y = np.array(history_prices, dtype=float)

            models = {
                "Linear Regression": LinearRegression(),
                "k-NN Regression": KNeighborsRegressor(n_neighbors=3),
                "SVR": SVR(kernel="rbf", C=100, gamma="scale"),
                "Random Forest": RandomForestRegressor(
                    n_estimators=200, random_state=42
                ),
                "Gradient Boosting": GradientBoostingRegressor(
                    n_estimators=200, learning_rate=0.05, random_state=42
                ),
            }

            X_future = np.arange(
                len(history_prices),
                len(history_prices) + 10
            ).reshape(-1, 1)

            errors = {}

            for model_name, model in models.items():
                # Fit model on history
                model.fit(X, y)

                # Training RMSE (used for comparison)
                y_train_pred = model.predict(X)
                rmse = mean_squared_error(y, y_train_pred) ** 0.5
                errors[model_name] = float(rmse)

                # Forecast 10 future days
                future_pred = model.predict(X_future)

                # Clip to positive prices and round to int
                future_pred = np.clip(future_pred, 1, None)
                future_list = [int(round(v)) for v in future_pred]

                forecast_comparison[model_name] = future_list

            # Choose best model (lowest RMSE)
            best_model_name = min(errors, key=errors.get)
            forecast_prices = forecast_comparison.get(best_model_name,
                                                      forecast_prices)

        products.append({
            "id": product_id,
            "name": name,
            "price": price_str or f"₹{price_num:,}",
            "price_num": price_num,
            "site": site,
            "image": image,
            "merchant_url": merchant_url,
            "history_prices": history_prices,
            "forecast_prices": forecast_prices,              # best model
            "forecast_comparison": forecast_comparison,      # all 5 models
            "best_model": best_model_name                    # best algo name
        })

    return products
