from tinydb import TinyDB, Query
from datetime import datetime, timedelta

class DB:
    def __init__(self, path="data.json"):
        self.db = TinyDB(path)

        # Tables
        self.t_products = self.db.table("products")
        self.t_history = self.db.table("history")
        self.t_alerts = self.db.table("alerts")  # alert table

    def upsert_products(self, query, products):
        for p in products:
            self.t_products.upsert(p, Query().id == p["id"])

    def get_product(self, product_id):
        rows = self.t_products.search(Query().id == product_id)
        return rows[0] if rows else None

    def add_today_price(self, product_id, price_num: int):
        today = datetime.today().strftime("%Y-%m-%d")
        exists = self.t_history.search(
            (Query().product_id == product_id) &
            (Query().day == today)
        )
        if not exists:
            self.t_history.insert({
                "product_id": product_id,
                "day": today,
                "price_num": int(price_num)
            })

    def last_ndays(self, product_id, n=7):
        """
        Returns last 7 CALENDAR DAYS history.
        Fills missing days from last known price.
        Ensures graph NEVER shows only 1 day again.
        """
        # Fetch history from DB
        rows = sorted(
            self.t_history.search(Query().product_id == product_id),
            key=lambda r: r["day"]
        )

        # If no history, return empty
        if not rows:
            return [], []

        # Map: day -> price_num
        history_map = {r["day"]: int(r["price_num"]) for r in rows}

        # Get latest recorded price
        last_known_price = int(rows[-1]["price_num"])

        labels = []
        values = []

        today = datetime.today()

        for i in range(n):
            day = today - timedelta(days=(n - i - 1))
            d_str = day.strftime("%Y-%m-%d")
            labels.append(day.strftime("%d %b"))

            if d_str in history_map:
                last_known_price = history_map[d_str]

            values.append(last_known_price)

        return labels, values
