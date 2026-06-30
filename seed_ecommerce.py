"""
seed_ecommerce.py
-----------------
Creates and populates memory/ecommerce.db with a realistic
e-commerce schema for dataset generation.

Tables:
  customers, orders, order_items, products, categories,
  suppliers, inventory, shipping, returns, promotions

Run: python3 seed_ecommerce.py
"""

import sqlite3
import random
from pathlib import Path
from datetime import datetime, timedelta

DB_PATH = Path("memory/ecommerce.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

random.seed(42)

# Pin the reference date so 'last 30 days' queries return non-empty results
# regardless of when this script runs. Update this if you regenerate the
# dataset on a different date and want fresh timestamps.
BASE_DATE = datetime(2026, 6, 17)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# ── schema ─────────────────────────────────────────────────────────────────────
cursor.executescript("""
    DROP TABLE IF EXISTS returns;
    DROP TABLE IF EXISTS shipping;
    DROP TABLE IF EXISTS inventory;
    DROP TABLE IF EXISTS promotions;
    DROP TABLE IF EXISTS order_items;
    DROP TABLE IF EXISTS orders;
    DROP TABLE IF EXISTS products;
    DROP TABLE IF EXISTS categories;
    DROP TABLE IF EXISTS suppliers;
    DROP TABLE IF EXISTS customers;

    CREATE TABLE customers (
        customer_id     INTEGER PRIMARY KEY,
        customer_name   TEXT    NOT NULL,
        email           TEXT    NOT NULL,
        country         TEXT,
        city            TEXT,
        segment         TEXT,   -- 'Consumer', 'Corporate', 'Home Office'
        created_at      TEXT
    );

    CREATE TABLE categories (
        category_id     INTEGER PRIMARY KEY,
        category_name   TEXT    NOT NULL,
        parent_category TEXT
    );

    CREATE TABLE suppliers (
        supplier_id     INTEGER PRIMARY KEY,
        supplier_name   TEXT    NOT NULL,
        country         TEXT,
        lead_time_days  INTEGER
    );

    CREATE TABLE products (
        product_id      INTEGER PRIMARY KEY,
        product_name    TEXT    NOT NULL,
        category_id     INTEGER,
        supplier_id     INTEGER,
        unit_price      REAL,
        cost_price      REAL,
        sku             TEXT,
        FOREIGN KEY (category_id) REFERENCES categories(category_id),
        FOREIGN KEY (supplier_id) REFERENCES suppliers(supplier_id)
    );

    CREATE TABLE promotions (
        promotion_id    INTEGER PRIMARY KEY,
        promotion_name  TEXT    NOT NULL,
        discount_pct    REAL,
        start_date      TEXT,
        end_date        TEXT
    );

    CREATE TABLE orders (
        order_id        INTEGER PRIMARY KEY,
        customer_id     INTEGER NOT NULL,
        promotion_id    INTEGER,
        order_date      TEXT,
        status          TEXT,   -- 'pending', 'processing', 'shipped', 'delivered', 'cancelled'
        total_amount    REAL,
        region          TEXT,
        channel         TEXT,   -- 'web', 'mobile', 'marketplace', 'in-store'
        FOREIGN KEY (customer_id)  REFERENCES customers(customer_id),
        FOREIGN KEY (promotion_id) REFERENCES promotions(promotion_id)
    );

    CREATE TABLE order_items (
        item_id         INTEGER PRIMARY KEY,
        order_id        INTEGER NOT NULL,
        product_id      INTEGER NOT NULL,
        quantity        INTEGER,
        unit_price      REAL,
        discount_pct    REAL,
        FOREIGN KEY (order_id)   REFERENCES orders(order_id),
        FOREIGN KEY (product_id) REFERENCES products(product_id)
    );

    CREATE TABLE inventory (
        inventory_id    INTEGER PRIMARY KEY,
        product_id      INTEGER NOT NULL,
        warehouse       TEXT,
        quantity_on_hand INTEGER,
        reorder_level   INTEGER,
        last_updated    TEXT,
        FOREIGN KEY (product_id) REFERENCES products(product_id)
    );

    CREATE TABLE shipping (
        shipping_id     INTEGER PRIMARY KEY,
        order_id        INTEGER NOT NULL,
        carrier         TEXT,   -- 'FedEx', 'UPS', 'DHL', 'USPS'
        tracking_number TEXT,
        shipped_date    TEXT,
        delivered_date  TEXT,
        shipping_cost   REAL,
        status          TEXT,   -- 'in_transit', 'delivered', 'returned'
        FOREIGN KEY (order_id) REFERENCES orders(order_id)
    );

    CREATE TABLE returns (
        return_id       INTEGER PRIMARY KEY,
        order_id        INTEGER NOT NULL,
        product_id      INTEGER NOT NULL,
        return_date     TEXT,
        reason          TEXT,   -- 'defective', 'wrong_item', 'not_as_described', 'changed_mind'
        refund_amount   REAL,
        status          TEXT,   -- 'pending', 'approved', 'rejected', 'refunded'
        FOREIGN KEY (order_id)   REFERENCES orders(order_id),
        FOREIGN KEY (product_id) REFERENCES products(product_id)
    );
""")

# ── seed data ──────────────────────────────────────────────────────────────────

# categories
categories = [
    (1, "Electronics",    None),
    (2, "Computers",      "Electronics"),
    (3, "Mobile",         "Electronics"),
    (4, "Furniture",      None),
    (5, "Office",         "Furniture"),
    (6, "Home",           "Furniture"),
    (7, "Clothing",       None),
    (8, "Sports",         None),
    (9, "Books",          None),
    (10,"Stationery",     None),
]
cursor.executemany("INSERT INTO categories VALUES (?, ?, ?)", categories)

# suppliers
suppliers = [
    (1, "TechSource Inc",   "US",  7),
    (2, "GlobalParts Ltd",  "CN",  21),
    (3, "EuroSupply GmbH",  "DE",  14),
    (4, "AsiaTrade Co",     "JP",  18),
    (5, "LocalGoods LLC",   "US",   3),
]
cursor.executemany("INSERT INTO suppliers VALUES (?, ?, ?, ?)", suppliers)

# products
products = [
    (1,  "Laptop Pro 15",      2, 1, 1299.99,  820.00, "LAP-001"),
    (2,  "Wireless Mouse",     1, 2,   49.99,   18.00, "MOU-001"),
    (3,  "iPhone Case",        3, 2,   19.99,    5.00, "ACC-001"),
    (4,  "Standing Desk",      5, 3,  599.99,  310.00, "DSK-001"),
    (5,  "Office Chair",       5, 3,  349.99,  180.00, "CHR-001"),
    (6,  "USB-C Hub",          1, 2,   79.99,   28.00, "HUB-001"),
    (7,  "Monitor 27in",       1, 1,  449.99,  260.00, "MON-001"),
    (8,  "Desk Lamp",          6, 5,   59.99,   22.00, "LMP-001"),
    (9,  "Webcam HD",          1, 4,   99.99,   42.00, "CAM-001"),
    (10, "Keyboard Mech",      1, 4,  149.99,   65.00, "KEY-001"),
    (11, "Running Shoes",      8, 5,  129.99,   55.00, "SHO-001"),
    (12, "Python Programming", 9, 5,   49.99,   15.00, "BOK-001"),
    (13, "Notebook Pack",     10, 5,   19.99,    6.00, "NTB-001"),
    (14, "Tablet 10in",        3, 4,  399.99,  210.00, "TAB-001"),
    (15, "Smart Watch",        3, 1,  299.99,  140.00, "WCH-001"),
]
cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", products)

# promotions
promotions = [
    (1, "Summer Sale",      15.0, "2026-06-01", "2026-06-30"),
    (2, "Flash Deal",       25.0, "2026-05-15", "2026-05-16"),
    (3, "Loyalty Discount",  5.0, "2026-01-01", "2026-12-31"),
    (4, "Clearance",        40.0, "2026-04-01", "2026-04-30"),
]
cursor.executemany("INSERT INTO promotions VALUES (?, ?, ?, ?, ?)", promotions)

# customers (30)
customer_data = [
    ("Alice Tan",      "alice@email.com",    "US", "New York",    "Consumer"),
    ("Budi Santoso",   "budi@email.com",     "ID", "Jakarta",     "Consumer"),
    ("James Wright",   "james@email.com",    "UK", "London",      "Corporate"),
    ("Sara Kim",       "sara@email.com",     "US", "Los Angeles", "Consumer"),
    ("Lena Fischer",   "lena@email.com",     "DE", "Berlin",      "Corporate"),
    ("Carlos Mendez",  "carlos@email.com",   "MX", "Mexico City", "Consumer"),
    ("Yuki Tanaka",    "yuki@email.com",     "JP", "Tokyo",       "Home Office"),
    ("Priya Patel",    "priya@email.com",    "IN", "Mumbai",      "Consumer"),
    ("Omar Hassan",    "omar@email.com",     "AE", "Dubai",       "Corporate"),
    ("Emily Chen",     "emily@email.com",    "US", "Chicago",     "Home Office"),
    ("Marco Rossi",    "marco@email.com",    "IT", "Milan",       "Corporate"),
    ("Anna Kowalski",  "anna@email.com",     "PL", "Warsaw",      "Consumer"),
    ("David Okafor",   "david@email.com",    "NG", "Lagos",       "Consumer"),
    ("Sophie Martin",  "sophie@email.com",   "FR", "Paris",       "Corporate"),
    ("Li Wei",         "liwei@email.com",    "CN", "Shanghai",    "Consumer"),
    ("Fatima Al-Said", "fatima@email.com",   "SA", "Riyadh",      "Corporate"),
    ("Tom Anderson",   "tom@email.com",      "AU", "Sydney",      "Home Office"),
    ("Nina Petrov",    "nina@email.com",     "RU", "Moscow",      "Consumer"),
    ("Jake Murphy",    "jake@email.com",     "US", "Houston",     "Consumer"),
    ("Aisha Diallo",   "aisha@email.com",    "SN", "Dakar",       "Consumer"),
    ("Chen Wei",       "chenwei@email.com",  "CN", "Beijing",     "Corporate"),
    ("Maria Garcia",   "maria@email.com",    "ES", "Madrid",      "Consumer"),
    ("John Smith",     "john@email.com",     "US", "Seattle",     "Corporate"),
    ("Hana Suzuki",    "hana@email.com",     "JP", "Osaka",       "Consumer"),
    ("Pierre Dupont",  "pierre@email.com",   "FR", "Lyon",        "Home Office"),
    ("Amara Nwosu",    "amara@email.com",    "NG", "Abuja",       "Consumer"),
    ("Diego Torres",   "diego@email.com",    "AR", "Buenos Aires","Consumer"),
    ("Mei Lin",        "meilin@email.com",   "SG", "Singapore",   "Corporate"),
    ("Ryan O'Brien",   "ryan@email.com",     "IE", "Dublin",      "Home Office"),
    ("Zara Ahmed",     "zara@email.com",     "PK", "Karachi",     "Consumer"),
]

country_region = {
    "US": "North America", "MX": "LATAM", "AR": "LATAM",
    "UK": "EMEA", "DE": "EMEA", "IT": "EMEA", "PL": "EMEA",
    "FR": "EMEA", "RU": "EMEA", "SA": "EMEA", "AE": "EMEA",
    "NG": "EMEA", "SN": "EMEA", "ES": "EMEA", "IE": "EMEA", "PK": "EMEA",
    "ID": "APAC", "JP": "APAC", "IN": "APAC", "CN": "APAC",
    "AU": "APAC", "SG": "APAC",
}

for i, (name, email, country, city, segment) in enumerate(customer_data, 1):
    days_ago = random.randint(30, 730)
    created = (BASE_DATE - timedelta(days=days_ago)).strftime("%Y-%m-%d")
    cursor.execute(
        "INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?, ?)",
        (i, name, email, country, city, segment, created)
    )

# orders (200)
statuses = ["delivered", "delivered", "delivered", "shipped", "processing", "pending", "cancelled"]
channels = ["web", "web", "mobile", "marketplace", "in-store"]

orders_data = []
for i in range(1, 201):
    customer_id = random.randint(1, 30)
    customer_country = customer_data[customer_id - 1][2]
    region = country_region.get(customer_country, "EMEA")
    status = random.choice(statuses)
    channel = random.choice(channels)
    days_ago = random.randint(1, 90)
    order_date = (BASE_DATE - timedelta(days=days_ago)).strftime("%Y-%m-%d")
    promotion_id = random.choice([None, None, None, 1, 2, 3, 4])
    total_amount = round(random.uniform(20, 2500), 2)
    orders_data.append((i, customer_id, promotion_id, order_date, status, total_amount, region, channel))

cursor.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?)", orders_data)

# order_items (400 — ~2 items per order)
item_id = 1
for order in orders_data:
    order_id = order[0]
    num_items = random.randint(1, 3)
    for _ in range(num_items):
        product_id = random.randint(1, 15)
        quantity = random.randint(1, 4)
        unit_price = products[product_id - 1][4]
        discount_pct = random.choice([0, 0, 0, 5, 10, 15, 25])
        cursor.execute(
            "INSERT INTO order_items VALUES (?, ?, ?, ?, ?, ?)",
            (item_id, order_id, product_id, quantity, unit_price, discount_pct)
        )
        item_id += 1

# inventory
warehouses = ["US-East", "US-West", "EU-Central", "APAC-Hub"]
inv_id = 1
for product_id in range(1, 16):
    for warehouse in warehouses:
        qty = random.randint(0, 500)
        reorder = random.randint(20, 100)
        cursor.execute(
            "INSERT INTO inventory VALUES (?, ?, ?, ?, ?, ?)",
            (inv_id, product_id, warehouse, qty, reorder,
             BASE_DATE.strftime("%Y-%m-%d"))
        )
        inv_id += 1

# shipping (for delivered/shipped orders)
carriers = ["FedEx", "UPS", "DHL", "USPS"]
ship_id = 1
for order in orders_data:
    if order[4] in ("delivered", "shipped"):
        order_date = datetime.strptime(order[3], "%Y-%m-%d")
        shipped = (order_date + timedelta(days=random.randint(1, 3))).strftime("%Y-%m-%d")
        delivered = (order_date + timedelta(days=random.randint(4, 10))).strftime("%Y-%m-%d") if order[4] == "delivered" else None
        cursor.execute(
            "INSERT INTO shipping VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (ship_id, order[0], random.choice(carriers),
             f"TRK{ship_id:06d}", shipped, delivered,
             round(random.uniform(5, 50), 2),
             "delivered" if delivered else "in_transit")
        )
        ship_id += 1

# returns (20 returns)
return_reasons = ["defective", "wrong_item", "not_as_described", "changed_mind"]
return_statuses = ["refunded", "refunded", "approved", "pending", "rejected"]
for i in range(1, 21):
    order_id = random.randint(1, 200)
    product_id = random.randint(1, 15)
    days_ago = random.randint(1, 30)
    return_date = (BASE_DATE - timedelta(days=days_ago)).strftime("%Y-%m-%d")
    cursor.execute(
        "INSERT INTO returns VALUES (?, ?, ?, ?, ?, ?, ?)",
        (i, order_id, product_id, return_date,
         random.choice(return_reasons),
         round(random.uniform(20, 500), 2),
         random.choice(return_statuses))
    )

conn.commit()

# ── summary ────────────────────────────────────────────────────────────────────
for table in ["customers", "categories", "suppliers", "products",
              "promotions", "orders", "order_items", "inventory",
              "shipping", "returns"]:
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    print(f"{table:<15}: {cursor.fetchone()[0]} rows")

conn.close()
print(f"\nE-commerce database seeded at {DB_PATH}")
