"""
augment_prompts.py  —  RELIABLE EDITION
---------------------------------------
Generates a diverse, schema-aligned NL prompt set for the de-swarm e-commerce
pipeline. Outputs ~800-1200 prompts covering the FULL schema.

Output: dataset/ecommerce_prompts.txt
       (filename matches generate_dataset_v3.py default — no env var needed)

RELIABILITY GUARANTEES (vs. the old version):

  1. Schema-aligned values — every status, segment, channel, carrier,
     warehouse, etc. matches what seed_ecommerce.py actually inserts.
     No more "completed orders" queries returning 0 rows.

  2. Full schema coverage — prompts exercise every table:
     customers, orders, order_items, products, categories, suppliers,
     inventory, shipping, returns, promotions.

  3. Balanced complexity — explicit 1-table / 2-table / 3+-table
     generation so the SFT training distribution isn't lopsided.

  4. Real randomization in paraphrases — iterates over MULTIPLE value
     combinations per template (was: hardcoded to single values).

  5. Filtered edge cases — removed prompts that reference nonexistent
     tables/columns (would teach the student to hallucinate).

  6. Determinism — pinned seed (42) for reproducibility.

  7. Category-tagged stats — prints per-category counts so you can
     verify distribution before kicking off the Groq run.

Usage:
  python3 augment_prompts.py
  python3 augment_prompts.py --seed 123        # different shuffle, same prompts
  python3 augment_prompts.py --target 2000     # stratified oversample to N prompts
  python3 augment_prompts.py --balance 300     # cap each category at 300 prompts
  python3 augment_prompts.py --output dataset/custom.txt

IMPROVEMENTS (v2):
  - Fixed: output file now ends with trailing newline (was missing — wc -l
    reported N-1 instead of N).
  - Fixed: "completed Electronics orders" → "delivered Electronics orders"
    (completed is not a valid orders.status value).
  - Fixed: "refunded Furniture orders" → "returned Furniture orders"
    (refunded is a returns.status value, not an orders.status).
  - Fixed: removed 2 duplicate prompts that appeared in multiple tiers.
  - Fixed: 20+ prompts moved to correct complexity tier (was miscategorized,
    producing inaccurate stats — didn't affect SQL generation, just stats).
  - Added: --balance N flag to cap per-category counts (mirrors
    build_sft_dataset.py logic at prompt-generation time).
  - Improved: --target oversample now uses stratified sampling (preserves
    category distribution instead of uniform random).
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

# ── pinned seed for reproducibility ───────────────────────────────────────────
DEFAULT_SEED = 42

# ── schema-aligned variable pools ─────────────────────────────────────────────
# Every value here MUST exist in seed_ecommerce.py — verify before editing.

# Time windows used in prompts. Keep these within the 90-day window that
# seed_ecommerce.py populates (orders are 1-90 days old).
TIMEFRAMES = ["last 7 days", "last 14 days", "last 30 days", "last 60 days", "last 90 days"]

# Top-N limits — small enough to actually return data on a 200-order dataset.
LIMITS = [3, 5, 10]

# Geographic scopes — these are the REGION values populated in orders.region.
REGIONS = ["North America", "EMEA", "APAC", "LATAM"]

# 2-letter country codes — match customers.country values in seed_ecommerce.
COUNTRIES = ["US", "UK", "DE", "JP", "FR", "IN", "AU", "SG", "ID", "CN", "MX", "AR"]

# Top-level categories only — subcategories (Computers, Mobile, Office, Home)
# have parent_category set and are rarely referenced in prompts.
CATEGORIES = ["Electronics", "Furniture", "Clothing", "Sports", "Books", "Stationery"]

# Products — sample of the 15 products in seed_ecommerce.
PRODUCTS = [
    "Laptop Pro 15", "Wireless Mouse", "Standing Desk", "Office Chair",
    "USB-C Hub", "Monitor 27in", "Desk Lamp", "Webcam HD",
    "Keyboard Mech", "Tablet 10in", "Smart Watch",
]

# Customers — sample of the 30 customer names in seed_ecommerce.
CUSTOMERS = [
    "Alice Tan", "Budi Santoso", "James Wright", "Sara Kim",
    "Lena Fischer", "Carlos Mendez", "Yuki Tanaka", "Priya Patel",
    "Emily Chen", "Marco Rossi",
]

# Order statuses — EXACTLY what seed_ecommerce inserts into orders.status.
# (Old version used "completed" / "refunded" — neither exists.)
ORDER_STATUSES = ["delivered", "shipped", "processing", "pending", "cancelled"]

# Returns-specific: returns.status column has its own values.
RETURN_STATUSES = ["refunded", "approved", "pending", "rejected"]
RETURN_REASONS  = ["defective", "wrong_item", "not_as_described", "changed_mind"]

# Shipping — shipping.carrier values from seed_ecommerce.
CARRIERS = ["FedEx", "UPS", "DHL", "USPS"]
SHIPPING_STATUSES = ["in_transit", "delivered", "returned"]

# Inventory — inventory.warehouse values from seed_ecommerce.
WAREHOUSES = ["US-East", "US-West", "EU-Central", "APAC-Hub"]

# Customers.segment values from seed_ecommerce.
SEGMENTS = ["Consumer", "Corporate", "Home Office"]

# Orders.channel values from seed_ecommerce.
CHANNELS = ["web", "mobile", "marketplace", "in-store"]

# Suppliers — sample (5 suppliers exist in schema).
SUPPLIERS = ["TechSource Inc", "GlobalParts Ltd", "EuroSupply GmbH", "AsiaTrade Co"]

# Promotions — sample (4 promotions exist in schema).
PROMOTIONS = ["Summer Sale", "Flash Deal", "Loyalty Discount", "Clearance"]

# HAVING thresholds — kept realistic for a 200-order dataset with
# orders.total_amount ranging 20-2500.
REVENUE_THRESHOLDS = [500, 1000, 1500, 2000, 5000]
ORDER_COUNT_THRESHOLDS = [2, 3, 5, 10]
QUANTITY_THRESHOLDS = [10, 20, 50, 100]


# ──────────────────────────────────────────────────────────────────────────────
# Strategy 1 — Variable Randomization
# Generates explicit complexity tiers: 1-table, 2-table, 3+-table.
# ──────────────────────────────────────────────────────────────────────────────

def _tier_1_single_table() -> list[str]:
    """Single-table aggregations — easiest tier."""
    prompts: list[str] = []

    # Orders-only queries
    for tf in TIMEFRAMES:
        prompts += [
            f"Show total revenue by region for the {tf}",
            f"Show number of orders by status for the {tf}",
            f"Show total revenue by channel for the {tf}",
            f"Show average order value by region for the {tf}",
            f"Show number of cancelled orders by region for the {tf}",
        ]

    for status in ORDER_STATUSES:
        for tf in TIMEFRAMES:
            prompts.append(f"Show total revenue from {status} orders for the {tf}")
            prompts.append(f"Show number of {status} orders by region for the {tf}")

    for channel in CHANNELS:
        for tf in TIMEFRAMES:
            prompts.append(f"Show total revenue from the {channel} channel for the {tf}")

    # NOTE: customer-segment, country, category, and product revenue queries
    # require joins (orders + customers, or orders + order_items + products)
    # so they've been moved to _tier_2_two_tables() and _tier_3_multi_table().

    # Inventory-only (single-table on inventory)
    for warehouse in WAREHOUSES:
        prompts += [
            f"Show products below reorder level at {warehouse}",
            f"Show total quantity on hand at {warehouse}",
            f"Show number of distinct products at {warehouse}",
        ]

    # Categories-only (single-table on categories)
    prompts += [
        "Show categories that have subcategories",
        "Show all top-level categories",
    ]

    # Promotions-only (single-table on promotions)
    prompts += [
        f"Show all promotions and their discount percentages",
        f"Show promotions active in the last 30 days",
        f"Show the promotion with the highest discount percentage",
    ]

    # Orders-with-promotion-id (single-table on orders — promotion_id is on orders)
    for tf in TIMEFRAMES:
        prompts += [
            f"Show total revenue from orders with promotions for the {tf}",
            f"Show total revenue from orders without promotions for the {tf}",
            f"Show number of orders with a promotion for the {tf}",
        ]

    # Returns-only (single-table on returns)
    for tf in TIMEFRAMES:
        prompts += [
            f"Show total refund amount by return reason for the {tf}",
            f"Show number of returns by reason for the {tf}",
            f"Show total refund amount by status for the {tf}",
        ]
    for reason in RETURN_REASONS:
        prompts.append(f"Show total refund amount for {reason} returns in the last 30 days")

    # Shipping-only (single-table on shipping)
    for tf in TIMEFRAMES:
        prompts += [
            f"Show average shipping cost by carrier for the {tf}",
            f"Show number of shipments by carrier for the {tf}",
            f"Show average delivery time in days by carrier for the {tf}",
        ]
    for carrier in CARRIERS:
        prompts.append(f"Show total shipping cost for {carrier} in the last 30 days")

    # Suppliers-only (single-table on suppliers)
    prompts += [
        "Show suppliers with average lead time above 14 days",
        "Show suppliers based in the US",
        "Show total number of suppliers by country",
    ]

    # Products-only (single-table on products)
    prompts += [
        "Show products where cost price exceeds 50 percent of unit price",
        "Show 5 products with the lowest unit price",
        "Show 5 products with the highest cost price",
    ]

    # Regions-with-revenue-threshold (single-table on orders)
    for threshold in REVENUE_THRESHOLDS:
        prompts.append(f"Show regions with total revenue above {threshold} in the last 30 days")

    # Revenue-by-channel-and-region (single-table on orders — both cols on orders)
    prompts += [
        "Show total revenue by channel and region for the last 30 days",
        "Show total revenue by channel and region for the last 60 days",
    ]

    # Warehouse-with-most-reorders (single-table on inventory)
    prompts += [
        "Show which warehouse has the most products below reorder level",
    ]

    return prompts


def _tier_2_two_tables() -> list[str]:
    """Two-table joins — medium complexity."""
    prompts: list[str] = []

    # orders + customers
    for tf in TIMEFRAMES:
        prompts += [
            f"Show total revenue by customer segment for the {tf}",
            f"Show top 5 customers by total revenue for the {tf}",
            f"Show number of orders by country for the {tf}",
            f"Show average order value by customer segment for the {tf}",
            f"Show top 5 countries by number of orders for the {tf}",
        ]

    for n in LIMITS:
        for tf in TIMEFRAMES:
            prompts.append(f"Show top {n} customers by total revenue for the {tf}")
            prompts.append(f"Show top {n} countries by number of orders for the {tf}")

    for customer in CUSTOMERS:
        for tf in TIMEFRAMES:
            prompts.append(f"Show total amount spent by {customer} in the {tf}")
            prompts.append(f"How many orders did {customer} place in the {tf}")

    # orders + customers (segment/country revenue)
    for segment in SEGMENTS:
        for tf in TIMEFRAMES:
            prompts.append(f"Show total revenue from {segment} customers for the {tf}")
            prompts.append(f"Show number of orders from {segment} customers for the {tf}")

    for country in COUNTRIES:
        for tf in TIMEFRAMES:
            prompts.append(f"Show total revenue from customers in {country} for the {tf}")

    # orders + order_items (revenue from line items)
    for tf in TIMEFRAMES:
        prompts += [
            f"Show average discount applied per order for the {tf}",
        ]

    # orders + promotions (join for promotion name)
    for tf in TIMEFRAMES:
        prompts += [
            f"Show number of orders by promotion name for the {tf}",
            f"Show average order value for orders with vs without promotions for the {tf}",
            f"Show which promotion generated the most revenue in the {tf}",
        ]

    # products + categories
    prompts += [
        "Show total number of products by category",
        "Show products in the Electronics category",
    ]

    # products + suppliers
    prompts += [
        "Show total number of products supplied by each supplier",
        "Show products supplied by TechSource Inc",
        "Show average product cost price by supplier",
        "Show suppliers with average lead time above 14 days and their products",
    ]

    # inventory + products (for unit_price to compute inventory value)
    for warehouse in WAREHOUSES:
        prompts.append(f"Show total inventory value at {warehouse}")

    # orders + shipping
    for tf in TIMEFRAMES:
        prompts += [
            f"Show orders with no shipping record for the {tf}",
            f"Show average days between order date and delivered date by carrier for the {tf}",
            f"Show total shipping cost by region for the {tf}",
        ]
    for carrier in CARRIERS:
        for tf in TIMEFRAMES:
            prompts.append(f"Show number of {carrier} shipments by region in the {tf}")

    # orders + returns
    for tf in TIMEFRAMES:
        prompts += [
            f"Show total revenue lost to returns for the {tf}",
            f"Show return rate by region for the {tf}",
        ]
    for reason in RETURN_REASONS:
        for tf in TIMEFRAMES:
            prompts.append(f"Show number of {reason} returns by region in the {tf}")

    # HAVING clauses — orders + customers
    for threshold in REVENUE_THRESHOLDS:
        for tf in TIMEFRAMES:
            prompts.append(f"Show customers who spent more than {threshold} in the {tf}")

    for threshold in ORDER_COUNT_THRESHOLDS:
        for tf in TIMEFRAMES:
            prompts.append(f"Show customers who placed more than {threshold} orders in the {tf}")

    # Negation
    prompts += [
        "Show customers who have never placed an order",
        "Show products that have never been ordered",
        "Show orders with no associated shipping record",
    ]

    # Complex multi-filter (2-table)
    prompts += [
        "Show total revenue from Corporate customers in the EMEA region for the last 30 days",
        "Show customers who ordered from more than one channel in the last 60 days",
    ]

    return prompts


def _tier_3_multi_table() -> list[str]:
    """3+ table joins — hard tier. CRITICAL for balanced SFT distribution."""
    prompts: list[str] = []

    # orders + order_items + products
    for tf in TIMEFRAMES:
        prompts += [
            f"Show total revenue by product category for the {tf}",
            f"Show best selling products by quantity sold for the {tf}",
            f"Show total revenue by product name for the {tf}",
            f"Show average discount applied by product category for the {tf}",
            f"Show number of units sold by category for the {tf}",
            f"Show total gross profit by product category for the {tf}",
        ]

    for n in LIMITS:
        for tf in TIMEFRAMES:
            prompts.append(f"Show top {n} products by total revenue in the {tf}")
            prompts.append(f"Show top {n} products by quantity sold in the {tf}")
            prompts.append(f"Show top {n} categories by units sold in the {tf}")

    # orders + order_items + products (revenue by product)
    for product in PRODUCTS:
        for tf in TIMEFRAMES:
            prompts.append(f"Show total revenue from {product} for the {tf}")
            prompts.append(f"How many {product} units were sold in the {tf}")

    # orders + order_items + products + categories (revenue by category)
    for cat in CATEGORIES:
        for tf in TIMEFRAMES:
            prompts.append(f"Show total revenue from {cat} category for the {tf}")
            prompts.append(f"Show number of {cat} items sold for the {tf}")
            prompts.append(f"Show top 3 products in {cat} by revenue for the {tf}")

    # orders + customers + returns
    for tf in TIMEFRAMES:
        prompts += [
            f"Show customers with more than 2 returns in the {tf}",
            f"Show total refund amount by customer for the {tf}",
            f"Show return rate by customer segment for the {tf}",
        ]

    # orders + order_items + products + returns
    for tf in TIMEFRAMES:
        prompts += [
            f"Show return rate by product name for the {tf}",
            f"Show number of returns by product category for the {tf}",
            f"Show total refund amount by product category for the {tf}",
        ]

    # orders + shipping + customers
    for tf in TIMEFRAMES:
        prompts += [
            f"Show average delivery time by customer segment for the {tf}",
            f"Show total shipping cost by customer country for the {tf}",
        ]

    # orders + order_items + products + suppliers
    prompts += [
        "Show total revenue by supplier for the last 30 days",
        "Show top 5 products by gross profit margin in the last 60 days",
    ]

    # orders + order_items + products + inventory
    for warehouse in WAREHOUSES:
        prompts += [
            f"Show products below reorder level at {warehouse} that were sold in the last 30 days",
            f"Show total inventory value at {warehouse} by product category",
            f"Show inventory quantity by product for Electronics at {warehouse}",
        ]

    # orders + customers + promotions + order_items
    for tf in TIMEFRAMES:
        prompts += [
            f"Show total discount amount applied by promotion in the {tf}",
            f"Show number of orders by promotion and customer segment for the {tf}",
        ]

    # Complex multi-filter (3+ tables) — uses VALID statuses only
    prompts += [
        "Show delivered Electronics orders from US customers in the last 14 days",  # was: completed
        "Show returned Furniture orders from EMEA in the last 60 days",  # was: refunded
        "Show customers who have not returned any orders in the last 90 days",
    ]

    return prompts


def generate_variable_prompts() -> list[tuple[str, str]]:
    """
    Returns list of (prompt, category) tuples.
    Category is one of: '1_table', '2_tables', '3plus_tables'.
    """
    out: list[tuple[str, str]] = []
    for p in _tier_1_single_table():
        out.append((p, "1_table"))
    for p in _tier_2_two_tables():
        out.append((p, "2_tables"))
    for p in _tier_3_multi_table():
        out.append((p, "3plus_tables"))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Strategy 2 — Paraphrasing
# Now with REAL variable randomization — iterates over multiple value
# combinations per template instead of using a single hardcoded fill_vars.
# ──────────────────────────────────────────────────────────────────────────────

# Template → variable pool mapping.
# Each template's variables get filled with RANDOM values from these pools,
# producing many distinct prompts per template.
TEMPLATE_VAR_POOLS = {
    "tf": TIMEFRAMES,
    "region": REGIONS,
    "country": COUNTRIES,
    "category": CATEGORIES,
    "product": PRODUCTS,
    "customer": CUSTOMERS,
    "n": [str(n) for n in LIMITS],
    "threshold": [str(t) for t in REVENUE_THRESHOLDS],
    "order_count_threshold": [str(t) for t in ORDER_COUNT_THRESHOLDS],
    "status": ORDER_STATUSES,
    "return_status": RETURN_STATUSES,
    "return_reason": RETURN_REASONS,
    "carrier": CARRIERS,
    "warehouse": WAREHOUSES,
    "segment": SEGMENTS,
    "channel": CHANNELS,
    "supplier": SUPPLIERS,
    "promotion": PROMOTIONS,
}

PARAPHRASE_TEMPLATES = {
    "casual": [
        "how much rev did {region} make in {tf}?",
        "what's the total rev for {category} in {tf}?",
        "show me top {n} customers spending in {tf}",
        "which regions went over {threshold} in {tf}?",
        "how much did {country} customers spend in {tf}?",
        "how many {status} orders by country in {tf}?",
        "what's the avg order for {category}?",
        "break down rev by country and category for {tf}",
        "what got returned in {tf}?",
        "who ordered more than {order_count_threshold} times in {tf}?",
        "top {n} categories by order count in {tf}?",
        "total rev skipping pending and cancelled in {tf}",
        "who spent the most in {tf}?",
        "avg order by region, highest first in {tf}?",
        "bottom {n} countries by rev in {tf}?",
        "what categories did {region} customers buy in {tf}?",
        "how much did {country} customers spend on {category}?",
        "who spent over {threshold} on {category} in {tf}?",
        "show {status} orders by {carrier} in {tf}",
        "show inventory at {warehouse}",
        "show me {segment} customers' revenue in {tf}",
        "how many {category} items sold via {channel} in {tf}?",
        "show top {n} products by revenue in {tf}",
    ],
    "formal": [
        "Calculate total revenue aggregated by region for the {tf}",
        "Provide a breakdown of total revenue by product category for the {tf}",
        "Identify the top {n} customers ranked by total revenue for the {tf}",
        "Determine which regions exceeded {threshold} in total revenue for the {tf}",
        "Calculate total revenue attributable to customers from {country} for the {tf}",
        "Count {status} orders grouped by country of customer origin for the {tf}",
        "Calculate the mean order value for the {category} product category",
        "Generate a revenue summary grouped by country and product category for the {tf}",
        "Retrieve all product names associated with {return_reason} returns in the {tf}",
        "Identify customers who placed in excess of {order_count_threshold} orders during the {tf}",
        "Rank the top {n} product categories by cumulative order volume for the {tf}",
        "Calculate total revenue excluding pending and cancelled orders for the {tf}",
        "Identify the customer with the highest cumulative expenditure for the {tf}",
        "Rank regions by average order value in descending order for the {tf}",
        "Identify the {n} lowest-performing countries by total revenue for the {tf}",
        "List distinct product categories purchased by {region}-based customers for the {tf}",
        "Calculate {category} revenue attributable to {country}-based customers for the {tf}",
        "Identify customers whose {category} expenditure exceeded {threshold} in the {tf}",
        "Compute total revenue from {status} orders shipped via {carrier} for the {tf}",
        "Summarize inventory levels at the {warehouse} warehouse",
        "Aggregate revenue by customer segment for {segment} customers in the {tf}",
        "Calculate units sold for {category} products via the {channel} channel in the {tf}",
    ],
    "business": [
        "What is the gross revenue performance by region for the {tf}?",
        "What is the revenue contribution of each product category for the {tf}?",
        "Who are our top {n} revenue-generating customers for the {tf}?",
        "Which regions are performing above the {threshold} revenue threshold for the {tf}?",
        "What is the total customer spend from {country} for the {tf}?",
        "What is our {status} order count by country for the {tf}?",
        "What is the average transaction size for our {category} segment?",
        "What does our revenue look like broken down by country and product line for the {tf}?",
        "What is our product return volume by reason for the {tf}?",
        "Which customers are repeat purchasers (more than {order_count_threshold} orders) within the {tf}?",
        "What are our top {n} product lines by volume for the {tf}?",
        "What is our revenue excluding pending and cancelled orders for the {tf}?",
        "Who is our highest-value customer by spend in the {tf}?",
        "How does average order value rank across our regional segments for the {tf}?",
        "Which markets are underperforming in revenue for the {tf}?",
        "What product categories are gaining traction in the {region} market for the {tf}?",
        "What is the {category} revenue contribution from our {country} customer segment for the {tf}?",
        "Which customers exceeded the {threshold} spend threshold in {category} for the {tf}?",
        "What is our {carrier} shipment volume by region for the {tf}?",
        "What is our inventory position at the {warehouse} distribution center?",
        "What is the revenue contribution of our {segment} customers for the {tf}?",
        "What is the {category} sales volume via our {channel} channel for the {tf}?",
    ],
}


def _extract_template_vars(template: str) -> list[str]:
    """Extract {var} names from a template string."""
    import re
    return re.findall(r"\{(\w+)\}", template)


def _random_fill(template: str, rng: random.Random) -> str:
    """
    Fill template variables with RANDOM values from TEMPLATE_VAR_POOLS.
    Falls back to leaving the placeholder if a var has no pool (defensive).
    """
    var_names = _extract_template_vars(template)
    fill: dict[str, str] = {}
    for v in var_names:
        pool = TEMPLATE_VAR_POOLS.get(v)
        if pool:
            fill[v] = rng.choice(pool)
        else:
            # Unknown var — leave as-is so it's visible in output (and gets
            # caught by dedup naturally).
            fill[v] = "{" + v + "}"
    try:
        return template.format(**fill)
    except (KeyError, IndexError):
        return template


def generate_paraphrase_prompts(
    rng: random.Random,
    fills_per_template: int = 4,
) -> list[tuple[str, str]]:
    """
    Generate paraphrases by filling each template with MULTIPLE random value
    combinations. Returns list of (prompt, category) tuples.

    Category is 'paraphrase_<style>' for downstream filtering.
    """
    out: list[tuple[str, str]] = []
    for style, templates in PARAPHRASE_TEMPLATES.items():
        for template in templates:
            seen_for_template: set[str] = set()
            # Try up to fills_per_template * 2 attempts to get distinct fills
            attempts = 0
            while len(seen_for_template) < fills_per_template and attempts < fills_per_template * 4:
                filled = _random_fill(template, rng)
                # Quick lint: skip if any placeholder remains unfilled
                if "{" in filled and "}" in filled:
                    attempts += 1
                    continue
                key = filled.strip().lower()
                if key not in seen_for_template:
                    seen_for_template.add(key)
                    out.append((filled, f"paraphrase_{style}"))
                attempts += 1
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Strategy 3 — Edge Cases
# FILTERED: removed prompts that reference nonexistent tables/columns.
# Kept: ambiguous timeframes, extreme limits, negation logic, missing filters.
# ──────────────────────────────────────────────────────────────────────────────

def generate_edge_cases() -> list[tuple[str, str]]:
    """
    Returns (prompt, category) tuples.

    Removed (would teach the student to hallucinate):
      - "Show total revenue from the inventory table" (no revenue col)
      - "List all events from the activity_log table" (table doesn't exist)
      - "Show total discount applied per order" (no discount col on orders)
    """
    cases = [
        # Ambiguous timeframe — tests robustness to vague language
        "Show total revenue for this year",
        "Show all orders from last quarter",
        "What was revenue like in the last month",
        "Show total revenue for the previous 30 days",

        # Extreme limits
        "Show top 100 customers by revenue in the last 30 days",
        "Show top 1 product by total revenue in the last 7 days",

        # No timeframe — should return ALL data
        "Show total revenue by region",
        "Show all customers who have ever placed an order",
        "What is the total revenue from Electronics",
        "Show average order value by product category",
        "Who placed the most orders ever",
        "Show all products and their suppliers",

        # Negation logic
        "Show customers who have never placed a delivered order",
        "Show products that have never been returned",
        "Which regions had zero cancelled orders in the last 30 days",
        "Show customers with no returns in the last 90 days",
        "Show products never ordered via the mobile channel",

        # Combined multi-filters (these are HARD — good for stress testing)
        "Show delivered Electronics orders from US customers in the last 14 days",
        "Show returned Furniture orders from EMEA in the last 60 days",
        "Show pending Stationery orders from customers in Japan in the last 30 days",
        "Show shipped Sports products from APAC customers in the last 45 days",

        # Aggregation edge cases
        "Show the difference between delivered and cancelled revenue by region for the last 30 days",
        "Show total revenue as a percentage of overall revenue by product category for the last 30 days",
        "Show month over month revenue change by region for the last 90 days",
        "Show the ratio of returned to delivered orders by category",

        # Superlative queries
        "Who is the single most valuable customer based on total amount spent",
        "Which product has the highest gross profit margin",
        "Which warehouse has the most products below reorder level",
        "Which carrier has the fastest average delivery time",

        # Pagination-style (LIMIT without ranking)
        "Show the first 10 customers alphabetically",
        "Show 5 products with the lowest unit price",
        "Show 5 products with the highest cost price",
    ]
    return [(p, "edge_case") for p in cases]


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def _dedup(prompts: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Deduplicate by normalized prompt text. Keep first occurrence's category."""
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for prompt, cat in prompts:
        key = " ".join(prompt.lower().split())
        if key in seen:
            continue
        seen.add(key)
        unique.append((prompt, cat))
    return unique


def _oversample_stratified(
    prompts: list[tuple[str, str]],
    target: int,
    rng: random.Random,
) -> list[tuple[str, str]]:
    """
    Stratified oversample to hit target count while preserving category distribution.

    Old version used uniform random choice, which over-sampled whatever category
    had the most prompts. This version computes how many extras each category
    needs (proportional to its share) and samples within each category.
    """
    if len(prompts) >= target:
        return prompts

    # Group by category
    by_cat: dict[str, list[tuple[str, str]]] = {}
    for p in prompts:
        by_cat.setdefault(p[1], []).append(p)

    # Compute proportional target per category
    total = len(prompts)
    extra: list[tuple[str, str]] = []
    for cat, cat_prompts in by_cat.items():
        cat_share = len(cat_prompts) / total
        cat_target = round(target * cat_share) - len(cat_prompts)
        cat_target = max(0, cat_target)
        for _ in range(cat_target):
            extra.append(rng.choice(cat_prompts))

    # If still short (rounding), fill from largest category
    while len(prompts) + len(extra) < target:
        largest_cat = max(by_cat, key=lambda c: len(by_cat[c]))
        extra.append(rng.choice(by_cat[largest_cat]))

    return prompts + extra


def _balance(
    prompts: list[tuple[str, str]],
    cap: int,
    rng: random.Random,
) -> list[tuple[str, str]]:
    """
    Cap each category at `cap` prompts. Samples randomly if over cap.
    Mirrors the build_sft_dataset.py bucket-capping logic at prompt-generation
    time, so you can preview the balanced distribution before the Groq run.
    """
    by_cat: dict[str, list[tuple[str, str]]] = {}
    for p in prompts:
        by_cat.setdefault(p[1], []).append(p)

    out: list[tuple[str, str]] = []
    for cat, cat_prompts in by_cat.items():
        if len(cat_prompts) > cap:
            out.extend(rng.sample(cat_prompts, cap))
        else:
            out.extend(cat_prompts)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate schema-aligned NL prompts for the e-commerce pipeline.",
    )
    parser.add_argument(
        "--output",
        default="dataset/ecommerce_prompts.txt",
        help="Output file path (default: dataset/ecommerce_prompts.txt)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=0,
        help="If set, oversample with replacement to hit N prompts",
    )
    parser.add_argument(
        "--fills-per-template",
        type=int,
        default=4,
        help="Number of random fills per paraphrase template (default: 4)",
    )
    parser.add_argument(
        "--no-paraphrases",
        action="store_true",
        help="Skip paraphrase generation (faster, less diverse)",
    )
    parser.add_argument(
        "--balance",
        type=int,
        default=0,
        help="Cap each category at N prompts (0 = no cap). Mirrors build_sft_dataset.py.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # ── generate ──────────────────────────────────────────────────────────────
    all_prompts: list[tuple[str, str]] = []

    variable_prompts = generate_variable_prompts()
    all_prompts.extend(variable_prompts)
    print(f"  Variable randomization : {len(variable_prompts)} prompts")

    if not args.no_paraphrases:
        paraphrase_prompts = generate_paraphrase_prompts(rng, args.fills_per_template)
        all_prompts.extend(paraphrase_prompts)
        print(f"  Paraphrases            : {len(paraphrase_prompts)} prompts "
              f"({args.fills_per_template} fills/template)")

    edge_prompts = generate_edge_cases()
    all_prompts.extend(edge_prompts)
    print(f"  Edge cases             : {len(edge_prompts)} prompts")

    print(f"  ─────────────────────────────────────")
    print(f"  Total raw              : {len(all_prompts)} prompts")

    # ── dedup ─────────────────────────────────────────────────────────────────
    unique = _dedup(all_prompts)
    print(f"  Unique (after dedup)   : {len(unique)} prompts")

    # ── balance if requested ──────────────────────────────────────────────────
    if args.balance > 0:
        before = len(unique)
        unique = _balance(unique, args.balance, rng)
        print(f"  After balance (cap={args.balance}) : {len(unique)} prompts (was {before})")

    # ── oversample if requested ───────────────────────────────────────────────
    if args.target > 0 and len(unique) < args.target:
        unique = _oversample_stratified(unique, args.target, rng)
        print(f"  After oversample       : {len(unique)} prompts (target={args.target})")

    # ── shuffle ───────────────────────────────────────────────────────────────
    rng.shuffle(unique)

    # ── write ─────────────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Write one prompt per line (no category — that's for stats only)
    # IMPORTANT: trailing newline is required so wc -l reports the correct
    # count and so generate_dataset_v3.py's read loop doesn't miss the last
    # prompt on some Python versions.
    output_path.write_text("\n".join(p for p, _ in unique) + "\n")

    # ── stats ─────────────────────────────────────────────────────────────────
    cat_counts = Counter(cat for _, cat in unique)
    print()
    print(f"✅ Output written → {output_path}")
    print()
    print("Category distribution:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        pct = count / len(unique) * 100
        print(f"  {cat:<25} : {count:>4}  ({pct:.1f}%)")

    # Approximate complexity distribution (rough estimate via FROM/JOIN count)
    # — matches what build_sft_dataset.py will compute downstream
    n_1t = sum(1 for c in cat_counts if c == "1_table")
    n_2t = sum(1 for c in cat_counts if c == "2_tables")
    n_3t = sum(1 for c in cat_counts if c == "3plus_tables")
    print()
    print("Approx complexity distribution:")
    print(f"  1_table      : {cat_counts.get('1_table', 0):>4}")
    print(f"  2_tables     : {cat_counts.get('2_tables', 0):>4}")
    print(f"  3plus_tables : {cat_counts.get('3plus_tables', 0):>4}")
    print(f"  paraphrases  : {sum(v for k, v in cat_counts.items() if k.startswith('paraphrase_')):>4}")
    print(f"  edge_cases   : {cat_counts.get('edge_case', 0):>4}")

    print()
    print("Next step:")
    print(f"  SKIP_LLM_VALIDATION=true python3 generate_dataset_v3.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
