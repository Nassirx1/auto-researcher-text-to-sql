from __future__ import annotations

import argparse
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from utils import ensure_dir, set_seed


REGIONS = ["North", "South", "East", "West", "Central"]
FIRST_NAMES = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Sam", "Jamie"]
LAST_NAMES = ["Adams", "Baker", "Carter", "Diaz", "Evans", "Foster", "Garcia", "Hughes"]


def random_date(rng: random.Random, start: date, end: date) -> str:
    delta = (end - start).days
    return (start + timedelta(days=rng.randint(0, delta))).isoformat()


def money(rng: random.Random, low: float, high: float) -> float:
    return round(rng.uniform(low, high), 2)


def connect(path: Path) -> sqlite3.Connection:
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys = ON")
    return con


def make_name(rng: random.Random) -> str:
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


def generate_ecommerce(db_dir: Path, seed: int) -> None:
    rng = random.Random(seed)
    con = connect(db_dir / "ecommerce.db")
    cur = con.cursor()
    cur.executescript(
        """
        CREATE TABLE regions(region_id INTEGER PRIMARY KEY, region_name TEXT NOT NULL);
        CREATE TABLE customers(
            customer_id INTEGER PRIMARY KEY,
            customer_name TEXT NOT NULL,
            region_id INTEGER NOT NULL,
            signup_date TEXT NOT NULL,
            FOREIGN KEY(region_id) REFERENCES regions(region_id)
        );
        CREATE TABLE categories(category_id INTEGER PRIMARY KEY, category_name TEXT NOT NULL);
        CREATE TABLE products(
            product_id INTEGER PRIMARY KEY,
            category_id INTEGER NOT NULL,
            product_name TEXT NOT NULL,
            price REAL NOT NULL,
            FOREIGN KEY(category_id) REFERENCES categories(category_id)
        );
        CREATE TABLE orders(
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            order_date TEXT NOT NULL,
            order_status TEXT NOT NULL,
            FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
        );
        CREATE TABLE order_items(
            order_item_id INTEGER PRIMARY KEY,
            order_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            unit_price REAL NOT NULL,
            FOREIGN KEY(order_id) REFERENCES orders(order_id),
            FOREIGN KEY(product_id) REFERENCES products(product_id)
        );
        CREATE TABLE payments(
            payment_id INTEGER PRIMARY KEY,
            order_id INTEGER NOT NULL,
            payment_date TEXT NOT NULL,
            amount REAL NOT NULL,
            payment_method TEXT NOT NULL,
            FOREIGN KEY(order_id) REFERENCES orders(order_id)
        );
        CREATE TABLE refunds(
            refund_id INTEGER PRIMARY KEY,
            order_id INTEGER NOT NULL,
            refund_date TEXT NOT NULL,
            amount REAL NOT NULL,
            reason TEXT NOT NULL,
            FOREIGN KEY(order_id) REFERENCES orders(order_id)
        );
        CREATE TABLE marketing_campaigns(
            campaign_id INTEGER PRIMARY KEY,
            campaign_name TEXT NOT NULL,
            channel TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            spend REAL NOT NULL
        );
        CREATE TABLE web_sessions(
            session_id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            campaign_id INTEGER,
            session_date TEXT NOT NULL,
            converted INTEGER NOT NULL,
            FOREIGN KEY(customer_id) REFERENCES customers(customer_id),
            FOREIGN KEY(campaign_id) REFERENCES marketing_campaigns(campaign_id)
        );
        CREATE TABLE support_tickets(
            ticket_id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            created_date TEXT NOT NULL,
            issue_type TEXT NOT NULL,
            resolved INTEGER NOT NULL,
            satisfaction_score INTEGER,
            FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
        );
        """
    )
    cur.executemany("INSERT INTO regions VALUES (?, ?)", enumerate(REGIONS, start=1))
    categories = ["Electronics", "Home", "Fashion", "Beauty", "Sports", "Books"]
    cur.executemany("INSERT INTO categories VALUES (?, ?)", enumerate(categories, start=1))
    customers = [
        (i, make_name(rng), rng.randint(1, len(REGIONS)), random_date(rng, date(2023, 1, 1), date(2025, 12, 31)))
        for i in range(1, 501)
    ]
    cur.executemany("INSERT INTO customers VALUES (?, ?, ?, ?)", customers)
    products = []
    for i in range(1, 121):
        cat_id = rng.randint(1, len(categories))
        products.append((i, cat_id, f"{categories[cat_id - 1]} Product {i}", money(rng, 8, 450)))
    cur.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", products)
    campaigns = []
    for i in range(1, 31):
        start = date(2024, 1, 1) + timedelta(days=20 * i)
        campaigns.append((i, f"Campaign {i}", rng.choice(["email", "search", "social", "affiliate"]), start.isoformat(), (start + timedelta(days=30)).isoformat(), money(rng, 500, 7000)))
    cur.executemany("INSERT INTO marketing_campaigns VALUES (?, ?, ?, ?, ?, ?)", campaigns)

    item_id = payment_id = refund_id = session_id = ticket_id = 1
    for order_id in range(1, 2001):
        customer_id = rng.randint(1, 500)
        order_date = random_date(rng, date(2024, 1, 1), date(2025, 12, 31))
        status = rng.choices(["completed", "completed", "completed", "cancelled"], weights=[45, 30, 20, 5])[0]
        cur.execute("INSERT INTO orders VALUES (?, ?, ?, ?)", (order_id, customer_id, order_date, status))
        order_total = 0.0
        for _ in range(rng.randint(1, 4)):
            product = rng.choice(products)
            quantity = rng.randint(1, 5)
            order_total += quantity * product[3]
            cur.execute("INSERT INTO order_items VALUES (?, ?, ?, ?, ?)", (item_id, order_id, product[0], quantity, product[3]))
            item_id += 1
        if status == "completed":
            cur.execute("INSERT INTO payments VALUES (?, ?, ?, ?, ?)", (payment_id, order_id, order_date, round(order_total, 2), rng.choice(["card", "paypal", "bank_transfer"])))
            payment_id += 1
            if rng.random() < 0.08:
                cur.execute("INSERT INTO refunds VALUES (?, ?, ?, ?, ?)", (refund_id, order_id, random_date(rng, date.fromisoformat(order_date), date(2025, 12, 31)), round(order_total * rng.uniform(0.2, 1.0), 2), rng.choice(["damaged", "late", "wrong_item", "buyer_remorse"])))
                refund_id += 1

    for _ in range(3000):
        cur.execute("INSERT INTO web_sessions VALUES (?, ?, ?, ?, ?)", (session_id, rng.randint(1, 500), rng.randint(1, 30), random_date(rng, date(2024, 1, 1), date(2025, 12, 31)), 1 if rng.random() < 0.12 else 0))
        session_id += 1
    for _ in range(700):
        resolved = 1 if rng.random() < 0.82 else 0
        cur.execute("INSERT INTO support_tickets VALUES (?, ?, ?, ?, ?, ?)", (ticket_id, rng.randint(1, 500), random_date(rng, date(2024, 1, 1), date(2025, 12, 31)), rng.choice(["delivery", "refund", "product", "billing"]), resolved, rng.randint(1, 5) if resolved else None))
        ticket_id += 1
    con.commit()
    con.close()


def generate_saas(db_dir: Path, seed: int) -> None:
    rng = random.Random(seed)
    con = connect(db_dir / "saas.db")
    cur = con.cursor()
    cur.executescript(
        """
        CREATE TABLE accounts(account_id INTEGER PRIMARY KEY, account_name TEXT, industry TEXT, signup_date TEXT);
        CREATE TABLE users(user_id INTEGER PRIMARY KEY, account_id INTEGER, user_name TEXT, active INTEGER, created_date TEXT, FOREIGN KEY(account_id) REFERENCES accounts(account_id));
        CREATE TABLE plans(plan_id INTEGER PRIMARY KEY, plan_name TEXT, monthly_price REAL);
        CREATE TABLE subscriptions(subscription_id INTEGER PRIMARY KEY, account_id INTEGER, plan_id INTEGER, start_date TEXT, end_date TEXT, status TEXT, FOREIGN KEY(account_id) REFERENCES accounts(account_id), FOREIGN KEY(plan_id) REFERENCES plans(plan_id));
        CREATE TABLE invoices(invoice_id INTEGER PRIMARY KEY, account_id INTEGER, invoice_date TEXT, due_date TEXT, amount REAL, status TEXT, FOREIGN KEY(account_id) REFERENCES accounts(account_id));
        CREATE TABLE payments(payment_id INTEGER PRIMARY KEY, invoice_id INTEGER, payment_date TEXT, amount REAL, FOREIGN KEY(invoice_id) REFERENCES invoices(invoice_id));
        CREATE TABLE product_usage(usage_id INTEGER PRIMARY KEY, account_id INTEGER, user_id INTEGER, usage_date TEXT, feature_name TEXT, events INTEGER, FOREIGN KEY(account_id) REFERENCES accounts(account_id), FOREIGN KEY(user_id) REFERENCES users(user_id));
        CREATE TABLE support_tickets(ticket_id INTEGER PRIMARY KEY, account_id INTEGER, created_date TEXT, severity TEXT, resolved INTEGER, FOREIGN KEY(account_id) REFERENCES accounts(account_id));
        CREATE TABLE churn_events(churn_id INTEGER PRIMARY KEY, account_id INTEGER, churn_date TEXT, reason TEXT, FOREIGN KEY(account_id) REFERENCES accounts(account_id));
        """
    )
    plans = [(1, "Starter", 49.0), (2, "Growth", 149.0), (3, "Business", 399.0), (4, "Enterprise", 999.0)]
    cur.executemany("INSERT INTO plans VALUES (?, ?, ?)", plans)
    industries = ["Retail", "Finance", "Healthcare", "Software", "Manufacturing"]
    for account_id in range(1, 301):
        cur.execute("INSERT INTO accounts VALUES (?, ?, ?, ?)", (account_id, f"Account {account_id}", rng.choice(industries), random_date(rng, date(2023, 1, 1), date(2025, 6, 30))))
        plan = rng.choice(plans)
        status = "churned" if rng.random() < 0.12 else "active"
        start = random_date(rng, date(2023, 1, 1), date(2025, 6, 30))
        end = random_date(rng, date.fromisoformat(start), date(2025, 12, 31)) if status == "churned" else None
        cur.execute("INSERT INTO subscriptions VALUES (?, ?, ?, ?, ?, ?)", (account_id, account_id, plan[0], start, end, status))
        if status == "churned":
            cur.execute("INSERT INTO churn_events VALUES (?, ?, ?, ?)", (account_id, account_id, end, rng.choice(["price", "low_usage", "support", "competitor"])))
    for user_id in range(1, 1001):
        account_id = rng.randint(1, 300)
        cur.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?)", (user_id, account_id, make_name(rng), 1 if rng.random() < 0.78 else 0, random_date(rng, date(2023, 1, 1), date(2025, 12, 31))))
    for invoice_id in range(1, 1501):
        account_id = rng.randint(1, 300)
        plan = rng.choice(plans)
        invoice_date = random_date(rng, date(2024, 1, 1), date(2025, 12, 31))
        status = rng.choices(["paid", "paid", "paid", "unpaid", "void"], weights=[40, 25, 20, 12, 3])[0]
        amount = money(rng, plan[2] * 0.8, plan[2] * 1.4)
        cur.execute("INSERT INTO invoices VALUES (?, ?, ?, ?, ?, ?)", (invoice_id, account_id, invoice_date, (date.fromisoformat(invoice_date) + timedelta(days=30)).isoformat(), amount, status))
        if status == "paid":
            cur.execute("INSERT INTO payments VALUES (?, ?, ?, ?)", (invoice_id, invoice_id, random_date(rng, date.fromisoformat(invoice_date), date(2025, 12, 31)), amount))
    for usage_id in range(1, 6001):
        account_id = rng.randint(1, 300)
        cur.execute("INSERT INTO product_usage VALUES (?, ?, ?, ?, ?, ?)", (usage_id, account_id, rng.randint(1, 1000), random_date(rng, date(2024, 1, 1), date(2025, 12, 31)), rng.choice(["dashboard", "export", "automation", "api", "reporting"]), rng.randint(1, 200)))
    for ticket_id in range(1, 601):
        cur.execute("INSERT INTO support_tickets VALUES (?, ?, ?, ?, ?)", (ticket_id, rng.randint(1, 300), random_date(rng, date(2024, 1, 1), date(2025, 12, 31)), rng.choice(["low", "medium", "high", "critical"]), 1 if rng.random() < 0.84 else 0))
    con.commit()
    con.close()


def generate_banking(db_dir: Path, seed: int) -> None:
    rng = random.Random(seed)
    con = connect(db_dir / "banking.db")
    cur = con.cursor()
    cur.executescript(
        """
        CREATE TABLE branches(branch_id INTEGER PRIMARY KEY, branch_name TEXT, region TEXT);
        CREATE TABLE customers(customer_id INTEGER PRIMARY KEY, customer_name TEXT, region TEXT, signup_date TEXT);
        CREATE TABLE accounts(account_id INTEGER PRIMARY KEY, customer_id INTEGER, branch_id INTEGER, account_type TEXT, opened_date TEXT, balance REAL, FOREIGN KEY(customer_id) REFERENCES customers(customer_id), FOREIGN KEY(branch_id) REFERENCES branches(branch_id));
        CREATE TABLE transactions(transaction_id INTEGER PRIMARY KEY, account_id INTEGER, transaction_date TEXT, transaction_type TEXT, amount REAL, channel TEXT, suspicious INTEGER, FOREIGN KEY(account_id) REFERENCES accounts(account_id));
        CREATE TABLE cards(card_id INTEGER PRIMARY KEY, account_id INTEGER, card_type TEXT, issued_date TEXT, active INTEGER, FOREIGN KEY(account_id) REFERENCES accounts(account_id));
        CREATE TABLE loans(loan_id INTEGER PRIMARY KEY, customer_id INTEGER, branch_id INTEGER, loan_type TEXT, principal REAL, interest_rate REAL, risk_grade TEXT, status TEXT, FOREIGN KEY(customer_id) REFERENCES customers(customer_id), FOREIGN KEY(branch_id) REFERENCES branches(branch_id));
        CREATE TABLE fraud_alerts(alert_id INTEGER PRIMARY KEY, transaction_id INTEGER, alert_date TEXT, alert_type TEXT, resolved INTEGER, FOREIGN KEY(transaction_id) REFERENCES transactions(transaction_id));
        """
    )
    branches = [(i, f"Branch {i}", REGIONS[(i - 1) % len(REGIONS)]) for i in range(1, 21)]
    cur.executemany("INSERT INTO branches VALUES (?, ?, ?)", branches)
    for customer_id in range(1, 501):
        cur.execute("INSERT INTO customers VALUES (?, ?, ?, ?)", (customer_id, make_name(rng), rng.choice(REGIONS), random_date(rng, date(2020, 1, 1), date(2025, 12, 31))))
    for account_id in range(1, 801):
        cur.execute("INSERT INTO accounts VALUES (?, ?, ?, ?, ?, ?)", (account_id, rng.randint(1, 500), rng.randint(1, 20), rng.choice(["checking", "savings", "business"]), random_date(rng, date(2020, 1, 1), date(2025, 12, 31)), money(rng, 100, 60000)))
    alert_id = 1
    for transaction_id in range(1, 5001):
        amount_value = money(rng, 5, 12000)
        suspicious = 1 if (amount_value > 9000 or rng.random() < 0.025) else 0
        cur.execute("INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?, ?)", (transaction_id, rng.randint(1, 800), random_date(rng, date(2024, 1, 1), date(2025, 12, 31)), rng.choice(["deposit", "withdrawal", "purchase", "transfer"]), amount_value, rng.choice(["branch", "atm", "mobile", "web", "card"]), suspicious))
        if suspicious:
            cur.execute("INSERT INTO fraud_alerts VALUES (?, ?, ?, ?, ?)", (alert_id, transaction_id, random_date(rng, date(2024, 1, 1), date(2025, 12, 31)), rng.choice(["velocity", "large_amount", "geo_mismatch"]), 1 if rng.random() < 0.7 else 0))
            alert_id += 1
    for card_id in range(1, 701):
        cur.execute("INSERT INTO cards VALUES (?, ?, ?, ?, ?)", (card_id, rng.randint(1, 800), rng.choice(["debit", "credit"]), random_date(rng, date(2021, 1, 1), date(2025, 12, 31)), 1 if rng.random() < 0.9 else 0))
    for loan_id in range(1, 401):
        cur.execute("INSERT INTO loans VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (loan_id, rng.randint(1, 500), rng.randint(1, 20), rng.choice(["mortgage", "auto", "personal", "business"]), money(rng, 2000, 400000), round(rng.uniform(0.035, 0.16), 4), rng.choice(["A", "B", "C", "D"]), rng.choice(["current", "current", "delinquent", "paid_off"])))
    con.commit()
    con.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/databases")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    db_dir = ensure_dir(args.output_dir)
    generate_ecommerce(db_dir, args.seed + 1)
    generate_saas(db_dir, args.seed + 2)
    generate_banking(db_dir, args.seed + 3)
    print(f"Generated SQLite databases in {db_dir}")


if __name__ == "__main__":
    main()
