from __future__ import annotations

import argparse
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from evaluate_sql import execute_sql
from schema_utils import schema_context
from utils import project_path, set_seed, write_jsonl


Template = Callable[[random.Random], tuple[str, str, str]]


def ecommerce_templates() -> list[Template]:
    return [
        lambda r: ("easy", "How many customers are in each region?", "SELECT r.region_name, COUNT(*) AS customer_count FROM customers c JOIN regions r ON c.region_id = r.region_id GROUP BY r.region_name"),
        lambda r: ("easy", "How many ecommerce orders have each status?", "SELECT order_status, COUNT(*) AS order_count FROM orders GROUP BY order_status"),
        lambda r: (lambda threshold: ("easy", f"List products priced above {threshold} dollars.", f"SELECT product_name, price FROM products WHERE price > {threshold} ORDER BY price DESC"))(r.randint(50, 300)),
        lambda r: ("medium", "Which product categories generated the most completed order revenue?", "SELECT c.category_name, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue FROM order_items oi JOIN products p ON oi.product_id = p.product_id JOIN categories c ON p.category_id = c.category_id JOIN orders o ON oi.order_id = o.order_id WHERE o.order_status = 'completed' GROUP BY c.category_name ORDER BY revenue DESC"),
        lambda r: ("medium", "What is the refund rate by product category?", "SELECT c.category_name, ROUND(COUNT(DISTINCT rf.refund_id) * 1.0 / COUNT(DISTINCT o.order_id), 4) AS refund_rate FROM categories c JOIN products p ON c.category_id = p.category_id JOIN order_items oi ON p.product_id = oi.product_id JOIN orders o ON oi.order_id = o.order_id LEFT JOIN refunds rf ON o.order_id = rf.order_id GROUP BY c.category_name ORDER BY refund_rate DESC"),
        lambda r: ("medium", "Which regions produced the highest completed order revenue?", "SELECT r.region_name, ROUND(SUM(pay.amount), 2) AS revenue FROM payments pay JOIN orders o ON pay.order_id = o.order_id JOIN customers c ON o.customer_id = c.customer_id JOIN regions r ON c.region_id = r.region_id GROUP BY r.region_name ORDER BY revenue DESC"),
        lambda r: ("medium", "Which marketing channels have the highest conversion rate?", "SELECT mc.channel, ROUND(SUM(ws.converted) * 1.0 / COUNT(*), 4) AS conversion_rate FROM web_sessions ws JOIN marketing_campaigns mc ON ws.campaign_id = mc.campaign_id GROUP BY mc.channel ORDER BY conversion_rate DESC"),
        lambda r: ("hard", "What was month over month ecommerce revenue growth in 2025?", "WITH monthly AS (SELECT substr(payment_date, 1, 7) AS month, SUM(amount) AS revenue FROM payments WHERE payment_date >= '2025-01-01' GROUP BY substr(payment_date, 1, 7)), lagged AS (SELECT month, revenue, LAG(revenue) OVER (ORDER BY month) AS previous_revenue FROM monthly) SELECT month, ROUND(revenue, 2) AS revenue, ROUND((revenue - previous_revenue) * 100.0 / previous_revenue, 2) AS growth_pct FROM lagged WHERE previous_revenue IS NOT NULL"),
        lambda r: ("hard", "Which customers increased spending by more than 30 percent from 2024 to 2025?", "WITH spending AS (SELECT c.customer_id, c.customer_name, substr(pay.payment_date, 1, 4) AS year, SUM(pay.amount) AS revenue FROM customers c JOIN orders o ON c.customer_id = o.customer_id JOIN payments pay ON o.order_id = pay.order_id WHERE substr(pay.payment_date, 1, 4) IN ('2024', '2025') GROUP BY c.customer_id, c.customer_name, year), pivoted AS (SELECT customer_id, customer_name, SUM(CASE WHEN year = '2024' THEN revenue ELSE 0 END) AS revenue_2024, SUM(CASE WHEN year = '2025' THEN revenue ELSE 0 END) AS revenue_2025 FROM spending GROUP BY customer_id, customer_name) SELECT customer_name, ROUND(revenue_2024, 2) AS revenue_2024, ROUND(revenue_2025, 2) AS revenue_2025 FROM pivoted WHERE revenue_2024 > 0 AND revenue_2025 > revenue_2024 * 1.3 ORDER BY revenue_2025 DESC LIMIT 20"),
        lambda r: ("hard", "Which support issue types have the lowest average satisfaction score?", "SELECT issue_type, ROUND(AVG(satisfaction_score), 2) AS avg_satisfaction, COUNT(*) AS tickets FROM support_tickets WHERE satisfaction_score IS NOT NULL GROUP BY issue_type HAVING COUNT(*) >= 10 ORDER BY avg_satisfaction ASC"),
    ]


def saas_templates() -> list[Template]:
    return [
        lambda r: ("easy", "How many SaaS accounts are in each industry?", "SELECT industry, COUNT(*) AS account_count FROM accounts GROUP BY industry"),
        lambda r: ("easy", "List active SaaS subscriptions by plan.", "SELECT p.plan_name, COUNT(*) AS active_subscriptions FROM subscriptions s JOIN plans p ON s.plan_id = p.plan_id WHERE s.status = 'active' GROUP BY p.plan_name"),
        lambda r: ("easy", "How many unpaid invoices are there?", "SELECT COUNT(*) AS unpaid_invoices FROM invoices WHERE status = 'unpaid'"),
        lambda r: ("medium", "What monthly recurring revenue is represented by active subscriptions by plan?", "SELECT p.plan_name, ROUND(SUM(p.monthly_price), 2) AS mrr FROM subscriptions s JOIN plans p ON s.plan_id = p.plan_id WHERE s.status = 'active' GROUP BY p.plan_name ORDER BY mrr DESC"),
        lambda r: ("medium", "Which plans have the highest churned subscription count?", "SELECT p.plan_name, COUNT(*) AS churned_accounts FROM subscriptions s JOIN plans p ON s.plan_id = p.plan_id WHERE s.status = 'churned' GROUP BY p.plan_name ORDER BY churned_accounts DESC"),
        lambda r: ("medium", "What are total product usage events by feature?", "SELECT feature_name, SUM(events) AS total_events FROM product_usage GROUP BY feature_name ORDER BY total_events DESC"),
        lambda r: ("medium", "Which accounts have the most high severity support tickets?", "SELECT a.account_name, COUNT(*) AS high_tickets FROM support_tickets t JOIN accounts a ON t.account_id = a.account_id WHERE t.severity IN ('high', 'critical') GROUP BY a.account_name ORDER BY high_tickets DESC LIMIT 20"),
        lambda r: ("hard", "What is the monthly MRR trend by plan for paid invoices?", "SELECT substr(i.invoice_date, 1, 7) AS month, p.plan_name, ROUND(SUM(i.amount), 2) AS paid_revenue FROM invoices i JOIN subscriptions s ON i.account_id = s.account_id JOIN plans p ON s.plan_id = p.plan_id WHERE i.status = 'paid' GROUP BY month, p.plan_name ORDER BY month, p.plan_name"),
        lambda r: ("hard", "What is churn rate by plan?", "SELECT p.plan_name, ROUND(SUM(CASE WHEN s.status = 'churned' THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4) AS churn_rate FROM subscriptions s JOIN plans p ON s.plan_id = p.plan_id GROUP BY p.plan_name ORDER BY churn_rate DESC"),
        lambda r: ("hard", "Which accounts have low usage and unresolved support tickets?", "WITH usage_totals AS (SELECT account_id, SUM(events) AS total_events FROM product_usage GROUP BY account_id), ticket_totals AS (SELECT account_id, COUNT(*) AS unresolved_tickets FROM support_tickets WHERE resolved = 0 GROUP BY account_id) SELECT a.account_name, COALESCE(u.total_events, 0) AS total_events, COALESCE(t.unresolved_tickets, 0) AS unresolved_tickets FROM accounts a LEFT JOIN usage_totals u ON a.account_id = u.account_id LEFT JOIN ticket_totals t ON a.account_id = t.account_id WHERE COALESCE(u.total_events, 0) < 1000 AND COALESCE(t.unresolved_tickets, 0) > 0 ORDER BY unresolved_tickets DESC, total_events ASC LIMIT 20"),
    ]


def banking_templates() -> list[Template]:
    return [
        lambda r: ("easy", "How many banking customers are in each region?", "SELECT region, COUNT(*) AS customer_count FROM customers GROUP BY region"),
        lambda r: ("easy", "How many accounts exist by account type?", "SELECT account_type, COUNT(*) AS account_count FROM accounts GROUP BY account_type"),
        lambda r: (lambda threshold: ("easy", f"Count transactions above {threshold} dollars.", f"SELECT COUNT(*) AS large_transactions FROM transactions WHERE amount > {threshold}"))(r.randint(1000, 9000)),
        lambda r: ("medium", "What is transaction volume by branch?", "SELECT b.branch_name, ROUND(SUM(t.amount), 2) AS transaction_volume FROM transactions t JOIN accounts a ON t.account_id = a.account_id JOIN branches b ON a.branch_id = b.branch_id GROUP BY b.branch_name ORDER BY transaction_volume DESC"),
        lambda r: ("medium", "What is average transaction value by channel?", "SELECT channel, ROUND(AVG(amount), 2) AS avg_transaction_value FROM transactions GROUP BY channel ORDER BY avg_transaction_value DESC"),
        lambda r: ("medium", "Which branches have the highest total customer balances?", "SELECT b.branch_name, ROUND(SUM(a.balance), 2) AS total_balance FROM accounts a JOIN branches b ON a.branch_id = b.branch_id GROUP BY b.branch_name ORDER BY total_balance DESC"),
        lambda r: ("medium", "What is card usage count by card type?", "SELECT card_type, COUNT(*) AS card_count FROM cards WHERE active = 1 GROUP BY card_type"),
        lambda r: ("hard", "What suspicious transaction rate does each branch have?", "SELECT b.branch_name, ROUND(SUM(t.suspicious) * 1.0 / COUNT(*), 4) AS suspicious_rate FROM transactions t JOIN accounts a ON t.account_id = a.account_id JOIN branches b ON a.branch_id = b.branch_id GROUP BY b.branch_name ORDER BY suspicious_rate DESC"),
        lambda r: ("hard", "Which branches have the highest loan principal in risky grades?", "SELECT b.branch_name, ROUND(SUM(l.principal), 2) AS risky_principal FROM loans l JOIN branches b ON l.branch_id = b.branch_id WHERE l.risk_grade IN ('C', 'D') GROUP BY b.branch_name ORDER BY risky_principal DESC"),
        lambda r: ("hard", "What is monthly fraud alert volume by alert type?", "SELECT substr(alert_date, 1, 7) AS month, alert_type, COUNT(*) AS alert_count FROM fraud_alerts GROUP BY month, alert_type ORDER BY month, alert_type"),
    ]


def all_templates() -> dict[str, list[Template]]:
    return {
        "ecommerce": ecommerce_templates(),
        "saas": saas_templates(),
        "banking": banking_templates(),
    }


def build_example(
    db_id: str,
    index: int,
    rng: random.Random,
    template: Template,
    contexts: dict[str, str],
    database_dir: Path,
) -> dict[str, Any]:
    difficulty, question, sql = template(rng)
    variant = index % 97
    question = f"{question} Use reporting variant {variant}."
    sql = f"SELECT * FROM ({sql}) AS report_variant_{variant}"
    db_path = database_dir / f"{db_id}.db"
    result = execute_sql(db_path, sql)
    if not result["ok"]:
        raise ValueError(f"Gold SQL failed for {db_id}: {question}: {result['error']}")
    return {
        "id": f"{db_id}_{index:04d}",
        "db_id": db_id,
        "difficulty": difficulty,
        "question": question,
        "schema_context": contexts[db_id],
        "gold_sql": sql,
        "expected_result": result["rows"],
    }


def generate_examples(total: int, start_index: int, seed: int, database_dir: Path, include_sample_rows: bool) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    templates = all_templates()
    contexts = {
        db_id: schema_context(database_dir / f"{db_id}.db", include_sample_rows=include_sample_rows)
        for db_id in templates
    }
    examples: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    db_ids = list(templates)
    attempts = 0
    while len(examples) < total:
        attempts += 1
        if attempts > total * 50:
            raise RuntimeError("Could not generate enough unique SQL examples.")
        db_id = db_ids[len(examples) % len(db_ids)]
        template = templates[db_id][rng.randrange(len(templates[db_id]))]
        example = build_example(db_id, start_index + len(examples), rng, template, contexts, database_dir)
        key = (example["db_id"], example["question"], example["gold_sql"])
        if key in seen:
            continue
        seen.add(key)
        examples.append(example)
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--database-dir", default="data/databases")
    parser.add_argument("--include-sample-rows", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    database_dir = project_path(args.database_dir)
    train_count, eval_count = (30, 15) if args.small else (300, 100)
    train = generate_examples(train_count, 1, args.seed, database_dir, args.include_sample_rows)
    train_keys = {(row["db_id"], row["question"], row["gold_sql"]) for row in train}
    eval_rows = []
    for row in generate_examples(eval_count * 10, 5001, args.seed + 999, database_dir, args.include_sample_rows):
        key = (row["db_id"], row["question"], row["gold_sql"])
        if key not in train_keys:
            eval_rows.append(row)
        if len(eval_rows) == eval_count:
            break
    if len(eval_rows) < eval_count:
        raise RuntimeError("Could not generate enough eval examples distinct from train examples.")
    write_jsonl("data/train.jsonl", train)
    write_jsonl("data/eval.jsonl", eval_rows)
    print(f"Wrote {len(train)} train examples and {len(eval_rows)} eval examples.")


if __name__ == "__main__":
    main()
