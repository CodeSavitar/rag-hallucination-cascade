import csv
from datetime import datetime, timezone
from pathlib import Path

# Spending guard for every paid API call (M3 onward). Adhith's Claude API credit is $18 and
# cannot be topped up, so this file is the one place that decides whether a call may run.
#
#   check_budget(model, input_tokens, max_output_tokens)  BEFORE a call: refuses if the call's
#                                                          worst case would cross the cap
#   record_spend(model, usage, purpose)                    AFTER a call: logs what it really cost

BASE_DIR = Path(__file__).resolve().parent.parent
LEDGER_CSV = BASE_DIR / 'data' / 'api_spend.csv'    # one row per paid call; committed, contains no secrets

CREDIT_USD = 18.00
SPEND_CAP_USD = 15.00    # $3 kept back: estimation error, plus the live demo in M6 will call the judge too

# USD per 1M tokens, from Anthropic's published prices as of 2026-10-06. Update if they change.
# Thinking tokens are billed as output, so they are covered by the output price.
PRICES_PER_MILLION = {
    'claude-sonnet-5-5': {'input': 2.00, 'output': 10.00},
    'claude-opus-5-5': {'input': 4.00, 'output': 20.00},
    'claude-haiku-5-5': {'input': 0.10, 'output': 0.50},    # prompts up to 100K tokens
}

LEDGER_COLUMNS = ['timestamp', 'purpose', 'model', 'input_tokens', 'output_tokens', 'cost_usd']


class BudgetExceeded(Exception):
    """Raised instead of making a call that could push spending past the cap."""


def cost_usd(model, input_tokens, output_tokens):
    prices = PRICES_PER_MILLION[model]    # an unknown model raises KeyError: never call an unpriced model
    return (input_tokens * prices['input'] + output_tokens * prices['output']) / 1_000_000


def spent_so_far():
    """Total USD recorded in the ledger."""
    if not LEDGER_CSV.exists():
        return 0.0
    with open(LEDGER_CSV, newline='', encoding='utf-8') as file:
        return sum(float(row['cost_usd']) for row in csv.DictReader(file))


def check_budget(model, input_tokens, max_output_tokens):
    """Raise BudgetExceeded if this call's WORST case would cross the cap.

    Worst case = the full prompt plus every output token the call is allowed (max_tokens).
    The real cost is almost always lower, so the guard errs on the side of stopping early.
    """
    worst_case = cost_usd(model, input_tokens, max_output_tokens)
    spent = spent_so_far()
    if spent + worst_case > SPEND_CAP_USD:
        raise BudgetExceeded(
            f"Refusing call: ${spent:.4f} spent + worst case ${worst_case:.4f} would exceed the "
            f"${SPEND_CAP_USD:.2f} cap (credit ${CREDIT_USD:.2f})")
    return worst_case


def record_spend(model, input_tokens, output_tokens, purpose):
    """Append one paid call to the ledger, priced from the token counts the API returned."""
    cost = cost_usd(model, input_tokens, output_tokens)
    is_new = not LEDGER_CSV.exists()
    with open(LEDGER_CSV, 'a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=LEDGER_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow({
            'timestamp': datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            'purpose': purpose, 'model': model,
            'input_tokens': input_tokens, 'output_tokens': output_tokens,
            'cost_usd': f"{cost:.6f}",
        })
    return cost


if __name__ == '__main__':
    spent = spent_so_far()
    print(f"Spent ${spent:.4f} of the ${SPEND_CAP_USD:.2f} cap (credit ${CREDIT_USD:.2f}); "
          f"${SPEND_CAP_USD - spent:.4f} left under the cap")
