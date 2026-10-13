"""
src/1_prepare_data.py
=====================
Loads the `mteb/banking77` dataset, converts each sample into an Alpaca-style
instruction format, and writes the result to `data/processed_banking77.jsonl`.

The banking77 dataset contains:
  - text  : a customer banking query (natural language)
  - label : an integer (0-76) representing one of 77 intent categories

We frame the task as intent classification via instruction-following:
  instruction → "Classify the following banking query into one of 77 intents."
  input       → the customer query text
  output      → the intent label name (string, resolved from label2id map)

Run:
    python src/1_prepare_data.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

# ---------------------------------------------------------------------------
# Logging configuration: goes to both stdout and a log file
# ---------------------------------------------------------------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, "1_prepare_data.log"), mode="w"),
    ],
)
logger = logging.getLogger("prepare_data")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_NAME = "mteb/banking77"
OUTPUT_PATH = "data/processed_banking77.jsonl"
DATA_DIR = "data"

# The banking77 label-to-name mapping (77 intent categories).
# Source: https://huggingface.co/datasets/mteb/banking77
LABEL_NAMES: list[str] = [
    "activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support",
    "automatic_top_up", "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed",
    "cancel_transfer", "card_about_to_expire", "card_acceptance", "card_arrival",
    "card_delivery_estimate", "card_linking", "card_not_working", "card_payment_fee_charged",
    "card_payment_not_recognised", "card_payment_wrong_exchange_rate", "card_swallowed",
    "cash_withdrawal_charge", "cash_withdrawal_not_recognised", "change_pin",
    "compromised_card", "contactless_not_working", "country_support",
    "declined_card_payment", "declined_cash_withdrawal", "declined_transfer",
    "direct_debit_payment_not_recognised", "disposable_card_limits",
    "edit_personal_details", "exchange_charge", "exchange_rate", "exchange_via_app",
    "extra_charge_on_statement", "failed_transfer", "fiat_currency_support",
    "get_disposable_virtual_card", "get_physical_card", "getting_spare_card",
    "getting_virtual_card", "lost_or_stolen_card", "lost_or_stolen_phone",
    "order_physical_card", "passcode_forgotten", "pending_card_payment",
    "pending_cash_withdrawal", "pending_top_up", "pending_transfer",
    "pin_blocked", "receiving_money", "refund_not_showing_up",
    "request_refund", "reverted_card_payment?", "supported_cards_and_currencies",
    "terminate_account", "top_up_by_bank_transfer_charge", "top_up_by_card_charge",
    "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits",
    "top_up_reverted", "topping_up_by_card", "transaction_charged_twice",
    "transfer_fee_charged", "transfer_into_account", "transfer_not_received_by_recipient",
    "transfer_timing", "unable_to_verify_identity", "verify_my_identity",
    "verify_source_of_funds", "verify_top_up", "virtual_card_not_working",
    "visa_or_mastercard", "why_is_my_card_blocked", "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
]

INSTRUCTION_TEMPLATE = (
    "Classify the following banking customer query into one of 77 intent categories. "
    "Respond with only the intent label name."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def label_id_to_name(label_id: int) -> str:
    """Convert an integer label to its human-readable intent name."""
    try:
        return LABEL_NAMES[label_id]
    except IndexError:
        logger.warning("Unknown label id %d, using 'unknown_intent'", label_id)
        return "unknown_intent"


def convert_sample_to_alpaca(sample: dict[str, Any]) -> dict[str, str]:
    """
    Convert a single banking77 sample into Alpaca instruction format.

    Args:
        sample: dict with keys 'text' (str) and 'label' (int).

    Returns:
        dict with keys 'instruction', 'input', 'output'.
    """
    return {
        "instruction": INSTRUCTION_TEMPLATE,
        "input": sample["text"],
        "output": label_id_to_name(sample["label"]),
    }


def write_jsonl(records: list[dict[str, str]], path: str) -> None:
    """Write a list of dicts to a JSONL file (one JSON object per line)."""
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=" * 60)
    logger.info("Banking77 Dataset Preparation")
    logger.info("=" * 60)

    # 1. Create output directory
    os.makedirs(DATA_DIR, exist_ok=True)
    logger.info("Output directory ensured: %s", DATA_DIR)

    # 2. Load dataset
    logger.info("Loading dataset: %s", DATASET_NAME)
    try:
        from datasets import load_dataset, DatasetDict  # type: ignore
        raw: DatasetDict = load_dataset(DATASET_NAME)
    except ImportError as e:
        logger.error("Could not import `datasets`. Is the venv activated? %s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("Failed to load dataset '%s': %s", DATASET_NAME, e)
        sys.exit(1)

    logger.info("Dataset loaded. Available splits: %s", list(raw.keys()))
    for split_name, split_data in raw.items():
        logger.info("  Split '%-10s': %d samples, columns: %s",
                    split_name, len(split_data), split_data.column_names)

    # 3. Convert the train split for fine-tuning
    #    banking77 has 'train' and 'test' splits; the test split is kept for evaluation.
    all_records: list[dict[str, str]] = []

    for split_name, split_data in [("train", raw["train"])]:
        logger.info("Converting split '%s' (%d samples)...", split_name, len(split_data))
        try:
            converted = [convert_sample_to_alpaca(sample) for sample in split_data]
            all_records.extend(converted)
            logger.info("  Converted %d records from split '%s'.", len(converted), split_name)
        except Exception as e:
            logger.error("Error converting split '%s': %s", split_name, e)
            raise

    # 4. Log dataset statistics
    total = len(all_records)
    logger.info("-" * 40)
    logger.info("Dataset statistics:")
    logger.info("  Total records  : %d", total)
    logger.info("  Unique intents : %d", len({r["output"] for r in all_records}))

    # Distribution of intents
    from collections import Counter
    intent_counts = Counter(r["output"] for r in all_records)
    logger.info("  Intent distribution (top 10):")
    for intent, count in intent_counts.most_common(10):
        logger.info("    %-45s : %d", intent, count)

    # 5. Save to JSONL
    logger.info("-" * 40)
    logger.info("Saving to: %s", OUTPUT_PATH)
    try:
        write_jsonl(all_records, OUTPUT_PATH)
    except Exception as e:
        logger.error("Failed to write output file: %s", e)
        raise

    # Verify file was written
    file_size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    logger.info("Saved %d records to '%s' (%.1f KB).", total, OUTPUT_PATH, file_size_kb)

    # 6. Sanity-check: read back and validate first record
    logger.info("-" * 40)
    logger.info("Sanity check, first record:")
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        first_line = f.readline()
    first_record = json.loads(first_line)
    for key, value in first_record.items():
        logger.info("  %-12s: %s", key, value[:80] if len(value) > 80 else value)

    logger.info("=" * 60)
    logger.info("Data preparation complete. Ready for training.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
