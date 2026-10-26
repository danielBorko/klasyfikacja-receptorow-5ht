"""
Moduł sprawdzania jakości i poprawności sekwencji białkowych (FASTA).
"""

from typing import Tuple, List, Dict, Set
from Bio import SeqIO

# Standardowy alfabet 20 aminokwasów białkowych
STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def validate_sequence(sequence: str) -> Tuple[bool, str]:
    """
    Weryfikuje, czy sekwencja aminokwasowa nie jest pusta i zawiera tylko poprawne znaki.
    """
    seq_str = str(sequence).upper().strip()

    if not seq_str:
        return False, "Empty sequence"

    invalid_chars = set(seq_str) - STANDARD_AMINO_ACIDS
    if invalid_chars:
        return False, f"Invalid characters found: {', '.join(sorted(invalid_chars))}"

    return True, "Valid"


def check_dataset_integrity(records: List[Dict]) -> Dict:
    """
    Wykonuje pełny audit zbioru sekwencji pod kątem duplikatów i uszkodzonych rekordów.
    """
    seen_sequences: Set[str] = set()
    report = {
        "total_input": len(records),
        "valid_records": 0,
        "duplicates_removed": 0,
        "invalid_sequences": 0,
        "cleaned_records": []
    }

    for record in records:
        seq = str(record.get("sequence", "")).upper().strip()
        is_valid, msg = validate_sequence(seq)

        if not is_valid:
            report["invalid_sequences"] += 1
            continue

        if seq in seen_sequences:
            report["duplicates_removed"] += 1
            continue

        seen_sequences.add(seq)
        report["cleaned_records"].append(record)

    report["valid_records"] = len(report["cleaned_records"])
    return report