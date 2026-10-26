"""
Moduł ekstrakcji cech fizykochemicznych i składu aminokwasowego.
Korzysta ze wspólnych funkcji normalizacji etykiet.
"""

import os
from pathlib import Path
import pandas as pd
from Bio import SeqIO
from Bio.SeqUtils import molecular_weight
from Bio.SeqUtils.ProtParam import ProteinAnalysis


try:
    from receptors.labels import normalize_label
except ImportError:
    from labels import normalize_label

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# wyznaczenie ścieżek względem głównego folderu projektu
BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SPECIES_DIR = BASE_DIR / "dane" / "raw" / "fasta"
DEFAULT_OUTPUT_PATH = BASE_DIR / "dane" / "processed" / "biochemical_features.csv"

def extract_sequence_id(record):
    header = record.id
    if "|" in header:
        parts = header.split("|")
        if len(parts) >= 2:
            return parts[1]
    return header.split()[0]


def compute_biochemical_features(species_dir: Path = DEFAULT_SPECIES_DIR,
                                 output_path: Path = DEFAULT_OUTPUT_PATH) -> pd.DataFrame:
    """
    Oblicza cechy fizykochemiczne dla wszystkich sekwencji i zapisuje wynik na dysk.
    """
    wyniki = []

    species_dir_path = Path(species_dir)
    if not species_dir_path.exists():
        raise FileNotFoundError(f"Katalog źródłowy '{species_dir_path}' nie istnieje!")

    for species_path in species_dir_path.iterdir():
        if not species_path.is_dir():
            continue

        species = species_path.name

        for file_path in species_path.iterdir():
            if not file_path.name.endswith(".fasta"):
                continue

            try:
                records = list(SeqIO.parse(str(file_path), "fasta"))
                if not records:
                    continue

                record = records[0]
                seq = str(record.seq).upper().replace("X", "")
                seq_id = extract_sequence_id(record)
                gene = file_path.stem

                # Użycie scentralizowanej normalizacji etykiet
                label = normalize_label(gene)
                if label == "Other":
                    continue

                total = len(seq)
                if total == 0:
                    continue

                analysis = ProteinAnalysis(seq)

                aa_procent = {f"AA_{aa}": (seq.count(aa) / total) * 100 for aa in AMINO_ACIDS}
                aromatic_count = sum(seq.count(aa) for aa in "WYF")
                charged_count = sum(seq.count(aa) for aa in "DEKR")

                wyniki.append({
                    "sample_id": f"{species}_{seq_id}",
                    "uniprot_id": seq_id,
                    "species": species,
                    "gene": gene,
                    "label": label,
                    "length": total,
                    "MW": molecular_weight(seq, "protein"),
                    "PI": analysis.isoelectric_point(),
                    "hydro": analysis.gravy(),
                    "aromaticity": analysis.aromaticity(),
                    "instability": analysis.instability_index(),
                    "aromatic_ratio": (aromatic_count / total) * 100,
                    "charged_ratio": (charged_count / total) * 100,
                    **aa_procent,
                })
            except Exception as e:
                print(f"Błąd przetwarzania pliku {file_path}: {e}")

    df = pd.DataFrame(wyniki)

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path_obj, index=False)

    print(f"Zapisano cechy biochemiczne ({len(df)} rekordów) do: {output_path_obj}")
    return df


if __name__ == "__main__":
    compute_biochemical_features()