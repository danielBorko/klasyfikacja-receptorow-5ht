"""
Moduł generowania i cache'owania embeddingów z modelu językowego ESM-2.
"""

import os
from pathlib import Path
import torch
import pandas as pd
from Bio import SeqIO
from transformers import AutoTokenizer, EsmModel


try:
    from receptors.labels import normalize_label
except ImportError:
    from labels import normalize_label

MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

# Automatyczne wyznaczenie ścieżek względem głównego folderu projektu
BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SPECIES_DIR = BASE_DIR / "dane" / "raw" / "fasta"
DEFAULT_OUTPUT_PATH = BASE_DIR / "dane" / "processed" / "esm_embeddings.pt"


def generate_esm_embeddings(species_dir: Path = DEFAULT_SPECIES_DIR,
                            output_path: Path = DEFAULT_OUTPUT_PATH) -> pd.DataFrame:
    """
    Generuje embeddingi ESM-2 dla wszystkich sekwencji i je zapisuje
    """
    print(f"Ładowanie modelu ESM-2 ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EsmModel.from_pretrained(MODEL_NAME)
    model.eval()

    wyniki = []

    species_dir_path = Path(species_dir)
    if not species_dir_path.exists():
        raise FileNotFoundError(f"Katalog źródłowy '{species_dir_path}' nie istnieje!")

    print("Rozpoczynam ekstrakcję embeddingów ESM-2...")
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
                seq = str(record.seq).strip().upper().replace("X", "")
                if not seq:
                    continue

                gene = file_path.stem
                label = normalize_label(gene)

                if label == "Other":
                    continue

                # Generowanie embeddingu w trybie bez gradientów (oszczędność pamięci)
                inputs = tokenizer(seq, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)

                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

                feature_dict = {
                    "sample_id": f"{species}_{extract_sequence_id(record)}",
                    "species": species,
                    "label": label,
                    "embedding": embedding
                }
                wyniki.append(feature_dict)
            except Exception as e:
                print(f"Błąd przetwarzania pliku {file_path}: {e}")

    df_esm = pd.DataFrame(wyniki)

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(df_esm, output_path_obj)

    print(f"Przetworzono i zapisano {len(df_esm)} embeddingów ESM-2 do: {output_path_obj}")
    return df_esm


def extract_sequence_id(record):
    header = record.id
    if "|" in header:
        parts = header.split("|")
        if len(parts) >= 2:
            return parts[1]
    return header.split()[0]


if __name__ == "__main__":

    generate_esm_embeddings()