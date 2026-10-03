import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from Bio import SeqIO
from Bio.SeqUtils import molecular_weight
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import shap

dir_path = '../dane'
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def extract_sequence_id(record):
    header = record.id
    #UniProt format np. sp|O17470|
    if "|" in header:
        parts = header.split("|")
        if len(parts) >= 2:
            return parts[1]
    #UPI format
    return header.split()[0]


def map_to_specific_subtype(gene_name):
    """Precyzyjne mapowanie genów do konkretnych podtypów receptorów

    serotoninowych.
    """
    gene_lower = gene_name.lower().replace("-", "").replace("_", "")

    # Mapowanie dla C. elegans
    if "ser1" in gene_lower:
        return "5-HT1A"
    if "ser4" in gene_lower:
        return "5-HT2A"
    if "ser7" in gene_lower:
        return "5-HT7"
    if "ser5" in gene_lower:
        return "5-HT1B"
    if "ser2" in gene_lower:
        return "5-HT2B"

    #mapowanie dla kręgowców i duplikatów u ryb
    if "5ht1a" in gene_lower:
        return "5-HT1A"
    if "5ht1b" in gene_lower:
        return "5-HT1B"
    if "5ht1d" in gene_lower:
        return "5-HT1D"
    if "5ht1e" in gene_lower:
        return "5-HT1E"
    if "5ht1f" in gene_lower:
        return "5-HT1F"

    if "5ht2a" in gene_lower:
        return "5-HT2A"
    if "5ht2b" in gene_lower:
        return "5-HT2B"
    if "5ht2c" in gene_lower:
        return "5-HT2C"

    if "5ht4" in gene_lower:
        return "5-HT4"
    if "5ht5" in gene_lower:
        return "5-HT5A"
    if "5ht6" in gene_lower:
        return "5-HT6"
    if "5ht7" in gene_lower:
        return "5-HT7"

    return "Other"


def extract_features(dir_path):
    wyniki = []

    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Folder '{dir_path}' nie istnieje!")

    for species in os.listdir(dir_path):
        species_path = os.path.join(dir_path, species)

        # Sprawdzenie czy to na pewno folder
        if not os.path.isdir(species_path):
            continue

        for file in os.listdir(species_path):
            if not file.endswith(".fasta"):
                continue

            file_path = os.path.join(species_path, file)
            records = list(SeqIO.parse(file_path, "fasta"))
            if len(records) == 0:
                continue

            record = records[0]
            seq = str(record.seq).upper().replace("X", "")
            seq_id = extract_sequence_id(record)
            gene = file.replace(".fasta", "")

            # Przypisanie podtypu na etapie ekstrakcji
            label = map_to_specific_subtype(gene)

            if label == "Other":
                continue  # odrzucamy niesklasyfikowane białka

            total = len(seq)
            analysis = ProteinAnalysis(seq)

            # procentowy skład każdego z aminokwasów
            aa_procent = {
                f"AA_{aa}": (seq.count(aa) / total) * 100 for aa in AMINO_ACIDS
            }

            # dodatkowe cechy grup chemicznych
            aromatic_count = sum(seq.count(aa) for aa in "WYF")
            charged_count = sum(seq.count(aa) for aa in "DEKR")

            wyniki.append(
                {
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
                }
            )

    return pd.DataFrame(wyniki)



# 1. ekstrakcja danych
df = extract_features(dir_path)
print(f"Załadowano {len(df)} receptorów.")
print(df.head())

# 2. kontrola liczebności podtypów
print("\nLiczba próbek dla poszczególnych podtypów przed filtrowaniem:")
print(df["label"].value_counts())

# odrzucamy podtypy, które mają mniej niż 3 przykłady (warunek konieczny do podziału stratify)
min_samples = 3
subtypes_counts = df["label"].value_counts()
subtypes_to_keep = subtypes_counts[subtypes_counts >= min_samples].index
df = df[df["label"].isin(subtypes_to_keep)]

print(f"\nLiczba próbek po przefiltrowaniu (min. {min_samples} powtórzenia):")
print(df["label"].value_counts())

# 3. Przygotowanie macierzy cech (X) oraz wektora etykiet (y)
X = df.drop(columns=["gene", "species", "label", "uniprot_id"])
y = df["label"]
X = X.fillna(0)

# 4. podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)

# 5. budowa i trenowanie modelu Random Forest
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 6. predykcja i ocena
y_pred = model.predict(X_test)
print("\n WYNIKI MODELU ")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nRaport klasyfikacji:")
print(classification_report(y_test, y_pred))

# 7. macierz pomyłek
cm = confusion_matrix(y_test, y_pred)
# etykiety na osiach
labels = sorted(df["label"].unique())

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
)
plt.title("Macierz pomyłek dla klasyfikacji podtypów receptorów 5-HT")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()


# 8. SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# posortowane nazwy klas
unique_labels = [
    str(cls) for cls in getattr(model, "classes_", sorted(df["label"].unique()))
]

# Zmiana skrótu aminokwasu na nazwę
AA_MAP = {
    "A": "Alanina",
    "ALA": "Alanina",
    "R": "Arginina",
    "ARG": "Arginina",
    "N": "Asparagina",
    "ASN": "Asparagina",
    "D": "Kwas asparaginianowy",
    "ASP": "Kwas asparaginianowy",
    "C": "Cysteina",
    "CYS": "Cysteina",
    "E": "Kwas glutaminowy",
    "GLU": "Kwas glutaminowy",
    "Q": "Glutamina",
    "GLN": "Glutamina",
    "G": "Glicyna",
    "GLY": "Glicyna",
    "H": "Histydyna",
    "HIS": "Histydyna",
    "I": "Izoleucyna",
    "ILE": "Izoleucyna",
    "L": "Leucyna",
    "LEU": "Leucyna",
    "K": "Lizyna",
    "LYS": "Lizyna",
    "M": "Metionina",
    "MET": "Metionina",
    "F": "Fenyloalanina",
    "PHE": "Fenyloalanina",
    "P": "Prolina",
    "PRO": "Prolina",
    "S": "Seryna",
    "SER": "Seryna",
    "T": "Treonina",
    "THR": "Treonina",
    "W": "Tryptofan",
    "TRP": "Tryptofan",
    "Y": "Tyrozyna",
    "TYR": "Tyrozyna",
    "V": "Walina",
    "VAL": "Walina",
}


def clean_feature_name(name):
    # sprawdza czy cecha zaczyna się od AA_
    if name.startswith("AA_"):
        code = name[3:].upper()
        # pełną nazwę aminokwasu
        return AA_MAP.get(code, name[3:])
    return name


# Podmiana nazw kolumn w kopii ramki testowej
X_test_display = X_test.copy()
X_test_display.columns = [clean_feature_name(col) for col in X_test_display.columns]

# wykres dla każdego receptora
for idx, class_name in enumerate(unique_labels):
    plt.figure(figsize=(10, 6))

    # wyciąganie macierzy SHAP zależnie od wersji biblioteki
    if isinstance(shap_values, list):
        vals = shap_values[idx]
    elif len(shap_values.shape) == 3:
        vals = shap_values[:, :, idx]
    else:
        vals = shap_values[:, :, idx].values

    # Generowanie wykresu
    shap.summary_plot(vals, X_test_display, show=False)

    plt.title(
        f"Wykres SHAP Summary Plot dla podtypu {class_name}",
        fontsize=12,
        pad=15,
        weight="bold",
    )
    plt.tight_layout()

    # Zapis pliku bez niedozwolonych znaków w nazwie
    safe_filename = re.sub(r"[^\w\-]", "_", str(class_name))
    filename = f"wykres_shap_{safe_filename}.png"

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

