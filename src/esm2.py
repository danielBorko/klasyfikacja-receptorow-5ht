import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from transformers import AutoTokenizer, EsmModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

#nazwa najlżejszego modelu ESM-2 (8 milionów parametrów)
MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

#zamiana liter (aminokwasów) na indeksy liczbowe
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model ewolucyjny ESM-2
model = EsmModel.from_pretrained(MODEL_NAME)



dir_path = '../dane'

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

    gene_lower = gene_name.lower().replace("-", "").replace("_", "")

    #Mapowanie dla C. elegans
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

    # mapowanie dla kręgowców i duplikatów u ryb
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

def get_esm_embedding(sequence):
    clean_seq = sequence.strip().upper().replace("X", "")
    inputs = tokenizer(clean_seq, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


wyniki = []

print("Rozpoczynam ekstrakcję cech przy użyciu ESM-2 dla wszystkich plików...")
for species in os.listdir(dir_path):
    species_path = os.path.join(dir_path, species)
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
        seq = str(record.seq)
        gene = file.replace(".fasta", "")
        label = map_to_specific_subtype(gene)

        if label == "Other":
            continue

        #  wektor ewolucyjny
        embedding = get_esm_embedding(seq)

        # słownik z metadanymi i cechami wektora
        feature_dict = {f"E_{i}": val for i, val in enumerate(embedding)}
        feature_dict["label"] = label
        feature_dict["species"] = species

        wyniki.append(feature_dict)

df_esm = pd.DataFrame(wyniki)
print(f"\nPrzetworzono pomyślnie {len(df_esm)} receptorów.")

# filtrowanie klas  -   min 3 próbki do prawidłowego podziału stratify
min_samples = 3
subtypes_counts = df_esm["label"].value_counts()
subtypes_to_keep = subtypes_counts[subtypes_counts >= min_samples].index
df_esm = df_esm[df_esm["label"].isin(subtypes_to_keep)]

# przygotowanie danych do modelu
X = df_esm.drop(columns=["label", "species"])
y = df_esm["label"]

# podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)

# klasyfikacja przy użyciu Random Forest
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nWYNIKI MODELU UNIMODALNEGO (ESM-2) ")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nRaport klasyfikacji:")
print(classification_report(y_test, y_pred))

# macierz pomyłek
cm = confusion_matrix(y_test, y_pred)
labels = sorted(df_esm["label"].unique())

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=labels, yticklabels=labels)
plt.title("Macierz pomyłek dla klasyfikacji za pomocą osadzeń ESM-2")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("macierz_esm.png", dpi=300)
plt.show()