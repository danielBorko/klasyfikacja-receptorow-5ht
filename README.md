# Klasyfikacja podtypów receptorów serotoninowych i analiza wyjaśnialności

Projekt inżynierski dedykowany wieloklasowej sekwencyjnej klasyfikacji podtypów receptorów serotoninowych (5-HT) z uwzględnieniem koncepcji ewolucyjnych i filogenetycznych. Projekt łączy metody tradycyjnego uczenia maszynowego oparte na profilach fizykochemicznych z nowoczesnymi modelami językowymi białek (Protein Language Models) oraz interpretowalnością modeli (XAI)
##Metodologia Badawcza

Projekt porównuje (na razie) dwie strategie reprezentacji sekwencji białkowych (pliki `.fasta`):

1. **Podejście Biochemiczne (`biochemical.py`)**
   - Ekstrakcja klasycznych cech fizykochemicznych: skład aminokwasowy (kompozycja AA), masa cząsteczkowa (MW), punkt izoelektryczny (pI) itp.
   - Trening klasyfikatora (np. Random Forest).
   - Analiza wpływu poszczególnych właściwości i aminokwasów na podtypy receptorów za pomocą wartości SHAP.

2. **Podejście Głębokie / Językowe (`esm2.py`)**
   - Wykorzystanie modelu językowego białek **ESM-2** (Evolutionary Scale Modeling) do generowania wielowymiarowych osadzeń (*embeddings*).
   - Klasyfikacja receptorów na podstawie reprezentacji kontekstowej sekwencji.
  
##Struktura Projektu

```text
├── data/                    # Surowe i przetworzone pliki sekwencji (.fasta)
│   └── A_carolinensis/      # Przykładowy zestaw sekwencji dla danego gatunku
└── src/
    ├── biochemical.py  # Ekstrakcja cech biochemicznych, klasyfikacja i SHAP
    └── esm2.py         # Ekstrakcja reprezentacji ESM-2, klasyfikacja