# 🔍 FLUX Latent Subspaces (TML Project)

Projekt realizowany w ramach przedmiotu **Trustworthy Machine Learning (TML)**. Badamy i interpretujemy przestrzenie ukryte (latent spaces) w modelu dyfuzyjnym **FLUX.1-dev**. 

Repozytorium bazuje na pracy *Pach et al. (ICML 2026) - Latent Color Subspaces (LCS)*, przenosząc metodykę ingerencji z kolorów na inne abstrakcyjne cechy wizualne, ze szczególnym naciskiem na modyfikację **tekstury** obiektów.

## 📂 Struktura Repozytorium

Projekt wykorzystuje modułową architekturę izolującą eksperymenty poszczególnych członków grupy badawczej, co eliminuje konflikty w kodzie:

*   `colors_baseline/` - Czysty, oryginalny rdzeń kodu z repozytorium LCS. Służy wyłącznie jako nienaruszalna biblioteka plików referencyjnych.
*   `texture/` - Folder badawczy adaptujący logikę na potrzeby analizy tekstur (od gładkiego szkła po szorstki beton).
*   `[inna_cecha]/` - Przestrzeń robocza dla kolejnych badanych czynników wizualnych.

## ⚙️ Logiczny ciąg wydarzeń (Workflow)

Praca z repozytorium dzieli się na trzy główne etapy. 

**Krok 1: Ekstrakcja wektorów z modelu (Generacja danych)**
Zmuszamy model FLUX do wygenerowania obrazów płaskich powierzchni z konkretnych materiałów. Enkoder VAE przetwarza te wygenerowane obrazy i zapisuje ich skompresowaną reprezentację matematyczną (tensory `.pt`).

**Krok 2: Znalezienie podprzestrzeni (PCA)**
Analizujemy zebrane tensory materiałów za pomocą Analizy Głównych Składowych (PCA). Celem jest wyznaczenie wektora kierunkowego (głównej składowej PC1), który odpowiada w przestrzeni ukrytej za płynne przejście między badanymi wartościami (np. oś gładkość $\rightarrow$ szorstkość).

**Krok 3: Aktywna Interwencja**
Wstrzykujemy wyliczony wektor bezpośrednio w proces odszumiania (np. w 9. kroku dyfuzji). Skrypt izoluje wskazany obiekt za pomocą masek uwagi, wymuszając na modelu nałożenie obliczonej cechy (np. faktury kory drzewa) wyłącznie na docelowy przedmiot, zachowując oryginalne tło.

## 🚀 Przykład użycia

Poniżej znajduje się kompletny przykład uruchomienia całego pipeline'u z poziomu terminala dla cechy tekstury.

**1. Przygotowanie środowiska i autoryzacja:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r colors_baseline/requirements.txt
export HF_TOKEN="twój_token_huggingface"
