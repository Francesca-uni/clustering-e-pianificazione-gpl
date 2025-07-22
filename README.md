# Clustering e Pianificazione GPL

Questo progetto nasce con l’obiettivo di sviluppare un sistema intelligente per la **pianificazione delle consegne di gas GPL sfuso**, tramite **clustering geospaziale**, **previsioni di domanda** e **ottimizzazione dei percorsi**.

## Obiettivi

- Geocodificare e normalizzare i dati geografici dei clienti
- Identificare zone operative tramite clustering (DBSCAN + KMeans)
- Pianificare i giri di consegna basati su domanda e prossimità
- Ottimizzare i percorsi tramite algoritmo Nearest Neighbor
- Simulare scenari multi-veicolo con vincoli logistici

## Fasi principali

### 1. Preprocessing e geocodifica
- Conversione coordinate GPS da DMS a decimali
- Geocodifica tramite API OpenCage per 350 clienti con GPS mancante
- Salvataggio nel file `clienti_validi_geocodificati.csv`

### 2. Clustering
- Algoritmo: **DBSCAN** (`eps=0.3`, `min_samples=4`)
- Valutazione con **Silhouette Score = 0.483**
- Riaffinamento con **K-Means** sui cluster sovraccarichi
- Riassegnazione intelligente di alcuni outlier

### 3. Pianificazione consegne
- Esplosione date consegna previste
- Raggruppamento per `data` e `cluster`
- Ripianificazione giri poco efficienti (giri < 3 clienti)
- Output: `piano_consegne_finale.csv`

### 4. Ottimizzazione percorso
- Algoritmo: **Nearest Neighbor**
- Punto di partenza: deposito (85010 Vaglio Basilicata)
- Output: `percorso_ottimizzato_nearest_neighbor.csv`

### 5. Multi-veicolo
- Suddivisione giri con vincoli (`min=3`, `max=8` clienti)
- Assegnazione veicolo e zona
- Output finale: `percorso_multi_veicolo.csv`



