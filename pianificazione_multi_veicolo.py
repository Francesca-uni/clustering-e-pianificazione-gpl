import pandas as pd
from geopy.distance import geodesic
import math

# Caricamento dati
df_coord = pd.read_csv("clienti_validi_geocodificati.csv")
df_piano = pd.read_csv("piano_consegne_finale.csv")

# Coordinate del deposito 
deposito_coord = (40.656361, 15.880113)

# Funzione NEAREST NEIGHBOR 
def tsp_nearest_neighbor(df_clienti, deposito_coord):
    clienti = df_clienti.to_dict('records')
    if not clienti:
        return 0, []
    percorso = []
    non_visitati = clienti.copy()
    current_coord = deposito_coord
    distanza_totale = 0

    while non_visitati:
        nearest = min(non_visitati, key=lambda c: geodesic(current_coord, (c['latitudine'], c['longitudine'])).km)
        distanza = geodesic(current_coord, (nearest['latitudine'], nearest['longitudine'])).km
        distanza_totale += distanza
        percorso.append(nearest)
        current_coord = (nearest['latitudine'], nearest['longitudine'])
        non_visitati.remove(nearest)

    percorso_codici = [c['Codice Cliente'] for c in percorso]
    return round(distanza_totale, 2), percorso_codici

# Parametri di configurazione
max_clienti_per_veicolo = 8
min_clienti_per_veicolo = 3

# Suddivide la lista di clienti in gruppi di dimensione compresa tra min_size e max_size.
def suddividi_in_gruppi(clienti, min_size=3, max_size=8):
    gruppi = []
    i = 0
    while i < len(clienti):
        end = i + max_size
        # Se l'ultimo gruppo ha meno di min_size clienti, li accorpa al gruppo precedente (se esiste),
        if end >= len(clienti):
            if len(clienti) - i < min_size and gruppi:
                gruppi[-1].extend(clienti[i:])
            else:
                gruppi.append(clienti[i:])
            break
        else:
            gruppi.append(clienti[i:end])
            i = end
    return gruppi

# Costruzione dei giri MULTI-VEICOLO 
# Mappa i codici cluster in etichette leggibili (es. "0_0" → "Zona A1") 
def mappa_zona(cluster_label):
    if cluster_label.startswith("0_"):
        return f"Zona A{int(cluster_label.split('_')[1]) + 1}"
    elif cluster_label.startswith("1_"):
        return f"Zona B{int(cluster_label.split('_')[1]) + 1}"
    elif cluster_label == "2":
        return "Zona C"
    elif cluster_label == "3":
        return "Zona D"
    elif cluster_label == "outlier":
        return "Zona X"
    else:
        return f"Zona ? ({cluster_label})"
    
risultati = []

for giorno in sorted(df_piano['Data Consegna'].unique()):
    sottoinsieme = df_piano[df_piano['Data Consegna'] == giorno]

    for _, riga in sottoinsieme.iterrows():
        clienti = eval(riga['Clienti']) if isinstance(riga['Clienti'], str) else riga['Clienti']
        cluster = riga['Cluster']
        gruppi = suddividi_in_gruppi(clienti, min_clienti_per_veicolo, max_clienti_per_veicolo)

        for i, gruppo in enumerate(gruppi):
            df_gruppo = df_coord[df_coord['Codice Cliente'].isin(gruppo)]
            dist, percorso = tsp_nearest_neighbor(df_gruppo, deposito_coord)

            id_veicolo = f'V{giorno}_C{cluster}_N{i+1}'
            velocita_media_kmh = 40
            tempo_per_cliente_min = 10

            # Tempo guida in minuti
            tempo_guida_min = (dist / velocita_media_kmh) * 60
            # Tempo totale = guida + consegne
            tempo_totale_min = tempo_guida_min + len(percorso) * tempo_per_cliente_min


            risultati.append({
                'Data Consegna': giorno,
                'Cluster': cluster,
                'Zona Consegna': mappa_zona(str(cluster)),
                'Veicolo': id_veicolo,
                'Numero Clienti': len(percorso),
                'Distanza Stimata (km)': dist,
                'Percorso Ottimo': percorso,
                'Tempo Totale Stimato (min)': round(tempo_totale_min),
                'Tempo Guida Stimato (min)': round(tempo_guida_min)
            })

# Esportazione finale 
df_risultati = pd.DataFrame(risultati)
df_risultati.to_csv("percorso_multi_veicolo.csv", index=False)

