import pandas as pd
from datetime import date
from geopy.distance import geodesic
from opencage.geocoder import OpenCageGeocode

df_cluster = pd.read_csv("clienti_con_cluster_riparato.csv")

# Funzione di parsing per convertire stringhe in liste di oggetti datetime.date
def parse_date_objects(x):
    try:
        return eval(x, {"datetime": __import__("datetime"), "date": date})
    except Exception as e:
        print(f"Errore nella riga: {x[:50]}... -> {e}")
        return []

# Tramite l'explode ogni cliente può comparire in più righe (una per ogni data)
df_cluster['Date_consegna_previste'] = df_cluster['Date_consegna_previste'].apply(parse_date_objects)
df_exploded = df_cluster.explode('Date_consegna_previste')
df_exploded['Date_consegna_previste'] = pd.to_datetime(df_exploded['Date_consegna_previste'])

# Crea i giri di consegna raggruppando per data e cluster (cioè zona geografica)
piano_consegne = df_exploded.groupby(['Date_consegna_previste', 'cluster_finale'])['Codice Cliente'].apply(list).reset_index()
piano_consegne.columns = ['Data Consegna', 'Cluster', 'Clienti']

# Aggiungi il numero di clienti per ciascun giro
piano_consegne['Numero Clienti'] = piano_consegne['Clienti'].apply(len)

piano_consegne_filtrato = piano_consegne[piano_consegne['Numero Clienti'] >= 3]

# Giri esclusi (n < 3)
giri_esclusi = piano_consegne[piano_consegne['Numero Clienti'] < 3]

# Estrai tutti i clienti da quei giri per ripianificarli
clienti_esclusi = df_exploded.merge(
    giri_esclusi[['Data Consegna', 'Cluster']],
    left_on=['Date_consegna_previste', 'cluster_finale'],
    right_on=['Data Consegna', 'Cluster'],
    how='inner'
)

# Ripianificazione
def trova_data_vicina(row, df_validi, tolleranza=7):
    data_attuale = row['Date_consegna_previste']
    cluster = row['cluster_finale']
    
    # Cerca giri buoni nello stesso cluster, data diversa
    possibili = df_validi[
        (df_validi['Cluster'] == cluster) &
        (df_validi['Data Consegna'] != data_attuale)
    ]
    
    # Calcola distanza in giorni
    possibili['delta'] = (possibili['Data Consegna'] - data_attuale).abs().dt.days
    
    vicine = possibili[possibili['delta'] <= tolleranza]
    
    if not vicine.empty:
        return vicine.sort_values('delta').iloc[0]['Data Consegna']
    else:
        return None
    
# Assegna una nuova data ai clienti esclusi, cercando giri esistenti nello stesso cluster entro 7 giorni  
clienti_esclusi['Nuova_Data'] = clienti_esclusi.apply(
    lambda r: trova_data_vicina(r, piano_consegne_filtrato),
    axis=1
)

# Clienti ripianificabili
ripianificati = clienti_esclusi[clienti_esclusi['Nuova_Data'].notna()].copy()
ripianificati['Date_consegna_previste'] = ripianificati['Nuova_Data']

# Unisci ai clienti del piano valido
clienti_validi = df_exploded.merge(
    piano_consegne_filtrato[['Data Consegna', 'Cluster']],
    left_on=['Date_consegna_previste', 'cluster_finale'],
    right_on=['Data Consegna', 'Cluster'],
    how='inner'
)

# Unione dei clienti ripianificati e validi
clienti_finali = pd.concat([clienti_validi, ripianificati], ignore_index=True)

# Nuovo piano finale
piano_consegne_finale = clienti_finali.groupby(['Date_consegna_previste', 'cluster_finale'])['Codice Cliente'].apply(list).reset_index()
piano_consegne_finale.columns = ['Data Consegna', 'Cluster', 'Clienti']
piano_consegne_finale['Numero Clienti'] = piano_consegne_finale['Clienti'].apply(len)

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

piano_consegne_finale['Zona Consegna'] = piano_consegne_finale['Cluster'].astype(str).apply(mappa_zona)

print(piano_consegne_finale.sort_values('Data Consegna'))
piano_consegne_finale.sort_values('Data Consegna').to_csv("piano_consegne_finale.csv", index= False)

# Filtra solo quelli con coordinate valide
df_coord = df_cluster[df_cluster['valid_coordinates'] == True][['Codice Cliente', 'latitudine', 'longitudine']]


# Geocodifica dell'indirizzo del deposito ossia il punto di partenza di ogni giro di consegne
key = "d5cd143fc4ec4e5caab50b49f85f9bb7"
geocoder = OpenCageGeocode(key)

indirizzo = "85010 Vaglio Basilicata (PZ), Strada Statale 407, Italy"
result = geocoder.geocode(indirizzo)

if result and len(result):
    lat = result[0]['geometry']['lat']
    lng = result[0]['geometry']['lng']
    print(f'Coordinate del deposito: {lat}, {lng}')
else:
    print('Indirizzo non trovato.')


# Coordinate del punto di partenza di ciascun giro
deposito_coord = (lat, lng)

#deposito_coord = (40.656361, 15.880113)  

# Ottimizzazione dei percorsi tramite l'algoritmo euristico Nearest Neighbor
# che calcola un percorso approssimato minimo partendo dal deposito

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

    percorso_codici = [cliente['Codice Cliente'] for cliente in percorso]
    return round(distanza_totale, 2), percorso_codici


giorni = piano_consegne_finale['Data Consegna'].unique()

# Costruisce i percorsi finali per un solo veicolo che visita 
# più zone (cluster) nello stesso giorno, partendo dal deposito

risultati = [] # lista che conterrà i risultati

for giorno in sorted(giorni):
    sottoinsieme = piano_consegne_finale[piano_consegne_finale['Data Consegna'] == giorno]
    percorso_totale = []
    distanza_totale = 0
    cluster_sequence = []

    for _, riga in sottoinsieme.iterrows():
        clienti = riga['Clienti']
        cluster = riga['Cluster']
        df_clienti_giro = df_coord[df_coord['Codice Cliente'].isin(clienti)]

        dist, percorso = tsp_nearest_neighbor(df_clienti_giro, deposito_coord)

        percorso_totale += percorso
        distanza_totale += dist
        cluster_sequence += [cluster] * len(percorso)

    risultati.append({
        'Data Consegna': giorno,
        'Numero Cluster Serviti': len(sottoinsieme),
        'Numero Clienti': len(percorso_totale),
        'Distanza Totale Stimata (km)': round(distanza_totale, 2),
        'Percorso Ottimo (NN)': percorso_totale,
        'Cluster Percorso': cluster_sequence
    })

# Salva 
df_risultati = pd.DataFrame(risultati)
df_risultati = df_risultati.sort_values('Data Consegna')
df_risultati.to_csv("percorso_ottimizzato_nearest_neightbor.csv", index=False)