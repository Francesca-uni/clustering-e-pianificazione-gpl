import pandas as pd
import re
import time
from opencage.geocoder import OpenCageGeocode

# Dataset originale fornito dall'azienda, contenente dati anagrafici e geografici dei clienti
df = pd.read_excel("estrazione per minervas REV01.xlsx")
df.columns = df.columns.str.strip().str.replace("'", "").str.replace("à", "a")
df['Data'] = pd.to_datetime(df['Data'], dayfirst=True)
df = df.sort_values(['Codice Cliente', 'Data'])

# Dataset generato a seguito del modello di previsione 
# Contiene le date di consegna previste per ogni cliente (colonna 'Date_consegna_previste')
df_risultato_finale = pd.read_csv("risultati_random_forest.csv")

# Funzione per estrarre coordinate decimali dalla colonna GPS (in formato DMS)
def dms_to_decimal(dms_str):
    """
    Converte una stringa tipo "40°31'18.01''N" o "15°4'34.58''E" in float decimale
    """
    if pd.isnull(dms_str):
        return None
    match = re.match(r"(\d+)°(\d+)'([\d\.]+)''([NSEW])", dms_str.strip())
    if not match:
        return None
    degrees, minutes, seconds, direction = match.groups()
    decimal = float(degrees) + float(minutes)/60 + float(seconds)/3600
    if direction in ['S', 'W']:
        decimal *= -1
    return decimal

# Rimuovere duplicati basati sul Codice Cliente
df_unique = df.drop_duplicates(subset=['Codice Cliente'])

# Unione dei dati GPS, Via e Località dal DataFrame originale (df)
df_risultato_finale = df_risultato_finale.merge(df_unique[['Codice Cliente', 'GPS', 'Via', 'Localita']], on='Codice Cliente', how='left')

def normalizza_indirizzo(via):
    if pd.isnull(via): return via
    via = via.upper()
    via = via.replace("C/DA", "CONTRADA").replace("S.S.", "STRADA STATALE")
    via = re.sub(r"[^A-Z0-9\s]", " ", via)  # rimuove simboli strani
    return via.strip()

df_risultato_finale['Via_clean'] = df_risultato_finale['Via'].apply(normalizza_indirizzo)

# Conversione delle coordinate da DMS nel formato decimale
df_risultato_finale['latitudine'] = df_risultato_finale['GPS'].apply(lambda x: dms_to_decimal(x.split()[0]) if pd.notnull(x) else None)
df_risultato_finale['longitudine'] = df_risultato_finale['GPS'].apply(lambda x: dms_to_decimal(x.split()[1]) if pd.notnull(x) else None)

def estrai_localita_pulita(localita):
    if pd.isnull(localita): return localita
    match = re.search(r'\d+\s+(.*)', localita)
    return match.group(1) if match else localita

df_risultato_finale['Localita_clean'] = df_risultato_finale['Localita'].apply(estrai_localita_pulita)

# Aggiungi un controllo per verificare se le coordinate sono plausibili
def is_valid_coordinates(lat, lon):
    if lat is not None and lon is not None:
        return 36.0 <= lat <= 47.0 and 6.0 <= lon <= 18.0  # Intervallo geografico approssimato per l'Italia
    return False

# Applica il controllo per le coordinate
df_risultato_finale['valid_coordinates'] = df_risultato_finale.apply(lambda x: is_valid_coordinates(x['latitudine'], x['longitudine']), axis=1)

# Filtra i clienti con coordinate valide
df_valid_coordinates = df_risultato_finale[df_risultato_finale['valid_coordinates']]

# Visualizza i clienti che non hanno coordinate valide
df_invalid_coordinates = df_risultato_finale[df_risultato_finale['latitudine'].isna() | df_risultato_finale['longitudine'].isna()]
# Stampa i clienti con coordinate non valide
print(f"Clienti con coordinate non valide:\n{df_invalid_coordinates[['Codice Cliente', 'Via', 'Localita']]}")


# Filtra i clienti senza coordinate
clienti_senza_coord = df_risultato_finale[df_risultato_finale['latitudine'].isna() | df_risultato_finale['longitudine'].isna()]

# Geocodifica solo per i clienti senza coordinate valide
key = "d5cd143fc4ec4e5caab50b49f85f9bb7"
geocoder = OpenCageGeocode(key)

def get_coordinates(row):
    try:
        address = f"{row['Via_clean']}, {row['Localita_clean']}, Italy"
        print(f"Geocodificando indirizzo: {address}")
        result = geocoder.geocode(address)
        if result:
            return result[0]['geometry']['lat'], result[0]['geometry']['lng']
        else:
            return None, None
    except Exception as e:
        print(f"Errore geocodifica con OpenCage per {address}: {e}")
        return None, None

# Popolazione con le coordinate tramite la geocodifica
for index, row in clienti_senza_coord.iterrows():
    lat, lon = get_coordinates(row)  # Usa la funzione di geocodifica
    df_risultato_finale.at[index, 'latitudine'] = lat
    df_risultato_finale.at[index, 'longitudine'] = lon
    time.sleep(2)  # Pausa tra le richieste per evitare il blocco dell'API

def geocodifica_con_cap(row):
    cap_match = re.search(r'(\d{5})', str(row['Localita']))
    if not cap_match:
        return None, None
    cap = cap_match.group(1)
    address = f"{cap}, Italy"
    print(f"Geocodificando per CAP: {address}")
    try:
        result = geocoder.geocode(address)
        if result:
            return result[0]['geometry']['lat'], result[0]['geometry']['lng']
        return None, None
    except Exception as e:
        print(f"Errore con geocodifica per CAP {cap}: {e}")
        return None, None

clienti_ancora_senza_coord = clienti_senza_coord[
    clienti_senza_coord['latitudine'].isna() | clienti_senza_coord['longitudine'].isna()
]

for index, row in clienti_ancora_senza_coord.iterrows():
    lat, lon = geocodifica_con_cap(row)
    if lat and lon:
        df_risultato_finale.at[index, 'latitudine'] = lat
        df_risultato_finale.at[index, 'longitudine'] = lon
    time.sleep(1)  # anche 1 secondo può bastare per evitare blocchi API



# Ricalcola il campo di validità e aggiorna df_valid_coordinates    
df_risultato_finale['valid_coordinates'] = df_risultato_finale.apply(
    lambda x: is_valid_coordinates(x['latitudine'], x['longitudine']), axis=1
)
df_valid_coordinates = df_risultato_finale[df_risultato_finale['valid_coordinates']]
df_invalid_coordinates = df_risultato_finale[~df_risultato_finale['valid_coordinates']]


# Verifica i risultati finali
print(df_valid_coordinates[['Codice Cliente', 'Via', 'Localita', 'latitudine', 'longitudine']].head())

# Salva i risultati con coordinate valide
df_valid_coordinates.to_csv("clienti_validi_geocodificati.csv", index=False)

