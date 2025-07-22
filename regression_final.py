# Librerie generali
import pandas as pd
import numpy as np
from datetime import timedelta

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Lettura e pulizia iniziale
df = pd.read_excel("estrazione per minervas REV01.xlsx")
df.columns = df.columns.str.strip().str.replace("'", "").str.replace("à", "a")
df['Data'] = pd.to_datetime(df['Data'], dayfirst=True)
df = df.sort_values(['Codice Cliente', 'Data'])

# Filtra clienti con almeno 5 consegne
consegne_per_cliente = df.groupby("Codice Cliente")['Data'].count().reset_index()
clienti_validi = consegne_per_cliente[consegne_per_cliente['Data'] >= 5]['Codice Cliente']
df_filtrato = df[df['Codice Cliente'].isin(clienti_validi)].copy()

# Feature engineering
df_filtrato['Data_prec'] = df_filtrato.groupby('Codice Cliente')['Data'].shift(1)
df_filtrato['Quantita_prec'] = df_filtrato.groupby('Codice Cliente')['Quantita [litri]'].shift(1)
df_filtrato['Giorni_trascorsi'] = (df_filtrato['Data'] - df_filtrato['Data_prec']).dt.days
df_filtrato = df_filtrato.dropna(subset=['Giorni_trascorsi', 'Quantita_prec'])
df_filtrato['Consumo_giornaliero'] = df_filtrato['Quantita_prec'] / df_filtrato['Giorni_trascorsi']
df_filtrato['Mese'] = df_filtrato['Data'].dt.month
df_filtrato['Stagione'] = df_filtrato['Mese'].apply(lambda x: 'Inverno' if x in [12,1,2] else 'Estate' if x in [6,7,8] else 'Intermedio')

# Preprocessing
categorical_features = ['Localita', 'Ragione sociale', 'Via', 'Stagione']
numerical_features = ['Quantita_prec', 'Consumo_giornaliero']
df_filtrato[numerical_features] = df_filtrato[numerical_features].replace([np.inf, -np.inf], np.nan)
df_filtrato = df_filtrato.dropna(subset=categorical_features + numerical_features + ['Giorni_trascorsi'])

X = df_filtrato[numerical_features + categorical_features]
y = df_filtrato['Giorni_trascorsi']

preprocessor = ColumnTransformer([
    ('num', 'passthrough', numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modello 1: Random Forest
rf_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])
rf_model.fit(X_train, y_train)
df_filtrato['Giorni_previsti_RF'] = rf_model.predict(X)

# Modello 2: Regressione Lineare
lr_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
lr_model.fit(X_train, y_train)
df_filtrato['Giorni_previsti_LR'] = lr_model.predict(X)

# Calcolo media per cliente e classificazione
def classificazione(g):
    return 'Urgente' if g < 15 else ('Normale' if g <= 30 else 'Lento')

for modello in ['RF', 'LR']:
    media = df_filtrato.groupby('Codice Cliente')[f'Giorni_previsti_{modello}'].mean().reset_index()
    media.columns = ['Codice Cliente', 'Media_giorni_previsti']
    media['Classe_cliente'] = media['Media_giorni_previsti'].apply(classificazione)

    ultima_data = df_filtrato.groupby('Codice Cliente')['Data'].max().reset_index()
    ultima_data.columns = ['Codice Cliente', 'Data_ultima_consegna']
    ultima_quantita = df_filtrato.sort_values('Data').groupby('Codice Cliente').tail(1)[['Codice Cliente', 'Quantita [litri]']]
    ultima_quantita.columns = ['Codice Cliente', 'Quantita_ultima_consegna']

    media_completa = media.merge(ultima_data, on='Codice Cliente').merge(ultima_quantita, on='Codice Cliente')

    def genera_date(row):
        giorni = int(round(row['Media_giorni_previsti']))
        if giorni <= 0: giorni = 1
        moltiplicatore = 1 + (row['Quantita_ultima_consegna'] / 1000) * 0.1
        giorni = int(round(giorni * moltiplicatore))
        start = row['Data_ultima_consegna'] + timedelta(days=giorni)
        end = row['Data_ultima_consegna'] + pd.DateOffset(years=5)
        data_limite = pd.to_datetime("2025-05-31")
        current = start
        future_dates = []
        while current <= end:
            if current >= data_limite:
                future_dates.append(current.date())
            current += timedelta(days=giorni)
        return future_dates


    media_completa['Date_consegna_previste'] = media_completa.apply(genera_date, axis=1)

    # Calcolo del consumo medio giornaliero
    media_completa['Consumo_medio_giornaliero'] = media_completa['Quantita_ultima_consegna'] / media_completa['Media_giorni_previsti']
    media_completa['Consumo_medio_giornaliero'] = media_completa['Consumo_medio_giornaliero'].round(1)
    # Riduzione delle date a massimo 10 per cliente
    media_completa['Date_consegna_previste'] = media_completa['Date_consegna_previste'].apply(lambda x: x[:10])

    # Ordinamento per priorità
    media_completa['Classe_ordine'] = media_completa['Classe_cliente'].map({'Urgente': 0, 'Normale': 1, 'Lento': 2})
    # Dopo il calcolo di media_completa, salva tutti i clienti (non solo 3)
    media_completa.to_csv("hotEncoding.csv", index=False)


    # Selezione di un solo cliente per ogni classe
    clienti_selezionati = media_completa.sort_values(by='Classe_ordine').groupby('Classe_cliente').head(1)

    # Stampa finale
    risultato = clienti_selezionati[['Codice Cliente', 'Classe_cliente', 'Quantita_ultima_consegna', 'Consumo_medio_giornaliero', 'Date_consegna_previste']]
    print(f"\nRisultati modello: {modello}")
    print(risultato.reset_index(drop=True))

    # Valutazione dei modelli
def valuta_modello(nome, y_true, y_pred):
    print(f"\nValutazione modello: {nome}")
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(" - RMSE:", round(rmse, 2))
    print(" - MAE :", round(mae, 2))
    print(" - R²  :", round(r2, 3))

valuta_modello("Random Forest", y_test, rf_model.predict(X_test))
valuta_modello("Regressione Lineare", y_test, lr_model.predict(X_test))