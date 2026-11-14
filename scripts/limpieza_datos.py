import pandas as pd
import numpy as np
from pathlib import Path

df = pd.read_csv('data/raw/atp_matches_2025.csv')

df_filtrado = df[
    (df['surface'] == 'Hard') &
    (df['best_of'] == 3) &
    (~df['score'].str.contains(r'RET|W/O|DEF', case=False, na=False))
].copy()

ganadores = df_filtrado[[
    'winner_name',
    'winner_age',
    'winner_ht',
    'winner_ioc',
    'w_2ndWon',
    'w_2ndIn',
    'w_2ndLost'
]].copy()

ganadores.columns = [
    'Jugador', 'Edad', 'Altura', 'Pais',
    '2do_Ganados', '2do_Dentro', '2do_Perdidos'
]
ganadores['Resultado'] = 1

perdedores = df_filtrado[[
    'loser_name',
    'loser_age',
    'loser_ht',
    'loser_ioc',
    'l_2ndWon',
    'l_2ndIn',
    'l_2ndLost'
]].copy()

perdedores.columns = [
    'Jugador', 'Edad', 'Altura', 'Pais',
    '2do_Ganados', '2do_Dentro', '2do_Perdidos'
]
perdedores['Resultado'] = 0

df_largo = pd.concat([ganadores, perdedores], ignore_index=True)

columnas_numericas = ['2do_Ganados', '2do_Dentro', '2do_Perdidos']
for col in columnas_numericas:
    df_largo[col] = pd.to_numeric(df_largo[col], errors='coerce')

resumen = df_largo.groupby('Jugador').agg(
    Partidos_Jugados=('Resultado', 'count'),
    Partidos_Ganados=('Resultado', 'sum'),
    Puntos_2do_Ganados=('2do_Ganados', 'sum'),
    Puntos_2do_Dentro=('2do_Dentro', 'sum'),
    Puntos_2do_Perdidos=('2do_Perdidos', 'sum')
).reset_index()

resumen['Puntos_2do_Disputados'] = (
    resumen['Puntos_2do_Ganados'] + resumen['Puntos_2do_Perdidos']
)

UMBRAL_MINIMO_PARTIDOS = 10
resumen_filtrado = resumen[
    resumen['Partidos_Jugados'] >= UMBRAL_MINIMO_PARTIDOS
].copy()

resumen_filtrado['Y_Win_Rate'] = (
    resumen_filtrado['Partidos_Ganados'] /
    resumen_filtrado['Partidos_Jugados']
) * 100

resumen_filtrado['X_Efectividad_2do_Servicio'] = np.where(
    resumen_filtrado['Puntos_2do_Disputados'] > 0,
    (
        resumen_filtrado['Puntos_2do_Ganados'] /
        resumen_filtrado['Puntos_2do_Disputados']
    ) * 100,
    np.nan
)

resumen_filtrado['Tasa_victorias'] = resumen_filtrado['Tasa_victorias'].round(2)
resumen_filtrado['X_Efectividad_2do_Servicio'] = (
    resumen_filtrado['X_Efectividad_2do_Servicio'].round(2)
)

Path('data/processed').mkdir(parents=True, exist_ok=True)
resumen_filtrado.to_csv('data/processed/matriz_atp_2025_spss.csv', index=False)

print(f"Número de unidades de observación n: {len(resumen_filtrado)} jugadores")