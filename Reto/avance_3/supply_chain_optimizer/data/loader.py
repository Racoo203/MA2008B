import pandas as pd
import numpy as np

# --- Constants and Data ---

PLANTS = [
    "Agave lechuguilla", "Agave salmiana", "Agave scabra", "Agave striata",
    "Opuntia cantabrigiensis", "Opuntia engelmani", "Opuntia robusta",
    "Opuntia streptacanta", "Prosopis laevigata", "Yucca filifera"
]

STORES = ["Vivero", "Moctezuma", "Venado", "Laguna seca"]

SUPPLIER_PRICES_DATA = [
    [None, None, None, 26],
    [None, None, None, 26],
    [None, 26, None, 26],
    [None, 26, 25, None],
    [None, 17, 18, None],
    [None, None, 18, 21],
    [None, 17, 18, 18],
    [None, None, 18, None],
    [26.5, None, None, None],
    [26, None, None, None]
]

SPECIES_DATA = {
    "Especie": PLANTS,
    "Nombre común": [
        "Lechuguilla", "Maguey verde", "Maguey azul", "Maguey verde",
        "Cuijo", "Cuijo", "Tapón", "Cardón", "Mezquite", "Palma china"
    ],
    "HA": [33, 157, 33, 33, 39, 30, 58, 51, 69, 21],
    "% POR HA": [
        6.297709924, 29.96183206, 6.297709924, 6.297709924,
        7.442748092, 5.72519084, 11.06870229, 9.732824427,
        13.16793893, 4.007633588
    ],
    "N° de individuos 75 ha": [
        2475, 11775, 2475, 2475, 2925, 2250, 4350, 3825, 5175, 1575
    ],
    "Altura (cm)": [
        "30-50", "30-50", "15-25", "20-30", "20-30", "20-30",
        "20-30", "20-35", "25-30", "20-30"
    ],
    "Capacidad en la camioneta": [
        "VAN EN LA CAMIONETA", "VAN EN LA CAMIONETA", "VAN EN LA CAMIONETA",
        "VAN EN LA CAMIONETA", "1 REJA", "1 REJA", "2 REJAS", "2 REJAS",
        "2 REJAS", "1 REJA"
    ]
}

SPACE_DATA = {
    1: (-56, -1298, 5.4), 2: (-602, -1030, 7.52), 3: (-394, -1042, 8),
    4: (-195, -1046, 8), 5: (1, -1046, 7.56), 6: (-814, -932, 4.19),
    7: (-824, -643, 6.28), 8: (-600, -600, 7.6), 9: (-399, -639, 8),
    10: (-195, -632, 8), 11: (0, -600, 7.67), 12: (153, -492, 1.47),
    13: (-794, -258, 7.97), 14: (-195, -258, 5.98), 15: (56, -334, 5.4),
    16: (408, -158, 5.64), 17: (189, -43, 6.11), 18: (0, 0, 7.11),
    19: (600, 0, 4.92), 20: (-931, 75, 1.38), 21: (-808, 75, 8),
    22: (-604, 79, 7.82), 23: (-403, 136, 5.53), 24: (-206, 133, 5.64),
    25: (800, 200, 5.05), 26: (1000, 400, 4.75), 27: (-927, 445, 1.28),
    28: (-801, 434, 6.64), 29: (-597, 431, 6.54), 30: (-349, 380, 6.76),
    31: (-417, -269, 7.34)
}

# SELECTED_POLYGONS = [1, 3, 4, 5, 20, 23, 24, 18, 17, 16, 19, 25, 26]
SELECTED_POLYGONS = [2,3,4,18]

# --- DataFrame Creation Functions ---

def __create_species_df():
    df = pd.DataFrame(SPECIES_DATA)
    df.to_csv("./data/especies_reforestacion.csv", index=True)
    return df

def __create_space_df():
    ha_df = pd.DataFrame(SPACE_DATA).T
    ha_df.columns = ["X", "Y", "Hectareas"]
    ha_df.reset_index(inplace=True)
    ha_df.rename(columns={'index': 'Poligono'}, inplace=True)
    ha_df = ha_df[ha_df['Poligono'].isin(SELECTED_POLYGONS)]
    ha_df.to_csv('./data/ha.csv', index=False)
    return ha_df

def __create_supplier_prices_df():
    supplier_prices = pd.DataFrame(SUPPLIER_PRICES_DATA, index=PLANTS, columns=STORES)
    supplier_prices = supplier_prices.reset_index()
    supplier_prices.rename(columns={'index': 'specie'}, inplace=True)
    supplier_prices = supplier_prices.melt(id_vars='specie', var_name='supplier', value_name='price')
    supplier_prices.dropna(inplace=True)
    supplier_prices.to_csv('./data/supplier_prices.csv', index=False)
    return supplier_prices

def __create_demand_df(species_df, ha_df):
    species_names = species_df['Especie'].sort_values()
    polygon_ids = ha_df['Poligono'].sort_values().values
    demand_df = pd.DataFrame(index=species_names, columns=polygon_ids)
    for _, specie in species_df.iterrows():
        for _, row in ha_df.iterrows():
            demand = row['Hectareas'] * specie['N° de individuos 75 ha'] / 75.09
            demand_df.at[specie['Especie'], row['Poligono']] = demand
    demand_df = demand_df.astype(float).round(0).astype(int)
    demand_df.reset_index(inplace=True)
    demand_df.rename(columns={'Especie': 'specie'}, inplace=True)
    demand_df = demand_df.melt(id_vars='specie', var_name='polygon', value_name='demand')
    demand_df.to_csv('./data/demand.csv', index=False)
    return demand_df

# --- Main Execution ---

def create_datasets():
    species_df = __create_species_df()
    ha_df = __create_space_df()
    prices_df = __create_supplier_prices_df()
    demand_df = __create_demand_df(species_df, ha_df)

    return demand_df, prices_df, ha_df