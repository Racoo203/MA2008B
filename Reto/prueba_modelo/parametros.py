# prueba/parametros.py

# Parámetros generales
capv = 524           # Capacidad del camión
ct = 4500            # Costo por activación del transporte
capr = 4000         # Capacidad del almacén (ilimitada en esta prueba)
cp = 1               # No se considera costo por siembra
tc = 30              # Tiempo de carga
td = 30              # Tiempo de descarga
jl = 360             # Jornada laboral (minutos)
Lmax = 3             # Máximo número de viajes
M = 99999            # Constante grande
n = 10               # Número de días

# Área ocupada por especie (no relevante en esta prueba)
#a_e = {1: 0.1, 2: 0.1, 3: 0.1}
a_e = {
    1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1,
    6: 0.1, 7: 0.1, 8: 0.1, 9: 0.1, 10: 0.1
}


'''
# Costos de compra (cc[e,j])
cc = {
    (1, 1): 26, (2, 1): 23, (3, 1): 26,
    (1, 2): 999999, (2, 2): 25, (3, 2): 25,
    (1, 3): 999999, (2, 3): 27, (3, 3): 21,
    (1, 4): 22, (2, 4): 22, (3, 4): 999999
}
'''

cc = {
    (1, 1): 999999, (1, 2): 999999, (1, 3): 999999, (1, 4): 26,
    (2, 1): 999999, (2, 2): 999999, (2, 3): 999999, (2, 4): 26,
    (3, 1): 999999, (3, 2): 999999, (3, 3): 26,     (3, 4): 26,
    (4, 1): 999999, (4, 2): 999999, (4, 3): 25,     (4, 4): 999999,
    (5, 1): 999999, (5, 2): 17,     (5, 3): 18,     (5, 4): 999999,
    (6, 1): 999999, (6, 2): 999999, (6, 3): 18,     (6, 4): 21,
    (7, 1): 999999, (7, 2): 17,     (7, 3): 18,     (7, 4): 999999,
    (8, 1): 999999, (8, 2): 999999, (8, 3): 18,     (8, 4): 999999,
    (9, 1): 26.5,   (9, 2): 999999, (9, 3): 999999, (9, 4): 999999,
    (10,1): 26,     (10,2): 999999, (10,3): 999999, (10,4): 999999
}

# Disponibilidad (FactE[e,j] = 1 si el proveedor j ofrece la especie e)
FactE = {(e, j): 1 if cc[(e, j)] < 999999 else 0 for e in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] for j in [1, 2, 3, 4]}


# Tiempo de preparación (tpe[e])
#tpe = {1: 1, 2: 1, 3: 1}

tpe = {
    1: 1, 2: 1, 3: 1, 4: 1, 5: 1,
    6: 1, 7: 1, 8: 1, 9: 1, 10: 1
}

# Área de polígonos (agg[z])
agg = {2: 1.0, 3: 1.0, 4: 1.0}

# Densidad de siembra por especie
dee = {1: 100, 2: 100, 3: 100}

# Demanda por especie y polígono
#D = {
#    (1, 2): 509, (2, 2): 576, (3, 2): 464,
#    (1, 3): 1811, (2, 3): 2049, (3, 3): 1650,
#    (1, 4): 509, (2, 4): 576, (3, 4): 464
#}

D = {
    # Polígono 2
    (1, 2): 264,  (2, 2): 1254, (3, 2): 264,  (4, 2): 264,  (5, 2): 312,
    (6, 2): 240,  (7, 2): 463,  (8, 2): 408,  (9, 2): 551,  (10, 2): 168,
    # Polígono 3
    (1, 3): 264,  (2, 3): 1254, (3, 3): 264,  (4, 3): 264,  (5, 3): 312,
    (6, 3): 240,  (7, 3): 463,  (8, 3): 408,  (9, 3): 551,  (10, 3): 168,
    # Polígono 4
    (1, 4): 249,  (2, 4): 1185, (3, 4): 249,  (4, 4): 249,  (5, 4): 294,
    (6, 4): 227,  (7, 4): 438,  (8, 4): 385,  (9, 4): 521,  (10, 4): 159,
}

# Punto de reorden (mínimo en inventario antes de pedir)
s = {1: 10, 2: 10, 3: 10}

# Distancias entre nodos (C[i,j]) incluyendo almacén (1) y polígonos (2,3,4)
C = {
    (1, 1): 0.0, (1, 2): 0.626, (1, 3): 0.441, (1, 4): 0.868,
    (2, 1): 0.626, (2, 2): 0.0, (2, 3): 0.209, (2, 4): 0.242,
    (3, 1): 0.441, (3, 2): 0.209, (3, 3): 0.0, (3, 4): 0.441,
    (4, 1): 0.868, (4, 2): 0.242, (4, 3): 0.441, (4, 4): 0.0
}


# Función para declarar parámetros dentro del modelo
def declarar_parametros_en_modelo(model, E, J, Z, a_e, cc, D, tpe, C, s):
    from pyomo.environ import Param, NonNegativeReals
    model.cc = Param(E, J, initialize=cc, within=NonNegativeReals)
    model.a_e = Param(E, initialize=a_e, within=NonNegativeReals)
    model.D = Param(E, Z, initialize=D, within=NonNegativeReals)
    model.tpe = Param(E, initialize=tpe, within=NonNegativeReals)
    model.C = Param(Z + [1], Z + [1], initialize=C, within=NonNegativeReals)
    model.s = Param(E, initialize=s, within=NonNegativeReals)
    model.capv = Param(initialize=capv, within=NonNegativeReals)
