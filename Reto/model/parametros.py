# prueba/parametros.py

# Parámetros logísticos y operativos
capv = 240                  # Capacidad del camión por viaje
ct = 4500                   # Costo fijo por uso del camión por día
capr = 100000               # Capacidad de almacenamiento del receptor (en m²)
cp = 20                     # Costo de plantación por unidad
tc = 30                     # Tiempo de carga por operación (min)
td = 30                     # Tiempo de descarga por operación (min)
jl = 360                    # Jornada laboral diaria (min)
Lmax = 3                    # Máximo número de viajes
M = 999999                  # Big-M
pen = 100000000             # Penalización por demanda insatisfecha
#n = 10                     # Número de días en el horizonte

# Área por unidad de especie
a_e = {
    1: 0.1,
    2: 0.1,
    3: 0.1
}

# Costos de compra (especie, proveedor)
cc = {
    (1, 1): 26, (1, 2): 999999,
    (2, 1): 23, (2, 2): 25,
    (3, 1): 26, (3, 2): 25
}

# Tiempo de preparación antes de siembra por especie
tpe = {
    1: 1,
    2: 1,
    3: 1
}

# Demanda por especie y polígono
D = {
    (1, 2): 509, (1, 3): 576, (1, 4): 464,
    (2, 2): 1811, (2, 3): 2049, (2, 4): 1650,
    (3, 2): 509, (3, 3): 576, (3, 4): 464
}

# Punto de reorden por especie
s = {
    1: 50,
    2: 50,
    3: 50
}

# Distancias entre nodos
C = {
    (1, 1): 0,     (1, 2): 0.626, (1, 3): 0.441, (1, 4): 0.868,
    (2, 1): 0.626, (2, 2): 0,     (2, 3): 0.209, (2, 4): 0.242,
    (3, 1): 0.441, (3, 2): 0.209, (3, 3): 0,     (3, 4): 0.441,
    (4, 1): 0.868, (4, 2): 0.242, (4, 3): 0.441, (4, 4): 0
}

# Función para declarar en Pyomo
def declarar_parametros_en_modelo(model, E, J, Z, a_e, cc, D, tpe, C, s):
    from pyomo.environ import Param, NonNegativeReals
    model.cc = Param(E, J, initialize=cc, within=NonNegativeReals)
    model.a_e = Param(E, initialize=a_e, within=NonNegativeReals)
    model.D = Param(E, Z, initialize=D, within=NonNegativeReals)
    model.tpe = Param(E, initialize=tpe, within=NonNegativeReals)
    model.C = Param(Z + [1], Z + [1], initialize=C, within=NonNegativeReals)
    model.s = Param(E, initialize=s, within=NonNegativeReals)
