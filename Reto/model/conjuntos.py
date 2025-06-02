# prueba/conjuntos.py

def construir_conjuntos():
    E = [1, 2, 3]  # especies
    J = [1, 2]     # proveedores
    Z = [2, 3, 4]  # polígonos de siembra
    R = [1]        # polígono receptor (Base)
    T = list(range(1, 34))  # días 1 a 33
    Q = list(range(1, 8))   # edades posibles 1 a 7
    L = list(range(1, 4))   # máximo 3 viajes
    N = Z + R               # nodos VRP

    especies = {
        1: "Agave lechuguilla",
        2: "Agave salmiana",
        3: "Agave scabra"
    }

    viveros = {
        1: "Proveedor 1",
        2: "Proveedor 2"
    }

    E_nopal = []   # No hay especies nopal especificadas
    E_bolsa = E    # Todas requieren tratamiento tipo bolsa

    return E, J, Z, R, T, Q, L, N, especies, viveros, E_nopal, E_bolsa
