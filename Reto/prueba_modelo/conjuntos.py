# prueba/conjuntos.py

def construir_conjuntos():
    E = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # especies
    J = [1, 2, 3, 4]     # proveedores
    Z = [2, 3, 4]  # polígonos de siembra
    R = [1]        # polígono receptor (Base)
    T = list(range(1, 47))  # días 1 a 23
    Q = list(range(1, 8))   # edades posibles 1 a 7
    L = list(range(1, 4))   # máximo 3 viajes
    N = Z + R               # nodos VRP

    especies = {
        1: "Agave lechuguilla",
        2: "Agave salmiana",
        3: "Agave scabra",
        4: "Agave striata",
        5: "Opuntia cantabrigiensis",
        6: "Opuntia engelmani",
        7: "Opuntia robusta",
        8: "Opuntia streptacantha",
        9: "Prosopis laevigata",
        10: "Yucca filifera"
    }

    viveros = {
        1: "Vivero",
        2: "Moctezuma",
        3: "Venado",
        4: "Laguna seca"
    }

    E_nopal = []   # No hay especies nopal especificadas
    E_bolsa = E    # Todas requieren tratamiento tipo bolsa

    return E, J, Z, R, T, Q, L, N, especies, viveros, E_nopal, E_bolsa

construir_conjuntos()

