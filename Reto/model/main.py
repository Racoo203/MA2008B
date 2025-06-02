# prueba/main.py

from pyomo.environ import ConcreteModel

from conjuntos import construir_conjuntos
from parametros import (
    capv, ct, a_e, capr, cc, cp, tc, td, jl,
    tpe, D, C, Lmax, M, s, pen, declarar_parametros_en_modelo
)
from variables import definir_variables
from restricciones import definir_restricciones
from funcion_objetivo import definir_funcion_objetivo
from resolucion import resolver_modelo, exportar_resultados_por_dia, debug_infactibilidad

# ===============================
# Construcción del modelo
# ===============================
E, J, Z, R, T, Q, L, N, especies, viveros, E_nopal, E_bolsa = construir_conjuntos()
model = ConcreteModel()

definir_variables(model, E, J, T, Q, Z, L, N)
declarar_parametros_en_modelo(model, E, J, Z, a_e, cc, D, tpe, C, s)
definir_restricciones(model, E, J, T, Q, Z, L, N, capv, jl, tc, td, tpe, a_e, capr, D, M, C)

# Cambia a True si quieres usar función objetivo multiobjetivo
tipo = 'simple'
definir_funcion_objetivo(model, E, J, T, Q, L, Z, a_e, cp, ct, pen, tipo = tipo)

# ===============================
# Resolución
# ===============================
resultados = resolver_modelo(model, solver_name="cbc")

if resultados.solver.termination_condition.name == "infeasible":
    debug_infactibilidad(model, E, J, T, Q, Z, L)

exportar_resultados_por_dia(model, E, J, T, Q, Z, L)

# ===============================
# Verificación
# ===============================
print("Especies:", E)
print("Viveros:", J)
print("Polígonos:", N)
