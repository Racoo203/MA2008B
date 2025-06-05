# prueba/main.py

from pyomo.environ import ConcreteModel
from conjuntos import construir_conjuntos
from parametros import (
    capv, ct, a_e, capr, cc, FactE, cp, tc, td, jl,
    tpe, agg, dee, D, C, Lmax, M, n, s, declarar_parametros_en_modelo
)
from variables import definir_variables
from restricciones import definir_restricciones
from funcion_objetivo import definir_funcion_objetivo, definir_variables_desviacion
from resolucion import resolver_modelo, exportar_resultados_por_dia, debug_infactibilidad

# Modo de función objetivo: 'metas' o 'simple'
modo_objetivo = 'simple'  # cambiar a 'simple' si se desea minimizar solo costos

# Construir conjuntos
E, J, Z, R, T, Q, L, N, especies, viveros, E_nopal, E_bolsa = construir_conjuntos()

# Inicializar modelo
model = ConcreteModel()

# Definir componentes
definir_variables(model, E, J, T, Q, Z, L, N)
declarar_parametros_en_modelo(model, E, J, Z, a_e, cc, D, tpe, C, s)
definir_restricciones(model, E, J, T, Q, Z, L, N, capv, jl, tc, td, tpe, a_e, capr, D, M, s)

# Función objetivo
if modo_objetivo == 'metas':
    definir_variables_desviacion(model)
    definir_funcion_objetivo(model, E, J, T, Q, Z, L, a_e, cp, ct, modo='metas')
else:
    definir_funcion_objetivo(model, E, J, T, Q, Z, L, a_e, cp, ct, modo='simple')

# Resolver
resultados = resolver_modelo(model)

# Diagnóstico si hay problema
if resultados.solver.termination_condition.name == "infeasible":
    debug_infactibilidad(model, E, J, T, Q, Z, L)

# Exportar resultados
exportar_resultados_por_dia(model, E, J, T, Q, Z, L)

# Verificación visual básica
print("Especies:", E)
print("Viveros:", J)
print("Polígonos:", Z + R)
