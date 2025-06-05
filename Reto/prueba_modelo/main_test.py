# prueba/main_test.py

from pyomo.environ import ConcreteModel
from conjuntos import construir_conjuntos
from parametros import (
    capv, ct, a_e, capr, cc, FactE, cp, tc, td, jl, tpe,
    agg, dee, D, C, Lmax, M, n, s, declarar_parametros_en_modelo
)
from variables import definir_variables
from restricciones import definir_restricciones
from funcion_objetivo import definir_funcion_objetivo
from resolucion import resolver_modelo, exportar_resultados_por_dia, debug_infactibilidad

# Construir los conjuntos
E, J, Z, R, T, Q, L, N, especies, viveros, E_nopal, E_bolsa = construir_conjuntos()

# Crear modelo concreto
model = ConcreteModel()

# Definir variables y parámetros
definir_variables(model, E, J, T, Q, Z, L, N)
declarar_parametros_en_modelo(model, E, J, Z, a_e, cc, D, tpe, C, s)

# Agregar restricciones
definir_restricciones(model, E, J, T, Q, Z, L, N, capv, jl, tc, td, tpe, a_e, capr, D, M, s)

# Función objetivo SIMPLE (solo costo total): use False si quieres la multiobjetivo
definir_funcion_objetivo(model, E, J, T, Q, a_e, cp, ct, modo="simple", w1=1, w2=1, w3=1, B=100000, I_max=100000, t_ideal=10)

# Resolver modelo
resultados = resolver_modelo(model, solver_name="cbc", mostrar_resultado=True)

# Diagnóstico si es infactible
if resultados.solver.termination_condition.name == "infeasible":
    debug_infactibilidad(model, E, J, T, Q)

# Exportar resultados
exportar_resultados_por_dia(model, E, J, T, Q, Z)

# Verificación básica
print("Especies:", E)
print("Viveros:", J)
print("Polígonos:", N)
