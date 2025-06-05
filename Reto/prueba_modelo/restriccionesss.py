# prueba/restricciones.py

from pyomo.environ import Constraint, ConstraintList

def definir_restricciones(model, E, J, T, Q, Z, L, N, capv, jl, tc, td, tpe, a_e, capr, D, M, s):

    # ========= FASE 1: COMPRA Y ACOPIO EN ALMACÉN (NODO 1) =========

    # Restricciones temporales
    # Permitir transporte el primer día igual a la compra del mismo día
    model.restr_compra_transporte_dia1 = Constraint(E, J, rule=lambda m, e, j:
        m.xtransf[e, j, T[0]] <= m.xcomp[e, j, T[0]])
    
    # Permitir transporte el primer día solo si hay compra
    model.restr_compra_minima = Constraint(E, J, rule=lambda m, e, j:
        m.xcomp[e, j, T[0]] >= 1)
    model.restr_transf_minima = Constraint(E, J, rule=lambda m, e, j:
        m.xtransf[e, j, T[0]] >= 1)
    # -------------------------------------------------

    # Restricción 1: Relación entre compra y transporte
    model.restr_compra_transporte = Constraint(E, J, T[1:], rule=lambda m, e, j, t:
        m.xtransf[e, j, t] <= m.xcomp[e, j, t - 1])

    # Restricción 2: Capacidad diaria del camión (entrada)
    model.restr_capacidad_entrada = Constraint(T, rule=lambda m, t:
        sum(m.xtransf[e, j, t] for e in E for j in J) <= capv * m.T[t])

    # Restricción 3: Tiempo logístico diario (entrada)
    model.restr_tiempo_entrada = Constraint(T, rule=lambda m, t:
        sum((tc + td + tpe[e]) * m.xtransf[e, j, t] / capv for e in E for j in J) <= jl)

    # Restricción 4: Ingreso a inventario del almacén
    model.restr_ingreso_inventario = Constraint(E, T, rule=lambda m, e, t:
        m.inv[e, 1, t, 1] == sum(m.xtransf[e, j, t] for j in J))

    # Restricción 5: Capacidad del almacén (polígono 1)
    model.restr_capacidad_almacen = Constraint(T, rule=lambda m, t:
        sum(a_e[e] * m.inv[e, 1, t, q] for e in E for q in Q) <= capr)

    # Restricción 6: Inventario inicial en cero
    model.restr_inventario_inicial = Constraint(E, Q, rule=lambda m, e, q:
        m.inv[e, 1, T[0], q] >= 0) # Cambiado para prueba 

    # ========= FASE 2: DISTRIBUCIÓN, RUTEO Y SIEMBRA ==========

    # Restricción 7: Evolución del inventario
    model.restr_evolucion_inventario = Constraint(E, T[1:], Q[1:], rule=lambda m, e, t, q:
        m.inv[e, 1, t, q] == m.inv[e, 1, t - 1, q - 1] - m.xsem[e, 1, t - 1, q - 1])

    # Restricción 8: Solo siembra con edad 3 a 7
    #model.restr_aclimatacion = Constraint(E, T, Q, rule=lambda m, e, t, q:
    #    m.xsem[e, 1, t, q] == 0 if q < 3 or q > 7 else Constraint.Skip)
    # Descomenta la línea de arriba para volver a activar la restricción
    # Se realizó un cambio para permitir siembra de cualquier edad en la prueba

    # Restricción 9: Enlace entre entregas y siembra
    model.restr_entrega_siembra = Constraint(E, Z, rule=lambda m, e, z:
        sum(m.X[l, z, e, t] for l in L for t in T) == sum(m.xsem[e, 1, t, q] for t in T for q in Q))
    # Se cammbio Q[2:7] a Q para permitir siembra de cualquier edad (cambio temporal para prueba)

    # Restricción 10: Cobertura exacta de demanda (demanda total)
    model.restr_cobertura_demanda = Constraint(E, Z, rule=lambda m, e, z:
        sum(m.X[l, z, e, t] for l in L for t in T) + m.demanda_insatisfecha[e, z] == D[e, z])

    # Restricción 11: Capacidad por viaje (por día)
    model.restr_capacidad_viaje = Constraint(L, T, rule=lambda m, l, t:
        sum(m.X[l, z, e, t] for z in Z for e in E) <= capv)

    # Restricción 12: Entrega solo si se visita (por día)
    model.restr_entrega_si_visita = Constraint(L, Z, E, T, rule=lambda m, l, z, e, t:
        m.X[l, z, e, t] <= M * sum(m.Y[i, z, l, t] for i in N))

    # Restricción 13: Salida única desde receptor (nodo 1, por viaje y día)
    model.restr_salida_base = Constraint(L, T, rule=lambda m, l, t:
        sum(m.Y[1, j, l, t] for j in Z) == 1)

    # Restricción 14: Regreso único al receptor (por viaje y día)
    model.restr_regreso_base = Constraint(L, T, rule=lambda m, l, t:
        sum(m.Y[j, 1, l, t] for j in Z) == 1)

    # Restricción 15: Flujo continuo (por día)
    model.restr_flujo = Constraint(Z, L, T, rule=lambda m, i, l, t:
        sum(m.Y[i, j, l, t] for j in N if j != i) == sum(m.Y[j, i, l, t] for j in N if j != i))

    # Restricción 16: Eliminación de subtours (MTZ, por día)
    model.restr_mtz = Constraint(Z, Z, L, T, rule=lambda m, i, j, l, t:
        m.u[i, l, t] - m.u[j, l, t] + len(Z) * m.Y[i, j, l, t] <= len(Z) - 1 if i != j else Constraint.Skip)

    # ========= PUNTO DE REORDEN ==========

    #model.restr_reorden_activacion = Constraint(E, T, rule=lambda m, e, t:
    #    sum(m.inv[e, 1, t, q] for q in Q) <= s[e] + M * (1 - m.R[e, t]))

    #model.restr_reorden_compra = Constraint(E, T[:-1], rule=lambda m, e, t:
    #    sum(m.xcomp[e, j, t + 1] for j in J) >= 1e-3 * m.R[e, t])