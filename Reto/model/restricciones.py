# prueba/restricciones.py

from pyomo.environ import Constraint, ConstraintList

def definir_restricciones(model, E, J, T, Q, Z, L, N, capv, jl, tc, td, tpe, a_e, capr, D, M, C):

    # Restricciones tempolares

    model.restr_inventario_dia1 = Constraint(E, [Q[0]], rule=lambda m, e, q:
        m.inv[e, 1, T[0], q] == sum(m.xcomp[e,j,T[0]] for j in J)) 

    # ========= FASE 1: COMPRA Y ACOPIO EN ALMACÉN (NODO 1) =========

    # Restricción 1: Relación entre compra y transporte
    model.restr_compra_transporte = Constraint(E, J, T[1:], rule=lambda m, e, j, t:
        m.xtransf[e, j, t] <= m.xcomp[e, j, t])

    # Restricción 3: Tiempo logístico diario (entrada)
    model.restr_tiempo_entrada = Constraint(T, rule=lambda m, t:
        (tc + td)*sum(m.xsem[e,z,l,t,q] for l in L for e in E for z in Z for q in Q) / capv + sum(m.xsem[e,z,l,t,q] * tpe[e] for l in L for e in E for z in Z for q in Q) + 3*sum(m.Y[i,z,l,t]*C[(i,z)] for i in N for z in N for l in L) <= jl)

    # Restricción 4: Ingreso a inventario del almacén
    model.restr_ingreso_inventario = Constraint(E, T, rule=lambda m, e, t:
        m.inv[e, 1, t, 1] == sum(m.xtransf[e, j, t] for j in J))

    # Restricción 5: Capacidad del almacén (polígono 1)
    model.restr_capacidad_almacen = Constraint(T, rule=lambda m, t:
        sum(a_e[e] * m.inv[e, 1, t, q] for e in E for q in Q) <= capr)

    # Restricción 6: Inventario inicial en cero
    model.restr_inventario_inicial = Constraint(E, Q[1:], rule=lambda m, e, q:
        m.inv[e, 1, T[0], q] == 0)

    # ========= FASE 2: DISTRIBUCIÓN, RUTEO Y SIEMBRA ==========

    # Restricción 7: Evolución del inventario
    model.restr_evolucion_inventario = Constraint(E, T[1:], Q[1:], rule=lambda m, e, t, q:
        m.inv[e, 1, t, q] == m.inv[e, 1, t - 1, q - 1] - sum(m.xsem[e, z, l, t - 1, q - 1] for z in Z for l in L))

    # Restricción 8: Solo siembra con edad 3 a 7
    model.restr_aclimatacion = Constraint(T, Q, rule=lambda m, t, q:
        sum(m.xsem[e, z, l, t, q] for z in Z for l in L for e in E) == 0 if q < 3 or q > 7 else Constraint.Skip)
    
    # Restricción 9: Plantar solo lo que hay en el almacén
    model.restr_limite_plantado = Constraint(E, T, Q, rule = lambda m, e, t, q:
        sum(m.xsem[e,z,l,t,q] for z in Z for l in L) <= m.inv[e, 1, t, q])

    # Restricción 10: Cobertura exacta de demanda
    model.restr_cobertura_demanda = Constraint(E, Z, rule=lambda m, e, z:
        (sum(m.xsem[e, z, l, t, q] for l in L for t in T for q in Q) + m.demandaNoSatisfecha[z, e])== D[(e, z)])

    # Restricción 11: Capacidad por viaje
    model.restr_capacidad_viaje = Constraint(L, T, rule=lambda m, l, t:
        sum(m.xsem[e, z, l, t, q] for z in Z for e in E for q in Q) <= capv)

    # Restricción 12: Entrega solo si se visita
    model.restr_entrega_si_visita = Constraint(L, Z, T, rule=lambda m, l, z, t:
        sum(m.xsem[e, z, l, t, q] for e in E for q in Q) <= M * sum(m.Y[i, z, l, t] for i in N))

    # Restricción 13: Salida única desde receptor (nodo 1)
    model.restr_salida_base = Constraint(L, T, rule=lambda m, l, t:
        sum(m.Y[1, j, l, t] for j in Z) == 1)

    # Restricción 14: Regreso único al receptor
    model.restr_regreso_base = Constraint(L, T, rule=lambda m, l, t:
        sum(m.Y[j, 1, l, t] for j in Z) == 1)

    # Restricción 15: Flujo continuo
    model.restr_flujo = Constraint(Z, L, T, rule=lambda m, i, l, t:
        sum(m.Y[i, j, l, t] for j in N if j != i) == sum(m.Y[j, i, l, t] for j in N if j != i))

    # Restricción 16: Eliminación de subtours (MTZ)
    model.restr_mtz = Constraint(Z, Z, L, T, rule=lambda m, i, j, l, t:
        m.u[i, l] - m.u[j, l] + len(Z) * m.Y[i, j, l, t] <= len(Z) - 1 if i != j else Constraint.Skip)

    # ========= PUNTO DE REORDEN ==========

    #model.restr_reorden_activacion = Constraint(E, T, rule=lambda m, e, t:
    #    sum(m.inv[e, 1, t, q] for q in Q) <= s[e] + M * (1 - m.R[e, t]))

    #model.restr_reorden_compra = Constraint(E, T[:-1], rule=lambda m, e, t:
    #    sum(m.xcomp[e, j, t + 1] for j in J) >= 1e-3 * m.R[e, t])
