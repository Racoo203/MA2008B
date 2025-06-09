# prueba/restricciones.py

from pyomo.environ import Constraint, ConstraintList

def definir_restricciones(model, E, J, T, Q, Z, L, N, capv, jl, tc, td, tpe, a_e, capr, D, M, s):

    # ========= FASE 1: COMPRA Y ACOPIO EN ALMACÉN (NODO 1) =========
    # Restricción 0.1: Permitir transporte el primer día igual a la compra del mismo día
    model.restr_compra_transporte_dia1 = Constraint(E, J, rule=lambda m, e, j:
        m.xtransf[e, j, T[0]] <= m.xcomp[e, j, T[0]])
    
    
    # Restricción 0.3: En el día 1 no hay plantas de edad mayor a 1
    model.restr_edad_plantas_dia1 = Constraint(Q, rule=lambda m, q:
        sum(m.inv[e, 1, T[0], q] for e in E) == 0 if q > 1 else Constraint.Skip)

    
    ## Restricción 0.5: Activar variable binaria de compra si hay compra ###
    model.restr_compra_binaria = Constraint(T,J, rule=lambda m, t, j:
        sum(m.xcomp[e, j, t] for e in E) <= 8000 * m.compra_realizada[t,j])

    # Restricción 1: Relación entre compra y transporte
    model.restr_compra_transporte = Constraint(E, J, T[1:], rule=lambda m, e, j, t:
        m.xtransf[e, j, t] <= m.xcomp[e, j, t-1])
    
    model.restr_tiempo_entrada = Constraint(T, rule=lambda m, t:
    (tc + td) * sum((m.xsem[e,z,q,t,l] / m.capv) for e in E for z in Z for q in Q for l in L) +  
    sum(m.xsem[e,z,q,t,l] * m.tpe[e] for e in E for z in Z for q in Q for l in L) +
    3 * sum(m.C[(i,j)] * m.Y[i,j,l,t] for i in N for j in N for l in L) <= jl)  

    # Restricción 4: Ingreso a inventario del almacén
    model.restr_ingreso_inventario = Constraint(E, T, rule=lambda m, e, t:
        m.inv[e, 1, t, 1] == sum(m.xtransf[e, j, t] for j in J))

    # Restricción 5: Capacidad del almacén (polígono 1)
    model.restr_capacidad_almacen = Constraint(T, rule=lambda m, t:
        sum(a_e[e] * m.inv[e, 1, t, q] for e in E for q in Q) <= capr)


    # ========= FASE 2: DISTRIBUCIÓN, RUTEO Y SIEMBRA ==========

    # Restricción 7: Evolución del inventario
    model.restr_evolucion_inventario = Constraint(E, T[1:], Q[1:], rule=lambda m, e, t, q:
        m.inv[e, 1, t, q] == m.inv[e, 1, t - 1, q - 1] - sum(m.xsem[e, z, q - 1, t - 1, l] for z in Z for l in L))

    # Restricción 8: Solo siembra con edad 3 a 7
    model.restr_aclimatacion = Constraint(Q, rule=lambda m, q:
        sum(m.xsem[e, z, q, t, l] for e in E for t in T for l in L for z in Z) == 0 if q < 3 or q > 7 else Constraint.Skip)

    # Restricción 9: Planta solo si hay inventario suficiente
    model.restr_plantar_inventario = Constraint(E, Q, T, rule=lambda m, e, q, t:
        sum(m.xsem[e, z, q, t, l] for l in L for z in Z) <= m.inv[e, 1, t, q])
    
    # Restricción 10: Cobertura exacta de demanda (demanda total)
    model.restr_cobertura_demanda = Constraint(E, Z, rule=lambda m, e, z:
        sum(m.xsem[e,z,q,t,l] for l in L for t in T for q in Q) + m.demanda_insatisfecha[e, z] == D[e, z])

    # Restricción 11: Capacidad por viaje (por día)
    model.restr_capacidad_viaje = Constraint(L, T, rule=lambda m, l, t:
        sum(m.xsem[e,z,q,t,l] for z in Z for e in E for q in Q) <= capv)
        
    # Restricción 12: Entrega solo si se visita (por día)
    # Se utiliza una constante M para linealizar la restricción con respecto a lo que se envia y lo que realmente se tiene
    model.restr_entrega_si_visita = Constraint(L, Z, T, rule=lambda m, l, z, t:
        sum(m.xsem[e, z, q, t, l] for e in E for q in Q) <= M * sum(m.Y[i, z, l, t] for i in N))

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
