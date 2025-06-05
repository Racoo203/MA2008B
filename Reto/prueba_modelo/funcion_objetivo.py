# prueba/funcion_objetivo.py

# Penalización para demanda insatisfecha
penalizacion = 10000000000000000

from pyomo.environ import Objective, minimize, Var, NonNegativeReals

def definir_variables_desviacion(model):
    """Variables de desviación para el enfoque por metas."""
    model.d1_pos = Var(domain=NonNegativeReals)
    model.d1_neg = Var(domain=NonNegativeReals)
    model.d2_pos = Var(domain=NonNegativeReals)
    model.d2_neg = Var(domain=NonNegativeReals)
    model.d3_pos = Var(domain=NonNegativeReals)
    model.d3_neg = Var(domain=NonNegativeReals)

def definir_funcion_objetivo(model, E, J, T, Q, Z, L, a_e, cp, ct, modo='metas',
                              w1=1, w2=1, w3=1, B=100000, I_max=100000, t_ideal=10):
    """
    Define la función objetivo del modelo.
    Si modo = 'metas', usa programación por metas.
    Si modo = 'simple', minimiza solo el costo total.
    """

    if modo == 'metas':
        # Meta 1: Costo total (compras + transporte + siembra)
        model.restr_presupuesto = model.Bmeta = \
            sum(model.xcomp[e, j, t] * model.cc[e, j] for e in E for j in J for t in T) + \
            sum(ct * model.T[t] for t in T) + \
            sum(cp * model.xsem[e, z, q, t, l] for e in E for t in T for q in Q for z in Z for l in L) + \
            penalizacion * sum(model.demanda_insatisfecha[e, z] for e in E for z in Z) + \
            model.d1_pos - model.d1_neg == B

        # Meta 2: Inventario total acumulado
        model.restr_inventario_total = model.Imeta = \
            sum(a_e[e] * model.inv[e, 1, t, q] for t in T for e in E for q in Q) + \
            model.d2_pos - model.d2_neg == I_max

        # Meta 3: Minimizar días con transporte
        model.restr_dias_transporte = model.Tmeta = \
            sum(model.T[t] for t in T) + model.d3_pos - model.d3_neg == t_ideal

        # Función objetivo: minimizar desviaciones positivas
        model.objetivo = Objective(
            expr=w1 * model.d1_pos + w2 * model.d2_pos + w3 * model.d3_pos,
            sense=minimize)

    else:  # modo == 'simple'
        model.objetivo = Objective(
            expr=sum(model.xcomp[e, j, t] * model.cc[e, j] for e in E for j in J for t in T) +
                 sum(ct * model.T[t] for t in T) +
                 #sum(ct * model.compra_realizada[t] for t in T) +
                 sum(cp * model.xsem[e, z, q, t, l] for e in E for t in T for q in Q for l in L for z in Z) + 
                 penalizacion * sum(model.demanda_insatisfecha[e, z] for e in E for z in Z),
            sense=minimize)
