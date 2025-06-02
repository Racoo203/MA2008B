# prueba/funcion_objetivo.py

from pyomo.environ import Objective, minimize

def definir_funcion_objetivo_simple(model, E, J, T, Q, L, Z, cp, ct, pen):
    """
    Función objetivo simple: Minimiza el costo total de compra, transporte y siembra.
    """
    model.objetivo = Objective(expr=
        sum(model.xcomp[e, j, t] * model.cc[e, j] for e in E for j in J for t in T) +
        sum(ct * model.T[t] for t in T) +
        sum(cp * model.xsem[e,z,l,t,q] for e in E for t in T for q in Q for l in L for z in Z) +
        sum(model.demandaNoSatisfecha[z, e]*pen for z in Z for e in E),
        sense=minimize
    )


def definir_funcion_objetivo_metas(model, E, J, T, Q, a_e, cp, ct, w1=1, w2=1, w3=1, B=100000, I_max=100000, t_ideal=10):
    """
    Función objetivo por programación por metas: balancea presupuesto, inventario y días de transporte.
    """
    model.restr_presupuesto = model.Bmeta = \
        sum(model.xcomp[e, j, t] * model.cc[e, j] for e in E for j in J for t in T) + \
        sum(ct * model.T[t] for t in T) + \
        sum(cp * model.xsem[e, 1, t, q] for e in E for t in T for q in Q) + \
        model.d1_pos - model.d1_neg == B

    model.restr_inventario_total = model.Imeta = \
        sum(a_e[e] * model.inv[e, 1, t, q] for t in T for e in E for q in Q) + \
        model.d2_pos - model.d2_neg == I_max

    model.restr_dias_transporte = model.Tmeta = \
        sum(model.T[t] for t in T) + model.d3_pos - model.d3_neg == t_ideal

    model.objetivo = Objective(expr=
        w1 * model.d1_pos +
        w2 * model.d2_pos +
        w3 * model.d3_pos,
        sense=minimize
    )


def definir_funcion_objetivo(model, E, J, T, Q, L, Z, a_e, cp, ct, pen, tipo="simple", **kwargs):
    """
    Selección del tipo de función objetivo:
    - tipo="simple": solo costos.
    - tipo="metas": programación por metas con desviaciones.
    """
    if tipo == "metas":
        definir_funcion_objetivo_metas(model, E, J, T, Q, a_e, cp, ct, **kwargs)
    else:
        definir_funcion_objetivo_simple(model, E, J, T, Q, L, Z, cp, ct, pen)
