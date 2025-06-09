# prueba/variables.py

from pyomo.environ import Var, NonNegativeReals, Binary, Integers, NonNegativeIntegers

def definir_variables(model, E, J, T, Q, Z, L, N):
    # Fase 1: Compra y acopio
    model.xcomp = Var(E, J, T, domain = NonNegativeReals)            # Compra de especie e al proveedor j en día t
    model.xtransf = Var(E, J, T, domain=NonNegativeReals)          # Transporte de especie e del proveedor j al almacén en t
    model.inv = Var(E, [1], T, Q, domain=NonNegativeReals)         # Inventario en receptor (nodo 1)

    # Fase 2: Ruteo y entrega
    model.Y = Var(N, N, L, T,  domain=Binary)                          # Enlace de ruta de i a j en viaje l
    model.u = Var(Z, L, T, domain=Integers)                           # Orden de visita para subtour elimination

    # Demanda insatisfecha
    model.demanda_insatisfecha = Var(E, Z, domain=NonNegativeReals)      # Demanda no cubierta de especie e en polígono z

    # Fase 2: Siembra desde el polígono receptor
    model.xsem = Var(E, Z, Q, T, L, domain=NonNegativeReals)        # Cantidad sembrada de especie e desde almacén en t con edad q

    # Variables auxiliares
    model.T = Var(T, domain=Binary)                                # 1 si hubo transporte el día t
    model.R = Var(E, T, domain=Binary)                              # 1 si se activa punto de reorden para especie e en día t

    # Variables de desviación para programación por metas
    model.d1_pos = Var(domain=NonNegativeReals)
    model.d1_neg = Var(domain=NonNegativeReals)
    model.d2_pos = Var(domain=NonNegativeReals)
    model.d2_neg = Var(domain=NonNegativeReals)
    model.d3_pos = Var(domain=NonNegativeReals)
    model.d3_neg = Var(domain=NonNegativeReals)

    # Variable Binaria par saber si se hizo una compra en el día t
    model.compra_realizada = Var(T,J, domain=Binary)  # 1 si se realizó compra en el día t
