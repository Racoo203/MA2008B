# prueba/resolucion.py

import json
import pandas as pd
from pyomo.environ import SolverFactory, TerminationCondition

def resolver_modelo(model, solver_name="cbc", mostrar_resultado=True, tiempo_maximo=600):
    solver = SolverFactory(solver_name)
    solver.options['seconds'] = tiempo_maximo  # Límite de tiempo en segundos
    resultados = solver.solve(model, tee=mostrar_resultado)
    if resultados.solver.termination_condition == TerminationCondition.optimal:
        print("✅ Solución óptima encontrada.")
    elif resultados.solver.termination_condition == TerminationCondition.infeasible:
        print("❌ El modelo es infactible.")
    else:
        print(f"⚠️ Estado: {resultados.solver.termination_condition}")

    return resultados

def exportar_resultados_por_dia(model, E, J, T, Q, Z, L, exportar_json=True, exportar_excel=True):
    calendario = {}

    for t in T:
        dia = f"Día {t}"
        calendario[dia] = {
            "compras": [],
            "transportes": [],
            "inventario": [],
            "siembra": []
        }

        for e in E:
            for j in J:
                val = model.xcomp[e, j, t].value
                if val and val > 1e-3:
                    calendario[dia]["compras"].append({
                        "especie": e,
                        "vivero": j,
                        "cantidad": round(val, 2)
                    })

                val_transf = model.xtransf[e, j, t].value
                if val_transf and val_transf > 1e-3:
                    calendario[dia]["transportes"].append({
                        "especie": e,
                        "vivero": j,
                        "cantidad": round(val_transf, 2)
                    })
            
        for l in L:
            for z in Z:
                for e in E:
                    for q in Q:
                        val_entrega = model.xsem[e, z, q, t, l].value
                        if val_entrega and val_entrega > 1e-3:
                            calendario[dia].setdefault("entregas", []).append({
                                "viaje": l,
                                "poligono": z,
                                "especie": e,
                                "cantidad": round(val_entrega, 2)
                            })

            val_T = model.T[t].value
            calendario[dia]["transporte_activado"] = bool(round(val_T)) if val_T is not None else False

        for q in Q:
            for e in E:
                inv_val = model.inv[e, 1, t, q].value
                if inv_val and inv_val > 1e-3:
                    calendario[dia]["inventario"].append({
                        "especie": e,
                        "edad": q,
                        "cantidad": round(inv_val, 2)
                    })

        for q in Q:
            for z in Z:
                for e in E:
                    for l in L:
                        sem_val = model.xsem[e, z, q, t, l].value
                        if sem_val and sem_val > 1e-3:
                            calendario[dia]["siembra"].append({
                                "especie": e,
                                "edad": q,
                                "cantidad": round(sem_val, 2)
                            })

    if exportar_json:
        with open("resultados_calendario.json", "w", encoding="utf-8") as f:
            json.dump(calendario, f, indent=2)
        print("✅ Resultados guardados en resultados_calendario.json")

    if exportar_excel:
        filas = []
        for dia, datos in calendario.items():
            for tipo in ["compras", "inventario", "siembra"]:
                for entrada in datos[tipo]:
                    entrada.update({
                        "día": dia,
                        "tipo": tipo
                    })
                    filas.append(entrada)
        df = pd.DataFrame(filas)
        df.to_excel("resultados_calendario.xlsx", index=False)
        print("✅ Resultados guardados en resultados_calendario.xlsx")

def debug_infactibilidad(model, E, J, T, Q, Z, L):
    print("\n🔍 DEBUG DE INFACTIBILIDAD:")

    vacias = {
        "xcomp": 0,
        "xtransf": 0,
        "inv": 0,
        "xsem": 0,
        "dias_transporte": 0
    }

    for e in E:
        for j in J:
            for t in T:
                if model.xcomp[e, j, t].value and model.xcomp[e, j, t].value > 1e-3:
                    vacias["xcomp"] += 1
                if model.xtransf[e, j, t].value and model.xtransf[e, j, t].value > 1e-3:
                    vacias["xtransf"] += 1
        for t in T:
            for q in Q:
                for z in Z:
                    for l in L:
                        if model.inv[e, 1, t, q].value and model.inv[e, 1, t, q].value > 1e-3:
                            vacias["inv"] += 1
                        if model.xsem[e, z, q, t, l].value and model.xsem[e, z, q, t, l].value > 1e-3:
                            vacias["xsem"] += 1

    vacias["dias_transporte"] = sum(1 for t in T if model.T[t].value and model.T[t].value > 0.5)

    for k, v in vacias.items():
        estado = "✅ OK" if v > 0 else "❌ VACÍO"
        print(f" - {k:12}: {v} valores distintos de cero → {estado}")

    if vacias["xcomp"] == 0:
        print("🔴 No se está comprando nada. Verifica restricciones de presupuesto o disponibilidad.")
    if vacias["xsem"] == 0 and vacias["inv"] > 0:
        print("🟡 Hay inventario pero no se está sembrando. Verifica restricción de edades.")
    if vacias["dias_transporte"] == 0:
        print("🔴 No hubo ningún día con transporte. Posible conflicto con t_ideal o jornada laboral.")
