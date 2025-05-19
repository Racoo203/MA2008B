import sympy as sp

# ==========================
# ENTRADA DE USUARIO
# ==========================
print("=== BUSQUEDA DORADA PARA FUNCION RACIONAL ===")
x = sp.Symbol('x')
expr_str = input("Introduce la expresión de la funcion (en x): ")

try:
    func_expr = sp.sympify(expr_str)
    funcion = sp.lambdify(x, func_expr, modules='numpy')
except Exception as e:
    print("Error al procesar los polinomios:", e)
    exit()

# ==========================
# PARÁMETROS INICIALES
# ==========================
r = 0.618
x1 = -5
x2 = 5
iteraciones = 25

# ==========================
# FUNCIÓN DE BÚSQUEDA DORADA
# ==========================
def busqueda_dorada(x1, x2):
    x3 = x2 - r * (x2 - x1)
    x4 = x1 + r * (x2 - x1)
    fx3 = funcion(x3)
    fx4 = funcion(x4)

    print(f"{x1:>8.4f} | {x2:>8.4f} | {x3:>8.4f} | {x4:>8.4f} | {fx3:>10.4f} | {fx4:>10.4f}")

    if fx3 < fx4:
        return x1, x4
    else:
        return x3, x2

# ==========================
# ITERACIÓN
# ==========================
print("\nITERACION DE BUSQUEDA DORADA:")
print("   x1     |    x2     |    x3     |    x4     |   f(x3)   |   f(x4)   ")
print("-" * 70)

for i in range(iteraciones):
    x1, x2 = busqueda_dorada(x1, x2)

# ==========================
# RESULTADO FINAL
# ==========================
x_opt = (x1 + x2) / 2
f_opt = funcion(x_opt)

print("\n=== RESULTADO FINAL ===")
print(f"Intervalo optimo encontrado: {x1:.6f} a {x2:.6f}")
print(f"x minimo estimado : {x_opt:.6f}")
print(f"f(x minimo)       : {f_opt:.6f}")
print(f"Expresion simbolica: f(x) = {func_expr}")