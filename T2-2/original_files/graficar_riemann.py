import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

def graficar_riemann(x_range, y_range, nx, ny, f, mask=None, color='steelblue', alpha=0.8):
    """
    Grafica paralelepípedos para suma de Riemann.

    Parámetros:
    -----------
    x_range : tuple (xmin, xmax)
    y_range : tuple (ymin, ymax)
    nx, ny : int - número de particiones en x e y
    f : función f(x, y) - altura
    mask : función mask(x, y) - retorna True si (x,y) está en la región (opcional)
    """
    xmin, xmax = x_range
    ymin, ymax = y_range

    # Crear particiones usando linspace (vértices de los rectángulos)
    x_vertices = np.linspace(xmin, xmax, nx + 1)
    y_vertices = np.linspace(ymin, ymax, ny + 1)

    # Calcular dx, dy
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    suma = 0
    for i in range(nx):
        for j in range(ny):
            # Vértices del rectángulo en la base
            x0, x1 = x_vertices[i], x_vertices[i + 1]
            y0, y1 = y_vertices[j], y_vertices[j + 1]

            # Centro del rectángulo
            X = (x0 + x1) / 2
            Y = (y0 + y1) / 2

            # Verificar máscara usando el centro
            if mask is not None and not mask(X, Y):
                continue

            # Altura evaluada en el centro
            z = f(X, Y)
            if z == 0:
                continue

            suma += z * dx * dy

            v = np.array([
                [x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0],
                [x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z],
            ])

            caras = [
                [v[0], v[1], v[5], v[4]], [v[2], v[3], v[7], v[6]],
                [v[0], v[3], v[7], v[4]], [v[1], v[2], v[6], v[5]],
                [v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]],
            ]

            ax.add_collection3d(Poly3DCollection(caras, alpha=alpha,
                                                 facecolors=color,
                                                 edgecolors='black',
                                                 linewidths=0.2,
                                                 zsort='average'))

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_zlabel('z', fontsize=12)
    ax.set_title(f'Suma de Riemann: {suma:.4f}', fontsize=14)

    # Vista desde el origen hacia valores positivos
    ax.view_init(elev=35.26, azim=225)
    ax.dist = 20  # Aumentar distancia de la cámara (default es 20)

    plt.tight_layout()
    return fig, ax, suma


# ========== EJEMPLO 0: FUNCIÓN LINEAL EN REGIÓN TRAPEZOIDAL ==========

def ejemplo_trapecio(nx=20, ny=20):
    """
    Ejemplo 0: Función lineal f(x,y) = x + y
    Región de integración: Trapecio con vértices en (0,0), (2,0), (1.5,1), (0.5,1)
    """
    # Función lineal
    def f(x, y):
        return x + y

    # Máscara para región trapezoidal
    # Trapecio invertido: base inferior (pequeña) de (0.5,0) a (1.5,0), base superior (grande) de (0,1) a (2,1)
    def mask(x, y):
        # Verificar que esté dentro del rango vertical
        if y < 0 or y > 1:
            return False

        # Borde izquierdo: línea de (0,1) a (0.5,0)
        # Ecuación: x = 0.5 - 0.5*y
        borde_izq = 0.5 - 0.5 * y

        # Borde derecho: línea de (2,1) a (1.5,0)
        # Ecuación: x = 1.5 + 0.5*y
        borde_der = 1.5 + 0.5 * y

        return borde_izq <= x <= borde_der

    fig, ax, suma = graficar_riemann((0, 2), (0, 1), nx, ny, f, mask, color='steelblue')
    print(f"Trapecio - Suma de Riemann: {suma:.4f}")
    plt.savefig('outputs/riemann_trapecio.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Crear carpeta outputs si no existe
    os.makedirs('outputs', exist_ok=True)

    ejemplo_trapecio()
