import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button


def calcular_tangentes_externas(c1, r1, c2, r2):
    """Calcula las tangentes externas entre dos círculos."""
    (x1, y1), (x2, y2) = c1, c2
    d = np.hypot(x2 - x1, y2 - y1)
    
    if d <= abs(r1 - r2):
        return []  # No hay tangentes externas si un círculo está dentro del otro
    
    vx, vy = (x2 - x1)/d, (y2 - y1)/d  # Vector unitario entre centros
    
    tangentes = []
    for sign in [1, -1]:
        cos_theta = (r1 - r2)/d
        sin_theta = np.sqrt(1 - cos_theta**2)
        
        # Vectores tangentes (rotación del vector entre centros)
        tx = vx * cos_theta - sign * vy * sin_theta
        ty = vy * cos_theta + sign * vx * sin_theta
        
        # Puntos de tangencia en ambos círculos
        p1 = (x1 + r1 * tx, y1 + r1 * ty)
        p2 = (x2 + r2 * tx, y2 + r2 * ty)
        tangentes.append((p1, p2))
    
    return tangentes

def interseccion_lineas(p1, p2, p3, p4):
    """Calcula la intersección de dos líneas definidas por p1-p2 y p3-p4."""
    A = np.array([
        [p2[0]-p1[0], p3[0]-p4[0]],
        [p2[1]-p1[1], p3[1]-p4[1]]
    ])
    b = np.array([p3[0]-p1[0], p3[1]-p1[1]])
    
    try:
        t, _ = np.linalg.solve(A, b)
        return (p1[0] + t*(p2[0]-p1[0]), p1[1] + t*(p2[1]-p1[1]))
    except np.linalg.LinAlgError:
        return None  # Líneas paralelas

class CirculoInteractivo:
    def __init__(self, ax, centro, radio, color):
        self.ax = ax
        self.centro = np.array(centro, dtype=float)
        self.radio = radio
        self.color = color
        self.circulo = Circle(centro, radio, fill=False, color=color, alpha=0.7, linewidth=2)
        self.ax.add_patch(self.circulo)
    
    def contiene_punto(self, punto):
        return np.hypot(*(self.centro - punto)) < self.radio
    
    def mover(self, nuevo_centro):
        self.centro[:] = nuevo_centro
        self.circulo.center = nuevo_centro

def actualizar_grafico():
    ax.clear()


    # Dibujar círculos
    for circ in circulos:
        ax.add_patch(Circle(circ.centro, circ.radio, fill=False, color=circ.color, alpha=0.7, linewidth=2))
    
    puntos_intersect = []
    pares = [(0, 1), (1, 2), (0, 2)]  # Combinaciones de pares de círculos
    
    # Primera pasada: Calcular todas las intersecciones
    for i, j in pares:
        c1, r1 = circulos[i].centro, circulos[i].radio
        c2, r2 = circulos[j].centro, circulos[j].radio
        
        tangentes = calcular_tangentes_externas(c1, r1, c2, r2)
        if len(tangentes) >= 2:
            p_inter = interseccion_lineas(*tangentes[0], *tangentes[1])
            if p_inter:
                puntos_intersect.append(p_inter)


        all_points = []
    for circ in circulos:
        all_points.append(circ.centro + [circ.radio, 0])  # Punto derecho
        all_points.append(circ.centro - [circ.radio, 0])  # Punto izquierdo
        all_points.append(circ.centro + [0, circ.radio])  # Punto superior
        all_points.append(circ.centro - [0, circ.radio])  # Punto inferior
    
    # Convertir a array y calcular min/max
    # Convertir a array y añadir puntos de intersección si existen
    all_points = np.array(all_points)

# Agregar puntos de intersección a los límites si existen
    if puntos_intersect:
        all_points = np.vstack([all_points, puntos_intersect])
        

    # Calcular límites dinámicos
    x_min, y_min = np.min(all_points, axis=0)
    x_max, y_max = np.max(all_points, axis=0)

    # Calcula el centro y el mayor tamaño
    x_centro = (x_min + x_max) / 2
    y_centro = (y_min + y_max) / 2
    half_width = (x_max - x_min) / 2
    half_height = (y_max - y_min) / 2
    half_size = max(half_width, half_height) + 2  # margen adicional

    # Establece límites simétricos para mantener el rectángulo proporcional
    ax.set_xlim(x_centro - half_size, x_centro + half_size)
    ax.set_ylim(y_centro - half_size, y_centro + half_size)
        
    
    # Segunda pasada: Dibujar tangentes extendidas hasta las intersecciones
    for idx, (i, j) in enumerate(pares):
        c1, r1 = circulos[i].centro, circulos[i].radio
        c2, r2 = circulos[j].centro, circulos[j].radio
        
        tangentes = calcular_tangentes_externas(c1, r1, c2, r2)
        if len(tangentes) >= 2 and idx < len(puntos_intersect):
            p_inter = puntos_intersect[idx]
            
            # Extender las tangentes hasta el punto de intersección
            for (p1, p2) in tangentes[:2]:  # Solo las dos primeras tangentes
                # Calcular dirección desde el punto de tangencia hacia la intersección
                dir_p1 = (p_inter[0] - p1[0], p_inter[1] - p1[1])
                dir_p2 = (p_inter[0] - p2[0], p_inter[1] - p2[1])
                
                # Puntos extendidos (un poco más allá de la intersección para mejor visualización)
                p1_ext = (p1[0] + dir_p1[0]*1.2, p1[1] + dir_p1[1]*1.2)
                p2_ext = (p2[0] + dir_p2[0]*1.2, p2[1] + dir_p2[1]*1.2)
                
                ax.plot([p1[0], p1_ext[0]], [p1[1], p1_ext[1]], '--', color='gray', alpha=0.7, linewidth=1.5)
                ax.plot([p2[0], p2_ext[0]], [p2[1], p2_ext[1]], '--', color='gray', alpha=0.7, linewidth=1.5)
                
                # Dibujar puntos de tangencia
                ax.plot(p1[0], p1[1], 'o', color=circulos[i].color, markersize=8, alpha=0.7)
                ax.plot(p2[0], p2[1], 'o', color=circulos[j].color, markersize=8, alpha=0.7)
    
    # Dibujar puntos de intersección y línea de Monge
    for idx, p_inter in enumerate(puntos_intersect):
        ax.plot(*p_inter, 'ro', markersize=10, markeredgecolor='black', label=f'Intersección {idx+1}')
    
    if len(puntos_intersect) == 3:
        x, y, z = puntos_intersect
        ax.plot([x[0], y[0], z[0]], [x[1], y[1], z[1]], 'b-', alpha=0.5, linewidth=2, label='Línea de Monge')
    
    ax.set_title("Tangentes Extendidas hasta Puntos de Intersección", fontsize=14)
    ax.legend(loc='upper right')
    plt.draw()

# Configuración inicial
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, aspect='equal', autoscale_on=True) 
circulos = [
        CirculoInteractivo(ax, (-4, 0), 0.5, 'blue'),
        CirculoInteractivo(ax, (4, 0), 2, 'green'),
        CirculoInteractivo(ax, (0, 5), 1, 'red')
]
circulo_seleccionado = None



# Variables para manejar desplazamiento
desplazamiento_inicial = None

def on_press(event):
    global circulo_seleccionado, desplazamiento_inicial
    if event.inaxes != ax:
        return
    for circ in circulos:
        if circ.contiene_punto((event.xdata, event.ydata)):
            circulo_seleccionado = circ
            desplazamiento_inicial = circ.centro - np.array([event.xdata, event.ydata])
            break

def on_motion(event):
    if circulo_seleccionado and event.inaxes == ax:
        factor_suavizado = 0.3  # Cambia entre 0.1 y 1 según lo que quieras
        nuevo_centro = circulo_seleccionado.centro + factor_suavizado * (np.array([event.xdata, event.ydata]) - circulo_seleccionado.centro)
        circulo_seleccionado.mover(nuevo_centro)
        actualizar_grafico()

def on_release(event):
    global circulo_seleccionado, desplazamiento_inicial
    circulo_seleccionado = None
    desplazamiento_inicial = None



fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event', on_motion)

# Botón para reiniciar
ax_reset = plt.axes([0.8, 0.01, 0.15, 0.05])
btn_reset = Button(ax_reset, 'Reiniciar')
def reset(event):
    global circulos
    # Limpiar el axis completamente
    ax.cla()

    # Recrear los círculos con sus posiciones y radios iniciales
    circulos = [
        CirculoInteractivo(ax, (-4, 0), 0.5, 'blue'),
        CirculoInteractivo(ax, (4, 0), 2, 'green'),
        CirculoInteractivo(ax, (0, 5), 1, 'red')
    ]
    
    # Redibujar todo
    actualizar_grafico()

btn_reset.on_clicked(reset)
plt.tight_layout()
actualizar_grafico()
plt.show()