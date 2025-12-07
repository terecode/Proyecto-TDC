import numpy as np
import matplotlib.pyplot as plt

# propiedades
k = 250
rho = 2500
cp = 900
alpha = k / (rho * cp)

#dimensiones
L = 0.0254
e = 0.004
vol = L ** 2* e

# parametros de tdc
h = 1900
Q_gen = 65/vol #w / m**3

T_inf = 20

#discretizacion espacial
Ny = 9
dy = e / (Ny-1)

Nx = 51
dx = L / (Nx-1)


def is_not_SS(T_anterior, T_actual, tol = 1e-3):
    return np.max(np.abs(T_actual - T_anterior)) >= tol

T = np.ones((Ny, Nx)) * T_inf
T_anterior = np.zeros((Ny, Nx))

n = 0
while is_not_SS(T_anterior, T, 1e-3):
    T_anterior = T.copy()
    for i in range(Nx):
        for j in range(Ny):
            #Esquina superior izquierda
            if i == 0 and j == 0:
                T[j,i] = (dx**2 * dy**2 * Q_gen + dx**2 * k * T[j+1, i] +  dy**2 * k * T[j, i+1] + dx**2 * dy * h * T_inf + dx * dy**2 * h * T_inf) / (dx**2 * k + dy**2 * k + dx**2 * dy * h + dx * dy**2 * h)
            # Esquina superior derecha
            elif i == Nx-1 and j == 0:
                T[j,i] = (dx**2 * dy**2 * Q_gen + dx**2 * k * T[j+1, i] +  dy**2 * k * T[j, i-1] + dx**2 * dy * h * T_inf + dx * dy**2 * h * T_inf) / (dx**2 * k + dy**2 * k + dx**2 * dy * h + dx * dy**2 * h)
            # Esquina inferior izquierda
            elif i == 0 and j == Ny-1:
                T[j,i] = (dx**2 * dy**2 * Q_gen + 2 * dx**2 * k * T[j-1, i] + dy**2 * k * T[j, i+1] + dx * dy**2 * h * T_inf) / (dx * dy**2 * h + 2 * dx**2 * k + dy**2 * k)
            # Esquina inferior derecha
            elif i == Nx-1 and j == Ny-1:
                T[j,i] = (dx**2 * dy**2 * Q_gen + 2 * dx**2 * k * T[j-1, i] + dy**2 * k * T[j, i-1] + dx * dy**2 * h * T_inf) / (dx * dy**2 * h + 2 * dx**2 * k + dy**2 * k)           
            
            # Pared izquierda
            elif i == 0:
                T[j,i] = (dx**2 * dy**2 * Q_gen 
                        + dx**2 * k * T[j+1,i] 
                        + dx**2 * k * T[j-1,i] 
                        + dy**2 * k * (T[j,i] - dx * h * (T[j,i] - T_inf) / k)
                        ) / (2*dx**2 * k + dy**2 * k)
            # Pared derecha
            elif i == Nx-1:
                T[j,i] = (dx**2 * dy**2 * Q_gen 
                        + dx**2 * k * T[j+1,i]
                        + dx**2 * k * T[j-1,i]
                        + dy**2 * k * (T[j,i] - dx * h * (T[j,i] - T_inf) / k)
                        ) / (2*dx**2 * k + dy**2 * k)
                
            # Superficie superior
            elif j == 0:
                T[j,i] = (dx**2 * dy**2 * Q_gen
                        + dx**2 * k * T[j+1,i]
                        + dy**2 * k * T[j,i+1]
                        + dy**2 * k * T[j,i-1]
                        + dx**2 * k * (T[j,i] - dy * h * (T[j,i] - T_inf) / k)
                        ) / (dx**2 * k + 2*dy**2 * k)
                          
            # Superficie inferior
            elif j == Ny-1:
                T[j,i] = (dx**2 * dy**2 * Q_gen
                        + dy**2 * k * T[j, i+1]
                        + dy**2 * k * T[j, i-1]
                        + dx**2 * k * T[j-1, i]   # ← Este es el reemplazo adiabático T[j+1] = T[j-1]
                        ) / (dx**2*k + 2*dy**2*k)
            # Interior
            else:
                T[j,i] = (dx**2 * dy**2 * Q_gen
                        + dx**2 * k * T[j+1,i]
                        + dx**2 * k * T[j-1,i]
                        + dy**2 * k * T[j,i+1]
                        + dy**2 * k * T[j,i-1]
                        ) / (2*dx**2 * k + 2*dy**2 * k)
    n+=1
    if n % 10 == 0:
        print(f"Iteracion {n}: max T = {np.max(T):.2f} °C, min T = {np.min(T):.2f} °C")

print(f"Convergio en {n} iteraciones")

# Mapa de calor de la última distribución de temperatura
plt.figure(figsize=(18, 10))
plt.imshow(T, aspect='auto', extent=[0, L*100, 0, e*1000], origin='upper', cmap='jet')
plt.colorbar(label='Temperatura [°C]')
plt.ylabel('Alto [mm]', fontsize=12)
plt.xlabel('Ancho [cm]', fontsize=12)
plt.title('Distribución de temperatura en el procesador - Estado estacionario', fontsize=13)
plt.show()