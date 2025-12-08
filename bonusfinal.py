import numpy as np
import matplotlib.pyplot as plt

# Parametros inciso a)
T_amb_C = 20.0
T_amb = T_amb_C + 273.15
T_cielo = 230.0
T_suelo = T_amb

sigma = 5.67e-8
G_solar = 800.0

# Dimensiones inciso a)
L_vidrio = 0.0021   
L_encapsulante = 0.0003  
L_celda  = 0.0003   
L_panel_vert = 2.416
L_panel_horiz = 1.09 

theta_grados = 45.0
theta_rad = np.deg2rad(theta_grados)

k_vidrio = 1.0
k_encapsulante = 0.4
k_celda  = 148.0

eps_vidrio = 0.85
rho_vidrio = 0.05
q_solar_entrada = (1.0 - rho_vidrio) * G_solar 

F_cielo_sup = (1.0 + np.cos(theta_rad)) / 2.0
F_suelo_sup = 1.0 - F_cielo_sup
F_suelo_inf = 1.0

# Funciones
def propiedades_aire(T_celsius):
    T = T_celsius
    tk = T + 273.15
    rho = 352.977/tk
    cp = 1003.7 - 0.032*T + 0.00035*T**2
    k = 0.0241 + 0.000076*T
    mu = (1.74 + 0.0049*T - 3.5e-6*T**2)*1e-5
    nu = mu/rho
    pr = nu/(k/(rho*cp))
    beta = 1/tk
    return k, nu, pr, beta

def calcular_h(T_sup, T_inf):
    # Superficie Superior
    T_pelicula_sup = (T_sup + T_amb) / 2
    ka, nu, pr, beta = propiedades_aire(T_pelicula_sup - 273.15)
    
    # Forzada
    Re_L = 1.0 * L_panel_vert / nu 
    if Re_L < 5e5:
        Nu_f = 0.664 * (Re_L**0.5) * (pr**(1/3))
    else:
        Nu_f = (0.037 * (Re_L**0.8) - 871) * (pr**(1/3))
    h_f = Nu_f * ka / L_panel_vert
    
    # Natural
    L_nat = (L_panel_vert * L_panel_horiz) / (2*(L_panel_vert + L_panel_horiz))
    g_eff = 9.81 * np.cos(theta_rad)
    Ra_nat = (g_eff * beta * abs(T_sup - T_amb) * (L_nat**3) * pr) / (nu**2)
    
    if Ra_nat < 1e7: Nu_n = 0.54 * (Ra_nat**(1/4))
    else:            Nu_n = 0.15 * (Ra_nat**(1/3))
    h_n_sup = Nu_n * ka / L_nat
    
    # Combinada
    n_exp = 3 + np.cos(theta_rad)
    h_conv_sup = (h_n_sup**n_exp + h_f**n_exp)**(1/n_exp)
    
    # Superficie Inferior
    T_pelicula_inf = (T_inf + T_amb) / 2
    kb, nub, prb, betab = propiedades_aire(T_pelicula_inf - 273.15)
    Ra_L_bot = (9.81 * np.cos(theta_rad) * betab * abs(T_inf - T_amb) * L_panel_vert**3 * prb) / (nub**2)
    
    num_cc = 0.387 * (Ra_L_bot**(1/6))
    den_cc = (1 + (0.492 / prb)**(9/16))**(8/27)
    Nu_churchill = (0.825 + num_cc / den_cc)**2
    h_conv_inf = Nu_churchill * kb / L_panel_vert
    
    return h_conv_sup, h_conv_inf

# Iteraciones

# Resistencias
R_vidrio = L_vidrio / k_vidrio
R_encapsulante = L_encapsulante / k_encapsulante
R_celda  = L_celda  / k_celda
R_cond_total = 2*R_vidrio + 2*R_encapsulante + R_celda

T_sup = 40.0 + 273.15
T_inf = 40.0 + 273.15
error = 1.0
tol = 1e-6
cont_iter = 0

print("Iniciando cálculo analítico iterativo...")

while error > tol and cont_iter < 1000:
    cont_iter += 1
    T_sup_ant = T_sup
    T_inf_ant = T_inf
    
    h_conv_sup, h_conv_inf = calcular_h(T_sup, T_inf)
    
    h_rad_sup = eps_vidrio * sigma * (
        F_cielo_sup * (T_sup**2 + T_cielo**2)*(T_sup + T_cielo) +
        F_suelo_sup * (T_sup**2 + T_suelo**2)*(T_sup + T_suelo)
    )
    
    h_rad_inf = eps_vidrio * sigma * F_suelo_inf * (T_inf**2 + T_suelo**2)*(T_inf + T_suelo)
    
    h_total_sup = h_conv_sup + h_rad_sup
    h_total_inf = h_conv_inf + h_rad_inf
    
    R_sup_total = 1.0 / h_total_sup
    R_inf_total = 1.0 / h_total_inf
    
    ganancia_rad_sup = eps_vidrio * sigma * (F_cielo_sup*T_cielo**4 + F_suelo_sup*T_suelo**4)
    ganancia_rad_inf = eps_vidrio * sigma * (F_suelo_inf*T_suelo**4)
    
    k_rad_sup = 4 * eps_vidrio * sigma * T_sup_ant**3
    k_rad_inf = 4 * eps_vidrio * sigma * T_inf_ant**3
    
    Matriz_A = np.zeros((2,2))
    Vector_b = np.zeros(2)
    
    Matriz_A[0,0] = h_conv_sup + k_rad_sup + (1.0/R_cond_total)
    Matriz_A[0,1] = -(1.0/R_cond_total)
    Vector_b[0]   = q_solar_entrada + h_conv_sup*T_amb + ganancia_rad_sup

    Matriz_A[1,0] = -(1.0/R_cond_total)
    Matriz_A[1,1] = h_conv_inf + k_rad_inf + (1.0/R_cond_total)
    Vector_b[1]   = h_conv_inf*T_amb + ganancia_rad_inf
    
    temps = np.linalg.solve(Matriz_A, Vector_b)
    T_sup = temps[0]
    T_inf = temps[1]
    
    error = max(abs(T_sup - T_sup_ant), abs(T_inf - T_inf_ant))

print(f"Convergió en {cont_iter} iteraciones.")
print(f"T_sup: {T_sup-273.15:.2f} °C")
print(f"T_inf: {T_inf-273.15:.2f} °C")

flujo_q = (T_sup - T_inf) / R_cond_total

x_interfaz = [0]
t_interfaz = [T_sup]

# Vidrio Superior
x_interfaz.append(L_vidrio)
t_interfaz.append(t_interfaz[-1] - flujo_q * R_vidrio)
# Encapsulante Superior
x_interfaz.append(x_interfaz[-1] + L_encapsulante)
t_interfaz.append(t_interfaz[-1] - flujo_q * R_encapsulante)
# Celda
x_interfaz.append(x_interfaz[-1] + L_celda)
t_interfaz.append(t_interfaz[-1] - flujo_q * R_celda)
# Encapsulante Inferior
x_interfaz.append(x_interfaz[-1] + L_encapsulante)
t_interfaz.append(t_interfaz[-1] - flujo_q * R_encapsulante)
# Vidrio Inferior
x_interfaz.append(x_interfaz[-1] + L_vidrio)
t_interfaz.append(t_interfaz[-1] - flujo_q * R_vidrio)

X_grafico = np.array(x_interfaz) * 1000 
T_grafico = np.array(t_interfaz) - 273.15 

# Grafico
plt.figure(figsize=(8,5))
plt.plot(X_grafico, T_grafico, 'o-', label='Solución Analítica 1D')

plt.axvspan(0, L_vidrio*1000, alpha=0.1, color='blue', label='Vidrio')
plt.axvspan(L_vidrio*1000, (L_vidrio+L_encapsulante)*1000, alpha=0.1, color='orange', label='Encap')
plt.axvspan((L_vidrio+L_encapsulante)*1000, (L_vidrio+L_encapsulante+L_celda)*1000, alpha=0.2, color='gray', label='Celda')
plt.axvspan((L_vidrio+L_encapsulante+L_celda)*1000, (L_vidrio+2*L_encapsulante+L_celda)*1000, alpha=0.1, color='orange')
plt.axvspan((L_vidrio+2*L_encapsulante+L_celda)*1000, (2*L_vidrio+2*L_encapsulante+L_celda)*1000, alpha=0.1, color='blue')

plt.xlabel('Profundidad (mm)')
plt.ylabel('Temperatura (°C)')
plt.title('Perfil Analítico de Temperatura (Bonus Track)')
plt.grid(True)
plt.legend()
plt.show()