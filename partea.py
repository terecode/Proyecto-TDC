import numpy as np
import matplotlib.pyplot as plt
#otro cambio
# 1. DATOS DEL PROBLEMA

# Temperaturas
T_amb_C = 20.0
T_amb_K = T_amb_C + 273.15
T_sky_K = 230
T_ground_K = T_amb_K

sigma = 5.67e-8 # Cte stefan boltzman
G_solar = 800.0  # W/m2, # flujo solar incidente en el vidrio superior

L_glass = 2e-3          # ESPESOR: 2 mm (vidrio sup e inf)
L_encap = 350e-6        # ESPESOR: 350 μm (encapsulantes sup/inf)
L_cell  = 300e-6        # ESPESOR: 300 μm (celda)
L_panel = 92e-3         # LARGO
W_panel = 92e-3         # ANCHO MALL

theta_deg = 45.0
theta_rad = np.deg2rad(theta_deg)

# Propiedades del vidrio (Tabla 2)
k_glass = 1.0          # W/m·K
eps_glass = 0.85
rho_glass = 0.05

# Propiedades del encapsulante
k_encap = 0.4          # W/m·K
eps_encap = 0.90

# Propiedades de la celda
k_cell = 148.0         # W/m·K
eps_cell = 0.90   # (realmente no la usamos en a), pero la dejo)


# 2. FACTORES DE VISTA Y RADIACIÓN
F_sky_top = (1.0 + np.cos(theta_rad)) / 2.0     #  Sup ↔ cielo:   F = (1 + cos θ) / 2
F_ground_top = 1.0 - F_sky_top                  #  Sup ↔ suelo:   F = 1 - (1 + cos θ) / 2
F_sky_bottom = 0.0                              #  Inf ↔ cielo:   F = 0
F_ground_bottom = 1.0                           #  Inf ↔ suelo:   F = 1


# Radiación linealizada:

#  Para la cara superior hay dos entornos: cielo y suelo.
h_rad_top = 4.0 * eps_glass * sigma * (F_sky_top * T_sky_K**3 + F_ground_top * T_ground_K**3)

# Cara inferior solo ve al suelo
h_rad_bot = 4.0 * eps_glass * sigma * (F_ground_bottom * T_ground_K**3)


# CALCULO DE NU

# 3. COEFICIENTES DE CONVECCIÓN (por ahora constantes)

def h_forzado_superior():
    # Aquí luego puedes meter correlación de placa inclinada
    return 15.0  # W/m2K

def h_natural_inferior():
    # Aquí luego correlación de convección natural
    return 5.0   # W/m2K

h_top_conv = h_forzado_superior()
h_bot_conv = h_natural_inferior()

h_top_total = h_top_conv + h_rad_top
h_bot_total = h_bot_conv + h_rad_bot


# 4. ENTRADA DE CALOR SOLAR
q_solar_in = (1.0 - rho_glass) * G_solar  # W/m2


# 5. RESISTENCIAS TÉRMICAS EN EL ESPESOR (5 CAPAS)

R_glass = L_glass / k_glass      # vidrio sup o inf
R_encap = L_encap / k_encap      # encapsulante sup o inf
R_cell  = L_cell  / k_cell

# Vidrio sup + encap sup + celda + encap inf + vidrio inf
R_cond_total = 2*R_glass + 2*R_encap + R_cell


# 6. SISTEMA 2x2 PARA Tsup Y Tinf (linealizado)
# Balance cara superior: q_solar_in = h_top_total (Tsup - T_amb) + (Tsup - Tinf)/R_cond_total
# Balance cara inferior: (Tsup - Tinf)/R_cond_total = h_bot_total (Tinf - T_amb)

A = np.zeros((2, 2))
b = np.zeros(2)

A[0, 0] = h_top_total + 1.0 / R_cond_total
A[0, 1] = -1.0 / R_cond_total
b[0] = q_solar_in + h_top_total * T_amb_K

A[1, 0] = 1.0 / R_cond_total
A[1, 1] = -1.0 / R_cond_total - h_bot_total
b[1] = -h_bot_total * T_amb_K

Tsup_K, Tinf_K = np.linalg.solve(A, b)

Tsup_C = Tsup_K - 273.15
Tinf_C = Tinf_K - 273.15

print(f"T_superficie superior = {Tsup_C:.2f} °C")
print(f"T_superficie inferior = {Tinf_C:.2f} °C")

q_cond = (Tsup_K - Tinf_K) / R_cond_total
print(f"Flujo conductivo a través del panel = {q_cond:.2f} W/m2")


# 7. DISTRIBUCIÓN DE TEMPERATURA EN LAS 5 CAPAS

N_glass1  = 10   # vidrio superior
N_encap1  = 10   # encapsulante superior
N_cell    = 10   # celda
N_encap2  = 10   # encapsulante inferior
N_glass2  = 10   # vidrio inferior

x_glass1 = np.linspace(0, L_glass, N_glass1, endpoint=False)
x_encap1 = np.linspace(L_glass, L_glass+L_encap, N_encap1, endpoint=False)
x_cell   = np.linspace(L_glass+L_encap, L_glass+L_encap+L_cell, N_cell, endpoint=False)
x_encap2 = np.linspace(L_glass+L_encap+L_cell, L_glass+2*L_encap+L_cell, N_encap2, endpoint=False)
x_glass2 = np.linspace(L_glass+2*L_encap+L_cell, 2*L_glass+2*L_encap+L_cell, N_glass2+1)  # +1 incluye cara inf

# Vidrio superior
T_glass1_K = Tsup_K - q_cond/k_glass * x_glass1

# Encapsulante superior
T_int_g1_e1_K = Tsup_K - q_cond/k_glass * L_glass
x_rel_encap1 = x_encap1 - L_glass
T_encap1_K = T_int_g1_e1_K - q_cond/k_encap * x_rel_encap1

# Celda
T_int_e1_cell_K = T_int_g1_e1_K - q_cond/k_encap * L_encap
x_rel_cell = x_cell - (L_glass + L_encap)
T_cell_K = T_int_e1_cell_K - q_cond/k_cell * x_rel_cell

# Encapsulante inferior
T_int_cell_e2_K = T_int_e1_cell_K - q_cond/k_cell * L_cell
x_rel_encap2 = x_encap2 - (L_glass + L_encap + L_cell)
T_encap2_K = T_int_cell_e2_K - q_cond/k_encap * x_rel_encap2

# Vidrio inferior
T_int_e2_g2_K = T_int_cell_e2_K - q_cond/k_encap * L_encap
x_rel_glass2 = x_glass2 - (L_glass + 2*L_encap + L_cell)
T_glass2_K = T_int_e2_g2_K - q_cond/k_glass * x_rel_glass2

print(f"Último nodo vidrio inferior = {T_glass2_K[-1]-273.15:.2f} °C (≈ Tinf)")

x_total = np.concatenate([x_glass1, x_encap1, x_cell, x_encap2, x_glass2])
T_total_C = np.concatenate([
    T_glass1_K - 273.15,
    T_encap1_K - 273.15,
    T_cell_K   - 273.15,
    T_encap2_K - 273.15,
    T_glass2_K - 273.15])


# 8. GRÁFICO

plt.figure(figsize=(8,4))
plt.plot(x_total*1000, T_total_C, marker='o')

x0 = 0.0
x1 = L_glass*1000
x2 = (L_glass+L_encap)*1000
x3 = (L_glass+L_encap+L_cell)*1000
x4 = (L_glass+2*L_encap+L_cell)*1000
x5 = (2*L_glass+2*L_encap+L_cell)*1000

plt.axvspan(x0, x1, alpha=0.15, label='Vidrio sup')
plt.axvspan(x1, x2, alpha=0.15, color='orange', label='Encap sup')
plt.axvspan(x2, x3, alpha=0.15, color='green', label='Celda')
plt.axvspan(x3, x4, alpha=0.15, color='orange', label='Encap inf')
plt.axvspan(x4, x5, alpha=0.15, label='Vidrio inf')

plt.xlabel('x (mm)')
plt.ylabel('Temperatura (°C)')
plt.title('Distribución de temperatura en el espesor (5 capas, NOCT, sin generación interna)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

T_celda_media_C = np.mean(T_cell_K - 273.15)
print(f"Temperatura media de la celda ≈ {T_celda_media_C:.2f} °C")