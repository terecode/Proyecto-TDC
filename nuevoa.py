import numpy as np
import matplotlib.pyplot as plt

# --- 1. PARÁMETROS Y CONSTANTES ---

# Balance Óptico Automático (Condiciones NOCT: eta=0)
G_solar = 800.0  # W/m2

# Propiedades Ópticas (Tabla 2)
# Vidrio 
rho_vidrio = 0.05
alpha_vidrio = 0.01
tau_vidrio = 1.0 - rho_vidrio - alpha_vidrio

# Encapsulante 
rho_enc = 0.01
tau_enc = 0.98
alpha_enc = 1.0 - rho_enc - tau_enc

# Celda 
rho_celda = 0.05
tau_celda = 0.01
alpha_celda = 1.0 - rho_celda - tau_celda

# Cálculo de Flujos de Calor Absorbido (q'')
q_abs_vidrio = G_solar * alpha_vidrio
G_transmitida_vidrio = G_solar * tau_vidrio

q_abs_enc = G_transmitida_vidrio * alpha_enc
G_transmitida_enc = G_transmitida_vidrio * tau_enc

q_abs_celda = G_transmitida_enc * alpha_celda

# --- Dimensiones del Espesor (capas) ---
esp_vidrio_sup = 0.0021  # 2.1 mm
esp_enc_sup    = 0.0003  # 300 um
esp_celda      = 0.0003  # 300 um
esp_enc_inf    = 0.0003  # 300 um
esp_vidrio_inf = 0.0021  # 2.1 mm

Espesor_total = esp_vidrio_sup + esp_enc_sup + esp_celda + esp_enc_inf + esp_vidrio_inf

# Flujos de Calor Volumétrico (q_dot = q'' / espesor de la capa)
flujo_q_vidrio_dot = q_abs_vidrio / esp_vidrio_sup
flujo_q_enc_dot    = q_abs_enc    / esp_enc_sup
flujo_q_celda_dot  = q_abs_celda  / esp_celda

print(f"--- Balance Óptico Calculado ---")
print(f"Calor abs. Vidrio Volumétrico: {flujo_q_vidrio_dot:.2f} W/m3")
print(f"Calor abs. Encap Volumétrico:  {flujo_q_enc_dot:.2f} W/m3")
print(f"Calor abs. Celda Volumétrico:  {flujo_q_celda_dot:.2f} W/m3")
print(f"--------------------------------")

# Geometría en X
Ancho_celda = 0.090  # 90 mm
Ancho_borde = 0.001  # 1 mm
Ancho_total = Ancho_celda + 2*Ancho_borde  # 92 mm

# Materiales y Ambiente
k_vidrio = 1.0 
k_enc    = 0.4
k_celda  = 148.0
T_amb    = 20.0 + 273.15
T_cielo  = 230.0
T_suelo  = 20.0 + 273.15
sigma    = 5.67e-8
emisividad = 0.85

# Parámetros para h
L_caracteristico = 2.416 
L_horiz = 1.09
Angulo = 45 * np.pi / 180
V_viento = 1.0 
F_cielo = (1 + np.cos(Angulo)) / 2
F_suelo_sup = 1 - F_cielo
F_suelo_inf = 1.0

# --- 2. GENERACIÓN DE MALLA (GRID) - DIFERENCIAS FINITAS ---

# En X mantenemos tu malla no uniforme
nx_borde, nx_celda = 3, 30
nx = 2*nx_borde + nx_celda 

# Coordenadas X (caras y centros de nodo)
caras_x = np.concatenate([
    np.linspace(0, Ancho_borde, nx_borde+1)[:-1],
    np.linspace(Ancho_borde, Ancho_borde+Ancho_celda, nx_celda+1)[:-1],
    np.linspace(Ancho_borde+Ancho_celda, Ancho_total, nx_borde+1)
])
dx = np.diff(caras_x)
X_nodo = (caras_x[:-1] + caras_x[1:]) / 2  # centros

# En Z imponemos nodos cada 300 µm
dz_const = 300e-6  # 300 micrómetros

n_vidrio = int(round(esp_vidrio_sup / dz_const))  # 7
n_enc    = int(round(esp_enc_sup    / dz_const))  # 1
n_celda  = int(round(esp_celda      / dz_const))  # 1

nz = 2*n_vidrio + 2*n_enc + n_celda  # 7 + 1 + 1 + 1 + 7 = 17

# Verificar que cierre con el espesor total
assert abs(nz*dz_const - Espesor_total) < 1e-9, "Chequea espesores y dz_const"

# Coordenadas de nodos en Z (centros)
nodos_z = (np.arange(nz) + 0.5) * dz_const
dz = np.full(nz, dz_const)

# Índices de capas en Z
idx_vid_sup_end = n_vidrio                    # 0 .. 6  -> vidrio superior
idx_enc_sup_end = idx_vid_sup_end + n_enc     # 7       -> encapsulante sup
idx_celda_end   = idx_enc_sup_end + n_celda   # 8       -> celda
idx_enc_inf_end = idx_celda_end + n_enc       # 9       -> encapsulante inf
# 10 .. 16 -> vidrio inferior

# --- Mapas de Propiedades (k y q_dot) en cada nodo (j,i) ---
Mapa_K = np.zeros((nz, nx))
Mapa_Q_dot = np.zeros((nz, nx))

for j in range(nz):
    # Primero definimos material base según la capa en Z
    if j < idx_vid_sup_end:          # Vidrio superior
        k_val_base = k_vidrio
        q_dot_base = flujo_q_vidrio_dot
    elif j < idx_enc_sup_end:        # Encapsulante superior
        k_val_base = k_enc
        q_dot_base = flujo_q_enc_dot
    elif j < idx_celda_end:          # Capa de la celda
        # acá decidimos según X si hay celda o encapsulante
        k_val_base = 0.0   # se ajusta más abajo según X
        q_dot_base = 0.0
    elif j < idx_enc_inf_end:        # Encapsulante inferior
        k_val_base = k_enc
        q_dot_base = 0.0
    else:                            # Vidrio inferior
        k_val_base = k_vidrio
        q_dot_base = 0.0

    for i in range(nx):
        x_c = X_nodo[i]

        if (j >= idx_enc_sup_end) and (j < idx_celda_end):
            # Capa de celda: diferenciamos entre zona de celda y marco
            if (x_c > Ancho_borde) and (x_c < Ancho_borde + Ancho_celda):
                # Dentro del área activa de la celda
                Mapa_K[j, i] = k_celda
                Mapa_Q_dot[j, i] = flujo_q_celda_dot
            else:
                # Marco lateral (encapsulante sin generación)
                Mapa_K[j, i] = k_enc
                Mapa_Q_dot[j, i] = 0.0
        else:
            Mapa_K[j, i] = k_val_base
            Mapa_Q_dot[j, i] = q_dot_base

# --- 3. FUNCIONES AUXILIARES (propiedades aire y cálculo de h) ---
def props_aire(T_celsius):
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

def calc_h(T_top, T_bot):
    # --- Superficie Superior (mixta forzada + natural) ---
    T_film_top = (T_top + T_amb) / 2
    ka, nu, pr, beta = props_aire(T_film_top - 273.15)
    
    # Forzada
    Re_h = V_viento*L_horiz/nu
    Nu_f = 0.664*Re_h**0.5*pr**(1/3)  # flujo laminar
    h_f = Nu_f*ka/L_horiz
    
    # Natural
    T_op_nat = T_top
    T_op_nat_celsius = T_op_nat - 273.15
    ka_op, nu_op, pr_op, beta_op = props_aire(T_op_nat_celsius)
    Ra = 9.81*beta_op*abs(T_op_nat-T_amb)*L_caracteristico**3/(nu_op*(ka_op/((352.977/T_op_nat)*1005)))
    h_n_top = 0.14*(Ra*np.cos(Angulo))**(1/3)*ka_op/L_caracteristico
    
    # Combinada
    n_exp = 3 + np.cos(Angulo)
    h_top = (h_n_top**n_exp + h_f**n_exp)**(1/n_exp)
    
    # --- Superficie Inferior (natural) ---
    T_film_bot = (T_bot + T_amb) / 2
    T_op_nat_bot = T_bot
    T_op_nat_bot_celsius = T_op_nat_bot - 273.15
    kb_op, nub_op, prb_op, betab_op = props_aire(T_op_nat_bot_celsius)
    
    Ra_b = 9.81*betab_op*abs(T_op_nat_bot-T_amb)*L_caracteristico**3/(nub_op*(kb_op/((352.977/T_op_nat_bot)*1005)))
    h_bot = 0.27*(Ra_b*np.cos(Angulo))**(1/4)*kb_op/L_caracteristico
    return h_top, h_bot

# --- 4. SOLVER ITERATIVO (SOR) - DIFERENCIAS FINITAS CONSERVATIVAS ---

T = np.ones((nz, nx)) * (40.0 + 273.15)  # K
error = 1.0
tol = 1e-5
max_iter = 10000
cnt = 0
omega = 1.2  # factor de sobrerrelajación para acelerar convergencia

print("Iniciando Método Iterativo (Diferencias Finitas con SOR)...")

while error > tol and cnt < max_iter:
    cnt += 1
    T_old_global = T.copy()
    
    # h_top y h_bot calculadas con temperatura promedio
    T_sup_avg = np.mean(T[0, :])
    T_inf_avg = np.mean(T[-1, :])
    h_top, h_bot = calc_h(T_sup_avg, T_inf_avg)
    
    for j in range(nz):
        dzj = dz[j]
        for i in range(nx):
            dxi = dx[i]
            k_P = Mapa_K[j, i]
            
            sigma_a_nb = 0.0
            # En DF conservativa, el término fuente es q_dot (por unidad de volumen),
            # pero como usamos la formulación volumétrica, lo multiplicamos por el volumen del nodo.
            rhs = Mapa_Q_dot[j, i] * dxi * dzj

            # --- Vecinos ESTE/OESTE (X) ---
            # Oeste
            if i > 0:
                k_W = Mapa_K[j, i-1]
                dist_x = (dxi + dx[i-1]) / 2.0
                k_interfaz_W = 2.0 * k_P * k_W / (k_P + k_W) if (k_P + k_W) != 0 else k_P
                a_W = k_interfaz_W * dzj / dist_x
                sigma_a_nb += a_W
                rhs += a_W * T[j, i-1]
            
            # Este
            if i < nx - 1:
                k_E = Mapa_K[j, i+1]
                dist_x = (dxi + dx[i+1]) / 2.0
                k_interfaz_E = 2.0 * k_P * k_E / (k_P + k_E) if (k_P + k_E) != 0 else k_P
                a_E = k_interfaz_E * dzj / dist_x
                sigma_a_nb += a_E
                rhs += a_E * T_old_global[j, i+1]

            # --- Vecinos en Z (NORTE/SUR) ---
            # Nodos internos: j-1 (N), j+1 (S)
            if j > 0 and j < nz - 1:
                # Norte
                k_N = Mapa_K[j-1, i]
                dist_z = (dzj + dz[j-1]) / 2.0
                k_interfaz_N = 2.0 * k_P * k_N / (k_P + k_N) if (k_P + k_N) != 0 else k_P
                a_N = k_interfaz_N * dxi / dist_z
                sigma_a_nb += a_N
                rhs += a_N * T[j-1, i]
            
            # --- Borde Superior (j = 0) ---
            if j == 0:
                T_val = T_old_global[j, i]
                dz_sup = dzj / 2.0
                
                # Conducción al nodo inferior (j=1)
                k_S = Mapa_K[j+1, i]
                dist_z_S = (dzj + dz[j+1]) / 2.0
                k_interfaz_S = 2.0 * k_P * k_S / (k_P + k_S) if (k_P + k_S) != 0 else k_P
                a_S = k_interfaz_S * dxi / dist_z_S
                sigma_a_nb += a_S
                rhs += a_S * T_old_global[j+1, i]
                
                # Conducción hasta la superficie
                a_cond_surf = k_P * dxi / dz_sup
                sigma_a_nb += a_cond_surf
                
                # Convección + radiación linealizada
                h_rad = emisividad * sigma * (
                    (T_val**2 + T_cielo**2) * (T_val + T_cielo) * F_cielo +
                    (T_val**2 + T_suelo**2) * (T_val + T_suelo) * F_suelo_sup
                )
                h_eff = h_top + h_rad
                a_conv_rad = h_eff * dxi
                sigma_a_nb += a_conv_rad
                
                # Términos de fuente de borde (formulación linealizada)
                rhs += a_cond_surf * T_old_global[j, i]
                rhs += a_conv_rad * T_amb * (h_top / h_eff)
                rhs += a_conv_rad * (h_rad / h_eff) * (F_cielo*T_cielo + F_suelo_sup*T_suelo)

            # Nodos internos hacia abajo (SUR) – excepto borde inferior
            if j < nz - 1 and j > 0:
                k_S = Mapa_K[j+1, i]
                dist_z = (dzj + dz[j+1]) / 2.0
                k_interfaz_S = 2.0 * k_P * k_S / (k_P + k_S) if (k_P + k_S) != 0 else k_P
                a_S = k_interfaz_S * dxi / dist_z
                sigma_a_nb += a_S
                rhs += a_S * T_old_global[j+1, i]

            # --- Borde Inferior (j = nz-1) ---
            if j == nz - 1:
                T_val = T_old_global[j, i]
                dz_inf = dzj / 2.0
                
                # Conducción al nodo superior (j = nz-2)
                k_N = Mapa_K[j-1, i]
                dist_z_N = (dzj + dz[j-1]) / 2.0
                k_interfaz_N = 2.0 * k_P * k_N / (k_P + k_N) if (k_P + k_N) != 0 else k_P
                a_N = k_interfaz_N * dxi / dist_z_N
                sigma_a_nb += a_N
                rhs += a_N * T[j-1, i]
                
                # Conducción hasta la superficie inferior
                a_cond_surf = k_P * dxi / dz_inf
                sigma_a_nb += a_cond_surf
                
                # Convección + radiación linealizada inferior
                h_rad = emisividad * sigma * (T_val**2 + T_suelo**2) * (T_val + T_suelo) * F_suelo_inf
                h_eff = h_bot + h_rad
                a_conv_rad = h_eff * dxi
                sigma_a_nb += a_conv_rad
                
                rhs += a_cond_surf * T_old_global[j, i]
                rhs += a_conv_rad * T_amb * (h_bot / h_eff)
                rhs += a_conv_rad * (h_rad / h_eff) * T_suelo

            # Actualizar temperatura del nodo con SOR
            T_P_new = rhs / sigma_a_nb
            T[j, i] = T[j, i] + omega * (T_P_new - T[j, i])
    
    error = np.max(np.abs(T - T_old_global))
    if cnt % 100 == 0 or cnt == 1:
        print(f"Iter {cnt}: Error {error:.2e}, T_max {np.max(T)-273.15:.2f} °C")
    
    if np.max(T) > 500:
        print("¡ADVERTENCIA! Divergencia detectada (T > 500 K). Deteniendo la simulación.")
        break

# --- 5. RESULTADOS Y GRÁFICOS ---
T_celsius = T - 273.15
print(f"\nConvergencia alcanzada en {cnt} iteraciones.")
print(f"Temp Máxima: {np.max(T_celsius):.2f} °C")
print(f"Temp Promedio Superficie Sup: {np.mean(T_celsius[0,:]):.2f} °C")

# Mapa de calor 2D
plt.figure(figsize=(10, 4))
plt.imshow(
    T_celsius, aspect='auto', cmap='inferno',
    extent=[0, Ancho_total*1000, Espesor_total*1000, 0]
)
plt.colorbar(label='Temperatura (°C)')
plt.title('Distribución de Temperatura 2D en el Panel (Diferencias Finitas)')
plt.xlabel('Ancho (mm)')
plt.ylabel('Profundidad Z (mm)')
plt.show()

# Perfil de temperatura en el espesor, al centro en X
indice_centro = nx // 2
profundidades = nodos_z * 1000  # mm
temps_centro = T_celsius[:, indice_centro]

plt.figure()
plt.plot(profundidades, temps_centro, 'o-', label='Centro Celda')
plt.xlabel('Profundidad (mm)')
plt.ylabel('Temperatura (°C)')
plt.title('Perfil de Temperatura a través del espesor (Diferencias Finitas)')
plt.grid(True)
plt.legend()
plt.show()
