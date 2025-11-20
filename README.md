# Proyecto Final – ICM2223: Transferencia de Calor

Este proyecto modela la **distribución de temperatura** en un conjunto de celda fotovoltaica compuesto por:
- Vidrio superior
- Encapsulante
- Celda fotovoltaica

El modelo considera **conducción**, **convección natural y forzada**, y **radiación** hacia el cielo y el suelo.

## Contenidos del proyecto
- Cálculo de flujos de calor en condiciones **NOCT**, **nominales** y **anómalas (sombreado)**.
- Obtención de temperaturas en estado estacionario.
- Dependencia de **eficiencia** e **intensidad eléctrica** con la temperatura.
- Cálculo de **generación de calor** en la celda por efecto Joule.
- Visualización de la distribución de temperaturas.

## Archivos incluidos
- `partea.py`: Script principal con la modelación térmica del conjunto.
- Otros archivos o funciones auxiliares si es necesario.

## Resumen del problema
- El panel recibe un flujo incidente definido por las condiciones de operación.
- La superficie superior intercambia calor con el ambiente por convección combinada y radiación.
- La superficie inferior intercambia calor por convección natural y radiación al suelo.
- Las propiedades térmicas de aire, vidrio, encapsulante y celda están dadas en el enunciado.
- Se asume un único elemento diferencial en la celda.

## Casos solicitados
1. **Condición NOCT:** Sin generación eléctrica (η = 0, I = 0).
2. **Condición nominal:** η(T) e I(T) dependen de la temperatura.
3. **Condición anómala:** Sombreado con flujo reducido a 100 W/m².

## Bonus (opcional)
- Solución analítica del caso NOCT.
- Cálculo del coeficiente global U del módulo para distintas velocidades de viento.

