#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 00:48:58 2022

@author: philip
"""
#---------------------------------------------------------------------------------------#
import pywt #Libreria para efectuar la DWT.
import scipy #Libreria para formar arrays mediantes datos de un archivo.
import numpy as np #Libreria que cumple similares funciones a libreria Scipy.
from scipy.io import wavfile #Importación de función wavfile para poder leer arhivos de audio .wav.
import matplotlib.pyplot as plt #Libreria necesaria para visualizar de manera grafica las respectivas transformadas.

### INICIO: En esta etapa se comienza con la lectura de datos, elaboración de variables, array y forma de figura  ###

# Mediante la funciones propias de la libreria Scipy, se prodece 
# a leer el archivo de audio a trabajar, donde esta data será convertida a un array denonimado 'señal',
# ademas, la funcion de scipy no entrega el valor de la frecuencia de muestreo del archivo de audio, 
# esta será alojada en la variable 'fs'.
fs, señal = scipy.io.wavfile.read('Jilguero.wav'); 
print(len(señal)) #Tamaño del array 'señal'
# Visuzalización grafica de la señal original
plt.figure(figsize=(10, 0.5))
señal = señal/max(señal)
plt.plot(señal,color = 'k') 
plt.xlabel('Muestras')
plt.ylabel('Amplitud ')
plt.show()
#---------------------------------------------------------------------------------------#
# Mediante la funcion 'max_level' derivada de la libreria pydwt podemos 
# encontrar el número máximo de descomposicion que tendrá nuestra señal.
# 'len(señal)': corresponde al tamaño de datos que tiene la señal.
# 'db10': Wavelet madre que definirá el tipo de transformada discreta que tendrá la señal.
#---------------------------------------------------------------------------------------#
max_level = pywt.dwt_max_level(len(señal), 'db10');
# Se crea los estandares que tendrá la figura que visualizará la grafica de las DWT. 
# 'nrows': Número máximo de filas.
# 'ncols': Número máximo de columnas.
# 'figsize': Tamaño que tendrá cada figura gráfica.
# 'dpi': Cantidad de pixeles que tendrá la figura final.
fig, axarr = plt.subplots(nrows=max_level, ncols=2, figsize=(40,30), dpi = 100)
#---------------------------------------------------------------------------------------#
### PARTE 2: En esta etapa se comienza con la obtencion de los coeficientes aproximados y detallados producto de la DWT de la señal ###

i = 0; # 'i': Variable inicializada que representará el conteo del bucle.
while i <= max_level:
# 'while': Bucle escogido para iterar los respectivos coeficientes de aproximacion y detalles.
# 'i <= max_level': Condición que tendrá el bucle para iterar, con un limite dado por la variable 'max_level' presentada anteriormente.
#---------------------------------------------------------------------------------------#  
    (señal, coeff_d) = pywt.dwt(señal, 'db10', mode='symmetric', axis=-1 )  
# Mediante la función 'dwt' de la libreria 'pywt' se comienza a aplicar la DWT para la señal de audio.
# 'pywt.dwt': función que entregará los coeficientes de aproximación y detalles, estos serán alojados en las variables 'señal' y 'coeff_d', respectivamente.
# 'db10': Familia Wavelet que se usará para los coeficientes.
# 'symmetric': Modo de extrapolación, que duplica las muestras de la señal.
# 'axis=-1': Eje sobre el cual se calculará la DWT.
#---------------------------------------------------------------------------------------#
    señal = señal/max(señal); # Se normaliza los datos del array 'señal' en el rango [-1,1], ya que los datos de los coeficientes se leen como 16(int), valores muy grandes para trabajar.
    coeff_d = coeff_d/max(coeff_d); # Se normaliza los datos del array 'coeff_d' con el mismo fin que el array 'señal'.
#---------------------------------------------------------------------------------------#
### FINAL: Se comienzan a representar graficamente las DWT de las señales obtenidas para cada coeficiente ###
# 'axarr': función de la libreria 'Matplotlib' encargada de crear los graficos de los coeficientes obtenidos.
# Los vectores '[i,0]' y '[i,1]' indican la ubicación de cada represetación gráfica en la figura.
    axarr[i,0].plot(señal , 'k') # Se representará gráficamente las señales de los coeficientes aproximados, donde 'k' indica el color negro que tendrá la grafica.
    axarr[i, 1].plot(coeff_d, 'm') # Se representará gráficamente las señales de los coeficientes detallados, donde 'm' indica el color morado que tendrá la grafica.
    axarr[i,0].set_ylabel(f'Nivel {i+1}', fontsize=14, rotation=90) # Descripción que tendrá en el eje Y la gráfica, con sus respectivas dimensiones.
    axarr[i,0].set_xlabel('Muestras', fontsize=14) # Descripción que tendrá el eje X en cada grafico de la columna 1.
    axarr[i,1].set_xlabel('Muestras', fontsize=14) # Descripción que tendrá el eje X en cada grafico de la columna 2.
    if i == 0: # Condicional, cuya función es ubicar la primera grafica de cada columna.
        axarr[i,0].set_title("Coeficientes de aproximación", fontsize=14) # Titulo que tendrán todos los gráficos de la columna 1, con sus respectvas dimensiones.
        axarr[i, 1].set_title("Coeficientes de detalles", fontsize=14)  # Titulo que tendrán todos los gráficos de la columna 2, con sus respectvas dimensiones.
    i = i + 1 # La variable iterante se adicina una unidad para  que el bucle continue. 
#---------------------------------------------------------------------------------------#      
plt.savefig('Coeficientes aproximados y detallados.png', dpi=100) # Se guarda la figura final.
plt.tight_layout() # Ajuste para la presentación de la figura final.
plt.show()  # Se representa visualmente la figura final en el entorno de desarollo. 

#---------------------------------------------------------------------------------------#
