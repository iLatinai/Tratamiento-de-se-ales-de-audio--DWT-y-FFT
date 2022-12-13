#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 03:48:44 2022

@author: philip
"""


#---------------------------------------------------------------------------------------#
import pywt #Libreria para efectuar la DWT.
import scipy #Libreria para formar arrays mediantes datos de un archivo.
import numpy as np #Libreria que cumple similares funciones a libreria Scipy.
from scipy.io import wavfile #Importación de función wavfile para poder leer arhivos de audio .wav.
import matplotlib.pyplot as plt #Libreria necesaria para visualizar de manera grafica las respectivas transformadas.
import scipy.io.wavfile as wavesdec # Función deribada de la libreria 'Scipy' encargada de elaborar las FFT.
import scipy.fftpack as fourier  #Libreria necesaria para visualizar de manera grafica las respectivas transformadas.


### INICIO: En esta etapa se comienza con la lectura de datos, elaboración de variablesy array  ###

# Mediante la funciones propias de la libreria Scipy, se prodece 
# a leer el archivo de audio a trabajar, donde esta data será convertida a un array denonimado 'señal',
# ademas, la funcion de scipy no entrega el valor de la frecuencia de muestreo del archivo de audio, 
# esta será alojada en la variable 'fs'.
fs, señal = scipy.io.wavfile.read('Jilguero.wav');
#---------------------------------------------------------------------------------------#
# Mediante la funcion 'max_level' derivada de la libreria pydwt podemos 
# encontrar el número máximo de descomposicion que tendrá nuestra señal.
# 'len(señal)': corresponde al tamaño de datos que tiene la señal.
# 'db10': Wavelet madre que definirá el tipo de transformada discreta que tendrá la señal.
#---------------------------------------------------------------------------------------#
max_level = pywt.dwt_max_level(len(señal), 'db10');

#---------------------------------------------------------------------------------------#
### PARTE 2: En esta estapa de aplica la DWT y luego la FFT para cada coeficiente obtenido de la señal ###

i = 0;  # 'i': Variable inicializada que representará el conteo del bucle.
while i <max_level:
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
# PARTE 3: Se comienzan a elaborar las FFT para los coeficientes aproximados y de detalles obtenidos de la DWT ###
     señal = señal/max(señal) # Se normaliza los datos del array 'señal' en el rango [-1,1], ya que los datos de los coeficientes se leen como 16(int), valores muy grandes para trabajar.
     L = len(señal) # Se obtiene el largo del array 'señal'.
#---------------------------------------------------------------------------------------#    
# 'apro': Vector de los coeficientes de aproximación reconstruidos.
# 'pywdt.upcoef': Función encargada de reconstruir los coeficientes.
# 'a': Variable encargada de indicar el tipo de reconstrucción del coeficiente aproximado.
# 'db10': Familia Wavelet que se usará para los coeficientes.
# 'level': Nivel de reconstrucción que tendrá la transformada multinivel de DWT.
# 'take': Tamaño que tendrá la reconstrucción, estará dado por la variable 'L', correspondiente al tamaño del array 'señal'.
     apro = pywt.upcoef('a',señal,'db10',level=i+1, take = L)
     FFT = fourier.fft(apro)# 'fourier.fft': Función encargada de aplicar la FFT para cada coeficiente  de aproximación del vector 'apro'.
     FFT = abs(FFT) # Se aplica valor absoluto al array 'FFT' para obtener solo los datos positivos de los coeficientes.
     FFT = FFT[0:L//2]  # Se divide cada valor del vector 'FFT' en la mitad debido a que la FFT de una señal es simetrica.
     F = fs*np.arange(0, L//2)/L # Se obtiene la frecuencia que tendrá cada coeficiente aproximados, este tambien se divide en la mitad para que sea compatible con los datos obtenidos de la variable 'FFT'.
#---------------------------------------------------------------------------------------#    
     coeff_d = coeff_d/max(coeff_d) # Se normaliza los datos del array 'coeff_d' con el mismo fin que el array 'señal'.
     L1 = len(coeff_d) # Se obtiene el largo del array 'coeff_d'.
# 'co': Vector de los coeficientes de detalle reconstruidos.
# 'pywdt.upcoef': Función encargada de reconstruir los coeficientes.
# 'd': Variable encargada de indicar el tipo de reconstrucción del coeficiente detallado.
# 'db10': Familia Wavelet que se usará para los coeficientes.
# 'level': Nivel de reconstrucción que tendrá la transformada multinivel de DWT.
# 'take': Tamaño que tendrá la reconstrucción, estará dado por la variable 'L1', correspondiente al tamaño del array 'coeff_d'.    
     co = pywt.upcoef('d',coeff_d,'db10',level=i+1, take = L1)
     FFT1 = fourier.fft(co)# 'fourier.fft': Función encargada de aplicar la FFT para cada coeficiente  de detalle del vector 'co'.
     FFT1 = abs(FFT1) # Se aplica valor absoluto al array 'FFT1' para obtener solo los datos positivos de los coeficientes.
     FFT1 = FFT1[0:L1//2]  # Se divide cada valor del vector en la mitad debido a que la FFT de una señal es simetrica.
     F1 = fs*np.arange(0, L1//2)/L1 # Se obtiene la frecuencia que tendrá cada coeficiente detallado, este tambien se divide en la mitad para que sea compatible con los datos obtenidos de la variable 'FFT1'.
#---------------------------------------------------------------------------------------#    
### FINAL: Se comienzan a representar graficamente las FFT de las señales obtenidas para cada DWT obtenida ### 
     plt.plot(F, FFT,color = 'k') # Función encargada de representar en el eje Y y eje X la frecuencia y FFT de los coeficientes de aproximación, respectivamente. Donde 'k', indica color negro, al igual que las graficas de la DWT.
     plt.xlabel('Frecuencia (Hz)', fontsize='14')# Descripción que habrá en cada eje x de las graficas, con su respectiva dimension.
     plt.ylabel('Amplitud - Energia', fontsize='14')# Descripción que habrá en cada eje Y de las graficas, con su respectiva dimension.
     plt.plot(F1, FFT1, color = 'm') # Función encargada de representar en el eje Y y eje X la frecuencia y FFT de los coeficientes de detalle, respectivamente. Donde 'm', indica color morado, al igual que las graficas de la DWT.
     plt.savefig(f'FFT-Nivel_{i+1}.png', dpi=100)# Función encargada de guardad cada archivo de imagen de manera enumerada.
     plt.show()# Se representa visualmente la figura final en el entorno de desarollo. 
     i = i + 1 # La variable iterante se adicina una unidad para  que el bucle continue. 
#---------------------------------------------------------------------------------------#    
    
    