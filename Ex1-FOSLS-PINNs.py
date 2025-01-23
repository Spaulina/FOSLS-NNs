#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:02:49 2025


@author: paulina

@ Grupo Boldo:
    Laura Sobarzo
    Francisca Alvarez
    Paulina Sepulveda
    
Resuelve: 
    div (sigma * grad(u)) = f
    u(0)= u(1) = 0

Como un sistema de primer orden:
      u'-v = 0 en (0,1)
    div(v) = f en (0,1)
      u(0) = 0
      u(1) = 0
    
"""

import tensorflow as tf  #2.14.0
import numpy as np

import keras
tf.random.set_seed(42)
tf.keras.backend.set_floatx('float32')


# ARQUITECTURA DE LA RED NEURONAL:
def init_model(num_hidden_layers = 4, num_neurons_per_layer = 20 ):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(1,)))

    for _ in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(num_neurons_per_layer,
                                        activation = tf.keras.activations.get('tanh'),
                                       kernel_initializer = 'glorot_normal')) #Forma de inicialización de los pesos.

    model.add(tf.keras.layers.Dense(2))
    return model

#########################################
#   SETTING
# Exact solution function (using TensorFlow operations)
def u_exact(x):
    
    return x*(1-x)

def sigma(x):
    
    return tf.constant(1.0)


def v_flux(x):
    # sigma*(grad(u)):
        
    return sigma(x)*(1-2*x)

def f_fun(x):
    # lado derecho
    return tf.constant(-2.0)


############################################

# Definir la función de Loss

def Loss(model, N=100):
    # Generar puntos de dominio (discretización del intervalo)
    x = tf.linspace(0.0, 1.0, N)[:, None]  # Puntos en el intervalo [0, 1]
    
    
    #points = tf.constant([x0,1-x0], dtype=tf.float32)
    #weights = tf.constant(weights, dtype=tf.float32)


    # Calcular derivadas de u
    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch(x)
        u = model(x)[:, 0:1]  # Primera salida: u(x)
    u_x = tape1.gradient(u, x)  # Primera derivada de u respecto a x
    del tape1  # Liberar memoria

    # Calcular derivadas de v
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(x)
        v = model(x)[:, 1:2]  # Segunda salida: v(x)
    v_x = tape2.gradient(v, x)  # Primera derivada de v respecto a x
    del tape2  # Liberar memoria

    # Calcular el residuo
    f_values = f_fun(x)  # Evaluar función fuente
    residuo1 = v_x - f_values  # Diferencia entre u_x y f(x)
    residuo2 = v- u_x         # Diferencia entre v_x y u

    # Pérdida combinada
    loss1 = tf.reduce_mean(residuo1**2)
    loss2 = tf.reduce_mean(residuo2**2)  # Norma L2 combinada
    
    
    # Condición de frontera en x=0
    x0 = tf.constant([[0.0]], dtype=tf.float32)  # Punto x=0
    u0 = model(x0)[:, 0:1]  # Evaluar u(0)
    x1 = tf.constant([[1.0]], dtype=tf.float32)  # Punto x=0
    u1 = model(x1)[:,0:1]
    boundary_loss = tf.reduce_mean(u0**2 + u1**2)  # Penalizar u(0) ≠ 0


    return loss1 + loss2 + boundary_loss, loss1, loss2, boundary_loss



model= init_model()
##################################################
from time import time 

@tf.function
def train_step(model, optimizer):
    with tf.GradientTape() as tape:
        # Calculate the combined loss
        loss, loss1, loss2, lossbc = Loss(model)
        tf.debugging.check_numerics(loss, message='Loss has invalid values')

    # Compute gradients and apply them
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Calculate error (make sure error_H1 returns a scalar)
    #print('before')
    #er1 = error_H1(model)
    #print('after')
    return loss, loss1, loss2, lossbc

# Create a dictionary to store history
hist = {'loss': [], 'loss_u': [], 'loss_v':[], 'loss_BC':[]}

# Start the timer
t0 = time()

# Define number of epochs
epochs = 10000

# Initialize optimizer (assuming Adam here, adjust as needed)
optimizer = keras.optimizers.Adam(0.0001)

# Use a while loop instead of for loop (or just a tf.function-based optimization)
i = 0
while i < epochs:
    # Perform a training step
    loss, loss1, loss2, lossbc = train_step(model, optimizer)

    # Append the loss and error to the history (using .numpy() here, which is now valid)
    hist['loss'].append(loss.numpy())  # This works because we are outside of tf.function
    hist['loss_u'].append(loss1.numpy())  # This works because we are outside of tf.function
    hist['loss_v'].append(loss2.numpy())  # This works because we are outside of tf.function
    hist['loss_BC'].append(lossbc.numpy())  # This works because we are outside of tf.function

    # Print progress every 500 epochs
    if i % 500 == 0:
        print(f'Epoch {i}, Loss: {loss.numpy()}')

    i += 1

# End the timer and print the training time`b
print(f"Training time: {time() - t0} seconds")

import matplotlib.pyplot as plt
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.xscale('log')
plt.plot(hist['loss'], label='loss')
plt.plot(hist['loss_u'], label='loss_u')
plt.plot(hist['loss_v'], label='loss_v')
plt.plot(hist['loss_BC'], label='loss_bc')

plt.legend()
plt.show()

import matplotlib.pyplot as plt
# Grafica
x_min = 0.0
x_max = 1.0
x = np.linspace(x_min, x_max, 21)[:, None]

# Convert x to a TensorFlow tensor
x_tf = tf.convert_to_tensor(x, dtype=tf.float32)

# Obtener las predicciones del modelo para u y v
u_pred= model(x_tf)[:,0:1].numpy()#.T  # Convertir tensor a NumPy y transponer para obtener u y v
v_pred= model(x_tf)[:,1:2].numpy()#.T  # Convertir tensor a NumPy y transponer para obtener u y v


#v_real = 1-2*x_tf
v_real = v_flux(x_tf)

# Get model predictions and real solution
#u_real =x_tf*(1-x_tf)   # Convert tensor to NumPy for plotting
u_real = u_exact(x_tf)
# Plot predictions and real values
plt.figure(figsize=(10,8))
plt.plot(x, u_pred, label='u_approx(x)')
plt.plot(x, u_real, '*',label='u_real(x)')
plt.legend()
plt.xlabel('x')
plt.ylabel('u')
plt.show()

# Plot predictions and real values
plt.figure(figsize=(10,8))
plt.plot(x, v_pred, label='v_approx(x)')
plt.plot(x, v_real, '*',label='v_real(x)')

plt.legend()
plt.xlabel('x')
plt.ylabel('v')
plt.show()
