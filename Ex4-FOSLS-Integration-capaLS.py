##!/usr/bin/env python3
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


Neurons=20
# ARQUITECTURA DE LA RED NEURONAL:
def init_model(num_hidden_layers = 3, num_neurons_per_layer = 20 ):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(1,)))

    for _ in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(num_neurons_per_layer,
                                        activation = tf.keras.activations.get('tanh'),
                                       kernel_initializer = 'glorot_normal')) #Forma de inicialización de los pesos.

    model.add(tf.keras.layers.Dense(Neurons))
    return model

#########################################
#   SETTING
# Exact solution function (using TensorFlow operations)
def u_exact(x):
    
    return x*(1-x)

def sigma(x):
    
    return 1+0*x


def v_flux(x):
    # sigma*(grad(u)):
        
    return sigma(x)*(1-2*x)

def f_fun(x):
    # lado derecho
    return -2+0*x


############################################

import tensorflow as tf

def residuoLastLayer(model,N=100):
    
    x = tf.linspace(0.0, 1.0, N + 1)[:, None]  # Puntos en el intervalo [0, 1]
    x0 = x[:-1]  # Extremos inferiores de los intervalos
    x1 = x[1:]   # Extremos superiores de los intervalos

    # Longitudes y puntos medios de los intervalos
    interval_lengths = x1 - x0
    

    # Generar puntos aleatorios dentro de cada subintervalo
    random_points = tf.random.uniform(shape=(len(x0), 1), minval=0.0, maxval=1.0)
    complement_points = 1.0 - random_points

    # Mapear los puntos al intervalo original
    # es necesario ordenar
    
    mapped_points1 = x0 + interval_lengths * (random_points)
    mapped_points2 = x0 + interval_lengths * (complement_points)
    
    print('mapped points ',mapped_points1)
    print('mapped points',mapped_points2)
    
    
    # Unificar los puntos mapeados para evaluación por lotes
    points_all = tf.concat([mapped_points1, mapped_points2], axis=0)  # (2 * N, 1)
    
    print(points_all)
    
    # Create tensors for the boundary points
    point0 = tf.constant([[0.]], dtype=tf.float32)  # Shape (1, 1)
    point1 = tf.constant([[1.]], dtype=tf.float32)    # Shape (1, 1)
#points_all = tf.sort(points_all, axis=0)

    
    # Pesos (iguales para ambos puntos en cada intervalo)
    weights = tf.constant(0.5, dtype=tf.float32)  # Pesos fijos para la regla de cuadratura

    # Calcular derivadas de u y v
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(points_all)
        model_penuntimo = model(points_all)
        # Calcular el jacobiano completo
    jacobianos = tape.batch_jacobian(model_penuntimo, points_all)

    # Squeeze para eliminar la última dimensión (si es de tamaño 1)
    jacobian_squeezed = tf.squeeze(jacobianos, axis=-1)
    
    sig = sigma(points_all)
    # Concatenar las matrices de manera eficiente
    u00 = model(point0) # model en cero
    u11 = model(point1)
    # Concatenar las matrices de manera eficiente
    concatenated_u0 = tf.concat( [ u00,                      0*u00], axis=1) # [u(0), 0 ]
    concatenated_top = tf.concat([-1*sig*jacobian_squeezed,     model_penuntimo], axis=1)
    concatenated_bottom = tf.concat([0 * jacobian_squeezed, jacobian_squeezed], axis=1)
    concatenated_u1 = tf.concat( [ u11,                  0*u11], axis=1)
    
    # Concatenar la parte superior e inferior para obtener la matriz final
    final_matrix = tf.concat([concatenated_u0,
                              concatenated_top, 
                              concatenated_bottom,
                              concatenated_u1], axis=0)
    
    #print(final_matrix)
    # Crear f_vals (dos veces xx)
   
    f_vals = f_fun(points_all)
    
    f_final = tf.concat([point0, 0 * points_all, f_vals , 0+0*point1], axis=0)  # Concatenar ceros con f_vals
    results = tf.linalg.lstsq(final_matrix, f_final, l2_regularizer=0.007, fast=True)
    
    print("Results from least squares:", results)

    # Dividing results into components
    peso_u = results[:Neurons]
    peso_v = results[Neurons:]
    
    # Liberar el GradientTape
    del tape
    
    # Calcular el producto punto correctamente usando tf.reduce_sum
    # revisar si es jacobiano * alpha_u
    
    u_x = tf.matmul(jacobian_squeezed, peso_u) # Producto punto entre peso_u y la derivada de u
    v_x = tf.matmul(jacobian_squeezed, peso_v)  # Producto punto entre peso_v y la derivada de v
    
    
    u = tf.matmul(model_penuntimo, peso_u) 
    v = tf.matmul(model_penuntimo, peso_v) 
    

    # Calcular el residuo para cada punto
    sig = sigma(points_all)
    #f_values = f_fun(points_all)  # Evaluar función fuente en puntos mapeados
    residuo1 = weights * (v_x - f_vals) ** 2
    residuo2 = weights * (v - sigma(points_all)*u_x) ** 2

    # Integrar usando cuadratura
    integral_1 = tf.reduce_sum(interval_lengths * tf.reduce_sum(residuo1[:len(x0)] + residuo1[len(x0):], axis=0))
    integral_2 = tf.reduce_sum(interval_lengths * tf.reduce_sum(residuo2[:len(x0)] + residuo2[len(x0):], axis=0))

    # Condiciones de frontera
    u_0 = tf.matmul(model(point0), peso_u) # Evaluar u(0) 
    u_1 = tf.matmul(model(point1), peso_u)
    
    boundary_loss = tf.reduce_mean(u_0**2) + tf.reduce_mean(u_1**2) # Penalizar u(0) ≠ 0
    
    v_flux_ex = v_flux(points_all)
    u_ex = u_exact(points_all)
    # Compute the squared error
    squared_error = (v - v_flux_ex)**2 + (u-u_ex)**2
    errorL2 = tf.reduce_sum(weights*(squared_error))
    total_loss = integral_1 + integral_2 + boundary_loss
    # Retornar la pérdida combinada
    return total_loss, integral_1, integral_2, boundary_loss, errorL2, peso_u, peso_v


# Calcular la pérdida del residuo
model = init_model()
loss, loss1, loss2, bc, errorL2, peso_u, peso_v = residuoLastLayer(model, N=3)
print("Pérdida total:", loss.numpy())
print("Residuo 1 (integral):", loss1.numpy())
print("Residuo 2 (integral):", loss2.numpy())
print("Condición de frontera (pérdida):", bc.numpy())
print("Error L2 (pérdida):", errorL2.numpy())


from time import time

# Create a tf.function wrapper for the training step
@tf.function(reduce_retracing=True)
def train_step(model, optimizer):
    with tf.GradientTape(persistent=True) as tape:
        # Calculate the combined loss
        loss, loss1, loss2, lossbc, errorl2, peso_u, peso_v = residuoLastLayer(model, N=50)
        tf.debugging.check_numerics(loss, message='Loss has invalid values')

    # Compute gradients and apply them
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    del tape
    # Calculate error (make sure error_H1 returns a scalar)
    #print('before')
    #er1 = error_H1(model)
    #print('after')
    return loss, loss1, loss2, lossbc, errorl2, peso_u, peso_v

# Create a dictionary to store history
hist = {'loss': [], 'loss_u': [], 'loss_v':[], 'loss_BC':[], 'errorl2':[]}

# Start the timer
t0 = time()

# Define number of epochs
epochs = 10000

# Initialize optimizer (assuming Adam here, adjust as needed)
optimizer = keras.optimizers.Adam(0.001)

# Use a while loop instead of for loop (or just a tf.function-based optimization)
i = 0
while i < epochs:
    # Perform a training step
    loss, loss1, loss2, lossbc, errorl2, peso_u, peso_v = train_step(model, optimizer)

    # Append the loss and error to the history (using .numpy() here, which is now valid)
    hist['loss'].append(loss.numpy())  # This works because we are outside of tf.function
    hist['loss_u'].append(loss1.numpy())  # This works because we are outside of tf.function
    hist['loss_v'].append(loss2.numpy())  # This works because we are outside of tf.function
    hist['loss_BC'].append(lossbc.numpy())  # This works because we are outside of tf.function
    hist['errorl2'].append(errorl2.numpy())
    # Print progress every 500 epochs
    if i % 50 == 0:
        print(f'Epoch {i}, Loss: {loss.numpy()}')

    i += 1

# End the timer and print the training time`b
print(f"Training time: {time() - t0} seconds")

import matplotlib.pyplot as plt
#plt.plot(hist['error'], label='error')
#plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.xscale('log')
plt.plot(hist['loss'], label='loss')
plt.plot(hist['loss_u'], label='loss_u')
plt.plot(hist['loss_v'], label='loss_v')
plt.plot(hist['loss_BC'], label='loss_bc')
plt.plot(hist['errorl2'], label='errorl2')

plt.legend()
plt.show()

import matplotlib.pyplot as plt
# Grafica
x_min = 0.0
x_max = 1.0
x = np.linspace(x_min, x_max, 21)[:, None]

# Convert x to a TensorFlow tensor
x_tf = tf.convert_to_tensor(x, dtype=tf.float32)

u_pred = tf.matmul(model(x_tf),peso_u).numpy()
v_pred = tf.matmul(model(x_tf),peso_v).numpy()


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
