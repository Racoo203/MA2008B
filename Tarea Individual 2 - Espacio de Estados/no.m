clc; clear; close all;

% Definir la variable simbólica
s = tf('s');

% Definir la función de transferencia en lazo abierto
k = 1; % Puedes cambiar este valor de ganancia
G = k * (1) / (s^3 + 0.2 * s^2 + 1);

% Graficar el diagrama de Nyquist
figure;
nyquist(G);
grid on;

% Etiquetas y título
title('Diagrama de Nyquist de G(s) = k(1 - s) / (s + 1)');
xlabel('Re(G(s))');
ylabel('Im(G(s))');
