#!/usr/bin/env python3
# coding: utf-8

# Aluno: Gabriel Bicalho Maroun

# script pra gerar animacoes de desenhos usando series de Fourier
# basicamente a gente pega uma imagem, extrai o contorno dela,
# e recria aquele contorno por meio de circulos girando

import argparse
from math import tau

import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad_vec

# configura os argumentos da linha de comando
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, type=str,
	help="caminho da imagem de entrada")
ap.add_argument("-o", "--output", required=True, type=str,
	help="caminho do arquivo de saida")
ap.add_argument("-f", "--frames", required=False, default=500, type=int,
	help="quantidade de frames (padrao: 500)")
ap.add_argument("-N", "--N", required=False, default=500, type=int,
	help="quantidade de coeficientes de Fourier (padrao: 500)")
args = vars(ap.parse_args())

# se N for menor que frames, aumenta pra qualidade melhor
if args["N"] < args["frames"]:
	args["N"] = args["frames"]

# carrega a imagem
img = cv2.imread(args["input"])

# converte pra escala de cinza e desfoca pra suavizar arestas
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(imgray, (7, 7), 0)

# binariza a imagem com Otsu pra extrair o contorno
(T, thresh) = cv2.threshold(blurred, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# acha todos os contornos
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# pega o maior contorno (nosso objeto principal)
largest_contour_idx = np.argmax([len(c) for c in contours])

# extrai os pontos do contorno maior
verts = [ tuple(coord) for coord in contours[largest_contour_idx].squeeze() ]

# separa coordenadas x e y
xs, ys = zip(*verts)

# centra no ponto (0,0) pra facilitar a animacao
xs = np.asarray(xs) - np.mean(xs)
ys = - np.asarray(ys) + np.mean(ys)

# parametriza o contorno de 0 a tau (2pi)
t_list = np.linspace(0, tau, len(xs))

# calcula os coeficientes de Fourier
# esses sao os "vetores magicos" que usamos pra reconstruir o contorno

def f(t, t_list, xs, ys):
    """
    Interpola linearmente entre os pontos do contorno.
    Acha o ponto na curva que corresponde ao tempo 't'
    """
    return np.interp(t, t_list, xs + 1j*ys)


def compute_cn(f, n):
    """
    Calcula o n-esimo coeficiente de Fourier usando integracao numerica.
    Cada coeficiente representa uma "amplitude e frequencia" de rotacao
    """
    coef = 1/tau*quad_vec(
        lambda t: f(t, t_list, xs, ys)*np.exp(-n*t*1j), 
        0, 
        tau, 
        limit=100,
        full_output=False)[0]
    return coef

def get_circle_coords(center, r, N=50):
    """
    Gera os pontos que formam um circulo pra visualizar na animacao.
    A gente desenha circulos em azul pra mostrar o tamanho de cada vetor
    """
    theta = np.linspace(0, tau, N)
    x, y = center[0] + r * np.cos(theta), center[1] + r * np.sin(theta)
    return x, y

def get_next_pos(c, fr, t, drawing_time = 1):
    """
    Calcula a posicao de um vetor que ta girando.
    Cada vetor de Fourier gira numa frequencia diferente
    """
    angle = (fr * tau * t) / drawing_time
    return c * np.exp(1j*angle)


# computa todos os coeficientes de Fourier
N = args["N"]
coefs = [ (compute_cn(f, 0), 0) ] + [ (compute_cn(f, j), j) for i in range(1, N+1) for j in (i, -i) ]

# cria a janela da animacao
fig, ax = plt.subplots()

# prepara os elementos visuais:
# - circulos azuis mostram o tamanho de cada vetor
# - linhas verdes conectam um vetor ao outro (tipo setas)
# - linha vermelha eh o desenho final que ta sendo feito
circles = [ax.plot([], [], 'b-')[0] for i in range(-N, N+1)]
circle_lines = [ax.plot([], [], 'g-')[0] for i in range(-N, N+1)]
drawing, = ax.plot([], [], 'r-', linewidth=2)

# define o tamanho da area de visualizacao
ax.set_xlim(-500, 500)
ax.set_ylim(-500, 500)

ax.set_axis_off()
ax.set_aspect('equal')
fig.set_size_inches(15, 15)

# guarda os pontos do desenho
draw_x, draw_y = [], []

# funcao que calcula cada frame da animacao
def animate(i, coefs, time): 
    # pega o tempo correspondente ao frame atual
    t = time[i]
    
    # rotaciona todos os vetores de Fourier
    coefs = [ (get_next_pos(c, fr, t=t), fr) for c, fr in coefs ]
    center = (0, 0)
    
    # desenha cada circulo e conecta um ao outro
    for i, elts in enumerate(coefs):
        c, _ = elts
        r = np.linalg.norm(c)
        x, y = get_circle_coords(center=center, r=r, N=80)
        
        # linha verde do centro atual ate o ponto
        circle_lines[i].set_data([center[0], center[0]+np.real(c)], [center[1], center[1]+np.imag(c)])
        
        # desenha o circulo azul
        circles[i].set_data(x, y) 
        
        # move o centro pro ponto final deste vetor
        center = (center[0] + np.real(c), center[1] + np.imag(c))
    
    # o ponto final eh onde a gente ta desenhando
    draw_x.append(center[0])
    draw_y.append(center[1])

    # atualiza a linha vermelha com todos os pontos
    drawing.set_data(draw_x, draw_y)

# cria a animacao
drawing_time = 1
frames = args["frames"]
time = np.linspace(0, drawing_time, num=frames)    
anim = animation.FuncAnimation(fig, animate, frames = frames, interval = 5, fargs=(coefs, time)) 

# salva a animacao
anim.save(args["output"], fps = 15)


