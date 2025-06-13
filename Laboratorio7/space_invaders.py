import gymnasium as gym
print([env for env in gym.envs.registry.keys() if "SpaceInvaders" in env])
import pygame
import time

# Inicializa pygame
pygame.init()
screen = pygame.display.set_mode((400, 100))
pygame.display.set_caption("Controlador Space Invaders - Usa flechas y barra espaciadora")

# Inicializa el entorno
env = gym.make("SpaceInvaders-v5", render_mode="human")
observation, info = env.reset()

done = False
total_reward = 0

# Diccionario para mapear teclas a acciones
key_action = {
    pygame.K_LEFT: 4,      # Mover izquierda
    pygame.K_RIGHT: 3,     # Mover derecha
    pygame.K_SPACE: 1,     # Disparar
}

action = 0  # Acción por defecto: NOOP

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
            break
        elif event.type == pygame.KEYDOWN:
            if event.key in key_action:
                action = key_action[event.key]
            elif event.key == pygame.K_ESCAPE:
                done = True
                break
        elif event.type == pygame.KEYUP:
            if event.key in [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_SPACE]:
                action = 0  # NOOP cuando sueltas la tecla

    # Ejecuta la acción en el entorno
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = done or terminated or truncated

    env.render()
    time.sleep(0.02)  # Controla la velocidad del bucle

env.close()
pygame.quit()
print(f"Juego terminado. Recompensa total: {total_reward}")