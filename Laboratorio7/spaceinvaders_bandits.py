"""
Space Invaders - Técnicas de Exploración/Explotación (Bandits Adaptados)
========================================================================

Este script demuestra cómo aplicar cuatro técnicas clásicas de balance exploración-explotación
(Greedy, Epsilon-Greedy, Softmax y UCB) en el entorno Space Invaders de Gymnasium.

ADAPTACIÓN:
-----------
Space Invaders no es un problema de bandido clásico, sino un entorno de refuerzo complejo.
Aquí simplificamos el problema: usamos una tabla Q muy básica (un valor por acción, sin considerar el estado)
y actualizamos Q de forma muy simple (promedio incremental de recompensas por acción). El objetivo es ilustrar el
mecanismo de selección de acción de cada técnica, NO resolver el juego ni aprender una política óptima.

TÉCNICAS IMPLEMENTADAS:
-----------------------
1. Greedy:     Siempre elige la acción con mayor valor Q.
2. Epsilon-Greedy:  Con probabilidad ε explora (acción aleatoria), con 1-ε explota (mejor acción).
3. Softmax:    Selecciona acciones según una distribución softmax sobre Q.
4. UCB:        Elige la acción con mayor Upper Confidence Bound (balancea valor y exploración).

Al final de cada episodio se muestra la recompensa total obtenida.
"""

import gymnasium as gym
import numpy as np

# -----------------------------------------------
# Definición de las políticas de selección de acción
# -----------------------------------------------

def greedy_policy(Q):
    """Selecciona la acción con mayor valor Q."""
    return np.argmax(Q)

def epsilon_greedy_policy(Q, epsilon):
    """Con probabilidad epsilon elige aleatorio, si no, la mejor acción."""
    if np.random.rand() < epsilon:
        return np.random.randint(len(Q))
    else:
        return np.argmax(Q)

def softmax_policy(Q, tau=1.0):
    """Selecciona acción según softmax de Q (tau controla aleatoriedad)."""
    exp_Q = np.exp(Q / tau)
    probs = exp_Q / np.sum(exp_Q)
    return np.random.choice(len(Q), p=probs)

def ucb_policy(Q, N, t, c=2):
    """Upper Confidence Bound: favorece acciones menos exploradas."""
    ucb_values = Q + c * np.sqrt(np.log(t + 1) / (N + 1e-5))
    return np.argmax(ucb_values)

# -----------------------------------------------
# Ejecución de un episodio usando una política
# -----------------------------------------------

def run_episode(policy_name, env_name="SpaceInvaders-v5", steps=1000, epsilon=0.1, tau=1.0, c=2):
    env = gym.make(env_name, render_mode="human")
    obs, info = env.reset()
    n_actions = env.action_space.n
    Q = np.zeros(n_actions)   # Valor estimado de cada acción
    N = np.zeros(n_actions)   # Veces que se ha elegido cada acción
    total_reward = 0

    print(f"\n--- Ejecutando política: {policy_name} ---")
    print(f"Acciones posibles: {n_actions}")

    for t in range(steps):
        # Selección de acción según la política elegida
        if policy_name == "greedy":
            action = greedy_policy(Q)
        elif policy_name == "epsilon-greedy":
            action = epsilon_greedy_policy(Q, epsilon)
        elif policy_name == "softmax":
            action = softmax_policy(Q, tau)
        elif policy_name == "ucb":
            action = ucb_policy(Q, N, t, c)
        else:
            raise ValueError("Política no reconocida")

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        N[action] += 1
        # Actualización muy simple de Q: promedio incremental
        Q[action] += (reward - Q[action]) / N[action]

        env.render()
        if terminated or truncated:
            break

    env.close()
    print(f"Recompensa total obtenida: {total_reward}\n")
    print(f"Q final: {Q}\nN (veces elegida cada acción): {N}")

# -----------------------------------------------
# Menú interactivo
# -----------------------------------------------

def main():
    print("Técnicas de exploración/explotación en Space Invaders")
    print("---------------------------------------------------")
    print("1. Greedy")
    print("2. Epsilon-Greedy")
    print("3. Softmax")
    print("4. UCB")
    choice = input("Selecciona la técnica a ejecutar (1-4): ").strip()

    if choice == "1":
        run_episode("greedy")
    elif choice == "2":
        epsilon = float(input("Valor de epsilon (ej: 0.1): ") or 0.1)
        run_episode("epsilon-greedy", epsilon=epsilon)
    elif choice == "3":
        tau = float(input("Valor de tau (ej: 1.0): ") or 1.0)
        run_episode("softmax", tau=tau)
    elif choice == "4":
        c = float(input("Valor de c (ej: 2): ") or 2)
        run_episode("ucb", c=c)
    else:
        print("Opción no válida.")

if __name__ == "__main__":
    main()