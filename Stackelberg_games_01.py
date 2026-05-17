import numpy as np
import matplotlib.pyplot as plt

# ========================= ПАРАМЕТРЫ МОДЕЛИ =========================
params_common = {
    'c_attack': 0.5, 'mu': 0.1, 'theta': 0.05, 'I': 0.8, 'k': 0.5,
    'L_trap': 5.0, 'a_max': 1.0, 'alpha_max': 0.5, 'beta_max': 1.0,
    'lambda': 1.0, 'Rmax': 100.0,
}

scenarios = {
    'A (High friction)': {
        'V_data': 2.0, 'eta': 2.5, 'c_alpha': 4.0, 'c_beta': 1.2,
        'xC0': 0.85, 'xP0': 0.1, 'xD0': 0.05
    },
    'B (High dynamics)': {
        'V_data': 6.0, 'eta': 0.2, 'c_alpha': 0.5, 'c_beta': 0.1,
        'xC0': 0.65, 'xP0': 0.3, 'xD0': 0.05
    },
    'C (Strategic)': {
        'V_data': 10.0, 'eta': 1.2, 'c_alpha': 2.0, 'c_beta': 0.8,
        'xC0': 0.75, 'xP0': 0.2, 'xD0': 0.05
    }
}

T, dt = 100.0, 0.5
t_eval = np.arange(0, T+dt, dt)
n_steps = len(t_eval)
delta_relax, eps, max_iter = 0.3, 1e-3, 100

# ====================== ДИНАМИКА ===========================
def dynamics(t, X, alpha, beta, a, p):
    xC, xP, xD = X
    dxC = -alpha * xC - p['mu'] * a * xC
    F = p['k'] * np.exp(p['eta'] * alpha)
    dxP = alpha * xC - p['theta'] * F * xP
    dxD = beta * (1 - xC - xP)
    return np.array([dxC, dxP, dxD])

def costate_dynamics(t, Psi, X, alpha, beta, a, p):
    psi1, psi2 = Psi
    xC, xP, xD = X
    F = p['k'] * np.exp(p['eta'] * alpha)
    dpsi1 = -p['I'] * a * (1 - beta) + psi1*(alpha + p['mu']*a) - psi2*alpha
    dpsi2 = psi2 * p['theta'] * F
    return np.array([dpsi1, dpsi2])

def optimal_a(beta, p):
    a = p['V_data'] * (1 - beta) / (2 * p['c_attack'])
    return np.clip(a, 0, p['a_max'])

def update_controls(alpha_old, beta_old, X, Psi, p):
    xC, xP, xD = X
    psi1, psi2 = Psi
    # ---- Альфа (22) ----
    rhs_alpha = -p['c_alpha'] - (psi2 - psi1) * xC
    alpha_new = 0.0
    if rhs_alpha > 0:
        denom = p['lambda'] * p['k'] * p['eta']
        if denom > 0:
            val = rhs_alpha / denom
            if val > 0:
                alpha_new = (1.0/p['eta']) * np.log(val)
    alpha_new = np.clip(alpha_new, 0, p['alpha_max'])
    # ---- Бета ----
    if xC > 1e-6:
        beta_new = 1 - (2 * p['c_attack'] * p['c_beta']) / (p['I'] * xC * p['V_data'])
    else:
        beta_new = 1.0
    beta_new = np.clip(beta_new, 0, p['beta_max'])
    # релаксация
    alpha_new = delta_relax * alpha_new + (1 - delta_relax) * alpha_old
    beta_new  = delta_relax * beta_new  + (1 - delta_relax) * beta_old
    return alpha_new, beta_new

# ======================== FBSM =========================
def run_fbsm(params, X0):
    alpha = np.full(n_steps, 0.2)
    beta  = np.full(n_steps, 0.1)
    X_hist = np.zeros((n_steps, 3))
    Psi_hist = np.zeros((n_steps, 2))

    for it in range(max_iter):
        # Прямой проход
        X_curr = X0.copy()
        X_hist[0] = X_curr
        for i in range(n_steps-1):
            t = t_eval[i]
            a_curr = optimal_a(beta[i], params)
            h = dt
            k1 = dynamics(t, X_curr, alpha[i], beta[i], a_curr, params)
            k2 = dynamics(t, X_curr + 0.5*h*k1, alpha[i], beta[i], a_curr, params)
            k3 = dynamics(t, X_curr + 0.5*h*k2, alpha[i], beta[i], a_curr, params)
            k4 = dynamics(t, X_curr + h*k3, alpha[i], beta[i], a_curr, params)
            X_next = X_curr + (h/6)*(k1+2*k2+2*k3+k4)
            X_curr = np.clip(X_next, 0, 1)
            X_hist[i+1] = X_curr

        # Обратный проход
        Psi_curr = np.array([0.0, 0.0])
        Psi_hist[-1] = Psi_curr
        for i in range(n_steps-2, -1, -1):
            t = t_eval[i]
            a_curr = optimal_a(beta[i], params)
            X_curr = X_hist[i]
            h = -dt
            k1 = costate_dynamics(t, Psi_curr, X_curr, alpha[i], beta[i], a_curr, params)
            k2 = costate_dynamics(t, Psi_curr + 0.5*h*k1, X_curr, alpha[i], beta[i], a_curr, params)
            k3 = costate_dynamics(t, Psi_curr + 0.5*h*k2, X_curr, alpha[i], beta[i], a_curr, params)
            k4 = costate_dynamics(t, Psi_curr + h*k3, X_curr, alpha[i], beta[i], a_curr, params)
            Psi_next = Psi_curr + (h/6)*(k1+2*k2+2*k3+k4)
            Psi_curr = Psi_next
            Psi_hist[i] = Psi_curr

        # Обновление управлений
        alpha_new = np.zeros(n_steps)
        beta_new = np.zeros(n_steps)
        for i in range(n_steps):
            alpha_new[i], beta_new[i] = update_controls(alpha[i], beta[i],
                                                        X_hist[i], Psi_hist[i], params)

        err_alpha = np.max(np.abs(alpha_new - alpha))
        err_beta  = np.max(np.abs(beta_new - beta))
        print(f"Iter {it+1}: err_alpha={err_alpha:.5f}, err_beta={err_beta:.5f}")
        if err_alpha < eps and err_beta < eps:
            break
        alpha, beta = alpha_new, beta_new

    # Финальная траектория a(t)
    a_opt = np.zeros(n_steps)
    X_final = np.zeros((n_steps, 3))
    X_curr = X0.copy()
    X_final[0] = X_curr
    for i in range(n_steps-1):
        a_opt[i] = optimal_a(beta[i], params)
        X_next = X_curr + dt * dynamics(t_eval[i], X_curr, alpha[i], beta[i], a_opt[i], params)
        X_curr = np.clip(X_next, 0, 1)
        X_final[i+1] = X_curr
    a_opt[-1] = optimal_a(beta[-1], params)

    return t_eval, alpha, beta, a_opt, X_final

# ======================== ВЫВОД ГРАФИКОВ (отдельные окна) =========================
def plot_results(t, alpha, beta, a, X, scenario_name):
    # Создаём новое окно для сценария
    plt.figure(figsize=(12, 8))
    
    # График 1: интенсивность миграции α(t) и обман β(t)
    plt.subplot(2, 2, 1)
    plt.plot(t, alpha, 'b-', linewidth=2)
    plt.ylabel(r'$\alpha(t)$ (migration intensity)')
    plt.xlabel('Time')
    plt.grid(True)
    plt.title('Migration intensity')
    
    plt.subplot(2, 2, 2)
    plt.plot(t, beta, 'r-', linewidth=2)
    plt.ylabel(r'$\beta(t)$ (deception probability)')
    plt.xlabel('Time')
    plt.grid(True)
    plt.title('Deception effort')
    
    # График 2: интенсивность атаки a(t)
    plt.subplot(2, 2, 3)
    plt.plot(t, a, 'g-', linewidth=2)
    plt.ylabel(r'$a(t)$ (attack intensity)')
    plt.xlabel('Time')
    plt.grid(True)
    plt.title('Attacker response')
    
    # График 3: доли узлов
    plt.subplot(2, 2, 4)
    plt.plot(t, X[:,0], 'k-', label=r'$x_C$ (classical)')
    plt.plot(t, X[:,1], 'b--', label=r'$x_P$ (PQC)')
    plt.plot(t, X[:,2], 'r:', label=r'$x_D$ (decoy)')
    plt.ylabel('Node fractions')
    plt.xlabel('Time')
    plt.legend(loc='best')
    plt.grid(True)
    plt.title('System composition')
    
    plt.suptitle(f'Stackelberg equilibrium for scenario {scenario_name}', fontsize=14)
    plt.tight_layout()
    plt.show()

# ======================== ЗАПУСК =========================
if __name__ == "__main__":
    for name, sc_params in scenarios.items():
        params = params_common.copy()
        params.update(sc_params)
        X0 = np.array([params['xC0'], params['xP0'], params['xD0']])
        print(f"\n=== Running scenario: {name} ===")
        t, alpha, beta, a, X = run_fbsm(params, X0)
        plot_results(t, alpha, beta, a, X, name)