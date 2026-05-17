import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
delta_relax, eps, max_iter = 0.3, 1e-3, 150 # Слегка увеличил max_iter для надежности

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
    # Упрощенная аналитическая реакция Атакующего
    a = p['V_data'] * (1 - beta) / (2 * p['c_attack'])
    return np.clip(a, 0, p['a_max'])

def update_controls(alpha_old, beta_old, X, Psi, p):
    xC, xP, xD = X
    psi1, psi2 = Psi
    
    # ---- Альфа (Миграция) ----
    rhs_alpha = -p['c_alpha'] - (psi2 - psi1) * xC
    alpha_new = 0.0
    if rhs_alpha > 0:
        denom = p['lambda'] * p['k'] * p['eta']
        if denom > 0:
            val = rhs_alpha / denom
            if val > 0:
                alpha_new = (1.0/p['eta']) * np.log(val)
    alpha_new = np.clip(alpha_new, 0, p['alpha_max'])
    
    # ---- Бета (Децепция) ----
    if xC > 1e-4: # Увеличен порог для избежания сингулярности в конце горизонта
        beta_new = 1 - (2 * p['c_attack'] * p['c_beta']) / (p['I'] * xC * p['V_data'])
    else:
        beta_new = 1.0
    beta_new = np.clip(beta_new, 0, p['beta_max'])
    
    # Релаксация
    alpha_new = delta_relax * alpha_new + (1 - delta_relax) * alpha_old
    beta_new  = delta_relax * beta_new  + (1 - delta_relax) * beta_old
    return alpha_new, beta_new

# ======================== FBSM =========================
def run_fbsm(params, X0):
    alpha = np.full(n_steps, 0.1)
    beta  = np.full(n_steps, 0.1)
    X_hist = np.zeros((n_steps, 3))
    Psi_hist = np.zeros((n_steps, 2))

    for it in range(max_iter):
        X_curr = X0.copy()
        X_hist[0] = X_curr
        # Прямой проход (RK4)
        for i in range(n_steps-1):
            t = t_eval[i]
            a_curr = optimal_a(beta[i], params)
            h = dt
            k1 = dynamics(t, X_curr, alpha[i], beta[i], a_curr, params)
            k2 = dynamics(t, X_curr + 0.5*h*k1, alpha[i], beta[i], a_curr, params)
            k3 = dynamics(t, X_curr + 0.5*h*k2, alpha[i], beta[i], a_curr, params)
            k4 = dynamics(t, X_curr + h*k3, alpha[i], beta[i], a_curr, params)
            X_curr = np.clip(X_curr + (h/6)*(k1+2*k2+2*k3+k4), 0, 1)
            X_hist[i+1] = X_curr

        Psi_curr = np.array([0.0, 0.0])
        Psi_hist[-1] = Psi_curr
        # Обратный проход (RK4)
        for i in range(n_steps-2, -1, -1):
            t = t_eval[i]
            a_curr = optimal_a(beta[i], params)
            X_curr = X_hist[i]
            h = -dt
            k1 = costate_dynamics(t, Psi_curr, X_curr, alpha[i], beta[i], a_curr, params)
            k2 = costate_dynamics(t, Psi_curr + 0.5*h*k1, X_curr, alpha[i], beta[i], a_curr, params)
            k3 = costate_dynamics(t, Psi_curr + 0.5*h*k2, X_curr, alpha[i], beta[i], a_curr, params)
            k4 = costate_dynamics(t, Psi_curr + h*k3, X_curr, alpha[i], beta[i], a_curr, params)
            Psi_curr = Psi_curr + (h/6)*(k1+2*k2+2*k3+k4)
            Psi_hist[i] = Psi_curr

        # Обновление управлений
        alpha_new, beta_new = np.zeros(n_steps), np.zeros(n_steps)
        for i in range(n_steps):
            alpha_new[i], beta_new[i] = update_controls(alpha[i], beta[i], X_hist[i], Psi_hist[i], params)

        err_alpha = np.max(np.abs(alpha_new - alpha))
        err_beta  = np.max(np.abs(beta_new - beta))
        
        if err_alpha < eps and err_beta < eps:
            print(f"Сходимость достигнута на итерации {it+1}")
            break
        alpha, beta = alpha_new, beta_new

    # Финальный проход для a(t)
    a_opt = np.array([optimal_a(b, params) for b in beta])
    return alpha, beta, a_opt, X_hist

# ======================== ИНТЕГРАЛЬНЫЕ МЕТРИКИ =========================
def calculate_metrics(alpha, beta, a, X, p):
    xC = X[:, 0]
    # Интеграл затрат Защитника: Интеграл (c_alpha * alpha + c_beta * beta) dt
    cost_integrand = p['c_alpha'] * alpha + p['c_beta'] * beta
    total_cost = np.trapz(cost_integrand, t_eval)
    
    # Интеграл ущерба от атаки: Интеграл (I * xC * a * (1 - beta)) dt
    damage_integrand = p['I'] * xC * a * (1 - beta)
    total_damage = np.trapz(damage_integrand, t_eval)
    
    return total_cost, total_damage

# ======================== ГРАФИКИ И ВЫВОД =========================
def plot_comparison(results_dict):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Comparative analysis of Stackelberg equilibrium strategies', fontsize=16)
    
    colors = {'A (High friction)': 'red', 'B (High dynamics)': 'blue', 'C (Strategic)': 'green'}
    
    for name, res in results_dict.items():
        axes[0].plot(t_eval, res['alpha'], label=name, color=colors[name], linewidth=2)
        axes[1].plot(t_eval, res['beta'], label=name, color=colors[name], linewidth=2)
        axes[2].plot(t_eval, res['a'], label=name, color=colors[name], linewidth=2)
        
    axes[0].set_title(r'Migration intensity $\alpha(t)$ (Defender)'); axes[0].set_ylabel('Intensity')
    axes[1].set_title(r'Active deception $\beta(t)$ (Defender)')
    axes[2].set_title(r'Attack intensity $a(t)$ (Attacker)')
    
    for ax in axes:
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel('Time $t$')
        ax.legend()
        
    plt.tight_layout()
    plt.savefig('stackelberg_comparison.png', dpi=300)
    plt.show()

def export_summary_csv(results_dict):
    data = []
    for name, res in results_dict.items():
        data.append({
            'Scenario': name,
            'Total Cost (Budget spent)': round(res['cost'], 2),
            'Total Damage (HNDL)': round(res['damage'], 2),
            'Max Alpha': round(np.max(res['alpha']), 3),
            'Max Beta': round(np.max(res['beta']), 3),
            'Max Attack (a)': round(np.max(res['a']), 3),
            'Final xC (Unprotected %)': round(res['X'][-1, 0] * 100, 2)
        })
    df = pd.DataFrame(data)
    df.to_csv('scenarios_summary.csv', index=False)
    print("\n=== Сводные результаты сохранены в 'scenarios_summary.csv' ===")
    print(df.to_string(index=False))

# ======================== ЗАПУСК =========================
if __name__ == "__main__":
    all_results = {}
    
    for name, sc_params in scenarios.items():
        params = params_common.copy()
        params.update(sc_params)
        X0 = np.array([params['xC0'], params['xP0'], params['xD0']])
        
        print(f"\n=== Запуск для сценария: {name} ===")
        alpha, beta, a, X = run_fbsm(params, X0)
        
        total_cost, total_damage = calculate_metrics(alpha, beta, a, X, params)
        
        all_results[name] = {
            'alpha': alpha, 'beta': beta, 'a': a, 'X': X,
            'cost': total_cost, 'damage': total_damage
        }

    # Визуализация и экспорт
    plot_comparison(all_results)
    export_summary_csv(all_results)