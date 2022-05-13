import numpy as np
from scipy.optimize import fsolve, minimize
from matplotlib import pyplot as plt

# Исходные данные
R = 8.31446261815324
c = 1  # 1 для уравнения Peng_Robinson
Omega_A = 0.457235  # Гипер-параметры уравнения Peng_Robinson
Omega_B = 0.077796  # Гипер-параметры уравнения Peng_Robinson
delta1 = ((1 + c) - np.sqrt((1 + c) ** 2 + 4 * c)) / 2  # Гипер-параметры уравнения Peng_Robinson
delta2 = -c / delta1  # Гипер-параметры уравнения Peng_Robinson

# Входные данные в СИ. Компоненты: C1 C2 iC4 NC5 FC9 FC16 FC42
Tc = np.array([190.6, 305.4, 369.8, 408.1, 425.2, 460.4, 872.53])
Pc = np.array([4600155, 4883865, 4245517.5, 3647700, 3799687.5, 3384255, 1025206])
Vc = np.array([0.099, 0.148, 0.203, 0.263, 0.255, 0.306, 1.4036]) * 10 ** (-3)
w = np.array([0.008, 0.098, 0.152, 0.176, 0.193, 0.227, 1.082281])
# z = np.array([0.4228, 0.1166, 0.1006, 0.0266, 0.1909, 0.1025, 0.04]) Начальное
z = np.array([0.363626099, 0.112816115, 0.102742864, 0.028596423, 0.216029174, 0.12243665, 0.053752675])
Mr = np.array([16.043, 30.07, 44.097, 58.124, 58.124, 72.151, 394]) / 1000


def MatBalans(Fv):
    return np.sum(z * (K - 1) / (1 + Fv * (K - 1)))  # Мат Баланс


#  Расчёты коэфф для уравнения состояния
def A_B_Calc(x, w, T, Tc, Pc, Omega_A, Omega_B):
    k = 0.37464 + 1.54226 * w - 0.26992 * w ** 2  # Параметры для расчёта уравнения состояния
    Tr = T / Tc
    alf = (1 + k * (1 - Tr ** 0.5)) ** 2
    a_c = Omega_A * R ** 2 * Tc ** 2 / Pc
    a = a_c * alf
    b = Omega_B * R * Tc / Pc

    def d_fun(i):   # Генерация коэффициентов попарного взаимодействия по корреляции о
        d = 1 - ((2 * (Vc[i] * Vc) ** (1 / 6)) / (Vc[i] ** (1 / 3) + Vc ** (1 / 3))) ** 1.2
        return d

    def S_fun(a, x):  # Расчёт значения S с учетом коэф попарного взаимодействия
        S = np.array([])
        for i in range(len(a)):
            s = np.sqrt(a[i]) * np.sum(x * (1 - d_fun(i)) * np.sqrt(a))
            S = np.append(S, s)
        return S

    S = S_fun(a, x)
    a_sum = np.sum(x * S)
    b_sum = np.sum(x * b)
    A = a_sum * P / (R ** 2 * T ** 2)
    B = b_sum * P / (R * T)
    return A, B, a_sum, b_sum, b, S


def fun(Z, A, B, c):   # Уравнение состояния
    return Z ** 3 - Z ** 2 * (1 - c * B) + Z * (A - B * (1 + c) - B ** 2 * (1 + 2 * c)) \
           - (A * B - c * (B ** 3 + B ** 2))


#  Расчёты коэфф сверхсжимаемости из уравнения состояния
def Z_calculate(A, B, c, Z0, *args):
    def fun(Z, *args):
        den = 1
        for a in args:
            den *= (Z - a)   # Делитель, чтобы ответы не повторялись и были найдены все корни
        return (Z ** 3 - Z ** 2 * (1 - c * B) + Z * (A - B * (1 + c) - B ** 2 * (1 + 2 * c)) \
                - (A * B - c * (B ** 3 + B ** 2))) / den

    result = []
    for i in range(3):
        sol = fsolve(fun, 1, args=tuple(result))[0]
        result.append(sol)

    return np.array(result)


def eos(k1, k2, k3):  # Аналитическое решение, которое работает
    q = (k1 ** 2 - 3 * k2) / 9
    r = (2 * k1 ** 3 - 9 * k1 * k2 + 27 * k3) / 54
    if r ** 2 > q ** 3:
        s = -np.sign(r) * (np.abs(r) + (r ** 2 - q ** 3) ** 0.5) ** (1 / 3)
        t = q / s if s != 0 else 0
        return [(s + t) - k1 / 3]
    else:
        theta = np.arccos(r / q ** (3 / 2))
        return [-2 * q ** 0.5 * np.cos(theta / 3) - k1 / 3,
                -2 * q ** 0.5 * np.cos((theta + 2 * np.pi) / 3) - k1 / 3,
                -2 * q ** 0.5 * np.cos((theta - 2 * np.pi) / 3) - k1 / 3]


def Z_check(Z, *args):
    Z_real = np.array([])
    for i in Z:
        z = np.isclose(fun(i, *args), [0]) * i
        z = (z > 0) * z
        Z_real = np.append(Z_real, z)
    return Z_Choose_MinG(Z_real, *args)


def Z_Choose_MinG(Z, A, B, c):
    delta1 = ((1 + c) - np.sqrt((1 + c) ** 2 + 4 * c)) / 2
    delta2 = -c / delta1
    n = np.size(Z)
    if n == 3:
        Z1 = Z[0]
        Z2 = Z[1]
        d_G1 = np.log((Z2 - B) / (Z1 - B)) + 1 / (delta2 - delta1) * A / B * np.log(
            (Z2 + delta2 * B) / (Z1 + delta2 * B) * (Z1 + delta1 * B) / (Z2 + delta1 * B)) - (Z2 - Z1)
        if d_G1 > 0:
            Z_find1 = Z2
        else:
            Z_find1 = Z1

        Z1 = Z_find1
        Z2 = Z[2]
        d_G2 = np.log((Z2 - B) / (Z1 - B)) + 1 / (delta2 - delta1) * A / B * np.log(
            (Z2 + delta2 * B) / (Z1 + delta2 * B) * (Z1 + delta1 * B) / (Z2 + delta1 * B)) - (Z2 - Z1)
        if d_G2 > 0:
            Z_find2 = Z2
        else:
            Z_find2 = Z1
        return Z_find2
    elif n == 2:
        Z1 = Z[0]
        Z2 = Z[1]
        d_G1 = np.log((Z2 - B) / (Z1 - B)) + 1 / (delta2 - delta1) * A / B * np.log(
            (Z2 + delta2 * B) / (Z1 + delta2 * B) * (Z1 + delta1 * B) / (Z2 + delta1 * B)) - (Z2 - Z1)
        if d_G1 > 0:
            Z_find1 = Z2
        else:
            Z_find1 = Z1
        return Z_find1


def fugacity(a_sum, b_sum, b, A, B, Z, S, delta1, delta2):
    ln_fi = b / b_sum * (Z - 1) - np.log(Z - B) - 1 / (delta2 - delta1) * A / B * (2 * S / a_sum - b / b_sum) * np.log(
        (Z + delta2 * B) / (Z + delta1 * B))
    return ln_fi


def MultiPhazeCalc(K):
    global x_G, x_L, Z_G, Z_L, Fv, check_L, check_G
    Fv = fsolve(MatBalans, np.array([0]))
    K = np.abs(K)

    x_L = z / (1 + Fv * (K - 1))
    A_L, B_L, a_sum_L, b_sum_L, b_L, S_L = A_B_Calc(x_L, w, T, Tc, Pc, Omega_A, Omega_B)
    k1 = - (1 - c * B_L)
    k2 = (A_L - B_L * (1 + c) - B_L ** 2 * (1 + 2 * c))
    k3 = - (A_L * B_L - c * (B_L ** 3 + B_L ** 2))
    Z_L = eos(k1, k2, k3)
    # Z_L = Z_calculate(A_L, B_L, c, 0)
    if len(Z_L) > 1:
        Z_L = Z_check(Z_L, A_L, B_L, c)
    else:
        Z_L = Z_L[0]

    x_G = K * x_L
    A_G, B_G, a_sum_G, b_sum_G, b_G, S_G = A_B_Calc(x_G, w, T, Tc, Pc, Omega_A, Omega_B)
    k1 = - (1 - c * B_G)
    k2 = (A_G - B_G * (1 + c) - B_G ** 2 * (1 + 2 * c))
    k3 = - (A_G * B_G - c * (B_G ** 3 + B_G ** 2))
    Z_G = eos(k1, k2, k3)

    # Z_G = Z_calculate(A_G, B_G, c, 2)
    if len(Z_G) > 1:
        Z_G = Z_check(Z_G, A_G, B_G, c)

    else:
        Z_G = Z_G[0][0]
    # Check results fugacity Calculation3

    check_L = P * x_L * np.exp(fugacity(a_sum_L, b_sum_L, b_L, A_L, B_L, Z_L, S_L, delta1, delta2))
    check_G = P * x_G * np.exp(fugacity(a_sum_G, b_sum_G, b_G, A_G, B_G, Z_G, S_G, delta1, delta2))
    check = np.abs(check_L - check_G)
    print(f"ZL:  {Z_L},ZG:  {Z_G},Разность летучестей:  {check}")  # Проблема в этом месте, поскольку при низких фазовых
    # мольных долях, большая погрешность. Прежде всего для тяжёлых и лёгких компонентов, особенно, близко к Давлению
    # насышения. Нужно увеличивать точность до 10^-21
    return check


Z_L_all = np.array([])
x_L_all = np.array([], ndmin=2)
Z_G_all = np.array([])
x_g_all = np.array([], ndmin=2)
K_all = np.array([])
Vmolar_L_all = np.array([])
Vmolar_G_all = np.array([])
V_L_all = np.array([])
V_G_all = np.array([])
B0_all = np.array([])
Mr_L_all = np.array([])
Mr_G_all = np.array([])
Density_L_all = np.array([])
Density_G_all = np.array([])
G_all = np.array([])
Fv_all = np.array([])
i = -1

P_all = np.linspace(101325, 101325 * 15, 15)  # Давление от 1 атм до 15 атм
T = np.array([20]) + 273.15  # Температура 20 цельсия переводится в кельвины
err = 1e-8

for P in P_all:
    i += 1
    ln_K = 5.37 * (1 + w) * (1 - Tc / T) + np.log(Pc / P)
    K = np.e ** ln_K
    eq = [1] * len(K)
    # fsolv = fsolve(MultiPhazeCalc, K)
    MultiPhazeCalc(K)
    while not all(np.abs(eq) < err):
        eq = check_L / check_G - 1
        K *= (eq + 1)
        fsolv = MultiPhazeCalc(K)

    Vmolar_L = Z_L * R * T / P
    Vmolar_G = Z_G * R * T / P

    V_L = Vmolar_L * (1 - Fv)
    V_G = Vmolar_G * Fv

    # Плотности
    Mr_L = np.sum(x_L * Mr)
    Density_L = Mr_L / Vmolar_L

    Mr_G = np.sum(x_G * Mr)
    Density_G = Mr_G / Vmolar_G

    Z_L_all = np.append(Z_L_all, Z_L)
    Z_G_all = np.append(Z_G_all, Z_G)
    K_all = np.append(K_all, K)
    Vmolar_L_all = np.append(Vmolar_L_all, Vmolar_L)
    Vmolar_G_all = np.append(Vmolar_G_all, Vmolar_G)
    V_L_all = np.append(V_L_all, V_L)
    V_G_all = np.append(V_G_all, V_G)
    Mr_L_all = np.append(Mr_L_all, Mr_L)
    Density_L_all = np.append(Density_L_all, Density_L)
    Mr_G_all = np.append(Mr_G_all, Mr_G)
    Density_G_all = np.append(Density_G_all, Density_G)
    Fv_all = np.append(Fv_all, Fv)

    B0 = V_L / V_L_all[0]  # Обьёмный коэффициент
    B0_all = np.append(B0_all, B0)
    G = (V_G_all[0] - V_G) / V_L_all[0]  # Газосодержание
    G_all = np.append(G_all, G)

print(f"V_L:  {V_L_all * 1000} \nV_G:   {V_G_all * 1000}")
print(f"Density_L:  {Density_L_all} \nDensity_G:   {Density_G_all}")

m_G_all = V_G_all * Density_G_all
m_L_all = V_L_all * Density_L_all

fig, ax = plt.subplots(3)
ax[0].plot(P_all[~np.isnan(Density_L_all)], G_all[~np.isnan(Density_L_all)], label='G')
ax[0].set_ylabel('Газосодержание м^3/м^3')
ax[0].set_xlabel('Давление Па')

ax[1].plot(P_all[~np.isnan(Density_L_all)], Density_L_all[~np.isnan(Density_L_all)], label='Density_L')
ax2 = ax[1].twinx()
ax2.plot(P_all[~np.isnan(Density_G_all)], Density_G_all[~np.isnan(Density_G_all)], label='Density_G')
ax[1].set_ylabel('Density')
ax[1].set_xlabel('Давление Па')

V_L_new = Z_L_all * R * T / P_all
ax[2].plot(P_all[~np.isnan(Density_L_all)], B0_all[~np.isnan(Density_L_all)], label='B0_allм^3')
ax[2].set_ylabel('Bo м^3/м^3')
ax[2].set_xlabel('Давление Па')
# plt.legend()
# plt.grid()
plt.show()

# Расчёт нормальных условий
T = np.array([20]) + 273.15
P = 101325
# z = x_L / np.sum(x_L)
ln_K = 5.37 * (1 + w) * (1 - Tc / T) + np.log(Pc / P)
K = np.e ** ln_K
eq = [1] * len(K)
# fsolv = fsolve(MultiPhazeCalc, K)
MultiPhazeCalc(K)
while not all(np.abs(eq) < err):
    eq = check_L / check_G - 1
    K *= (eq + 1)
    fsolv = MultiPhazeCalc(K)

Vmolar_L0 = Z_L * R * T / P
Vmolar_G0 = Z_G * R * T / P

V_L0 = Vmolar_L0 * (1 - Fv)
V_G0 = Vmolar_G0 * Fv

# Плотности
Mr_L = np.sum(x_L * Mr)
Density_L0 = Mr_L / Vmolar_L0

Mr_G0 = np.sum(x_G * Mr)
Density_G0 = Mr_G / Vmolar_G0

G0 = V_G0 / V_L0  # Газосодержание
m_G0 = V_G0 * Density_G0
m_L0 = V_L0 * Density_L0
print("Мольная доля газовой фазы после разгазирования Н.У. ", Fv)
