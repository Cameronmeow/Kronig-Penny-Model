import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const

# --- Page Title ---
st.title("Kronig-Penney Model Visualizer")
st.markdown("Numerical solution and band structure for a 1D Kronig-Penney potential.")

# --- Constants ---
hbar = const.hbar
m = const.m_e
e = const.e

# --- User Inputs ---
st.sidebar.header("Model Parameters")

a_angstrom = st.sidebar.slider("Well width a (in Å)", min_value=1.0, max_value=10.0,value=5.0, step=1.0)
b_angstrom = st.sidebar.slider("Barrier width b (in Å)", value=1.0, min_value=0.5, max_value=5.0, step=0.1)
V0_eV = st.sidebar.slider("Barrier height V₀ (in eV)", value=10.0, min_value=1.0, max_value=100.0, step=1.0)

E_min_eV = st.sidebar.slider("Minimum energy (eV)", min_value=0.0, max_value=10.0, step=0.1, value=0.01)
E_max_eV = st.sidebar.slider("Maximum energy (eV)",  min_value=10.0, max_value=100.0, step=1.0, value=30.0)
N_E = st.sidebar.slider("Number of energy points", min_value=500, max_value=5000, value=3000, step=100)

# Convert units
a = a_angstrom * 1e-10
b = b_angstrom * 1e-10
V0 = V0_eV * e
L = a + b
energies_eV = np.linspace(E_min_eV, E_max_eV, N_E)
energies_J = energies_eV * e

# --- Function to Calculate f(E) ---
def calculate_fE(E_J, V0_J, a, b, m, hbar):
    if E_J <= 0:
        return 2.0

    alpha_sq = 2 * m * E_J / hbar**2
    k1 = np.sqrt(alpha_sq)

    if abs(E_J - V0_J) < 1e-12 * e:
        epsilon = 1e-9 * V0_J
        fE_below = calculate_fE(E_J - epsilon, V0_J, a, b, m, hbar)
        fE_above = calculate_fE(E_J + epsilon, V0_J, a, b, m, hbar)
        return (fE_below + fE_above) / 2.0

    elif E_J < V0_J:
        beta_sq = 2 * m * (V0_J - E_J) / hbar**2
        k2 = np.sqrt(beta_sq)
        if abs(k1*k2) < 1e-20:
            return 2.0
        fE = (np.cos(k1 * a) * np.cosh(k2 * b) -
              ((k1**2 - k2**2) / (2 * k1 * k2)) * np.sin(k1 * a) * np.sinh(k2 * b))
    else:
        gamma_sq = 2 * m * (E_J - V0_J) / hbar**2
        kappa = np.sqrt(gamma_sq)
        if abs(k1*kappa) < 1e-20:
            return 2.0
        fE = (np.cos(k1 * a) * np.cos(kappa * b) -
              ((k1**2 + kappa**2) / (2 * k1 * kappa)) * np.sin(k1 * a) * np.sin(kappa * b))

    return np.clip(fE, -1.5, 1.5)

# --- Calculation ---
allowed_k = []
allowed_E_eV = []
fE_values = []

for i, E_J in enumerate(energies_J):
    fE = calculate_fE(E_J, V0, a, b, m, hbar)
    fE_values.append(fE)

    if -1.0 <= fE <= 1.0:
        E_eV = energies_eV[i]
        k_val = np.arccos(fE) / L
        allowed_k.append(k_val)
        allowed_E_eV.append(E_eV)
        if abs(k_val) > 1e-9:
            allowed_k.append(-k_val)
            allowed_E_eV.append(E_eV)

# --- Plotting: f(E) vs E ---
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(energies_eV, fE_values, color='teal', lw=1.5)
ax1.axhline(1.0, color='red', linestyle='--', lw=1, label='f(E)=+1')
ax1.axhline(-1.0, color='blue', linestyle='--', lw=1, label='f(E)=-1')
ax1.set_xlabel('Energy E (eV)')
ax1.set_ylabel('f(E) = cos(kL)')
ax1.set_title(f'f(E) vs Energy for Kronig-Penney Model')
ax1.set_ylim(-1.5, 1.5)
ax1.grid(True)
ax1.legend()

# Highlight allowed bands
for j in range(len(energies_eV) - 1):
    if abs(fE_values[j]) <= 1.0:
        ax1.fill_between([energies_eV[j], energies_eV[j+1]], -1.5, 1.5, color='lightgrey', alpha=0.5)

st.pyplot(fig1)

# --- Plotting: E vs k ---
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.scatter(allowed_k, allowed_E_eV, s=4, color='navy')
pi_L = np.pi / L
ax2.axvline(pi_L, color='r', linestyle='--', lw=1.2, label='±π/L')
ax2.axvline(-pi_L, color='r', linestyle='--', lw=1.2)
ax2.set_xlim(-pi_L, pi_L)
ax2.set_ylim(0, E_max_eV)
ax2.set_xlabel('Wavevector k ($m^{-1}$)')
ax2.set_ylabel('Energy E (eV)')
ax2.set_title('Band Structure: E vs k')
ax2.grid(True)
ax2.legend()

st.pyplot(fig2)