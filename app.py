import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const

# --- Page Title ---
st.title("Kronig-Penney Model Visualizer")
st.markdown("Numerical solution and band structure for a 1D Kronig-Penney potential.")

# --- Display Local Image ---
st.sidebar.header("MM 226 - Supervised Learning Project")
try:
    img = Image.open("IITB Logo Black (2).jpg")
    st.sidebar.image(img, caption='IIT Bombay' , use_container_width=True)
except FileNotFoundError:
    st.sidebar.warning("'diagram.png' not found in app folder. Please ensure the image file is present.")

# --- Constants ---
hbar = const.hbar
m = const.m_e
e = const.e

# --- User Inputs ---
st.sidebar.header("Model Parameters")
a_angstrom = st.sidebar.slider("Well width a (in Å)", min_value=1.0, max_value=10.0, value=5.0, step=1.0)
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
        if abs(k1 * k2) < 1e-20:
            return 2.0
        return np.clip(
            np.cos(k1 * a) * np.cosh(k2 * b) -
            ((k1**2 - k2**2) / (2 * k1 * k2)) * np.sin(k1 * a) * np.sinh(k2 * b),
            -1.5, 1.5
        )
    else:
        gamma_sq = 2 * m * (E_J - V0_J) / hbar**2
        kappa = np.sqrt(gamma_sq)
        if abs(k1 * kappa) < 1e-20:
            return 2.0
        return np.clip(
            np.cos(k1 * a) * np.cos(kappa * b) -
            ((k1**2 + kappa**2) / (2 * k1 * kappa)) * np.sin(k1 * a) * np.sin(kappa * b),
            -1.5, 1.5
        )

# --- Calculate f(E) and Allowed Bands ---
allowed_k = []
allowed_E_eV = []
fE_values = []
for i, E_J in enumerate(energies_J):
    fE = calculate_fE(E_J, V0, a, b, m, hbar)
    fE_values.append(fE)
    if -1.0 <= fE <= 1.0:
        E_eV = energies_eV[i]
        k_val = np.arccos(fE) / L
        allowed_k.extend((k_val, -k_val) if abs(k_val) > 1e-9 else (k_val,))
        allowed_E_eV.extend((E_eV, E_eV) if abs(k_val) > 1e-9 else (E_eV,))

# --- Plot f(E) vs E ---
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(energies_eV, fE_values, lw=1.5)
ax1.axhline(1.0, linestyle='--', lw=1)
ax1.axhline(-1.0, linestyle='--', lw=1)
ax1.set(xlabel='Energy E (eV)', ylabel='f(E) = cos(kL)', ylim=(-1.5, 1.5))
ax1.grid(True)
for j in range(len(energies_eV) - 1):
    if abs(fE_values[j]) <= 1.0:
        ax1.fill_between([energies_eV[j], energies_eV[j+1]], -1.5, 1.5, alpha=0.3)
st.pyplot(fig1)

# --- Plot Band Structure E vs k ---
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.scatter(allowed_k, allowed_E_eV, s=4)
pi_L = np.pi / L
ax2.axvline(pi_L, linestyle='--', lw=1)
ax2.axvline(-pi_L, linestyle='--', lw=1)
ax2.set(xlim=(-pi_L, pi_L), ylim=(0, E_max_eV), xlabel='Wavevector k (m⁻¹)', ylabel='Energy E (eV)')
ax2.grid(True)
st.pyplot(fig2)
