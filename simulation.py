import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import hsv_to_rgb

# Configure Matplotlib globally to use your Neon Green theme on Dark Background
plt.style.use('dark_background')
plt.rcParams['text.color'] = '#00ff00'          # Neon green text
plt.rcParams['axes.labelcolor'] = '#00ff00'     # Neon green X/Y axis labels
plt.rcParams['xtick.color'] = '#00ff00'         # Neon green X ticks
plt.rcParams['ytick.color'] = '#00ff00'         # Neon green Y ticks
plt.rcParams['axes.titlecolor'] = '#00ff00'     # Neon green Title
plt.rcParams['axes.facecolor'] = '#000000'      # Pure black plot canvas
plt.rcParams['figure.facecolor'] = '#000000'    # Pure black outer window
plt.rcParams['axes.edgecolor'] = '#333333'      # Muted gray for axis borders

# ==============================================================================
# 1. PHYSICAL CONSTANTS (Atomic Units)
# ==============================================================================
hbar = 1.0

# Changing this decreases the probability of tunneling

m_e = 1.0

Å = 1.8897261246257702      
eV = 0.03674932217565499    

# ==============================================================================
# 2. SIMULATION GRID & POTENTIAL SETUP
# ==============================================================================
N = 512
extent = 700 * Å
x = np.linspace(-extent/2, extent/2, N, endpoint=False)
y = np.linspace(-extent/2, extent/2, N, endpoint=False)
X, Y = np.meshgrid(x, y)

# Define interaction potential (Double Barrier)
a = 3 * Å      
b = 5 * Å      
V0 = 1.7 * eV  

# First barrier: 0 < x < a
V = np.where((X > 0) & (X < a), V0, 0.0)

# Second barrier: a+b < x < 2a+b
V = np.where((X > a + b) & (X < a + b + a), V + V0, V)

# ==============================================================================
# 3. INITIAL WAVEFUNCTION SETUP (Single Centered Particle)
# ==============================================================================
E0 = 0.8 * eV   
σ = 25.0 * Å    
k_x0 = np.sqrt(2 * m_e * E0) / hbar

# One particle traveling perfectly along the horizontal center (Y = 0)
psi = np.exp(-1/(4 * σ**2) * ((X + 150*Å)**2 + Y**2)) / np.sqrt(2*np.pi * σ**2) * np.exp(1j * k_x0 * X)

# ==============================================================================
# 4. SPLIT-STEP FOURIER ENGINE & RUNTIME PROGRESS BAR
# ==============================================================================
total_time = 4000
num_steps = 8000
dt = total_time / num_steps
store_steps = 800
steps_per_frame = num_steps // store_steps

dx = x[1] - x[0]
dy = y[1] - y[0]
kx = 2 * np.pi * np.fft.fftfreq(N, d=dx)
ky = 2 * np.pi * np.fft.fftfreq(N, d=dy)
Kx, Ky = np.meshgrid(kx, ky)

T = (hbar**2 / (2 * m_e)) * (Kx**2 + Ky**2)

exp_V = np.exp(-1j * V * (dt / 2.0) / hbar)
exp_T = np.exp(-1j * T * dt / hbar)

frames = []
frames.append(psi.copy())

print("Evaluating time evolution steps:")
for step in range(1, num_steps + 1):
    psi *= exp_V
    psi_k = np.fft.fft2(psi)
    psi_k *= exp_T
    psi = np.fft.ifft2(psi_k)
    psi *= exp_V
    
    if step % steps_per_frame == 0:
        frames.append(psi.copy())
    
    # Live rolling terminal progress bar
    if step % 80 == 0 or step == num_steps:
        percent = (step / num_steps) * 100
        bar_length = int(step / num_steps * 40)
        progress_bar = "█" * bar_length + "-" * (40 - bar_length)
        sys.stdout.write(f"\r[{progress_bar}] {percent:.1f}% Simulation Complete")
        sys.stdout.flush()

print("\nSimulation finished successfully. Rendering animation plot...")

# ==============================================================================
# 4B. EXPORT PROMPT
# ==============================================================================
print("\n" + "="*60)
print("Export Options:")
print("  1. GIF (Pillow)")
print("  2. MP4 (FFmpeg)")
print("  3. No Export")
print("="*60)
export_choice = input("Select export format (1/2/3): ").strip()

# ==============================================================================
# 5. HSV WAVEFUNCTION VISUALIZATION (Dark Theme Engine)
# ==============================================================================
fig, ax = plt.subplots(figsize=(8, 8))
max_amp = np.max(np.abs(frames[0])) * 0.3

def psi_to_rgb_dark_bg(psi_matrix, max_amplitude=max_amp):
    """Maps complex fields to a dark background canvas.
    Phase maps to Hue. Amplitude maps to Value/Brightness (0% Amp = Pure Black)."""
    amplitude = np.abs(psi_matrix)
    phase = np.angle(psi_matrix)
    
    H = (phase + np.pi) / (2 * np.pi)              
    S = np.ones_like(H)                             # 100% Saturation for vibrant colors
    V = np.minimum(amplitude / max_amplitude, 1.0) # Low amplitude = Dim/Black
    
    hsv = np.stack([H, S, V], axis=-1)
    return hsv_to_rgb(hsv)

spatial_extent_Å = [-extent/(2*Å), extent/(2*Å), -extent/(2*Å), extent/(2*Å)]
img = ax.imshow(psi_to_rgb_dark_bg(frames[0]), extent=spatial_extent_Å, origin='lower')

# ------------------------------------------------------------------------------
# POTENTIAL BARRIER OVERLAY (High Contrast against Dark Background)
# ------------------------------------------------------------------------------
b1_l, b1_r = 0, a/Å
b2_l, b2_r = (a+b)/Å, (2*a+b)/Å

# Draw the barriers as high-visibility translucent white blocks with explicit edges
ax.axvspan(b1_l, b1_r, color='#00ff00', alpha=0.5, linewidth=0.01, label='Barrier 1 (3 Å Wide)')

ax.axvspan(b2_l, b2_r, color='#00ff00', alpha=0.5, linewidth=0.01, label='Barrier 2 (3 Å Wide)')

# Viewport
ax.set_xlim(-300, 300)
ax.set_ylim(-300, 300)

ax.set_xlabel("Spatial Dimension X ($Å$)")
ax.set_ylabel("Spatial Dimension Y ($Å$)")
ax.set_title("Quantum Tunneling (Single Barrier)")

# Style the Legend to maintain the Neon Green color aesthetic
leg = ax.legend(loc='upper right', framealpha=0.2, edgecolor='#333333')
for text in leg.get_texts():
    text.set_color('#00ff00')

def animate_frame(idx):
    img.set_data(psi_to_rgb_dark_bg(frames[idx]))
    return [img]

# blit=False guarantees overlays like axvspan render immediately without needing resizes
ani = animation.FuncAnimation(
    fig, 
    animate_frame, 
    frames=len(frames), 
    interval=16, # 1000ms/intervalms to get fps
    blit=False
)

# ==============================================================================
# EXPORT BASED ON USER CHOICE
# ==============================================================================
if export_choice == '1':
    print("Encoding and saving animation to GIF... (This may take a minute)")
    ani.save('quantum_tunneling.gif', writer='pillow', fps=60)
    print("Export complete! Saved as 'quantum_tunneling.gif'")
elif export_choice == '2':
    print("Encoding and saving animation to MP4... (This may take a minute)")
    ani.save(
        'quantum_tunneling.mp4', 
        writer='ffmpeg', 
        fps=60, 
        extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p']
    )
    print("Export complete! Saved as 'quantum_tunneling.mp4'")
elif export_choice == '3':
    print("Skipping export.")
else:
    print("Invalid choice. Skipping export.")

plt.show()
