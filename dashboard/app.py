import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title='Comparative Physiology Platform', layout='wide')

st.title('Comparative Physiology Platform')
st.subheader('Simulated Pulse Dashboard')

# Sidebar controls
st.sidebar.header('Simulation Controls')

bpm = st.sidebar.slider("Simulated Heart Rate (BPM)", 40, 180, 75)
noise_level = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.15)
duration = st.sidebar.slider("Signal Duration (seconds)", 5, 30, 10)


# Sampling setup
sampling_rate = 100  # samples per second
t = np.linspace(0, duration, duration * sampling_rate)

# Convert BPM to Hz
heart_rate_hz = bpm / 60

# Fake pulse waveform
base_wave = np.sin(2 * np.pi * heart_rate_hz * t)

# Add sharper pulse peaks
pulse_peaks = np.maximum(base_wave, 0) ** 4

# Add noise
noise = noise_level * np.random.normal(size=len(t))

# Final fake signal
signal = pulse_peaks + noise

# Create dataframe
df = pd.DataFrame({
    "Time (s)": t,
    "Signal": signal
})

# Display metrics
col1, col2, col3 = st.columns(3)

col1.metric("Simulated BPM", bpm)

if noise_level < 0.25:
    quality = "Good"
elif noise_level < 0.6:
    quality = "Moderate"
else:
    quality = "Poor"

col2.metric("Signal Quality", quality)
col3.metric("Sampling Rate", f"{sampling_rate} Hz")

# Plot waveform
fig, ax = plt.subplots()
ax.plot(df["Time (s)"], df["Signal"])
ax.set_xlabel("Time (s)")
ax.set_ylabel("Signal amplitude")
ax.set_title("Simulated PPG Pulse Waveform")

st.pyplot(fig)

# Show raw data optionally
if st.checkbox("Show raw simulated data"):
    st.dataframe(df)