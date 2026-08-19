from typing import Tuple
from signal_source import list_serial_ports, get_live_signal

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import serial

from scipy.ndimage import uniform_filter1d

def send_servo_command(ser, finger: int, position: int) -> None:
    """
    Send a close/open command to the servo controller.
    finger: 0-4 (index to pinky)
    position: 0 = open, 180 = closed
    """
    command = f"{finger},{position}\n"
    ser.write(command.encode())

# -----------------------
# Configuration / Page
# -----------------------
st.set_page_config(page_title="Comparative Physiology Platform", layout="wide")
st.title("Comparative Physiology Platform")


# -----------------------
# Sidebar / Controls
# -----------------------
st.sidebar.header("Simulation Controls")

def sidebar_controls() -> dict:
    """Collect sidebar inputs and return as a dict of parameters."""
    params = {}

    # --- Mode toggle ---
    params["mode"] = st.sidebar.radio(
        "Data Source",
        ["Simulated", "Live Sensor"]
    )

    st.sidebar.divider()

    # --- Live sensor options (only shown in live mode) ---
    if params["mode"] == "Live Sensor":
        available_ports = list_serial_ports()
        if available_ports:
            params["port"] = st.sidebar.selectbox("Serial Port", available_ports)
        else:
            st.sidebar.warning("No serial ports found. Is the ESP32 plugged in?")
            params["port"] = None
        params["n_samples"] = st.sidebar.slider("Samples to Collect", 200, 1000, 500)

    # --- Simulation controls (always shown) ---
    params["bpm"] = st.sidebar.slider("Simulated Heart Rate (BPM)", 40, 180, 75)
    params["noise_level"] = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.15)
    params["duration"] = st.sidebar.slider("Signal Duration (seconds)", 5, 30, 10)
    params["tissue_thickness"] = st.sidebar.slider("Simulated Tissue Thickness", 0.0, 10.0, 2.0)
    params["vessel_stiffness"] = st.sidebar.slider("Simulated Vessel Stiffness", 0.0, 1.0, 0.3)
    params["motion_artifact"] = st.sidebar.slider("Motion Artifact", 0.0, 1.0, 0.1)
    params["random_seed"] = st.sidebar.number_input("Random Seed", value=42, step=1)
    return params


params = sidebar_controls()


# -----------------------
# Signal simulation
# -----------------------
SAMPLING_RATE = 100  # samples per second


def make_time_array(duration: int, sampling_rate: int = SAMPLING_RATE) -> np.ndarray:
    """Return a time vector for the simulation."""
    return np.linspace(0, duration, int(duration * sampling_rate))


def simulate_signal(t: np.ndarray, bpm: float, noise_level: float, tissue_thickness: float,
                    vessel_stiffness: float, motion_artifact: float, seed: int = 42) -> np.ndarray:
    """Generate a synthetic PPG-like signal."""
    np.random.seed(int(seed))

    heart_rate_hz = bpm / 60.0
    base_wave = np.sin(2 * np.pi * heart_rate_hz * t)
    pulse_peaks = np.maximum(base_wave, 0) ** 4

    attenuation = np.exp(-0.18 * tissue_thickness)
    stiffness_effect = 1.0 - (0.7 * vessel_stiffness)
    motion = motion_artifact * 0.5 * np.sin(2 * np.pi * 0.35 * t)
    noise = noise_level * np.random.normal(size=len(t))

    signal = (pulse_peaks * attenuation * stiffness_effect) + motion + noise
    return signal


def compute_signal_quality(noise_level: float, motion_artifact: float, vessel_stiffness: float) -> str:
    penalty = (noise_level * 0.5) + (motion_artifact * 0.3) + (vessel_stiffness * 0.2)
    if penalty < 0.25:
        return "Good"
    elif penalty < 0.55:
        return "Moderate"
    else:
        return "Poor"


# -----------------------
# Processing / Detection
# -----------------------

def detect_peaks_and_bpm(signal: np.ndarray, t: np.ndarray, sampling_rate: int = SAMPLING_RATE,
                         smoothing_sigma: float = 2.0) -> Tuple[np.ndarray, float, np.ndarray]:
    """Return peak indices, detected BPM, and the smoothed signal."""
    smoothed = gaussian_filter1d(signal, sigma=smoothing_sigma)

    # Remove DC baseline by subtracting rolling mean so peaks center around zero
    baseline = uniform_filter1d(smoothed, size=int(sampling_rate * 1.5))
    signal_ac = smoothed - baseline

    peaks, _ = find_peaks(
        signal_ac,
        distance=int(sampling_rate * 0.5),   # min 0.5s between peaks = max 120 BPM
        prominence=signal_ac.std() * 0.5     # must stand out from noise floor
    )

    peak_times = t[peaks]
    if len(peak_times) >= 2:
        intervals = np.diff(peak_times)
        avg_interval = np.mean(intervals)
        detected_bpm = 60.0 / avg_interval
    else:
        detected_bpm = 0.0

    return peaks, detected_bpm, smoothed


# -----------------------
# Presentation / Plotting
# -----------------------

def make_dataframe(t: np.ndarray, signal: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"Time (s)": t, "Signal": signal})


def render_metrics(params: dict, detected_bpm: float) -> None:
    """Display top-line metrics in Streamlit columns."""
    noise_level = params.get("noise_level", 0.0)
    tissue_thickness = params.get("tissue_thickness", 0.0)
    vessel_stiffness = params.get("vessel_stiffness", 0.0)
    motion_artifact = params.get("motion_artifact", 0.0)
    target_bpm = params.get("bpm", 0)

    quality = compute_signal_quality(noise_level, motion_artifact, vessel_stiffness)
    bpm_error = abs(detected_bpm - target_bpm)

    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    col1.metric("Simulated BPM", target_bpm)
    col2.metric("Signal Quality", quality)
    col3.metric("Sampling Rate", f"{SAMPLING_RATE} Hz")
    col4.metric("Tissue Thickness", f"{tissue_thickness:.1f}")
    col5.metric("Vessel Stiffness", f"{vessel_stiffness:.2f}")
    col6.metric("Detected BPM", f"{detected_bpm:.1f}")
    col7.metric("BPM Error", f"{bpm_error:.2f}")


def plot_signal(t: np.ndarray, raw: np.ndarray, smoothed: np.ndarray, peaks: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, raw, alpha=0.35, label="Raw signal")
    ax.plot(t, smoothed, linewidth=2, label="Filtered signal")
    if len(peaks) > 0:
        ax.plot(t[peaks], smoothed[peaks], "o", markersize=8, label="Detected peaks")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Signal amplitude")
    ax.set_title("PPG Signal Processing")
    ax.legend()
    st.pyplot(fig)


# -----------------------
# Main
# -----------------------

if "live_data" not in st.session_state:
    st.session_state.live_data = []


def main() -> None:
    mode = params.get("mode", "Simulated")

    if mode == "Live Sensor" and params.get("port"):
        st.subheader("Live Sensor Feed")
        with st.spinner("Reading from sensor..."):
            try:
                signal = get_live_signal(
                    params["port"],
                    n_samples=params.get("n_samples", 500)
                )
                t = np.linspace(0, len(signal) / SAMPLING_RATE, len(signal))
            except ConnectionError as e:
                st.error(str(e))
                return
    else:
        st.subheader("Simulated Pulse Dashboard")
        t = make_time_array(params["duration"])
        signal = simulate_signal(
            t,
            bpm=params["bpm"],
            noise_level=params["noise_level"],
            tissue_thickness=params["tissue_thickness"],
            vessel_stiffness=params["vessel_stiffness"],
            motion_artifact=params["motion_artifact"],
            seed=params["random_seed"]
        )

    peaks, detected_bpm, smoothed = detect_peaks_and_bpm(signal, t)
    render_metrics(params, detected_bpm)
    df = make_dataframe(t, signal)
    plot_signal(t, signal, smoothed, peaks)

    if st.checkbox("Show raw simulated data"):
        st.dataframe(df)


main()