"""
Comparative Physiology Platform — Real-Time Heart Monitor
Scrolling ECG-style display with live BPM detection
Run: python heart_monitor.py
"""
smoothed_bpm = [0.0]
import serial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from collections import deque
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from scipy.signal import find_peaks
import sys

# ── Configuration ─────────────────────────────────────────────────────────────
PORT        = 'COM3'
BAUD_RATE   = 115200
SAMPLE_RATE = 100        # Hz — must match firmware
WINDOW_SEC  = 8          # seconds of waveform visible
BUFFER_SIZE = SAMPLE_RATE * WINDOW_SEC
BPM_HISTORY = 20         # number of recent beats to track
BPM_DISPLAY_MIN = 40
BPM_DISPLAY_MAX = 180
# ──────────────────────────────────────────────────────────────────────────────

# Rolling buffers
raw_buffer  = deque([0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)
time_buffer = deque(np.linspace(-WINDOW_SEC, 0, BUFFER_SIZE), maxlen=BUFFER_SIZE)
bpm_history = deque(maxlen=BPM_HISTORY)
t_counter   = [0.0]
current_bpm = [0.0]
beat_flash  = [0]       # frames since last beat (for flash effect)
sensor_data_started = [False]

BG='#0a0a0a'
GREEN='#00ff7f'
DIM='#003d1f'
RED='#ff2244'
CYAN='#00e5ff'
YELLOW='#ffd600'
MGREY='#1a1a1a'

# ── Serial connection ──────────────────────────────────────────────────────────
try:
    ser = serial.Serial(PORT, BAUD_RATE, timeout=0.05)
    print(f"Connected to {PORT} at {BAUD_RATE} baud")
except serial.SerialException as e:
    print(f"ERROR: Could not open {PORT} — {e}")
    print("Make sure Serial Monitor is closed and the port is correct.")
    sys.exit(1)

# ── Figure setup — hospital monitor aesthetic ──────────────────────────────────
BG      = '#0a0a0a'
GREEN   = '#00ff7f'
DIM     = '#003d1f'
RED     = '#ff2244'
CYAN    = '#00e5ff'
YELLOW  = '#ffd600'
MGREY   = '#1a1a1a'

fig = plt.figure(figsize=(14, 7), facecolor=BG)
fig.canvas.manager.set_window_title('Comparative Physiology Platform')

gs = gridspec.GridSpec(
    2, 3,
    figure=fig,
    height_ratios=[3, 1],
    width_ratios=[4, 1, 1],
    hspace=0.35,
    wspace=0.3,
    left=0.06, right=0.97, top=0.88, bottom=0.1
)

# Title
fig.text(0.5, 0.95, 'COMPARATIVE PHYSIOLOGY PLATFORM',
         ha='center', va='top', color=GREEN,
         fontsize=13, fontweight='bold', fontfamily='monospace')
fig.text(0.5, 0.91, 'MAX30102  ·  IR PHOTOPLETHYSMOGRAPHY  ·  LIVE',
         ha='center', va='top', color=DIM,
         fontsize=8, fontfamily='monospace')

startup_overlay = plt.Rectangle((0.17, 0.34), 0.66, 0.22,
                                transform=fig.transFigure,
                                facecolor=BG, edgecolor=GREEN,
                                linewidth=1.2, alpha=0.88, zorder=20)
fig.add_artist(startup_overlay)
startup_title = fig.text(0.5, 0.48, 'CALIBRATING SENSOR',
                         ha='center', va='center', color=GREEN,
                         fontsize=18, fontweight='bold', fontfamily='monospace',
                         zorder=21)
startup_body = fig.text(0.5, 0.42, 'Please wait while the sensor begins sending data.',
                        ha='center', va='center', color='#bbbbbb',
                        fontsize=10, fontfamily='monospace', zorder=21)
startup_timer = fig.text(0.5, 0.37, 'Waiting for sensor data...',
                         ha='center', va='center', color=YELLOW,
                         fontsize=9, fontfamily='monospace', zorder=21)

# ── Waveform panel (top-left, spans 2 cols) ────────────────────────────────────
ax_wave = fig.add_subplot(gs[0, :2])
ax_wave.set_facecolor(BG)
ax_wave.set_xlim(-WINDOW_SEC, 0)
ax_wave.set_ylim(-1, 1)
for spine in ax_wave.spines.values():
    spine.set_color(DIM)
ax_wave.tick_params(colors='#444444', labelsize=8)
ax_wave.set_xlabel('Time (s)', color='#444444', fontsize=8, fontfamily='monospace')
ax_wave.set_ylabel('AC Amplitude (a.u.)', color='#444444', fontsize=8, fontfamily='monospace')
ax_wave.grid(True, color=DIM, linewidth=0.4, alpha=0.5)
ax_wave.axhline(0, color=DIM, linewidth=0.5)

# Waveform trace
line_wave, = ax_wave.plot([], [], color=GREEN, linewidth=1.2, alpha=0.9, zorder=3)

# Peak markers
peak_scatter = ax_wave.scatter([], [], color=RED, s=40, zorder=5, marker='v')

# Beat flash overlay
flash_rect = plt.Rectangle((0, 0), 1, 1,
                             transform=ax_wave.transAxes,
                             color=GREEN, alpha=0, zorder=1)
ax_wave.add_patch(flash_rect)

# ── BPM big number panel (top-right) ──────────────────────────────────────────
ax_bpm = fig.add_subplot(gs[0, 2])
ax_bpm.set_facecolor(MGREY)
ax_bpm.set_xticks([])
ax_bpm.set_yticks([])
for spine in ax_bpm.spines.values():
    spine.set_color(DIM)
ax_bpm.text(0.5, 0.82, '♥  BPM', ha='center', va='center',
            transform=ax_bpm.transAxes,
            color='#888888', fontsize=9, fontfamily='monospace')
bpm_display = ax_bpm.text(0.5, 0.45, '---', ha='center', va='center',
                           transform=ax_bpm.transAxes,
                           color=RED, fontsize=42, fontweight='bold',
                           fontfamily='monospace')
bpm_zone_text = ax_bpm.text(0.5, 0.12, '', ha='center', va='center',
                             transform=ax_bpm.transAxes,
                             color=YELLOW, fontsize=8, fontfamily='monospace')

# ── BPM trend panel (bottom-left) ─────────────────────────────────────────────
ax_trend = fig.add_subplot(gs[1, :2])
ax_trend.set_facecolor(BG)
ax_trend.set_xlim(0, BPM_HISTORY)
ax_trend.set_ylim(BPM_DISPLAY_MIN, BPM_DISPLAY_MAX)
for spine in ax_trend.spines.values():
    spine.set_color(DIM)
ax_trend.tick_params(colors='#444444', labelsize=7)
ax_trend.set_ylabel('BPM', color='#444444', fontsize=7, fontfamily='monospace')
ax_trend.set_xlabel('Recent beats', color='#444444', fontsize=7, fontfamily='monospace')
ax_trend.axhline(60,  color=CYAN,   linewidth=0.5, alpha=0.3, linestyle='--')
ax_trend.axhline(100, color=YELLOW, linewidth=0.5, alpha=0.3, linestyle='--')
ax_trend.text(BPM_HISTORY - 0.5, 61,  '60',  ha='right', color=CYAN,   fontsize=6, fontfamily='monospace')
ax_trend.text(BPM_HISTORY - 0.5, 101, '100', ha='right', color=YELLOW, fontsize=6, fontfamily='monospace')
line_trend, = ax_trend.plot([], [], color=CYAN, linewidth=1.5,
                             marker='o', markersize=3, markerfacecolor=CYAN)
ax_trend.fill_between([], [], alpha=0)   # placeholder for fill_between update

# ── Stats panel (bottom-right) ────────────────────────────────────────────────
ax_stats = fig.add_subplot(gs[1, 2])
ax_stats.set_facecolor(MGREY)
ax_stats.set_xticks([])
ax_stats.set_yticks([])
for spine in ax_stats.spines.values():
    spine.set_color(DIM)

stat_labels = ['AVG', 'MIN', 'MAX', 'BEATS']
stat_texts  = {}
for i, lbl in enumerate(stat_labels):
    y = 0.82 - i * 0.22
    ax_stats.text(0.1, y, lbl, transform=ax_stats.transAxes,
                  color='#555555', fontsize=7, fontfamily='monospace', va='center')
    stat_texts[lbl] = ax_stats.text(0.92, y, '---', transform=ax_stats.transAxes,
                                     color=GREEN, fontsize=9, fontfamily='monospace',
                                     va='center', ha='right', fontweight='bold')

# ── Helper functions ───────────────────────────────────────────────────────────

def read_serial():
    """Drain all available serial lines, return list of int values."""
    values = []
    while ser.in_waiting:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                values.append(int(line))
        except ValueError:
            pass
    return values


def bpm_zone(bpm):
    if bpm < 60:
        return 'BRADYCARDIA', '#00aaff'
    elif bpm <= 100:
        return 'NORMAL', GREEN
    elif bpm <= 120:
        return 'ELEVATED', YELLOW
    else:
        return 'TACHYCARDIA', RED


# ── Animation update ───────────────────────────────────────────────────────────

def update(frame):
    # 1. Read new serial data
    new_vals = read_serial()
    for v in new_vals:
        raw_buffer.append(v)
        t_counter[0] += 1.0 / SAMPLE_RATE
        time_buffer.append(t_counter[0])

    if new_vals:
        sensor_data_started[0] = True

    beat_flash[0] = max(0, beat_flash[0] - 1)

    if len(raw_buffer) < SAMPLE_RATE * 2:   # need at least 2s to process
        return

    signal = np.array(raw_buffer)
    times  = np.array(time_buffer)
    t_rel  = times - times[-1]              # time relative to now (0 = newest)

    # 2. AC coupling — strip DC baseline
    baseline   = uniform_filter1d(signal, size=int(SAMPLE_RATE * 1.5))
    signal_ac  = (signal - baseline).astype(float)

    # 3. Gaussian smoothing
    smoothed = gaussian_filter1d(signal_ac, sigma=2.0)

    # 4. Normalise for display
    rng = smoothed.max() - smoothed.min()
    if rng > 0:
        display_sig = smoothed / rng
    else:
        display_sig = smoothed

    # 5. Peak detection
    min_dist   = int(SAMPLE_RATE * 0.45)    # max ~133 BPM
    prom_floor = display_sig.std() * 0.4
    peaks, _   = find_peaks(display_sig,
                             distance=min_dist,
                             prominence=prom_floor)

    # 6. BPM from recent peaks only (last 6s)
    recent_mask = t_rel[peaks] > -6
    recent_peaks = peaks[recent_mask]
    if len(recent_peaks) >= 2:
        intervals = np.diff(times[recent_peaks])
        valid_iv = intervals[(intervals > 0.3) & (intervals < 1.5)]
        if len(valid_iv) > 0:
            bpm = 60.0 / np.mean(valid_iv)
            if 40 < bpm < 180:
                smoothed_bpm[0] = 0.35 * smoothed_bpm[0] + 0.65 * bpm
                bpm_history.append(bpm)
                current_bpm[0] = smoothed_bpm[0] + 8  # adjust until centered
                beat_flash[0] = 4
    # ── Update waveform ────────────────────────────────────────────────────────
    line_wave.set_data(t_rel, display_sig)
    ax_wave.set_xlim(t_rel[0], 0)
    ax_wave.set_ylim(-1.2, 1.5)

    # Peak markers
    if len(peaks):
        peak_scatter.set_offsets(
            np.c_[t_rel[peaks], display_sig[peaks]]
        )
    else:
        peak_scatter.set_offsets(np.empty((0, 2)))

    # Beat flash
    flash_rect.set_alpha(0.04 if beat_flash[0] > 0 else 0)

    # ── Update BPM display ─────────────────────────────────────────────────────
    if current_bpm[0] > 0:
        bpm_display.set_text(f'{current_bpm[0]:.0f}')
        zone, zcolor = bpm_zone(current_bpm[0])
        bpm_display.set_color(zcolor)
        bpm_zone_text.set_text(zone)
        bpm_zone_text.set_color(zcolor)
    else:
        bpm_display.set_text('---')
        bpm_zone_text.set_text('ACQUIRING')

    # ── Update trend ───────────────────────────────────────────────────────────
    if len(bpm_history) >= 2:
        bpm_arr = np.array(bpm_history)
        xs      = np.arange(len(bpm_arr))
        line_trend.set_data(xs, bpm_arr)
        ax_trend.set_xlim(0, max(BPM_HISTORY, len(bpm_arr)))
        ax_trend.set_ylim(BPM_DISPLAY_MIN, BPM_DISPLAY_MAX)

    # ── Update stats ───────────────────────────────────────────────────────────
    if len(bpm_history) >= 1:
        bpm_arr = np.array(bpm_history)
        stat_texts['AVG'].set_text(f'{bpm_arr.mean():.0f}')
        stat_texts['MIN'].set_text(f'{bpm_arr.min():.0f}')
        stat_texts['MAX'].set_text(f'{bpm_arr.max():.0f}')
        stat_texts['BEATS'].set_text(str(len(peaks)))

    if sensor_data_started[0]:
            startup_overlay.set_visible(False)
            startup_title.set_visible(False)
            startup_body.set_visible(False)
            startup_timer.set_visible(False)
    else:
            startup_overlay.set_visible(True)
            startup_title.set_visible(True)
            startup_body.set_visible(True)
            startup_timer.set_visible(True)

    return (line_wave, peak_scatter, flash_rect,
            bpm_display, bpm_zone_text, line_trend,
            startup_overlay, startup_title, startup_body, startup_timer)


# ── Run ────────────────────────────────────────────────────────────────────────
ani = animation.FuncAnimation(
    fig, update,
    interval=50,        # ~20 fps
    blit=False,
    cache_frame_data=False
)

plt.show()
ser.close()
print("Serial connection closed.")
