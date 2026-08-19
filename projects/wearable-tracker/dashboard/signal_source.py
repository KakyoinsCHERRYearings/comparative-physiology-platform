import serial
import serial.tools.list_ports
import numpy as np
from collections import deque
import time


def list_serial_ports() -> list:
    """Return available COM ports as a list of strings."""
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]


def get_live_signal(port: str, baud: int = 115200,
                    n_samples: int = 500,
                    sampling_rate: int = 100) -> np.ndarray:
    """
    Read n_samples from the ESP32 over serial.
    Returns a numpy array of raw sensor values.
    """
    ser = serial.Serial(port, baud, timeout=2)
    time.sleep(2)  # let connection settle

    buffer = deque(maxlen=n_samples)
    failures = 0

    while len(buffer) < n_samples:
        try:
            line = ser.readline().decode('utf-8').strip()
            value = float(line)
            buffer.append(value)
            failures = 0
        except Exception:
            failures += 1
            if failures > 50:
                ser.close()
                raise ConnectionError(
                    f"Too many bad readings from {port}. "
                    "Check wiring and baud rate."
                )

    ser.close()
    return np.array(buffer)