"""
Real-time AI noise reduction -- GUI

A PySide6 front-end around the same DSP pipeline as realtime_denoise.py:
input/output device pickers, live parameter controls, input/output VU
meters, and a bypass button for instant A/B comparison against the raw
signal.

Install:
    pip install PySide6 sounddevice numpy scipy pyrnnoise

Run:
    python realtime_denoise_gui.py

(realtime_denoise.py must be in the same folder -- this GUI imports its
DSP classes rather than duplicating them.)
"""

import sys
from collections import deque

import numpy as np
import sounddevice as sd

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QComboBox, QPushButton, QLabel, QSlider, QDoubleSpinBox, QCheckBox,
    QGroupBox, QGridLayout, QMessageBox,
)
from PySide6.QtGui import QPainter, QColor, QLinearGradient

from realtime_denoise import (
    DeEmphasis, AGC, BandpassFilter, RNNoiseBackend, DPDFNetBackend,
    soft_limit, MinGainFloor, MODE_PRESETS, SAMPLE_RATE, FRAME_SIZE,
)


def amplitude_to_db(x: float) -> float:
    return 20.0 * np.log10(max(x, 1e-6))


class VUMeter(QWidget):
    """Horizontal VU meter: current level bar + a white peak-hold tick,
    scaled -60dB (silence) to 0dB (full scale)."""

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.label = label
        self.level_db = -60.0
        self.peak_db = -60.0
        self.peak_hold_counter = 0
        self.setMinimumHeight(30)
        self.setMinimumWidth(220)

    def set_level(self, level_db: float):
        self.level_db = level_db
        if level_db > self.peak_db:
            self.peak_db = level_db
            self.peak_hold_counter = 20
        elif self.peak_hold_counter > 0:
            self.peak_hold_counter -= 1
        else:
            self.peak_db = max(self.peak_db - 0.5, self.level_db)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(20, 20, 20))

        def db_to_x(db):
            frac = max(0.0, min(1.0, (db + 60.0) / 60.0))
            return int(frac * w)

        level_x = db_to_x(self.level_db)
        grad = QLinearGradient(0, 0, w, 0)
        grad.setColorAt(0.0, QColor(60, 200, 60))
        grad.setColorAt(0.75, QColor(230, 200, 40))
        grad.setColorAt(0.92, QColor(220, 60, 40))
        p.fillRect(0, 0, level_x, h, grad)

        peak_x = db_to_x(self.peak_db)
        p.fillRect(max(0, peak_x - 2), 0, 2, h, QColor(255, 255, 255))

        p.setPen(QColor(230, 230, 230))
        p.drawText(6, h - 9, f"{self.label}   {self.level_db:5.1f} dB")


class Pipeline:
    """Owns the DSP chain plus level metering for one running stream.

    Note on thread safety: the GUI thread mutates these parameters (simple
    float/bool attributes, or an occasional full bandpass-filter rebuild)
    while the audio callback -- running on PortAudio's own thread -- reads
    them concurrently. There's no lock. Worst case from a mistimed read is
    an occasional tiny click, never a crash; that trade-off is worth it here
    for controls that feel instant.
    """

    def __init__(self, backend_name: str, model: str = "dpdfnet2_48khz_hr"):
        self.deemph = DeEmphasis(coeff=0.95)
        self.bandpass = BandpassFilter(300.0, 3000.0, SAMPLE_RATE)
        self.bandpass.enabled = False
        self.agc = AGC(sample_rate=SAMPLE_RATE)
        self.strength = 1.0
        self.min_gain = MinGainFloor(SAMPLE_RATE)
        self.bypass = False
        self.engine = self._make_backend(backend_name, model)
        self.in_level = -60.0
        self.out_level = -60.0

    @staticmethod
    def _make_backend(name, model):
        if name == "rnnoise":
            return RNNoiseBackend()
        elif name == "dpdfnet":
            return DPDFNetBackend(model=model)
        raise ValueError(f"Unknown backend: {name}")

    def process_frame(self, mono: np.ndarray) -> np.ndarray:
        in_peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
        self.in_level = amplitude_to_db(in_peak)

        if self.bypass:
            out = mono
        else:
            chunk = self.deemph.process(mono)
            chunk = self.bandpass.process(chunk)
            chunk = self.agc.process(chunk)
            pre_denoise = chunk
            chunk = self.engine.process(chunk)
            chunk = self.min_gain.process(pre_denoise, chunk)
            if self.strength < 1.0:
                chunk = self.strength * chunk + (1.0 - self.strength) * pre_denoise
            out = soft_limit(chunk)

        out_peak = float(np.max(np.abs(out))) if len(out) else 0.0
        self.out_level = amplitude_to_db(out_peak)
        return out


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Realtime AI Noise Reduction")
        self.stream = None
        self.pipeline = None
        self.in_buf = deque()
        self.out_buf = deque()

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # --- Devices / backend / mode ---
        dev_box = QGroupBox("Devices")
        dev_form = QFormLayout(dev_box)
        self.in_combo = QComboBox()
        self.out_combo = QComboBox()
        self._populate_devices()
        dev_form.addRow("Input:", self.in_combo)
        dev_form.addRow("Output:", self.out_combo)

        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["rnnoise", "dpdfnet"])
        dev_form.addRow("Backend:", self.backend_combo)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["none", "nbfm", "ssb"])
        self.mode_combo.currentTextChanged.connect(self.apply_mode_preset)
        dev_form.addRow("Mode preset:", self.mode_combo)

        layout.addWidget(dev_box)

        # --- Processing controls ---
        ctrl_box = QGroupBox("Processing")
        grid = QGridLayout(ctrl_box)
        row = 0

        self.deemph_check = QCheckBox("De-emphasis")
        self.deemph_check.setChecked(True)
        self.deemph_check.toggled.connect(self.on_deemph_toggled)
        self.deemph_spin = QDoubleSpinBox()
        self.deemph_spin.setRange(0.0, 0.99)
        self.deemph_spin.setSingleStep(0.01)
        self.deemph_spin.setValue(0.95)
        self.deemph_spin.valueChanged.connect(self.on_deemph_changed)
        grid.addWidget(self.deemph_check, row, 0)
        grid.addWidget(self.deemph_spin, row, 1)
        row += 1

        self.bandpass_check = QCheckBox("Bandpass")
        self.bandpass_check.toggled.connect(self.on_bandpass_toggled)
        self.bp_low_spin = QDoubleSpinBox()
        self.bp_low_spin.setRange(20.0, 10000.0)
        self.bp_low_spin.setValue(300.0)
        self.bp_high_spin = QDoubleSpinBox()
        self.bp_high_spin.setRange(200.0, 20000.0)
        self.bp_high_spin.setValue(3000.0)
        self.bp_low_spin.valueChanged.connect(self.on_bandpass_changed)
        self.bp_high_spin.valueChanged.connect(self.on_bandpass_changed)
        grid.addWidget(self.bandpass_check, row, 0)
        bp_row = QHBoxLayout()
        bp_row.addWidget(QLabel("Low:"))
        bp_row.addWidget(self.bp_low_spin)
        bp_row.addWidget(QLabel("High:"))
        bp_row.addWidget(self.bp_high_spin)
        bp_widget = QWidget()
        bp_widget.setLayout(bp_row)
        grid.addWidget(bp_widget, row, 1)
        row += 1

        self.agc_check = QCheckBox("AGC")
        self.agc_check.setChecked(True)
        self.agc_check.toggled.connect(self.on_agc_toggled)
        self.agc_target_spin = QDoubleSpinBox()
        self.agc_target_spin.setRange(0.01, 0.9)
        self.agc_target_spin.setSingleStep(0.01)
        self.agc_target_spin.setValue(0.15)
        self.agc_target_spin.valueChanged.connect(self.on_agc_target_changed)
        grid.addWidget(self.agc_check, row, 0)
        agc_row = QHBoxLayout()
        agc_row.addWidget(QLabel("Target RMS:"))
        agc_row.addWidget(self.agc_target_spin)
        agc_widget = QWidget()
        agc_widget.setLayout(agc_row)
        grid.addWidget(agc_widget, row, 1)
        row += 1

        grid.addWidget(QLabel("Strength (wet/dry):"), row, 0)
        self.strength_slider = QSlider(Qt.Horizontal)
        self.strength_slider.setRange(0, 100)
        self.strength_slider.setValue(100)
        self.strength_slider.valueChanged.connect(self.on_strength_changed)
        self.strength_label = QLabel("1.00")
        strength_row = QHBoxLayout()
        strength_row.addWidget(self.strength_slider)
        strength_row.addWidget(self.strength_label)
        strength_widget = QWidget()
        strength_widget.setLayout(strength_row)
        grid.addWidget(strength_widget, row, 1)
        row += 1

        grid.addWidget(QLabel("Min gain floor (dB):"), row, 0)
        self.min_gain_slider = QSlider(Qt.Horizontal)
        self.min_gain_slider.setRange(-60, 0)
        self.min_gain_slider.setValue(-60)
        self.min_gain_slider.valueChanged.connect(self.on_min_gain_changed)
        self.min_gain_label = QLabel("-60 (off)")
        min_gain_row = QHBoxLayout()
        min_gain_row.addWidget(self.min_gain_slider)
        min_gain_row.addWidget(self.min_gain_label)
        min_gain_widget = QWidget()
        min_gain_widget.setLayout(min_gain_row)
        grid.addWidget(min_gain_widget, row, 1)
        row += 1

        layout.addWidget(ctrl_box)

        # --- VU meters ---
        vu_box = QGroupBox("Levels")
        vu_layout = QVBoxLayout(vu_box)
        self.in_meter = VUMeter("IN")
        self.out_meter = VUMeter("OUT")
        vu_layout.addWidget(self.in_meter)
        vu_layout.addWidget(self.out_meter)
        layout.addWidget(vu_box)

        # --- Transport ---
        transport = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.toggle_stream)
        self.bypass_button = QPushButton("Bypass: OFF")
        self.bypass_button.setCheckable(True)
        self.bypass_button.toggled.connect(self.on_bypass_toggled)
        transport.addWidget(self.start_button)
        transport.addWidget(self.bypass_button)
        layout.addLayout(transport)

        self.status_label = QLabel("Stopped.")
        layout.addWidget(self.status_label)

        self.meter_timer = QTimer(self)
        self.meter_timer.timeout.connect(self.refresh_meters)
        self.meter_timer.start(50)

    def _populate_devices(self):
        devices = sd.query_devices()
        for idx, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                self.in_combo.addItem(f"{idx}: {d['name']}", idx)
            if d["max_output_channels"] > 0:
                self.out_combo.addItem(f"{idx}: {d['name']}", idx)

    # --- live parameter handlers ---
    def on_deemph_toggled(self, checked):
        if self.pipeline:
            self.pipeline.deemph.enabled = checked

    def on_deemph_changed(self, value):
        if self.pipeline:
            self.pipeline.deemph.a = value

    def on_bandpass_toggled(self, checked):
        if self.pipeline:
            self.pipeline.bandpass.enabled = checked

    def on_bandpass_changed(self, _value):
        if self.pipeline:
            low, high = self.bp_low_spin.value(), self.bp_high_spin.value()
            if high > low:
                self.pipeline.bandpass.set_band(low, high)

    def on_agc_toggled(self, checked):
        if self.pipeline:
            self.pipeline.agc.enabled = checked

    def on_agc_target_changed(self, value):
        if self.pipeline:
            self.pipeline.agc.target = value

    def on_strength_changed(self, value):
        s = value / 100.0
        self.strength_label.setText(f"{s:.2f}")
        if self.pipeline:
            self.pipeline.strength = s

    def on_min_gain_changed(self, value):
        label = "-60 (off)" if value <= -60 else str(value)
        self.min_gain_label.setText(label)
        if self.pipeline:
            self.pipeline.min_gain.min_gain_db = float(value)

    def on_bypass_toggled(self, checked):
        self.bypass_button.setText(f"Bypass: {'ON' if checked else 'OFF'}")
        if self.pipeline:
            self.pipeline.bypass = checked

    def apply_mode_preset(self, mode):
        if mode == "none":
            return
        preset_deemph, low, high = MODE_PRESETS[mode]
        self.deemph_check.setChecked(preset_deemph)
        self.bandpass_check.setChecked(True)
        self.bp_low_spin.setValue(low)
        self.bp_high_spin.setValue(high)

    # --- stream lifecycle ---
    def toggle_stream(self):
        if self.stream is None:
            self.start_stream()
        else:
            self.stop_stream()

    def start_stream(self):
        try:
            self.pipeline = Pipeline(backend_name=self.backend_combo.currentText())
        except Exception as e:
            QMessageBox.critical(self, "Backend error", str(e))
            return

        # sync the fresh pipeline to whatever the GUI currently shows
        self.pipeline.deemph.enabled = self.deemph_check.isChecked()
        self.pipeline.deemph.a = self.deemph_spin.value()
        self.pipeline.bandpass.set_band(self.bp_low_spin.value(), self.bp_high_spin.value())
        self.pipeline.bandpass.enabled = self.bandpass_check.isChecked()
        self.pipeline.agc.enabled = self.agc_check.isChecked()
        self.pipeline.agc.target = self.agc_target_spin.value()
        self.pipeline.strength = self.strength_slider.value() / 100.0
        self.pipeline.min_gain.min_gain_db = float(self.min_gain_slider.value())
        self.pipeline.bypass = self.bypass_button.isChecked()

        self.in_buf.clear()
        self.out_buf.clear()
        in_dev = self.in_combo.currentData()
        out_dev = self.out_combo.currentData()
        pipeline = self.pipeline
        in_buf, out_buf = self.in_buf, self.out_buf

        def callback(indata, outdata, frames, time_info, status):
            mono = indata[:, 0] if indata.ndim > 1 else indata.ravel()
            in_buf.extend(mono.tolist())
            while len(in_buf) >= FRAME_SIZE:
                chunk = np.asarray([in_buf.popleft() for _ in range(FRAME_SIZE)], dtype=np.float32)
                out = pipeline.process_frame(chunk)
                out_buf.extend(out.tolist())
            n = min(len(out_buf), frames)
            for i in range(n):
                outdata[i, 0] = out_buf.popleft()
            if n < frames:
                outdata[n:, 0] = 0.0

        try:
            self.stream = sd.Stream(
                samplerate=SAMPLE_RATE, blocksize=0, latency="high",
                channels=1, dtype="float32",
                device=(in_dev, out_dev), callback=callback,
            )
            self.stream.start()
        except Exception as e:
            QMessageBox.critical(self, "Stream error", str(e))
            self.stream = None
            self.pipeline = None
            return

        self.start_button.setText("Stop")
        self.status_label.setText("Running. Wear headphones to avoid feedback.")

    def stop_stream(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.pipeline = None
        self.start_button.setText("Start")
        self.status_label.setText("Stopped.")
        self.in_meter.set_level(-60.0)
        self.out_meter.set_level(-60.0)

    def refresh_meters(self):
        if self.pipeline:
            self.in_meter.set_level(self.pipeline.in_level)
            self.out_meter.set_level(self.pipeline.out_level)

    def closeEvent(self, event):
        self.stop_stream()
        event.accept()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(560, 560)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
