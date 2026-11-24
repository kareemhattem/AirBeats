# =============================================================================
# AirBeats — Real-Time Dashboard
# visualization/dashboard.py
#
# Change log vs. original:
#   - Removed threading.Lock.
#     All buffer mutations now happen exclusively on the GUI (main) thread:
#     drain_queue() is called from _update() which is called by FuncAnimation
#     which runs inside plt.show(). The acquisition thread only writes to the
#     queue; it never touches the deques directly. No shared mutable state
#     between threads → no lock needed.
#   - Added set_queue() / drain_queue() so the dashboard pulls packets from
#     a thread-safe queue instead of receiving them via push() calls from
#     a mixed-thread loop.
#   - Added schedule_callback() to let main.py register a periodic Tk after()
#     callback for command handling and rate reporting, without introducing
#     any additional plt.pause() calls.
#   - push() is kept as a convenience alias for single-packet injection in
#     tests or future phases that run entirely on the main thread.
#
# Layout (3 rows, 2 columns):
#   Row 0, col 0 : Pressure
#   Row 0, col 1 : IMU Accelerometer (X, Y, Z)
#   Row 1, col 0 : Flex Sensors 1-4
#   Row 1, col 1 : IMU Gyroscope (X, Y, Z)
#   Row 2        : Status bar (numerical readout + recording indicator)
# =============================================================================

import collections
import queue as queue_module

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import config

# Colour palette
_FLEX_COLORS    = ["#E74C3C", "#E67E22", "#F1C40F", "#2ECC71"]
_ACCEL_COLORS   = ["#3498DB", "#9B59B6", "#1ABC9C"]
_GYRO_COLORS    = ["#E74C3C", "#E67E22", "#2ECC71"]
_PRESSURE_COLOR = "#ECF0F1"


class Dashboard:
    """
    Real-time sensor dashboard built on matplotlib FuncAnimation.

    The dashboard runs entirely on the main (GUI) thread.
    Data arrives via a queue.Queue set with set_queue().
    FuncAnimation calls _update() every DASHBOARD_UPDATE_INTERVAL_MS ms.
    _update() drains the queue and refreshes all plot lines.

    Usage:
        dashboard = Dashboard()
        dashboard.set_queue(packet_queue)
        dashboard.start()
        # --- acquisition thread fills packet_queue ---
        plt.show(block=True)
    """

    def __init__(self):
        maxlen = config.DASHBOARD_HISTORY_SAMPLES

        # Circular buffers — one per channel.
        # All writes happen in _update() on the GUI thread → no lock needed.
        self._t        = collections.deque(maxlen=maxlen)
        self._pressure = collections.deque(maxlen=maxlen)
        self._flex     = [collections.deque(maxlen=maxlen) for _ in range(4)]
        self._accel    = [collections.deque(maxlen=maxlen) for _ in range(3)]
        self._gyro     = [collections.deque(maxlen=maxlen) for _ in range(3)]

        self._latest_packet = None
        self._is_recording  = False
        self._packet_queue  = None   # Set via set_queue()

        self._fig  = None
        self._anim = None

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    def set_queue(self, q: queue_module.Queue):
        """Provide the inter-thread queue to drain during _update()."""
        self._packet_queue = q

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def start(self):
        """Create the matplotlib figure and start the FuncAnimation timer."""
        self._build_figure()
        self._anim = animation.FuncAnimation(
            self._fig,
            self._update,
            interval=config.DASHBOARD_UPDATE_INTERVAL_MS,
            blit=False,
            cache_frame_data=False,
        )
        plt.show(block=False)
        plt.pause(0.05)   # One-time pause to let the window appear; not in the loop

    def push(self, packet: dict):
        """
        Directly insert one packet into the display buffers.
        Convenience method for single-threaded use and testing.
        Not called from the acquisition thread in the production architecture.
        """
        if packet is not None:
            self._append_packet(packet)

    def set_recording(self, state: bool):
        """Update the recording indicator on the status bar."""
        self._is_recording = state

    def close(self):
        if self._fig:
            plt.close(self._fig)

    def is_open(self) -> bool:
        return plt.fignum_exists(self._fig.number) if self._fig else False

    def schedule_callback(self, delay_ms: int, fn):
        """
        Schedule fn() to be called after delay_ms milliseconds by the Tk
        event loop. This is a zero-overhead alternative to plt.pause() for
        periodic tasks that need to run on the GUI thread.
        """
        if self._fig:
            self._fig.canvas.get_tk_widget().after(delay_ms, fn)

    # -------------------------------------------------------------------------
    # Figure construction
    # -------------------------------------------------------------------------

    def _build_figure(self):
        self._fig = plt.figure(figsize=(14, 8), facecolor="#1A1A2E")
        self._fig.canvas.manager.set_window_title("AirBeats — Live Dashboard")

        gs = self._fig.add_gridspec(
            3, 2,
            height_ratios=[1, 1, 0.3],
            hspace=0.45,
            wspace=0.35,
        )

        # Pressure
        self._ax_pressure = self._fig.add_subplot(gs[0, 0])
        self._line_pressure, = self._ax_pressure.plot(
            [], [], color=_PRESSURE_COLOR, linewidth=1.2
        )
        self._style_axis(self._ax_pressure, "Pressure (ADC)",
                         config.DASHBOARD_ANALOG_YMIN, config.DASHBOARD_ANALOG_YMAX)

        # Flex sensors
        self._ax_flex = self._fig.add_subplot(gs[1, 0])
        self._lines_flex = []
        for i in range(4):
            line, = self._ax_flex.plot(
                [], [], color=_FLEX_COLORS[i], linewidth=1.2, label=f"Flex {i+1}"
            )
            self._lines_flex.append(line)
        self._style_axis(self._ax_flex, "Flex Sensors (ADC)",
                         config.DASHBOARD_ANALOG_YMIN, config.DASHBOARD_ANALOG_YMAX)
        self._ax_flex.legend(loc="upper right", fontsize=7,
                             facecolor="#1A1A2E", labelcolor="#ECF0F1", framealpha=0.7)

        # Accelerometer
        self._ax_accel = self._fig.add_subplot(gs[0, 1])
        self._lines_accel = []
        for i, label in enumerate(["AX", "AY", "AZ"]):
            line, = self._ax_accel.plot(
                [], [], color=_ACCEL_COLORS[i], linewidth=1.2, label=label
            )
            self._lines_accel.append(line)
        self._style_axis(self._ax_accel, "Accelerometer (g)",
                         config.DASHBOARD_ACCEL_YMIN, config.DASHBOARD_ACCEL_YMAX)
        self._ax_accel.legend(loc="upper right", fontsize=7,
                              facecolor="#1A1A2E", labelcolor="#ECF0F1", framealpha=0.7)

        # Gyroscope
        self._ax_gyro = self._fig.add_subplot(gs[1, 1])
        self._lines_gyro = []
        for i, label in enumerate(["GX", "GY", "GZ"]):
            line, = self._ax_gyro.plot(
                [], [], color=_GYRO_COLORS[i], linewidth=1.2, label=label
            )
            self._lines_gyro.append(line)
        self._style_axis(self._ax_gyro, "Gyroscope (deg/s)",
                         config.DASHBOARD_GYRO_YMIN, config.DASHBOARD_GYRO_YMAX)
        self._ax_gyro.legend(loc="upper right", fontsize=7,
                             facecolor="#1A1A2E", labelcolor="#ECF0F1", framealpha=0.7)

        # Status bar
        self._ax_status = self._fig.add_subplot(gs[2, :])
        self._ax_status.axis("off")
        self._status_text = self._ax_status.text(
            0.5, 0.5, "Waiting for data...",
            transform=self._ax_status.transAxes,
            ha="center", va="center",
            fontsize=9, color="#ECF0F1", fontfamily="monospace",
        )

        self._fig.patch.set_facecolor("#1A1A2E")

    def _style_axis(self, ax, title: str, ymin: float, ymax: float):
        ax.set_facecolor("#16213E")
        ax.set_title(title, color="#ECF0F1", fontsize=9, pad=4)
        ax.set_ylim(ymin, ymax)
        ax.tick_params(colors="#7F8C8D", labelsize=7)
        ax.spines[:].set_color("#2C3E50")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color("#7F8C8D")
        ax.grid(True, color="#2C3E50", linewidth=0.5, alpha=0.7)
        ax.set_xlabel("Time (ms)", color="#7F8C8D", fontsize=7)

    # -------------------------------------------------------------------------
    # Buffer helpers
    # -------------------------------------------------------------------------

    def _append_packet(self, packet: dict):
        """Write one packet into all deques. Must be called on the GUI thread."""
        self._t.append(packet["timestamp_ms"])
        self._pressure.append(packet["pressure"])
        for i in range(4):
            self._flex[i].append(packet["flex"][i])
        for i in range(3):
            self._accel[i].append(packet["accelerometer"][i])
            self._gyro[i].append(packet["gyroscope"][i])
        self._latest_packet = packet
    # -------------------------------------------------------------------------
    # FuncAnimation callback — runs on the GUI thread every 50 ms
    # -------------------------------------------------------------------------

    def _update(self, frame):
        """
        Drain the inter-thread queue and redraw all plots.

        Draining strategy: consume every packet currently in the queue in one
        shot. At 50 Hz the queue holds at most ~2–3 packets between 50 ms
        redraws. We update the deques for all of them but only redraw once,
        which is correct — the deque already contains the latest values.
        """
        # --- Drain the queue ---
        if self._packet_queue is not None:
            drained = 0
            while True:
                try:
                    packet = self._packet_queue.get_nowait()
                    self._append_packet(packet)
                    drained += 1
                except queue_module.Empty:
                    break

        # --- Nothing to draw yet ---
        if not self._t:
            return

        t_arr = np.array(self._t)
        x_min = t_arr[0]
        x_max = max(t_arr[-1], t_arr[0] + 1)

        # Pressure
        self._line_pressure.set_data(t_arr, self._pressure)
        self._ax_pressure.set_xlim(x_min, x_max)

        # Flex
        for i in range(4):
            self._lines_flex[i].set_data(t_arr, self._flex[i])
        self._ax_flex.set_xlim(x_min, x_max)

        # Accelerometer
        for i in range(3):
            self._lines_accel[i].set_data(t_arr, self._accel[i])
        self._ax_accel.set_xlim(x_min, x_max)

        # Gyroscope
        for i in range(3):
            self._lines_gyro[i].set_data(t_arr, self._gyro[i])
        self._ax_gyro.set_xlim(x_min, x_max)

        # Status bar
        if self._latest_packet:
            p = self._latest_packet
            rec = " [REC]" if self._is_recording else ""

            # Prefer processed values when available, fall back to raw
            flex_display = p.get("flex_pct") or [v/1023.0 for v in p["flex"]]
            fp = [f"{v*100:.0f}%" for v in flex_display]

            accel_src = p.get("accel_cal") or p["accelerometer"]
            gyro_src  = p.get("gyro_cal")  or p["gyroscope"]

            filt_name = p.get("filter_name", "raw")

            # Phase 3: orientation and gesture
            roll  = p.get("roll", 0)
            pitch = p.get("pitch", 0)
            gesture = p.get("gesture_name", "")
            motion  = p.get("motion_intensity", 0)

            # Phase 4: musical state
            ms = p.get("musical_state", {})
            mode = ms.get("mode", "PLAY")
            scale = ms.get("scale", "")
            octave = ms.get("octave_shift", 0)
            bend = ms.get("pitch_bend", 0)
            mod = ms.get("modulation", 0)
            # Count fingers currently held bent (above close threshold).
            # This is more useful than the transient active_notes list which
            # only has entries on the exact frame a note-on fires.
            flex_vals = p.get("flex_pct") or [0, 0, 0, 0]
            n_held = sum(1 for v in flex_vals if v >= 0.7)

            status = (
                f"t={p['timestamp_ms']}ms  |  "
                f"P={p['pressure']:4d}  |  "
                f"Flex=[{fp[0]},{fp[1]},{fp[2]},{fp[3]}]  |  "
                f"R={roll:+5.1f}d  P={pitch:+5.1f}d  |  "
                f"M={motion:.2f}  |  "
                f"[{gesture}]  |  "
                f"Held:{n_held} {mode} oct={octave:+d} bend={bend:+5d} mod={mod:3d}  |  "
                f"{scale}  [{filt_name}]{rec}"
            )
            self._status_text.set_text(status)
            self._status_text.set_color("#E74C3C" if self._is_recording else "#ECF0F1")
