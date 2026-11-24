# =============================================================================
# AirBeats — Digital Signal Processing Filters
# processing/filters.py
#
# Implements four independent, selectable filter types as stateful classes.
# Each filter maintains internal state so it can be used sample-by-sample
# in a real-time acquisition loop — no batch processing required.
#
# ─────────────────────────────────────────────────────────────────────────────
# Filter overview and mathematical principles
# ─────────────────────────────────────────────────────────────────────────────
#
# 1. Moving Average (MA)
#    y[n] = (1/N) * sum(x[n], x[n-1], ..., x[n-N+1])
#    Keeps the last N samples in a circular buffer and returns their mean.
#    Equivalent to a rectangular FIR low-pass filter with a sinc frequency
#    response. Easy to understand, but introduces a group delay of (N-1)/2
#    samples. At 50 Hz with N=5 that is 40 ms of lag.
#
# 2. Exponential Moving Average (EMA)
#    y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
#    Single-pole IIR filter. alpha controls the trade-off:
#      alpha → 1  : almost no smoothing, tracks input closely
#      alpha → 0  : heavy smoothing, slow to respond
#    Lower latency than MA for the same degree of smoothing because
#    it weights recent samples more heavily than older ones.
#    The equivalent time constant is: tau = (1 - alpha) / (alpha * fs)
#
# 3. Butterworth Low-Pass (IIR, second-order)
#    Designed using the bilinear transform (scipy.signal.butter).
#    The Butterworth design maximises passband flatness — the magnitude
#    response is as flat as possible in the passband before rolling off.
#    Uses scipy.signal.lfilter_zi for proper initial conditions so the
#    filter output does not start with a large transient.
#    Cutoff frequency fc is specified in Hz. At fc, the response is -3 dB
#    (amplitude attenuated to 1/sqrt(2) ≈ 0.707 of the passband value).
#    With order=2 the roll-off is 40 dB/decade beyond fc.
#    This is the most rigorous filter for DSP analysis.
#
# 4. Median Filter
#    y[n] = median(x[n], x[n-1], ..., x[n-N+1])
#    Non-linear filter. Replaces the current sample with the median of a
#    sliding window. Very effective at removing impulse noise (spikes) while
#    preserving edges (sharp transitions). The flex sensors occasionally
#    produce spike values when fingers tap or bump the sensor — median filter
#    handles these better than any linear filter.
#
# ─────────────────────────────────────────────────────────────────────────────
# Usage pattern (real-time, one sample at a time):
#
#   f = MovingAverageFilter(window=5)
#   for packet in stream:
#       filtered_value = f.update(raw_value)
#
# Each filter class has an identical interface:
#   update(value: float) → float
#   reset()
#   name: str
#   parameters: dict
# ─────────────────────────────────────────────────────────────────────────────

import collections
import numpy as np
from scipy import signal


# =============================================================================
# Moving Average Filter
# =============================================================================

class MovingAverageFilter:
    """
    Simple N-point moving average (rectangular window FIR).

    Parameters
    ----------
    window : int
        Number of samples to average. Larger = more smoothing + more latency.
        Recommended range: 3–15 for 50 Hz flex/pressure data.

    Latency: (window - 1) / 2 samples = (window - 1) / (2 * fs) seconds.
    At 50 Hz, window=5 → 40 ms latency.
    """

    name = "Moving Average"

    def __init__(self, window: int = 5):
        if window < 1:
            raise ValueError("window must be >= 1")
        self._window  = window
        self._buf     = collections.deque(maxlen=window)
        self._sum     = 0.0
        self._count   = 0

    @property
    def parameters(self) -> dict:
        return {"window": self._window}

    def update(self, value: float) -> float:
        """
        Feed one new sample. Returns the current moving average.

        During the first (window - 1) calls, the buffer is not full.
        The filter returns the mean of however many samples it has seen
        so far — this is the correct "growing window" initialisation.
        """
        if len(self._buf) == self._window:
            # Buffer is full: subtract the oldest value before it is evicted
            self._sum -= self._buf[0]

        self._buf.append(value)
        self._sum += value
        return self._sum / len(self._buf)

    def reset(self):
        """Clear the buffer. Call when signal continuity is broken."""
        self._buf.clear()
        self._sum   = 0.0

    def set_window(self, window: int):
        """Change the window size and reset state."""
        self._window = window
        self._buf    = collections.deque(maxlen=window)
        self._sum    = 0.0


# =============================================================================
# Exponential Moving Average Filter
# =============================================================================

class EMAFilter:
    """
    Single-pole IIR exponential moving average.

    Parameters
    ----------
    alpha : float, 0 < alpha <= 1
        Smoothing factor.
        alpha = 1.0  → no filtering (output = input)
        alpha = 0.1  → heavy smoothing

    Relationship between alpha and cutoff frequency fc at sample rate fs:
        alpha ≈ 2 * pi * fc / fs    (approximate, valid for fc << fs)
    For fs=50 Hz and fc=5 Hz: alpha ≈ 2π×5/50 ≈ 0.63

    Recurrence: y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
    """

    name = "Exponential Moving Average"

    def __init__(self, alpha: float = 0.3):
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")
        self._alpha   = alpha
        self._y_prev  = None   # None = "not yet initialised"

    @property
    def parameters(self) -> dict:
        return {"alpha": self._alpha}

    def update(self, value: float) -> float:
        """
        Feed one new sample. Returns filtered output.

        On the very first call, output = input (zero-state initialisation).
        This avoids a large transient at startup.
        """
        if self._y_prev is None:
            self._y_prev = value
            return value
        y = self._alpha * value + (1.0 - self._alpha) * self._y_prev
        self._y_prev = y
        return y

    def reset(self):
        self._y_prev = None

    def set_alpha(self, alpha: float):
        self._alpha  = alpha
        self._y_prev = None


# =============================================================================
# Butterworth Low-Pass Filter
# =============================================================================

class ButterworthFilter:
    """
    Second-order Butterworth IIR low-pass filter.

    Parameters
    ----------
    cutoff_hz  : float — cutoff frequency in Hz (-3 dB point)
    sample_hz  : float — sample rate in Hz (must match your acquisition rate)
    order      : int   — filter order (default 2: 40 dB/decade roll-off)

    Uses scipy.signal.butter (bilinear transform design) for the coefficients
    and scipy.signal.lfilter with proper initial conditions (lfilter_zi) to
    avoid startup transients.

    The Butterworth design criterion: maximally flat magnitude in the passband.
    No ripple in passband or stopband (unlike Chebyshev or elliptic filters).

    For gesture control signals (0–10 Hz useful content, fs=50 Hz):
      cutoff_hz=10 Hz captures all intentional movement.
      cutoff_hz=5  Hz gives smoother output with slightly more lag.
    """

    name = "Butterworth Low-Pass"

    def __init__(self, cutoff_hz: float = 10.0, sample_hz: float = 50.0,
                 order: int = 2):
        self._cutoff    = cutoff_hz
        self._sample_hz = sample_hz
        self._order     = order
        self._b         = None
        self._a         = None
        self._zi        = None
        self._design()

    @property
    def parameters(self) -> dict:
        return {
            "cutoff_hz" : self._cutoff,
            "sample_hz" : self._sample_hz,
            "order"     : self._order,
        }

    def _design(self):
        """Compute filter coefficients and initial conditions."""
        # Normalised cutoff: Wn = fc / (fs/2), must be in (0, 1)
        wn = self._cutoff / (self._sample_hz / 2.0)
        wn = max(1e-4, min(0.9999, wn))   # clamp to valid range

        self._b, self._a = signal.butter(self._order, wn, btype="low",
                                          analog=False)
        # lfilter_zi computes the initial conditions for a step response of 1.
        # We will scale by the first input value at runtime to avoid a
        # large transient when the filter starts.
        zi_base   = signal.lfilter_zi(self._b, self._a)
        self._zi  = zi_base   # will be scaled on first update()
        self._initialised = False

    def update(self, value: float) -> float:
        """
        Feed one sample. Returns filtered output.

        Uses scipy.signal.lfilter with running state (zi) so that each call
        processes exactly one sample and the state carries over to the next call.
        This is the correct way to use lfilter in a real-time loop.
        """
        if not self._initialised:
            # Scale initial conditions by the first input value.
            # This makes the filter output start at the first input value
            # instead of zero, eliminating the startup transient.
            self._zi          = signal.lfilter_zi(self._b, self._a) * value
            self._initialised = True

        y, self._zi = signal.lfilter(self._b, self._a, [value], zi=self._zi)
        return float(y[0])

    def reset(self):
        """Reset filter state. Call when switching signals or after a gap."""
        self._initialised = False

    def set_cutoff(self, cutoff_hz: float):
        """Change the cutoff frequency and redesign the filter."""
        self._cutoff = cutoff_hz
        self._design()


# =============================================================================
# Median Filter
# =============================================================================

class MedianFilter:
    """
    Sliding-window median filter.

    Parameters
    ----------
    window : int — number of samples in the sliding window (must be odd)

    Removes spike/impulse noise that linear filters cannot handle well.
    Useful for flex sensors that occasionally produce a single bad reading
    when the finger taps a surface.

    Computational cost: O(window * log(window)) per sample due to sorting.
    At window=5 and 50 Hz this is negligible on a laptop CPU.
    """

    name = "Median"

    def __init__(self, window: int = 5):
        if window < 1:
            raise ValueError("window must be >= 1")
        if window % 2 == 0:
            window += 1   # Enforce odd window for unambiguous median
        self._window = window
        self._buf    = collections.deque(maxlen=window)

    @property
    def parameters(self) -> dict:
        return {"window": self._window}

    def update(self, value: float) -> float:
        """
        Feed one sample. Returns the median of the current window.
        Before the window is full, returns the median of available samples.
        """
        self._buf.append(value)
        return float(np.median(self._buf))

    def reset(self):
        self._buf.clear()

    def set_window(self, window: int):
        if window % 2 == 0:
            window += 1
        self._window = window
        self._buf    = collections.deque(maxlen=window)


# =============================================================================
# FilterBank — apply one filter to every channel of a multi-channel signal
# =============================================================================

class FilterBank:
    """
    Applies one filter instance per channel.

    Used to filter all flex sensors, all IMU axes, etc., with independent
    state per channel but the same filter type and parameters.

    Parameters
    ----------
    n_channels    : int   — number of independent channels
    filter_class  : class — one of the filter classes above
    **kwargs            — passed to the filter constructor for every channel

    Example:
        bank = FilterBank(4, EMAFilter, alpha=0.3)
        filtered = bank.update([flex1, flex2, flex3, flex4])
    """

    def __init__(self, n_channels: int, filter_class, **kwargs):
        self._filters = [filter_class(**kwargs) for _ in range(n_channels)]

    def update(self, values: list) -> list:
        """
        Apply each channel's filter to the corresponding value.
        Returns a list of filtered values, same length as input.
        """
        return [self._filters[i].update(values[i]) for i in range(len(values))]

    def reset(self):
        for f in self._filters:
            f.reset()

    @property
    def name(self) -> str:
        return self._filters[0].name if self._filters else "Unknown"

    @property
    def parameters(self) -> dict:
        return self._filters[0].parameters if self._filters else {}


# =============================================================================
# SignalProcessor — applies calibration then filtering to a live packet stream
# =============================================================================

class SignalProcessor:
    """
    Combines calibration and filtering into a single pipeline object.

    Takes a raw packet, applies calibration, then applies the selected filter
    to flex_pct, pressure_pct, accel_cal, and gyro_cal independently.

    The filter type and parameters can be changed at runtime without restarting
    acquisition.

    Usage:
        from processing.calibration import Calibration
        from processing.filters import SignalProcessor, EMAFilter

        cal  = Calibration()
        proc = SignalProcessor(cal)
        proc.set_filter(EMAFilter, alpha=0.3)

        for raw_packet in stream:
            processed = proc.process(raw_packet)
            # processed contains both calibrated and filtered values
    """

    def __init__(self, calibration):
        self._cal = calibration
        self._flex_bank    = None   # FilterBank for 4 flex channels
        self._pressure_f   = None   # Single filter for pressure
        self._accel_bank   = None   # FilterBank for 3 accel channels
        self._gyro_bank    = None   # FilterBank for 3 gyro channels
        self._filter_name  = "None"
        self._filter_params= {}

    def set_filter(self, filter_class, **kwargs):
        """
        Set the active filter. Creates new filter instances for all channels.
        Resets all existing filter state.

        Parameters
        ----------
        filter_class : one of MovingAverageFilter, EMAFilter,
                       ButterworthFilter, MedianFilter, or None to disable.
        **kwargs     : passed directly to the filter constructor.

        Example:
            proc.set_filter(ButterworthFilter, cutoff_hz=5.0, sample_hz=50.0)
            proc.set_filter(MovingAverageFilter, window=7)
            proc.set_filter(None)  # disable filtering
        """
        if filter_class is None:
            self._flex_bank    = None
            self._pressure_f   = None
            self._accel_bank   = None
            self._gyro_bank    = None
            self._filter_name  = "None"
            self._filter_params= {}
            return

        self._flex_bank    = FilterBank(4, filter_class, **kwargs)
        self._pressure_f   = filter_class(**kwargs)
        self._accel_bank   = FilterBank(3, filter_class, **kwargs)
        self._gyro_bank    = FilterBank(3, filter_class, **kwargs)

        # Retrieve name and params from a temporary instance
        tmp = filter_class(**kwargs)
        self._filter_name   = tmp.name
        self._filter_params = tmp.parameters

        print(f"[SignalProcessor] Filter set: {self._filter_name} "
              f"params={self._filter_params}")

    def process(self, raw_packet: dict) -> dict:
        """
        Apply calibration then filtering to one raw packet.

        Returns the calibrated dict (from Calibration.apply()) with
        additional keys if a filter is active:
          "flex_filtered"     : [float x4]  — filtered bend percentages
          "pressure_filtered" : float        — filtered pressure
          "accel_filtered"    : [float x3]  — filtered accelerometer
          "gyro_filtered"     : [float x3]  — filtered gyroscope
          "filter_name"       : str
        """
        # Step 1: Calibration
        cal_packet = self._cal.apply(raw_packet)

        # Step 2: Filtering (only if a filter has been set)
        if self._flex_bank is not None:
            cal_packet["flex_filtered"] = self._flex_bank.update(
                cal_packet["flex_pct"]
            )
            cal_packet["pressure_filtered"] = self._pressure_f.update(
                cal_packet["pressure_pct"]
            )
            cal_packet["accel_filtered"] = self._accel_bank.update(
                cal_packet["accel_cal"]
            )
            cal_packet["gyro_filtered"] = self._gyro_bank.update(
                cal_packet["gyro_cal"]
            )
            cal_packet["filter_name"] = self._filter_name

        return cal_packet
