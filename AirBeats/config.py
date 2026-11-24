# =============================================================================
# AirBeats — Central Configuration
# =============================================================================
# All tuneable parameters live here.
# To change the serial port: edit SERIAL_PORT below. That is the ONLY place
# the port is defined. Every other module imports it from here.
# =============================================================================

# -----------------------------------------------------------------------------
# Serial / Arduino
# -----------------------------------------------------------------------------
SERIAL_PORT      = "COM4"      # <-- Change only this if the port changes
SERIAL_BAUD      = 115200
SERIAL_TIMEOUT_S = 0.1         # Seconds before a read() call gives up.
                               # At 50 Hz (20 ms/sample), 0.1 s = 5 sample
                               # periods. Long enough to catch a real timeout;
                               # short enough not to stall the acquisition loop.
                               # The original 2.0 s would block for 2 seconds
                               # on every missed packet, which is unacceptable.

# -----------------------------------------------------------------------------
# Sampling
# -----------------------------------------------------------------------------
SAMPLE_RATE_HZ      = 50       # Must match SAMPLE_INTERVAL_MS on the Arduino
SAMPLE_INTERVAL_MS  = 20       # 1000 / SAMPLE_RATE_HZ

# -----------------------------------------------------------------------------
# Analog sensor channels
# -----------------------------------------------------------------------------
# ADC is 10-bit on the Arduino Mega: values span 0-1023.
ADC_RESOLUTION_BITS = 10
ADC_MIN             = 0
ADC_MAX             = 1023

# Human-readable channel names, in the same order they appear in the packet
ANALOG_CHANNEL_NAMES = ["Pressure", "Flex1", "Flex2", "Flex3", "Flex4"]
NUM_ANALOG_CHANNELS  = len(ANALOG_CHANNEL_NAMES)

# -----------------------------------------------------------------------------
# IMU channels
# -----------------------------------------------------------------------------
# Accelerometer: g (gravitational units, ~9.81 m/s^2 per unit)
# Gyroscope    : degrees per second
IMU_CHANNEL_NAMES = ["AccelX", "AccelY", "AccelZ", "GyroX", "GyroY", "GyroZ"]
NUM_IMU_CHANNELS  = len(IMU_CHANNEL_NAMES)

ACCEL_UNIT = "g"
GYRO_UNIT  = "deg/s"

# -----------------------------------------------------------------------------
# Full channel list (used for CSV headers and dashboards)
# -----------------------------------------------------------------------------
ALL_CHANNEL_NAMES = (
    ["timestamp_ms"]
    + ANALOG_CHANNEL_NAMES
    + IMU_CHANNEL_NAMES
)

# -----------------------------------------------------------------------------
# Data logging
# -----------------------------------------------------------------------------
import os

# Recordings are saved inside the project folder so they travel with the project
RECORDING_DIR = os.path.join(os.path.dirname(__file__), "recordings")

# CSV filename prefix — a timestamp is appended automatically
RECORDING_PREFIX = "airbeats_raw"

# -----------------------------------------------------------------------------
# Phase 2 — Calibration
# -----------------------------------------------------------------------------
import os as _os

# Path to the saved calibration JSON file.
# If the file does not exist the system uses the built-in defaults.
CALIBRATION_FILE = _os.path.join(_os.path.dirname(__file__), "calibration.json")

# Number of stationary samples to collect for IMU bias calibration.
# At 50 Hz: 150 samples = 3 seconds of still hand.
IMU_CALIBRATION_SAMPLES = 150

# -----------------------------------------------------------------------------
# Phase 2 — Signal Processing
# -----------------------------------------------------------------------------

# Default filter to apply at startup.
# Options: "none", "ema", "moving_average", "butterworth", "median"
DEFAULT_FILTER = "ema"

# EMA alpha: 0.3 gives moderate smoothing. Try 0.1–0.5.
EMA_ALPHA = 0.3

# Moving average window (samples). 5 = 100 ms at 50 Hz.
MA_WINDOW = 5

# Butterworth cutoff (Hz). Human finger gestures are well below 10 Hz.
BUTTERWORTH_CUTOFF_HZ = 10.0
BUTTERWORTH_ORDER     = 2

# Median window (samples, forced odd). 5 = 100 ms at 50 Hz.
MEDIAN_WINDOW = 5

# -----------------------------------------------------------------------------
# Phase 3 — Sensor Fusion
# -----------------------------------------------------------------------------
# Complementary filter coefficient (0-1).
# 0.98 = trust gyro 98%, accel 2%. Higher = smoother but slower drift correction.
COMPLEMENTARY_ALPHA = 0.98

# Accelerometer confidence threshold (g).
# If |accel_magnitude - 1.0| > this, the accelerometer is considered unreliable
# (hand is moving fast, linear acceleration contaminates gravity measurement).
ACCEL_CONFIDENCE_THRESHOLD_G = 0.3

# Motion detection threshold (deg/s).
# Gyro magnitude below this = hand is considered still.
MOTION_THRESHOLD_DEG_S = 15.0

# -----------------------------------------------------------------------------
# Phase 3 — Gesture Recognition
# -----------------------------------------------------------------------------
# Flex percentage thresholds (0.0-1.0).
# Below OPEN_THRESHOLD = finger extended, above CLOSE_THRESHOLD = finger bent.
GESTURE_OPEN_THRESHOLD  = 0.3
GESTURE_CLOSE_THRESHOLD = 0.7

# Hand tilt threshold (degrees from level).
GESTURE_TILT_THRESHOLD  = 30.0

# Pressure threshold (0.0-1.0) for PRESS gesture.
GESTURE_PRESS_THRESHOLD = 0.5

# Shake threshold (deg/s). Sustained gyro magnitude above this = SHAKE gesture.
SHAKE_THRESHOLD_DEG_S   = 100.0

# Debounce time (ms). A gesture must be held this long before it is reported.
# Prevents flicker during transitions. At 50 Hz, 200 ms = 10 samples.
GESTURE_DEBOUNCE_MS     = 200

# -----------------------------------------------------------------------------
# Phase 4 — Musical Mapping
# -----------------------------------------------------------------------------
# Default scale: "pentatonic_minor" is the most forgiving for improvisation.
# Options: pentatonic_minor, pentatonic_major, major, natural_minor, blues
DEFAULT_SCALE = "pentatonic_minor"

# Root note (MIDI note number). 60 = middle C.
ROOT_NOTE_MIDI = 60

# Base octave (0-8). This shifts all notes up or down by whole octaves.
# 0 = the octave containing the root note as written above.
# The actual octave depends on the root note. For root=60 (C4):
#   octave 0 → C4,  octave 1 → C5,  octave -1 → C3
BASE_OCTAVE = 0

# Finger-to-scale-degree mapping.
# Maps flex sensor index (0-3) to a scale degree (position in the scale).
# Degree 0 = root, degree 1 = second note, etc.
# With pentatonic_minor [0,3,5,7,10]:
#   finger 0 → degree 0 → root (C)
#   finger 1 → degree 1 → minor third (Eb)
#   finger 2 → degree 2 → perfect fourth (F)
#   finger 3 → degree 3 → perfect fifth (G)
FINGER_SCALE_DEGREES = [0, 1, 2, 3]

# Octave tilt threshold (degrees of roll for one octave shift).
# Roll beyond this → +1 or -1 octave. Beyond 2x this → +2 or -2.
OCTAVE_TILT_THRESHOLD_DEG = 30.0

# Pitch bend range (degrees of pitch for full bend).
# At this angle the pitch bend is at maximum (±8191 in MIDI).
PITCH_BEND_RANGE_DEG = 45.0

# Minimum MIDI velocity (1-127). Notes always play at least this loud.
MIN_VELOCITY = 20

# -----------------------------------------------------------------------------
# Phase 5 — Sound Generation
# -----------------------------------------------------------------------------
# Default instrument preset.
# Options: piano, pad, pluck, bass, bell
DEFAULT_INSTRUMENT = "piano"

# Audio sample rate (Hz). 44100 is CD quality.
AUDIO_SAMPLE_RATE = 44100

# Audio block size (samples). Smaller = lower latency but more CPU.
# 256 samples at 44100 Hz = 5.8 ms latency.
AUDIO_BLOCK_SIZE = 256

# Master output volume (0.0 to 1.0).
DEFAULT_VOLUME = 0.5

# Pitch bend range in semitones (±). Standard is 2.
PITCH_BEND_SEMITONES = 2

# -----------------------------------------------------------------------------
# Dashboard layout
# -----------------------------------------------------------------------------
DASHBOARD_UPDATE_INTERVAL_MS = 50    # Redraw every 50 ms (20 Hz visual refresh)
DASHBOARD_HISTORY_SECONDS    = 10    # How many seconds of history to show
DASHBOARD_HISTORY_SAMPLES    = SAMPLE_RATE_HZ * DASHBOARD_HISTORY_SECONDS

# Analog graph y-axis range (raw ADC values)
DASHBOARD_ANALOG_YMIN = 0
DASHBOARD_ANALOG_YMAX = 1023

# IMU graph y-axis ranges
DASHBOARD_ACCEL_YMIN = -4.0    # g
DASHBOARD_ACCEL_YMAX =  4.0
DASHBOARD_GYRO_YMIN  = -300.0  # deg/s
DASHBOARD_GYRO_YMAX  =  300.0
