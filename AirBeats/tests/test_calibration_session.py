# Calibration session state machine test — no hardware required.
# Simulates the acquisition thread's last_processed packets.
import sys, os, time
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

import config
from processing.calibration import Calibration

# Import CalibrationSession from main
sys.path.insert(0, os.path.join(_HERE, ".."))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "airbeats_main", os.path.join(_HERE, "..", "main.py"))
# main.py imports heavy deps; instead exec only the class definition.
# Simpler: parse and exec just the CalibrationSession block via the module
# source, skipping main() execution by not calling main().

# Extract the class source
src = open(os.path.join(_HERE, "..", "main.py"), encoding="utf-8").read()
start = src.index("class CalibrationSession:")
end = src.index("# =============================================================================\n# Keyboard input thread")
cls_src = src[start:end]

ns = {"__name__": "calib_test", "config": config}

# Redirect the save path to a temp file so tests never contaminate
# the real calibration.json.
import tempfile
_tmp_cal = tempfile.mktemp(suffix=".json")
class _SafeConfig:
    CALIBRATION_FILE = _tmp_cal
ns["config"] = _SafeConfig
_tmp_files = [_tmp_cal]

exec(cls_src, ns)
CalibrationSession = ns["CalibrationSession"]

print("=" * 60)
print("Calibration Session State Machine Test")
print("=" * 60)


class FakeAcq:
    """Simulates the acquisition thread's last_processed packets."""
    def __init__(self):
        self.last_processed = None
        self._values = [320, 310, 315, 300]

    def set_pose(self, sensor, straight=True):
        # Simulate the sensor reading for straight vs bent pose
        # Direction: Flex1/3/4 increase when bent; Flex2 decreases
        if sensor == 0:
            self._values[0] = 310 if straight else 380
        elif sensor == 1:
            self._values[1] = 300 if straight else 270
        elif sensor == 2:
            self._values[2] = 320 if straight else 335
        else:
            self._values[3] = 305 if straight else 345

    def tick(self):
        self.last_processed = {
            "flex": list(self._values),
            "pressure": 0,
            "timestamp_ms": int(time.time() * 1000),
        }


class FakeDashboard:
    """Captures scheduled callbacks so we can fire them manually."""
    def __init__(self):
        self.pending = []

    def schedule_callback(self, delay_ms, fn):
        self.pending.append((delay_ms, fn))


class FakeTimer:
    """Drives the FakeDashboard pending callbacks synchronously."""
    def __init__(self, dashboard, acq):
        self._dash = dashboard
        self._acq  = acq

    def run_pending(self, n=1):
        """Run the oldest pending callback, simulating an acquisition tick."""
        for _ in range(n):
            self._acq.tick()
            if self._dash.pending:
                delay, fn = self._dash.pending.pop(0)
                fn()
            else:
                time.sleep(0.01)


def run_session(collect_ticks_override=None):
    cal = Calibration()
    acq = FakeAcq()
    dash = FakeDashboard()
    timer = FakeTimer(dash, acq)
    sess = CalibrationSession.__new__(CalibrationSession)
    sess._cal = cal
    sess._acq = acq
    sess._dashboard = dash
    sess.state = sess.IDLE if hasattr(sess, 'IDLE') else 0
    # Initialize properly via start()
    sess.state = None  # force re-init

    # Reconstruct via __init__
    CalibrationSession.__init__(sess, cal, acq, dash)
    if collect_ticks_override:
        CalibrationSession.COLLECT_TICKS = collect_ticks_override

    acq.last_processed = {"flex": [320, 310, 315, 300], "pressure": 0, "timestamp_ms": 0}

    ok = sess.start()
    assert ok, "start() failed"
    assert sess.is_active, "session should be active after start()"

    for sensor in range(4):
        # --- Straight prompt: press Enter ---
        assert sess.state == CalibrationSession.PROMPT_STRAIGHT
        acq.set_pose(sensor, straight=True)
        sess.handle_command("")
        assert sess.state == CalibrationSession.COLLECT_STRAIGHT

        # Run collection ticks
        while sess.state == CalibrationSession.COLLECT_STRAIGHT:
            timer.run_pending(1)

        assert sess.state == CalibrationSession.PROMPT_BENT, \
            f"Expected PROMPT_BENT, got {sess.state}"

        # --- Bent prompt: press Enter ---
        acq.set_pose(sensor, straight=False)
        sess.handle_command("")
        assert sess.state == CalibrationSession.COLLECT_BENT

        while sess.state == CalibrationSession.COLLECT_BENT:
            timer.run_pending(1)

    # After 4 sensors we should be at CONFIRM
    assert sess.state == CalibrationSession.CONFIRM, \
        f"Expected CONFIRM, got {sess.state}"

    # Press Enter to save
    sess.handle_command("")
    assert sess.state == CalibrationSession.IDLE, "Should return to IDLE after save"
    return cal


# Run with small tick count for speed
cal = run_session(collect_ticks_override=8)

print()
print("Calibrated profiles after session:")
for i, f in enumerate(cal.flex):
    direction = "increases" if f.range_adc > 0 else "decreases"
    print(f"  {f.name}: straight={f.straight_adc:.1f} bent={f.bent_adc:.1f} "
          f"range={f.range_adc:+.1f} ({direction})")

# Verify expected values
assert abs(cal.flex[0].straight_adc - 310) < 1.0, f"Flex1 straight wrong: {cal.flex[0].straight_adc}"
assert abs(cal.flex[0].bent_adc - 380) < 1.0, f"Flex1 bent wrong: {cal.flex[0].bent_adc}"
assert abs(cal.flex[1].straight_adc - 300) < 1.0, f"Flex2 straight wrong: {cal.flex[1].straight_adc}"
assert abs(cal.flex[1].bent_adc - 270) < 1.0, f"Flex2 bent wrong: {cal.flex[1].bent_adc}"
print()
print("PASS: Calibration values match simulated sensor readings")

# Verify percent conversion
pct = cal.flex[1].to_percent(270)
assert pct == 1.0, f"Bent should be 100%: {pct}"
pct = cal.flex[1].to_percent(300)
assert pct == 0.0, f"Straight should be 0%: {pct}"
pct = cal.flex[1].to_percent(285)
assert 0.49 < pct < 0.51, f"Midpoint should be ~50%: {pct}"
print("PASS: Percent conversion correct (including decreasing-direction Flex2)")

# ── Test cancellation ─────────────────────────────────────────────────────────
print()
print("Testing cancellation...")
cal2 = Calibration()
orig_flex1 = (cal2.flex[0].straight_adc, cal2.flex[0].bent_adc)
acq = FakeAcq()
dash = FakeDashboard()
timer = FakeTimer(dash, acq)
sess = CalibrationSession(cal2, acq, dash)
acq.last_processed = {"flex": [320, 310, 315, 300], "pressure": 0, "timestamp_ms": 0}
sess.start()
sess.handle_command("x")
assert sess.state == CalibrationSession.IDLE, "Cancel should return to IDLE"
assert (cal2.flex[0].straight_adc, cal2.flex[0].bent_adc) == orig_flex1, \
    "Cancel must not modify profiles"
print("PASS: Cancellation leaves profiles untouched")

# ── Test rejection of tiny range ─────────────────────────────────────────────
print()
print("Testing range rejection...")
cal3 = Calibration()
acq = FakeAcq()
dash = FakeDashboard()

# Override the fake to produce nearly identical straight/bent values
class TinyRangeAcq(FakeAcq):
    def set_pose(self, sensor, straight=True):
        self._values[sensor] = 310 if straight else 311  # 1 ADC difference

acq = TinyRangeAcq()
timer = FakeTimer(dash, acq)
sess = CalibrationSession(cal3, acq, dash)
acq.last_processed = {"flex": [320, 310, 315, 300], "pressure": 0, "timestamp_ms": 0}
sess.start()

for sensor in range(4):
    assert sess.state == CalibrationSession.PROMPT_STRAIGHT
    acq.set_pose(sensor, straight=True)
    sess.handle_command("")
    while sess.state == CalibrationSession.COLLECT_STRAIGHT:
        timer.run_pending(1)
    acq.set_pose(sensor, straight=False)
    sess.handle_command("")
    while sess.state == CalibrationSession.COLLECT_BENT:
        timer.run_pending(1)

assert sess.state == CalibrationSession.CONFIRM
sess.handle_command("")  # save attempt
assert sess.state == CalibrationSession.IDLE

# All sensors should have been rejected (range 1 < MIN_RANGE_ADC 8)
# The original profiles should still be there
assert abs(cal3.flex[0].straight_adc - 307.0) < 0.1, \
    f"Flex1 should keep old profile: {cal3.flex[0].straight_adc}"
print("PASS: Tiny-range sensors rejected, old profiles kept")

# ── Test save/load round trip with new calibration ───────────────────────────
print()
print("Testing save/load round trip...")
import tempfile
tmp = tempfile.mktemp(suffix=".json")
cal4 = Calibration()
cal4.flex[0].straight_adc = 305.0
cal4.flex[0].bent_adc = 390.0
cal4.flex[0].is_calibrated = True
cal4.save(tmp)
cal5 = Calibration()
cal5.load(tmp)
assert abs(cal5.flex[0].straight_adc - 305.0) < 0.01
assert abs(cal5.flex[0].bent_adc - 390.0) < 0.01
os.unlink(tmp)
print("PASS: Save/load round trip works")

print()
print("=" * 60)
print("All calibration session tests passed.")
print("=" * 60)

# Clean up the temp calibration file
import os as _os2
if _os2.path.exists(_tmp_cal):
    _os2.unlink(_tmp_cal)
