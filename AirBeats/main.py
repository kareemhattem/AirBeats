# =============================================================================
# AirBeats — Main Entry Point
# main.py
#
# Thread architecture:
# ┌────────────────────────────────────────────────────────────────┐
# │  ACQUISITION THREAD  (daemon)                                  │
# │    SerialReader.read_packet()                                  │
# │      → SignalProcessor.process()  (calibration + filtering)   │
# │      → DataLogger.log()           (CSV, raw + processed)      │
# │      → packet_queue.put()         (to GUI thread)             │
# └─────────────────────────────┬──────────────────────────────────┘
#                               │  queue.Queue  (thread-safe)
# ┌─────────────────────────────▼──────────────────────────────────┐
# │  MAIN THREAD  (GUI thread)                                     │
# │    Dashboard._update() drains queue every 50 ms                │
# │    plt.show(block=True)                                        │
# └────────────────────────────────────────────────────────────────┘
#
# Commands (type in the terminal, press Enter):
#   r   → start recording
#   s   → stop recording
#   f   → cycle through available filters
#   q   → quit
# =============================================================================

import sys
import os
import threading
import queue
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import config
from acquisition.serial_reader import SerialReader
from storage.data_logger       import DataLogger
from visualization.dashboard   import Dashboard
from processing.calibration    import Calibration
from processing.filters        import (SignalProcessor, EMAFilter,
                                       MovingAverageFilter, ButterworthFilter,
                                       MedianFilter)
from sensor_fusion.fusion       import SensorFusion
from gesture_recognition.gestures import GestureRecognizer
from music.mapping              import MusicalMapping, SCALE_NAMES
from music.synthesizer          import Synthesizer, INSTRUMENT_NAMES
from music.midi_engine           import MidiEngine

# Filter cycle order for the 'f' keyboard command
_FILTER_CYCLE = [
    ("EMA",             EMAFilter,            {"alpha": config.EMA_ALPHA}),
    ("Moving Average",  MovingAverageFilter,  {"window": config.MA_WINDOW}),
    ("Butterworth",     ButterworthFilter,    {"cutoff_hz": config.BUTTERWORTH_CUTOFF_HZ,
                                               "sample_hz": config.SAMPLE_RATE_HZ,
                                               "order":     config.BUTTERWORTH_ORDER}),
    ("Median",          MedianFilter,         {"window": config.MEDIAN_WINDOW}),
    ("None",            None,                 {}),
]


# =============================================================================
# Interactive Flex Sensor Calibration Session
# =============================================================================
# A state machine driven by Tk after() callbacks (via dashboard.schedule_callback).
# Never blocks the acquisition thread or the audio thread.
#
# Flow per sensor (0..3):
#   PROMPT_STRAIGHT  → user straightens finger, presses Enter
#   COLLECT_STRAIGHT → 40 samples at 50 ms = 2 seconds, averaged
#   PROMPT_BENT      → user bends finger fully, presses Enter
#   COLLECT_BENT     → 40 samples at 50 ms = 2 seconds, averaged
# After all 4 sensors:
#   CONFIRM          → summary table shown, Enter saves, x cancels
#
# Safety: new values are held in a temp list. The live Calibration object is
# only modified (via set_flex_straight / set_flex_bent) and saved AFTER the
# user confirms. Cancelling at any point (x + Enter) leaves profiles intact.
#
# Validation: a sensor whose |bent - straight| < MIN_RANGE_ADC is rejected
# with an error message and keeps its previous profile.

class CalibrationSession:
    # Collection parameters
    COLLECT_INTERVAL_MS = 50     # 20 Hz sampling of the acquisition stream
    COLLECT_TICKS      = 40     # 40 x 50 ms = 2 seconds per pose
    MIN_RANGE_ADC      = 8.0    # Below this the calibration is unreliable

    # States
    IDLE, PROMPT_STRAIGHT, COLLECT_STRAIGHT, PROMPT_BENT, \
        COLLECT_BENT, CONFIRM, FINISHED = range(7)

    def __init__(self, cal, acq_thread, dashboard):
        self._cal        = cal
        self._acq        = acq_thread
        self._dashboard  = dashboard
        self.state       = self.IDLE
        self.sensor      = 0
        # temp storage: list of (straight_avg, bent_avg) or None if rejected
        self._measured   = [None, None, None, None]
        self._samples    = []
        self._tick_count = 0
        self._saved_ranges = [(f.straight_adc, f.bent_adc) for f in cal.flex]

    # ---------------------------------------------------------------------
    # Public control
    # ---------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        return self.state != self.IDLE

    def start(self) -> bool:
        """Begin calibration. Returns False if no data is flowing."""
        if self._acq.last_processed is None:
            print("[Calib] No sensor data yet — wait for acquisition to start.")
            return False
        self.state  = self.PROMPT_STRAIGHT
        self.sensor = 0
        self._measured = [None, None, None, None]
        print()
        print("=" * 60)
        print("  FLEX SENSOR CALIBRATION")
        print("  Type x + Enter at ANY prompt to cancel (profiles untouched).")
        print("=" * 60)
        self._show_prompt_straight()
        return True

    def handle_command(self, cmd) -> bool:
        """
        Called from _periodic_check with the latest keyboard command.
        cmd is None (no keypress this cycle), "" (bare Enter), or a letter.
        Returns True if the calibration session is active (command consumed).
        """
        if self.state == self.IDLE:
            return False

        # 'x' at any point cancels
        if cmd == "x":
            self._cancel()
            return True

        # cmd == "" means the user pressed bare Enter (advance the prompt).
        # cmd is None means no keypress this cycle — do nothing.
        if cmd == "":
            if self.state == self.PROMPT_STRAIGHT:
                self._begin_collect(self.COLLECT_STRAIGHT)
            elif self.state == self.PROMPT_BENT:
                self._begin_collect(self.COLLECT_BENT)
            elif self.state == self.CONFIRM:
                self._save()
        # During COLLECT states, keypresses are ignored (collection is fast)
        return True

    # ---------------------------------------------------------------------
    # Prompts
    # ---------------------------------------------------------------------

    def _show_prompt_straight(self):
        n = self.sensor + 1
        print()
        print(f"  [{n}/4] Flex{n}: hold finger {n} COMPLETELY STRAIGHT.")
        print(f"         When ready, press Enter to sample for 2 seconds.")

    def _show_prompt_bent(self):
        n = self.sensor + 1
        print()
        print(f"  [{n}/4] Flex{n}: BEND finger {n} as much as possible.")
        print(f"         When ready, press Enter to sample for 2 seconds.")

    # ---------------------------------------------------------------------
    # Collection (fast callback chain, 50 ms per tick)
    # ---------------------------------------------------------------------

    def _begin_collect(self, collect_state):
        self._samples    = []
        self._tick_count = 0
        self.state       = collect_state
        pose = "STRAIGHT" if collect_state == self.COLLECT_STRAIGHT else "BENT"
        print(f"         Sampling {pose} for 2 seconds... keep the finger still.")
        self._dashboard.schedule_callback(self.COLLECT_INTERVAL_MS,
                                          self._collect_tick)

    def _collect_tick(self):
        if self.state not in (self.COLLECT_STRAIGHT, self.COLLECT_BENT):
            return  # cancelled

        p = self._acq.last_processed
        if p is not None:
            self._samples.append(p["flex"][self.sensor])

        self._tick_count += 1

        # Progress indicator every 10 ticks (0.5 s)
        if self._tick_count % 10 == 0:
            print(f"         ... {self._tick_count}/{self.COLLECT_TICKS} samples")

        if self._tick_count >= self.COLLECT_TICKS:
            self._finish_collection()
        else:
            self._dashboard.schedule_callback(self.COLLECT_INTERVAL_MS,
                                              self._collect_tick)

    def _finish_collection(self):
        avg = sum(self._samples) / len(self._samples) if self._samples else 0.0
        n   = len(self._samples)

        if self.state == self.COLLECT_STRAIGHT:
            self._straight_avg = avg
            print(f"         Done: straight average = {avg:.1f} ADC ({n} samples)")
            self.state = self.PROMPT_BENT
            self._show_prompt_bent()
        else:
            self._bent_avg = avg
            print(f"         Done: bent average = {avg:.1f} ADC ({n} samples)")
            self._measured[self.sensor] = (self._straight_avg, self._bent_avg)
            self._validate_sensor()
            self.sensor += 1
            if self.sensor >= 4:
                self._show_summary()
                self.state = self.CONFIRM
                print()
                print(f"  Press Enter to SAVE these profiles, or x + Enter to CANCEL.")
            else:
                self.state = self.PROMPT_STRAIGHT
                self._show_prompt_straight()

    # ---------------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------------

    def _validate_sensor(self):
        straight, bent = self._measured[self.sensor]
        rng = bent - straight
        if abs(rng) < self.MIN_RANGE_ADC:
            print()
            print(f"  ! Flex{self.sensor+1} REJECTED: range is only {rng:+.1f} ADC "
                  f"(needs at least ±{self.MIN_RANGE_ADC:.0f}).")
            print(f"    straight={straight:.1f}, bent={bent:.1f}")
            print(f"    Previous profile for this sensor is KEPT.")
            self._measured[self.sensor] = None
        else:
            direction = "increases" if rng > 0 else "decreases"
            print(f"         Flex{self.sensor+1} OK: range {rng:+.1f} ADC "
                  f"(ADC {direction} when bent)")

    # ---------------------------------------------------------------------
    # Summary / save / cancel
    # ---------------------------------------------------------------------

    def _show_summary(self):
        print()
        print("  CALIBRATION SUMMARY")
        print(f"  {'Sensor':<8} {'Straight':>9} {'Bent':>9} {'Range':>9} {'Direction':<12}")
        print(f"  {'-'*8} {'-'*9} {'-'*9} {'-'*9} {'-'*12}")
        for i in range(4):
            if self._measured[i] is not None:
                s, b = self._measured[i]
                rng = b - s
                direction = "increases" if rng > 0 else "decreases"
                print(f"  Flex{i+1:<4} {s:>9.1f} {b:>9.1f} {rng:>+9.1f} {direction:<12}")
            else:
                s, b = self._saved_ranges[i]
                rng = b - s
                print(f"  Flex{i+1:<4} {s:>9.1f} {b:>9.1f} {rng:>+9.1f} "
                      f"{'(kept old)':<12}")

    def _save(self):
        applied = 0
        for i in range(4):
            if self._measured[i] is not None:
                s, b = self._measured[i]
                self._cal.set_flex_straight(i, s)
                self._cal.set_flex_bent(i, b)
                applied += 1
        if applied > 0:
            self._cal.save(config.CALIBRATION_FILE)
            print(f"\n  [Calib] {applied} sensor(s) calibrated and saved to "
                  f"{config.CALIBRATION_FILE}")
            print(f"  [Calib] Note: the new calibration takes effect on the next")
            print(f"  [Calib] acquisition restart (profiles are loaded at startup).")
        else:
            print("\n  [Calib] No sensors passed validation. Nothing saved.")
        self.state = self.FINISHED
        self.state = self.IDLE
        print(f"  [Calib] Returning to normal operation. Press d to verify.")

    def _cancel(self):
        self.state = self.IDLE
        print(f"\n  [Calib] Cancelled. Existing calibration profiles are untouched.")


# =============================================================================
# Keyboard input thread
# =============================================================================

class InputHandler:
    def __init__(self):
        self.command = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def _run(self):
        while True:
            try:
                self.command = input().strip().lower()
            except EOFError:
                break

    def get_command(self) -> str | None:
        cmd = self.command
        self.command = None
        return cmd


# =============================================================================
# Acquisition thread
# =============================================================================

class AcquisitionThread(threading.Thread):
    """
    Reads, processes, fuses, recognises, logs, and enqueues packets.
    No matplotlib or Tk calls ever happen in this thread.
    """

    def __init__(self, reader: SerialReader, logger: DataLogger,
                 processor: SignalProcessor,
                 fusion: SensorFusion,
                 recognizer: GestureRecognizer,
                 mapping: MusicalMapping,
                 midi_engine: MidiEngine,
                 packet_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True, name="AcquisitionThread")
        self._reader     = reader
        self._logger     = logger
        self._processor  = processor
        self._fusion     = fusion
        self._recognizer = recognizer
        self._mapping    = mapping
        self._midi       = midi_engine
        self._queue      = packet_queue
        self._stop       = stop_event

        self.packets_total   = 0
        self.packets_dropped = 0
        self._t_start        = None

        # Latest processed packet — stored for the GUI thread's diagnostic
        # command. The GUI thread only reads this dict during a 'd' command;
        # it never writes to it. Python's GIL makes dict reference assignment
        # atomic, so no lock is needed for this read-only diagnostic access.
        self.last_processed = None

    def run(self):
        self._t_start = time.perf_counter()

        while not self._stop.is_set():
            raw = self._reader.read_packet()
            if raw is None:
                continue

            # Phase 2: calibration + filtering
            processed = self._processor.process(raw)

            # Phase 3: sensor fusion (orientation + motion features)
            fusion_data = self._fusion.update(processed)
            processed.update(fusion_data)

            # Phase 3: gesture recognition
            gesture = self._recognizer.update(processed)
            processed["gesture_id"]        = gesture["id"]
            processed["gesture_name"]      = gesture["name"]
            processed["gesture_confidence"]= gesture["confidence"]

            # Phase 4: musical mapping
            musical_state = self._mapping.update(processed)
            processed["musical_state"] = musical_state

            # Phase 5: sound generation
            self._midi.update(musical_state)

            self.packets_total += 1

            # Store for diagnostics (read-only from GUI thread)
            self.last_processed = processed

            # Log to CSV
            self._logger.log(processed)

            # Send to dashboard
            try:
                self._queue.put_nowait(processed)
            except queue.Full:
                self.packets_dropped += 1

    @property
    def measured_rate_hz(self) -> float:
        if self._t_start is None or self.packets_total == 0:
            return 0.0
        elapsed = time.perf_counter() - self._t_start
        return self.packets_total / elapsed if elapsed > 0 else 0.0


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("  AirBeats — Gesture-Controlled Wearable Musical Instrument")
    print("  Phase 5: Sound Generation")
    print("=" * 60)
    print(f"  Port       : {config.SERIAL_PORT}")
    print(f"  Baud       : {config.SERIAL_BAUD}")
    print(f"  Rate       : {config.SAMPLE_RATE_HZ} Hz")
    print(f"  Filter     : {config.DEFAULT_FILTER}")
    print(f"  Fusion     : Complementary (alpha={config.COMPLEMENTARY_ALPHA})")
    print(f"  Scale      : {config.DEFAULT_SCALE}")
    print(f"  Instrument : {config.DEFAULT_INSTRUMENT}")
    print("-" * 60)
    print("  Commands (type then press Enter):")
    print("    r  → start recording")
    print("    s  → stop recording")
    print("    f  → cycle filter")
    print("    m  → cycle musical scale")
    print("    i  → cycle instrument")
    print("    t  → test tone (play C4 for 1 second directly)")
    print("    d  → diagnostic info (sensor values, note counts)")
    print("    c  → calibrate flex sensors (interactive, 4 sensors)")
    print("    z  → re-zero IMU reference (hold hand neutral, then press)")
    print("    q  → quit")
    print("=" * 60)

    # --- Calibration ---
    cal = Calibration()
    cal.load(config.CALIBRATION_FILE)
    cal.print_summary()

    # --- Signal processor ---
    proc = SignalProcessor(cal)
    _filter_index = [0]
    _apply_filter(proc, _filter_index[0])

    # --- Sensor fusion ---
    fusion = SensorFusion(
        sample_rate_hz=config.SAMPLE_RATE_HZ,
        alpha=config.COMPLEMENTARY_ALPHA,
        accel_confidence_threshold=config.ACCEL_CONFIDENCE_THRESHOLD_G,
    )
    print(f"[Fusion] Complementary filter initialised (alpha={config.COMPLEMENTARY_ALPHA})")

    # --- Gesture recognizer ---
    recognizer = GestureRecognizer()
    print(f"[Gestures] {len(recognizer._gestures)} gestures loaded")

    # --- Musical mapping ---
    mapping = MusicalMapping()
    mapping.print_summary()

    # --- Phase 5: Sound generation ---
    synth = Synthesizer()
    synth.start()
    if synth.is_running:
        midi_engine = MidiEngine(synth)
        print(f"[Synth] Instrument: {synth.instrument_name}")
    else:
        midi_engine = None
        print("[Synth] WARNING: Audio not available. Running in silent mode.")

    # --- Other components ---
    reader    = SerialReader()
    logger    = DataLogger()
    dashboard = Dashboard()
    keyboard  = InputHandler()

    packet_queue = queue.Queue(maxsize=200)
    stop_event   = threading.Event()

    # --- Connect ---
    if not reader.connect():
        print(f"\n[Main] Could not connect. Check SERIAL_PORT in config.py.")
        print(f"       Current port: {config.SERIAL_PORT}")
        sys.exit(1)

    dashboard.set_queue(packet_queue)
    dashboard.start()
    keyboard.start()

    acq_thread = AcquisitionThread(reader, logger, proc, fusion, recognizer,
                                   mapping, midi_engine, packet_queue, stop_event)
    acq_thread.start()

    # Calibration session (state machine, driven by _periodic_check)
    calib_session = CalibrationSession(cal, acq_thread, dashboard)

    print("\n[Main] Acquisition running.\n")

    last_report_time = time.time()

    import matplotlib.pyplot as plt

    def _periodic_check():
        nonlocal last_report_time

        cmd = keyboard.get_command()

        # Route commands to the calibration session first when it is active.
        # handle_command returns True if it consumed the command (including
        # the empty Enter key that advances prompts).
        if calib_session.is_active:
            calib_session.handle_command(cmd)
            # Do not process the same command twice below
            cmd = None

        if cmd == "r":
            if not logger.is_recording:
                logger.start()
                dashboard.set_recording(True)
                print(f"[Main] Recording started.")
            else:
                print("[Main] Already recording.")
        elif cmd == "s":
            if logger.is_recording:
                logger.stop()
                dashboard.set_recording(False)
            else:
                print("[Main] Not recording.")
        elif cmd == "f":
            _filter_index[0] = (_filter_index[0] + 1) % len(_FILTER_CYCLE)
            _apply_filter(proc, _filter_index[0])
            name = _FILTER_CYCLE[_filter_index[0]][0]
            print(f"[Main] Filter → {name}")
        elif cmd == "m":
            new_scale = mapping.cycle_scale()
            print(f"[Main] Scale → {new_scale}")
        elif cmd == "i":
            if synth.is_running:
                new_inst = synth.cycle_instrument()
                print(f"[Main] Instrument → {new_inst}")
        elif cmd == "t":
            # Test tone: play MIDI 60 (C4) directly through the synthesizer
            # for 1 second, bypassing all sensors and musical mapping.
            # This confirms the synth works and lets the user hear each
            # instrument preset clearly.
            if synth.is_running:
                print(f"[Main] Test tone: MIDI 60 (C4) for 1.0s — instrument={synth.instrument_name}")
                synth.note_on(60, velocity=100)
                # Schedule note-off after 1 second via Tk after()
                dashboard.schedule_callback(1000, lambda: synth.note_off(60))
            else:
                print("[Main] Synth not running, cannot play test tone.")
        elif cmd == "d":
            # ── Comprehensive flex-to-audio pipeline diagnostic ──────────
            # Traces: raw ADC → calibrated % → filtered % → trigger threshold
            #         → hysteresis state → MIDI note → synth voice
            # All data is read from the last processed packet (read-only).
            # Does NOT block the acquisition thread or audio thread.
            p = acq_thread.last_processed
            if p is None:
                print("[Main] No packets processed yet.")
            else:
                print()
                print("=" * 72)
                print("  FLEX SENSOR → MUSICAL NOTE PIPELINE DIAGNOSTIC")
                print("=" * 72)

                # --- Acquisition status ---
                print(f"  Packets: {acq_thread.packets_total} | "
                      f"Rate: {acq_thread.measured_rate_hz:.1f} Hz | "
                      f"Malformed: {reader.packets_malformed}")
                print()

                # --- Calibration profiles ---
                print("  CALIBRATION PROFILES (from config / calibration.json):")
                print(f"  {'Sensor':<8} {'Straight':>9} {'Bent':>9} {'Range':>9} {'Direction':<12}")
                for i in range(4):
                    fp = cal.flex[i]
                    direction = "increases" if fp.range_adc > 0 else "decreases"
                    print(f"  {fp.name:<8} {fp.straight_adc:>9.1f} {fp.bent_adc:>9.1f} "
                          f"{fp.range_adc:>+9.1f} {direction:<12}")
                print()

                # --- Per-sensor trace ---
                raw_flex      = p.get("flex", [0, 0, 0, 0])
                flex_pct      = p.get("flex_pct", [0, 0, 0, 0])
                flex_filtered = p.get("flex_filtered")
                finger_active = mapping._finger_active

                # Determine which value the mapping uses for triggering.
                # mapping.update() reads packet["flex_pct"] (or flex_filtered
                # if the filter is active — the mapping does NOT use filtered
                # values, it uses flex_pct directly from calibration).
                # Let's check what mapping actually received:
                # mapping.update() reads flex_pct from the packet dict.

                print("  PER-SENSOR TRACE:")
                print(f"  {'Ch':<4} {'Raw ADC':>8} {'Calib %':>8} {'Filt %':>8} "
                      f"{'>=70%':>6} {'<=30%':>6} {'Held':>6} {'Note':>8} {'Velocity':>9}")
                print(f"  {'─'*4} {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*6} {'─'*6} {'─'*8} {'─'*9}")

                for i in range(4):
                    raw_val = raw_flex[i] if i < len(raw_flex) else 0
                    pct_val = flex_pct[i] if i < len(flex_pct) else 0.0
                    filt_val = flex_filtered[i] if flex_filtered and i < len(flex_filtered) else None

                    above_on  = pct_val >= config.GESTURE_CLOSE_THRESHOLD
                    below_off = pct_val <= config.GESTURE_OPEN_THRESHOLD
                    held      = finger_active[i] if i < len(finger_active) else False

                    # What MIDI note would this finger produce?
                    midi_note = mapping._midi_note_for_finger(i, p.get("roll", 0))
                    note_name = mapping._note_name(midi_note)

                    filt_str = f"{filt_val*100:.1f}%" if filt_val is not None else "  N/A"
                    held_str = "YES" if held else "no"

                    print(f"  F{i+1:<3} {raw_val:>8} {pct_val*100:>7.1f}% {filt_str:>8} "
                          f"{'YES' if above_on else 'no':>6} {'YES' if below_off else 'no':>6} "
                          f"{held_str:>6} {note_name:>8} "
                          f"{mapping._finger_velocity[i]:>9}")

                print()

                # --- Thresholds ---
                print(f"  TRIGGER THRESHOLDS:")
                print(f"    Note ON  (close): {config.GESTURE_CLOSE_THRESHOLD:.2f} "
                      f"({config.GESTURE_CLOSE_THRESHOLD*100:.0f}%)")
                print(f"    Note OFF (open):  {config.GESTURE_OPEN_THRESHOLD:.2f} "
                      f"({config.GESTURE_OPEN_THRESHOLD*100:.0f}%)")
                print(f"    Hysteresis band:  {config.GESTURE_OPEN_THRESHOLD:.2f} – "
                      f"{config.GESTURE_CLOSE_THRESHOLD:.2f} "
                      f"({config.GESTURE_OPEN_THRESHOLD*100:.0f}%–"
                      f"{config.GESTURE_CLOSE_THRESHOLD*100:.0f}%)")
                print()

                # --- Musical state ---
                ms = p.get("musical_state", {})
                print(f"  MUSICAL STATE:")
                print(f"    Mode:          {ms.get('mode', '?')}")
                print(f"    Scale:         {ms.get('scale', '?')}")
                print(f"    Octave shift:  {ms.get('octave_shift', 0):+d}")
                print(f"    Pitch bend:    {ms.get('pitch_bend', 0):+d}")
                print(f"    Modulation:    {ms.get('modulation', 0)}")
                active = ms.get("active_notes", [])
                released = ms.get("released_notes", [])
                print(f"    Active events   (this frame): {len(active)}")
                for n in active:
                    print(f"      finger={n['finger']} note={n['midi_note']} "
                          f"({mapping._note_name(n['midi_note'])}) vel={n['velocity']}")
                print(f"    Released events(this frame): {len(released)}")
                for n in released:
                    print(f"      finger={n['finger']} note={n['midi_note']}")
                print()

                # --- MIDI engine / synth state ---
                print(f"  MIDI ENGINE:")
                if midi_engine:
                    print(f"    Total note-on events:  {midi_engine.total_notes_on}")
                    print(f"    Total note-off events: {midi_engine.total_notes_off}")
                    print(f"    Currently active notes: {midi_engine.active_note_count}")
                else:
                    print(f"    (not running)")
                print()

                print(f"  SYNTHESIZER:")
                if synth.is_running:
                    print(f"    Running: YES")
                    print(f"    Instrument: {synth.instrument_name}")
                    print(f"    Active voices: {synth.active_note_count}")
                    print(f"    Total notes played: {synth.notes_played}")
                else:
                    print(f"    Running: NO")
                print()

                # --- Diagnosis hint ---
                print("  DIAGNOSIS:")
                max_pct = max(flex_pct) if flex_pct else 0
                if max_pct < config.GESTURE_CLOSE_THRESHOLD:
                    print(f"    ! Max flex% is {max_pct*100:.1f}% — below the "
                          f"{config.GESTURE_CLOSE_THRESHOLD*100:.0f}% trigger threshold.")
                    print(f"    ! Notes will NOT trigger. The calibration ranges may")
                    print(f"    ! not match the physical sensor values when worn.")
                    print(f"    ! Check the 'Calib %' column above vs the raw ADC values.")
                else:
                    print(f"    Max flex% is {max_pct*100:.1f}% — above trigger threshold.")
                    n_held = sum(1 for v in flex_pct if v >= config.GESTURE_CLOSE_THRESHOLD)
                    print(f"    {n_held} finger(s) currently above trigger threshold.")
                print("=" * 72)
                print()
        elif cmd == "c":
            # ── Start interactive flex sensor calibration ───────────
            # Uses a state machine driven by _periodic_check callbacks.
            # Does NOT block the acquisition or audio threads.
            # Collects raw ADC from acq_thread.last_processed["flex"][i].
            if acq_thread.last_processed is None:
                print("[Main] No sensor data yet. Wait for acquisition to start.")
            else:
                calib_session.start()
        elif cmd == "z":
            # Re-zero the IMU reference frame. The user holds the hand in
            # the desired neutral playing pose, then presses z + Enter.
            # The fusion will average the next 50 packets (1 s) as the
            # new zero reference for roll/pitch.
            fusion.recapture_reference()
            print("  [Main] Hold your hand still in the neutral pose for 1 second...")
        elif cmd == "q":
            print("[Main] Quit.")
            _shutdown()
            return

        now = time.time()
        if now - last_report_time >= 5.0:
            print(
                f"[Main] {acq_thread.packets_total} packets | "
                f"{acq_thread.measured_rate_hz:.1f} Hz | "
                f"{reader.packets_malformed} malformed | "
                f"{acq_thread.packets_dropped} dropped | "
                f"Filter: {_FILTER_CYCLE[_filter_index[0]][0]} | "
                f"Inst: {synth.instrument_name if synth.is_running else 'N/A'} | "
                f"Notes: {midi_engine.active_note_count if midi_engine else 0} | "
                f"Rec: {logger.is_recording}"
            )
            last_report_time = now

        if dashboard.is_open():
            dashboard.schedule_callback(500, _periodic_check)
        else:
            print("[Main] Dashboard closed.")
            _shutdown()

    def _shutdown():
        stop_event.set()
        if logger.is_recording:
            logger.stop()
        if midi_engine:
            midi_engine.all_notes_off()
        if synth.is_running:
            synth.stop()
        reader.disconnect()
        plt.close("all")
        print("[Main] Shutdown complete.")

    dashboard.schedule_callback(500, _periodic_check)

    try:
        plt.show(block=True)
    except KeyboardInterrupt:
        print("\n[Main] Interrupted.")
    finally:
        stop_event.set()
        if logger.is_recording:
            logger.stop()
        if midi_engine:
            midi_engine.all_notes_off()
        if synth.is_running:
            synth.stop()
        reader.disconnect()
        print("[Main] Shutdown complete.")


def _apply_filter(proc: SignalProcessor, index: int):
    """Apply the filter at the given cycle index to the processor."""
    name, cls, kwargs = _FILTER_CYCLE[index]
    proc.set_filter(cls, **kwargs)


if __name__ == "__main__":
    main()
