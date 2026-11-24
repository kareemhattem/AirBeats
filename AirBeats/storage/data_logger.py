# =============================================================================
# AirBeats — Data Logger
# storage/data_logger.py
#
# Responsibilities:
#   - Create a new timestamped CSV file for each recording session
#   - Write one row per packet, preserving all raw sensor values
#   - Support start/stop recording so the user controls when data is saved
#   - Never overwrite an existing recording (filenames include date + time)
#   - Keep the write operation fast so it does not stall the acquisition loop
#
# CSV column order:
#   timestamp_ms, pressure, flex1, flex2, flex3, flex4,
#   accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
# =============================================================================

import csv
import os
from datetime import datetime

import config

# CSV column header — raw columns first, then processed columns.
# Processed columns are present only when calibration + filtering are active.
_RAW_COLUMNS = config.ALL_CHANNEL_NAMES   # timestamp_ms + 5 analog + 6 IMU

_PROCESSED_COLUMNS = [
    "flex1_pct", "flex2_pct", "flex3_pct", "flex4_pct",
    "pressure_pct",
    "accel_x_cal", "accel_y_cal", "accel_z_cal",
    "gyro_x_cal",  "gyro_y_cal",  "gyro_z_cal",
    "flex1_filt",  "flex2_filt",  "flex3_filt",  "flex4_filt",
    "pressure_filt",
    "accel_x_filt","accel_y_filt","accel_z_filt",
    "gyro_x_filt", "gyro_y_filt", "gyro_z_filt",
    # Phase 3: sensor fusion + gesture recognition
    "roll_deg", "pitch_deg", "yaw_gyro_deg",
    "gyro_mag", "accel_mag", "motion_intensity",
    "is_moving",
    "gesture_id", "gesture_name", "gesture_confidence",
    # Phase 4: musical mapping
    "musical_mode", "musical_scale", "musical_octave_shift",
    "musical_pitch_bend", "musical_modulation",
    "musical_active_notes",
]

ALL_CSV_COLUMNS = _RAW_COLUMNS + _PROCESSED_COLUMNS


class DataLogger:
    """
    Writes raw sensor packets to a CSV file.

    Usage:
        logger = DataLogger()
        logger.start()          # Opens a new file and writes the header
        logger.log(packet)      # Writes one row
        logger.stop()           # Flushes and closes the file
    """

    def __init__(self, recording_dir: str = None):
        """
        Parameters
        ----------
        recording_dir : str, optional
            Directory where CSV files are saved. Defaults to config.RECORDING_DIR.
        """
        self._dir      = recording_dir or config.RECORDING_DIR
        self._file     = None     # open file handle
        self._writer   = None     # csv.writer instance
        self._path     = None     # current file path (for display)
        self._recording = False
        self.rows_written = 0

        # Create the recordings directory if it does not exist yet
        os.makedirs(self._dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Session control
    # -------------------------------------------------------------------------

    def start(self) -> str:
        """
        Open a new CSV file and write the column header.

        Returns the full path of the created file so the caller can display it.
        """
        if self._recording:
            print("[DataLogger] Already recording. Call stop() first.")
            return self._path

        filename   = self._make_filename()
        self._path = os.path.join(self._dir, filename)

        # newline='' is required by the csv module on Windows to prevent
        # extra blank lines from appearing between rows.
        self._file   = open(self._path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)

        # Write header using the channel names from config
        self._writer.writerow(ALL_CSV_COLUMNS)

        self._recording   = True
        self.rows_written = 0

        print(f"[DataLogger] Recording started → {self._path}")
        return self._path

    def stop(self):
        """Flush buffered data and close the file."""
        if not self._recording:
            return

        self._file.flush()
        self._file.close()
        self._recording = False

        print(
            f"[DataLogger] Recording stopped. "
            f"{self.rows_written} rows saved to {self._path}"
        )

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def current_file(self) -> str | None:
        return self._path

    # -------------------------------------------------------------------------
    # Writing
    # -------------------------------------------------------------------------

    def log(self, packet: dict):
        """
        Write one sensor packet as a row in the CSV.

        Accepts both raw packets (from SerialReader) and processed packets
        (from SignalProcessor). Processed columns are written when available;
        otherwise those cells are left empty so the CSV structure is consistent.

        Raw data is always preserved in the first columns.
        """
        if not self._recording or packet is None:
            return

        # --- Raw columns (always present) ---
        row = [
            packet["timestamp_ms"],
            packet["pressure"],
            packet["flex"][0],
            packet["flex"][1],
            packet["flex"][2],
            packet["flex"][3],
            packet["accelerometer"][0],
            packet["accelerometer"][1],
            packet["accelerometer"][2],
            packet["gyroscope"][0],
            packet["gyroscope"][1],
            packet["gyroscope"][2],
        ]

        # --- Processed columns (present after Phase 2 calibration) ---
        fp   = packet.get("flex_pct")
        pp   = packet.get("pressure_pct")
        ac   = packet.get("accel_cal")
        gc   = packet.get("gyro_cal")
        ff   = packet.get("flex_filtered")
        pf   = packet.get("pressure_filtered")
        af   = packet.get("accel_filtered")
        gf   = packet.get("gyro_filtered")

        if fp is not None:
            row += [fp[0], fp[1], fp[2], fp[3]]
        else:
            row += ["", "", "", ""]

        row += [pp if pp is not None else ""]

        if ac is not None:
            row += [ac[0], ac[1], ac[2]]
        else:
            row += ["", "", ""]

        if gc is not None:
            row += [gc[0], gc[1], gc[2]]
        else:
            row += ["", "", ""]

        if ff is not None:
            row += [ff[0], ff[1], ff[2], ff[3]]
        else:
            row += ["", "", "", ""]

        row += [pf if pf is not None else ""]

        if af is not None:
            row += [af[0], af[1], af[2]]
        else:
            row += ["", "", ""]

        if gf is not None:
            row += [gf[0], gf[1], gf[2]]
        else:
            row += ["", "", ""]

        # --- Phase 3: sensor fusion + gesture recognition ---
        row += [
            packet.get("roll", ""),
            packet.get("pitch", ""),
            packet.get("yaw_gyro", ""),
            packet.get("gyro_magnitude", ""),
            packet.get("accel_magnitude", ""),
            packet.get("motion_intensity", ""),
            int(packet["is_moving"]) if "is_moving" in packet else "",
            packet.get("gesture_id", ""),
            packet.get("gesture_name", ""),
            packet.get("gesture_confidence", ""),
        ]

        # --- Phase 4: musical mapping ---
        ms = packet.get("musical_state", {})
        active_notes = ms.get("active_notes", [])
        active_str = ";".join(
            f"{n['finger']}:{n['midi_note']}:{n['velocity']}" for n in active_notes
        ) if active_notes else ""
        row += [
            ms.get("mode", ""),
            ms.get("scale", ""),
            ms.get("octave_shift", ""),
            ms.get("pitch_bend", ""),
            ms.get("modulation", ""),
            active_str,
        ]

        self._writer.writerow(row)
        self.rows_written += 1

        # Flush every 50 rows (~1 second at 50 Hz)
        if self.rows_written % 50 == 0:
            self._file.flush()

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _make_filename() -> str:
        """
        Generate a unique filename using the current date and time.
        Example: airbeats_raw_2024-11-23_14-05-32.csv
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return f"{config.RECORDING_PREFIX}_{timestamp}.csv"
