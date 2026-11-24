# =============================================================================
# AirBeats — Serial Reader
# acquisition/serial_reader.py
#
# Responsibilities:
#   - Open and maintain a serial connection to the Arduino
#   - Read one line at a time from the serial port
#   - Parse the CSV packet into a structured Python dictionary
#   - Validate the packet (correct field count, numeric values)
#   - Return None for malformed or incomplete packets instead of crashing
#   - Report connection errors without killing the application
#
# Expected packet format (12 comma-separated fields + newline):
#   timestamp_ms,pressure,flex1,flex2,flex3,flex4,ax,ay,az,gx,gy,gz
#
# Returned dictionary:
#   {
#       "timestamp_ms"  : int,
#       "pressure"      : int,
#       "flex"          : [int, int, int, int],
#       "accelerometer" : [float, float, float],
#       "gyroscope"     : [float, float, float],
#   }
# =============================================================================

import serial
import serial.tools.list_ports

# Config is imported from the project-level config.py.
# Because serial_reader.py lives in a sub-package (acquisition/), Python needs
# the parent package on sys.path. main.py handles that by running from the
# AirBeats/ directory, which makes "import config" work everywhere.
import config

# Total number of fields we expect in every valid packet
_EXPECTED_FIELDS = 12


class SerialReader:
    """
    Wraps a pyserial connection to the Arduino and parses incoming packets.

    Usage:
        reader = SerialReader()
        reader.connect()
        while True:
            packet = reader.read_packet()
            if packet is not None:
                # use packet
    """

    def __init__(self, port: str = None, baud: int = None):
        """
        Parameters
        ----------
        port : str, optional
            Serial port string, e.g. "COM4". Defaults to config.SERIAL_PORT.
        baud : int, optional
            Baud rate. Defaults to config.SERIAL_BAUD.
        """
        self.port    = port or config.SERIAL_PORT
        self.baud    = baud or config.SERIAL_BAUD
        self._serial = None          # serial.Serial instance, set in connect()
        self._connected = False

        # Counters for diagnostics
        self.packets_received  = 0
        self.packets_malformed = 0

    # -------------------------------------------------------------------------
    # Connection management
    # -------------------------------------------------------------------------

    def connect(self) -> bool:
        """
        Open the serial port and wait for the Arduino READY message.

        Returns True if the connection succeeded, False otherwise.
        """
        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baud,
                timeout=config.SERIAL_TIMEOUT_S,
            )
            print(f"[SerialReader] Connected to {self.port} at {self.baud} baud.")

            # Wait for the "READY" line the firmware sends after calibration.
            # This prevents us from parsing partial lines that arrive before
            # the Arduino has fully started.
            print("[SerialReader] Waiting for Arduino READY signal...")
            self._wait_for_ready()

            self._connected = True
            return True

        except serial.SerialException as e:
            print(f"[SerialReader] ERROR: Could not open {self.port}: {e}")
            self._list_available_ports()
            return False

    def disconnect(self):
        """Close the serial port cleanly."""
        if self._serial and self._serial.is_open:
            self._serial.close()
            self._connected = False
            print("[SerialReader] Disconnected.")

    @property
    def is_connected(self) -> bool:
        return self._connected and self._serial is not None and self._serial.is_open

    # -------------------------------------------------------------------------
    # Packet reading
    # -------------------------------------------------------------------------

    def read_packet(self) -> dict | None:
        """
        Read one line from the serial port and parse it.

        Returns a packet dictionary on success, None on failure.
        Never raises an exception — errors are logged and None is returned
        so the calling loop can continue without crashing.
        """
        if not self.is_connected:
            return None

        try:
            raw_line = self._serial.readline()
        except serial.SerialException as e:
            print(f"[SerialReader] Serial read error: {e}")
            self._connected = False
            return None

        if not raw_line:
            # Timeout — no data arrived within SERIAL_TIMEOUT_S
            return None

        # Decode bytes to string and strip whitespace / CRLF
        try:
            line = raw_line.decode("utf-8", errors="replace").strip()
        except Exception:
            self.packets_malformed += 1
            return None

        # Skip non-data lines (READY, ERROR messages, blank lines)
        if not line or line.startswith("READY") or line.startswith("ERROR"):
            if line.startswith("ERROR"):
                print(f"[SerialReader] Arduino reported: {line}")
            return None

        return self._parse_line(line)

    # -------------------------------------------------------------------------
    # Parsing
    # -------------------------------------------------------------------------

    def _parse_line(self, line: str) -> dict | None:
        """
        Parse a single CSV line into a packet dictionary.

        Returns the dict on success, None if the line is malformed.

        Diagnostic note on 18-field packets:
          An 18-field line contains exactly 12 + 6 fields, which is the
          signature of two packets being joined at a \r\n boundary in the
          serial receive buffer. This happens when the acquisition loop is
          blocked (e.g. by a GUI redraw) long enough for multiple packets to
          accumulate, and the OS serial driver merges them before readline()
          drains them. The fix is architectural: keep the acquisition loop
          free of blocking GUI calls. The parser correctly rejects these
          lines as malformed and counts them for diagnosis.
        """
        parts = line.split(',')

        # Validate field count before touching any values
        if len(parts) != _EXPECTED_FIELDS:
            self.packets_malformed += 1
            # Print the first 10 occurrences in full, then every 100th,
            # so the terminal is not flooded but the problem remains visible.
            if self.packets_malformed <= 10 or self.packets_malformed % 100 == 0:
                # Show a truncated version of the raw line so the field-
                # doubling pattern is visible for diagnosis.
                preview = line if len(line) <= 120 else line[:117] + "..."
                print(
                    f"[SerialReader] Malformed packet #{self.packets_malformed} "
                    f"({len(parts)} fields, expected {_EXPECTED_FIELDS}): "
                    f"'{preview}'"
                )
            return None

        try:
            packet = {
                "timestamp_ms" : int(parts[0]),
                "pressure"     : int(parts[1]),
                "flex"         : [int(parts[2]),
                                   int(parts[3]),
                                   int(parts[4]),
                                   int(parts[5])],
                "accelerometer": [float(parts[6]),
                                   float(parts[7]),
                                   float(parts[8])],
                "gyroscope"    : [float(parts[9]),
                                   float(parts[10]),
                                   float(parts[11])],
            }
        except ValueError as e:
            self.packets_malformed += 1
            if self.packets_malformed <= 5 or self.packets_malformed % 100 == 0:
                print(f"[SerialReader] Parse error on '{line}': {e}")
            return None

        self.packets_received += 1
        return packet

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _wait_for_ready(self, max_lines: int = 100):
        """
        Read lines until "READY" is received or max_lines is exhausted.
        This discards any partial data sent before the Arduino was ready.
        """
        for _ in range(max_lines):
            try:
                raw = self._serial.readline()
                line = raw.decode("utf-8", errors="replace").strip()
                print(f"[SerialReader] Arduino says: {line}")
                if line == "READY":
                    print("[SerialReader] Arduino is ready. Starting acquisition.")
                    return
            except Exception:
                return  # If anything goes wrong, just proceed

        print("[SerialReader] Warning: READY not received. Proceeding anyway.")

    @staticmethod
    def _list_available_ports():
        """Print available serial ports to help the user pick the right one."""
        ports = serial.tools.list_ports.comports()
        if ports:
            print("[SerialReader] Available serial ports:")
            for p in ports:
                print(f"  {p.device} — {p.description}")
        else:
            print("[SerialReader] No serial ports detected.")
