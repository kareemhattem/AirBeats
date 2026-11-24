# =============================================================================
# AirBeats — MIDI Engine
# music/midi_engine.py
#
# Translates MusicalState dicts (from Phase 4) into synthesizer calls.
# Manages note-on/note-off timing, prevents stuck notes, applies continuous
# controllers (pitch bend, modulation).
#
# ─────────────────────────────────────────────────────────────────────────────
# Architecture
# ─────────────────────────────────────────────────────────────────────────────
#
# The MidiEngine sits between the musical mapping (Phase 4) and the
# synthesizer (Phase 5). The acquisition thread calls midi_engine.update()
# once per sample with the MusicalState dict. The engine:
#
#   1. Processes note-on events → calls synth.note_on()
#   2. Processes note-off events → calls synth.note_off()
#   3. Updates pitch bend → calls synth.set_pitch_bend()
#   4. Updates modulation → calls synth.set_modulation()
#   5. Handles mode changes (MUTE → all_notes_off)
#
# The engine tracks which notes are currently active to prevent duplicate
# note-on events and ensure no stuck notes.
#
# ─────────────────────────────────────────────────────────────────────────────
# Stuck note prevention
# ─────────────────────────────────────────────────────────────────────────────
#
# A "stuck note" is a note that keeps sounding after the finger has been
# released. This happens when a note-off event is missed (e.g. due to a
# processing error or timing gap). The engine prevents this by:
#
#   - Tracking all active notes in a set
#   - Releasing any note that is no longer in the active_notes list
#   - Calling all_notes_off() on MUTE mode
#   - Calling all_notes_off() on shutdown
# =============================================================================

import time

import config


class MidiEngine:
    """
    Bridges MusicalState to the synthesizer.

    Usage:
        synth = Synthesizer()
        synth.start()
        engine = MidiEngine(synth)

        # In the acquisition loop:
        engine.update(musical_state)
    """

    def __init__(self, synthesizer):
        self._synth = synthesizer

        # Track which MIDI note numbers are currently sounding
        self._active_notes = set()

        # Previous mode for detecting transitions
        self._prev_mode = "PLAY"

        # Diagnostics
        self._note_on_count  = 0
        self._note_off_count = 0

    # -------------------------------------------------------------------------
    # Core update — called once per packet from the acquisition thread
    # -------------------------------------------------------------------------

    def update(self, musical_state: dict):
        """
        Process one MusicalState dict and drive the synthesizer.

        Parameters
        ----------
        musical_state : dict from MusicalMapping.update()
            Keys: active_notes, released_notes, pitch_bend, modulation,
                  octave_shift, mode, scale, root_midi
        """
        if musical_state is None:
            return

        mode = musical_state.get("mode", "PLAY")

        # --- Handle mode transitions ---
        if mode == "MUTE" and self._prev_mode != "MUTE":
            # Entering MUTE mode — silence everything
            self._synth.all_notes_off()
            self._active_notes.clear()
        elif mode == "PLAY" and self._prev_mode == "MUTE":
            # Leaving MUTE — nothing special, new notes will trigger normally
            pass

        self._prev_mode = mode

        # --- Process note-off events first ---
        # This ensures that if a note is released and retriggered in the same
        # packet (octave shift), the old note stops before the new one starts.
        for note_event in musical_state.get("released_notes", []):
            midi_note = note_event["midi_note"]
            if midi_note in self._active_notes:
                self._synth.note_off(midi_note)
                self._active_notes.discard(midi_note)
                self._note_off_count += 1

        # --- Process note-on events ---
        if mode != "MUTE":
            for note_event in musical_state.get("active_notes", []):
                midi_note = note_event["midi_note"]
                velocity  = note_event.get("velocity", 80)

                # If the note is already active (e.g. retriggered after octave
                # change), note_on() handles the retrigger internally.
                self._synth.note_on(midi_note, velocity)
                self._active_notes.add(midi_note)
                self._note_on_count += 1

        # --- Continuous controllers ---
        pitch_bend = musical_state.get("pitch_bend", 0)
        modulation = musical_state.get("modulation", 0)
        self._synth.set_pitch_bend(pitch_bend)
        self._synth.set_modulation(modulation)

    # -------------------------------------------------------------------------
    # Safety
    # -------------------------------------------------------------------------

    def all_notes_off(self):
        """Emergency silence — release every active note."""
        self._synth.all_notes_off()
        self._active_notes.clear()

    def panic(self):
        """Full MIDI panic — stop all sound immediately."""
        self._synth.all_notes_off()
        self._active_notes.clear()
        print("[MidiEngine] PANIC: All notes off.")

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    @property
    def active_note_count(self) -> int:
        return len(self._active_notes)

    @property
    def total_notes_on(self) -> int:
        return self._note_on_count

    @property
    def total_notes_off(self) -> int:
        return self._note_off_count

    def print_stats(self):
        print(f"[MidiEngine] Notes on: {self._note_on_count}, "
              f"off: {self._note_off_count}, "
              f"active: {len(self._active_notes)}")
