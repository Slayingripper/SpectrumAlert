"""Tests for RF fingerprint extraction and matching.

Two properties matter more than anything else here, and both are behavioural
consequences of the design choices in ``fingerprinting.py``:

* **Gain invariance.** A signature must survive a change of receiver gain, or a
  library is only ever usable on the receiver that captured it.
* **Refusing to guess.** An empty library must say "unknown", and an unknown
  transmitter must not be identified. A confident wrong device ID is worse than
  no answer, because the operator acts on it.
"""

import json
import math

import numpy as np
import pytest

from spectrum_alert.core.services.fingerprinting import (
    FEATURE_NAMES,
    SIGNATURE_LENGTH,
    FingerprintLibrary,
    FingerprintVector,
    LibraryEntry,
    cosine_distance,
    extract_signature,
    match_signature,
    mean_signature,
    scaled_distance,
    summarise,
)

SAMPLE_RATE = 2_048_000.0


def synth(n=16384, rate=240e3, seed=0, compression=1.0, bandwidth=12e3, tone=120e3):
    """FM-like capture with genuine envelope structure and per-capture noise.

    The carrier sits well above the peak frequency deviation on purpose: an FM
    signal whose instantaneous frequency crosses zero aliases, and the resulting
    garbage features look like a different radio on every capture.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / rate
    dev = 4e3
    swing = np.sin(2 * np.pi * dev * t) * bandwidth
    phase = 2 * np.pi * tone * t + np.cumsum(swing) / rate
    envelope = (1 - compression) + compression * np.abs(
        np.cos(np.pi * swing / (2 * bandwidth))
    )
    noise = (rng.normal(size=n) + 1j * rng.normal(size=n)) * 0.12
    return (envelope * np.exp(1j * phase) * 0.6 + noise).astype(np.complex64)


def signature(**kwargs):
    return extract_signature(synth(**kwargs), SAMPLE_RATE)


#: Three synthetic radiotypes. Parameters are chosen to be physically sensible
#: rather than maximally different: a very wide deviation around a low carrier
#: aliases, and an aliased signal has unstable features that look like a
#: different radio on every capture.
DEVICES = {
    "TX-A": dict(bandwidth=4e3, compression=1.0),
    "TX-B": dict(bandwidth=9e3, compression=0.3),
    "TX-C": dict(bandwidth=15e3, compression=0.7),
}

#: Not in any library: used to prove strangers are not identified.
STRANGER = dict(bandwidth=20e3, compression=0.5)


def captures(seeds, **kwargs):
    return [
        extract_signature(synth(seed=seed, **kwargs), SAMPLE_RATE) for seed in seeds
    ]


def burst(**kwargs):
    """Averaged signature from several captures of one transmitter."""
    seeds = kwargs.pop("seeds", range(4))
    kwargs.pop("seed", None)
    return mean_signature(captures(seeds, **kwargs))


def enrolled_burst(**kwargs):
    """Averaged signature plus the per-feature spread of its own captures."""
    return summarise(captures(range(4), **kwargs))


class TestExtraction:
    def test_vector_shape_and_finiteness(self):
        vector = signature()
        assert len(vector.values) == SIGNATURE_LENGTH
        assert vector.labels == FEATURE_NAMES
        assert all(math.isfinite(v) for v in vector.values)

    def test_rejects_empty_or_malformed_input(self):
        with pytest.raises(ValueError):
            extract_signature(np.array([]), SAMPLE_RATE)
        with pytest.raises(ValueError):
            extract_signature(np.zeros((4, 4)), SAMPLE_RATE)
        with pytest.raises(ValueError):
            extract_signature(synth(), 0)

    def test_const_envelope_signal_has_low_envelope_variation(self):
        """A constant-amplitude carrier should not look like a modulated one."""
        n, rate = 8192, 240e3
        t = np.arange(n) / rate
        constant = np.exp(2j * np.pi * 5e3 * t).astype(np.complex64)
        modulated = synth(n=n, rate=rate, seed=3)
        constant_cv = extract_signature(constant, rate).values[0]
        modulated_cv = extract_signature(modulated, rate).values[0]
        assert constant_cv < modulated_cv

    def test_spectral_features_do_not_need_supplied_spectrum(self):
        """The vector must be the same length with or without Welch input."""
        iq = synth()
        freqs = np.linspace(0, SAMPLE_RATE / 2, 4096)
        # rfft is real-input only; emulate a one-sided Welch output with fft.
        half = np.fft.fft(iq.astype(np.complex128))[: iq.size // 2]
        powers = np.abs(half) ** 2

        supplied = extract_signature(iq, SAMPLE_RATE, freqs, powers)
        derived = extract_signature(iq, SAMPLE_RATE)
        assert len(supplied.values) == len(derived.values) == SIGNATURE_LENGTH
        # Occupancy from the two paths should at least agree in magnitude.
        assert supplied.values[8] == pytest.approx(derived.values[8], abs=0.25)

    def test_features_are_scale_invariant(self):
        """The whole point: gain must not change the signature shape."""
        base = synth(seed=11)
        plain = extract_signature(base, SAMPLE_RATE)
        for scale in (0.01, 37.0, 900.0):
            scaled = extract_signature(base * scale, SAMPLE_RATE)
            for index in range(SIGNATURE_LENGTH):
                # Tolerance is numerical, not physical: these features are ratios
                # so they are analytically gain invariant, but float32 arithmetic
                # leaves a relative drift around 1e-6.
                assert scaled.values[index] == pytest.approx(
                    plain.values[index], rel=1e-4, abs=1e-9
                ), f"feature {FEATURE_NAMES[index]} moved with a x{scale} gain change"


class TestCosineDistance:
    def test_self_distance_is_zero(self):
        vector = signature()
        assert cosine_distance(vector, vector) == pytest.approx(0.0, abs=1e-12)

    def test_distance_is_symmetric_and_bounded(self):
        left = signature(seed=1, bandwidth=12e3)
        right = signature(seed=2, bandwidth=4e3)
        assert cosine_distance(left, right) == pytest.approx(
            cosine_distance(right, left)
        )
        assert 0.0 <= cosine_distance(left, right) <= 2.0

    def test_zero_vectors_are_unmatchable_not_identical(self):
        """An all-zero vector has no direction; it must not score as a match."""
        zeros = FingerprintVector(tuple([0.0] * SIGNATURE_LENGTH))
        assert cosine_distance(zeros, signature()) == pytest.approx(1.0)

    def test_mismatched_feature_sets_are_refused(self):
        left = signature()
        shorter = FingerprintVector(left.values[:-1], left.labels[:-1])
        with pytest.raises(ValueError):
            cosine_distance(left, shorter)


class TestScaledDistance:
    def test_identical_signatures_are_zero(self):
        vector = burst(seeds=[3], **DEVICES["TX-A"])
        scales = [1.0] * SIGNATURE_LENGTH
        assert scaled_distance(vector, vector, scales) == pytest.approx(0.0, abs=1e-12)

    def test_one_wild_feature_cannot_veto_an_otherwise_exact_match(self):
        """A near-cancellation feature can move by many spreads on its own.

        Two captures of the same radio sometimes disagree wildly on the third
        moment while agreeing everywhere else. Averaging that feature in raw
        made the whole distance saturate and the transmitter went unidentified.
        """
        base = burst(seeds=[7], **DEVICES["TX-A"])
        wild = FingerprintVector(
            tuple(v + (8.0 if i == 3 else 0.0) for i, v in enumerate(base.values)),
            base.labels,
        )
        scales = [1.0] * SIGNATURE_LENGTH
        assert scaled_distance(base, wild, scales) == pytest.approx(0.0, abs=1e-9)

    def test_features_differing_across_the_board_still_separate(self):
        """Trimming one feature must not make every radio look the same."""
        base = burst(seeds=[7], **DEVICES["TX-A"])
        broadly_different = FingerprintVector(
            tuple(v + 2.0 for v in base.values), base.labels
        )
        scales = [1.0] * SIGNATURE_LENGTH
        assert scaled_distance(base, broadly_different, scales) > 0.5

    def test_features_with_no_measured_spread_are_ignored(self):
        vector = burst(seeds=[3], **DEVICES["TX-A"])
        other = FingerprintVector(
            tuple(v + 5.0 for v in vector.values), vector.labels
        )
        assert scaled_distance(vector, other, [0.0] * SIGNATURE_LENGTH) == 1.0

    def test_refuses_mismatched_feature_sets(self):
        vector = burst(seeds=[3], **DEVICES["TX-A"])
        with pytest.raises(ValueError):
            scaled_distance(
                vector,
                FingerprintVector(vector.values[:-1], vector.labels[:-1]),
                [1.0] * SIGNATURE_LENGTH,
            )


class TestSummarise:
    def test_reports_within_capture_spread(self):
        vector, spread = summarise(captures(range(8), **DEVICES["TX-A"]))
        assert len(vector.values) == SIGNATURE_LENGTH
        assert len(spread) == SIGNATURE_LENGTH
        # Real captures are not identical, so at least one feature must show
        # measurable spread; a zero everywhere would silently disable the
        # noise weighting.
        assert any(value > 0 for value in spread)

    def test_single_capture_reports_zero_spread(self):
        _, spread = summarise(captures([5], **DEVICES["TX-A"]))
        assert all(value == 0.0 for value in spread)


class TestMeanSignature:
    def test_averages_and_preserves_length(self):
        a = signature(seed=5)
        b = signature(seed=6)
        averaged = mean_signature([a, b])
        assert len(averaged.values) == SIGNATURE_LENGTH
        for index in range(SIGNATURE_LENGTH):
            assert averaged.values[index] == pytest.approx(
                (a.values[index] + b.values[index]) / 2
            )

    def test_rejects_empty_and_mismatched_sets(self):
        with pytest.raises(ValueError):
            mean_signature([])
        vector = signature()
        with pytest.raises(ValueError):
            mean_signature([vector, FingerprintVector(vector.values[:-1],
                                                      vector.labels[:-1])])


class TestMatching:
    def _library(self):
        library = FingerprintLibrary()
        for label, kwargs in DEVICES.items():
            vector, spread = enrolled_burst(**kwargs)
            library.add(LibraryEntry(label=label, vector=vector, spread=spread))
        return library

    def test_empty_library_says_unknown(self):
        result = match_signature(signature(), FingerprintLibrary())
        assert result["status"] == "unknown"
        assert result["identified"] is False
        assert "empty" in result["reason"].lower()

    def test_known_device_is_identified(self):
        library = self._library()
        for label, kwargs in DEVICES.items():
            observation = burst(seeds=range(500, 504), **kwargs)
            result = match_signature(observation, library)
            assert result["identified"] is True, result["reason"]
            assert result["match"]["label"] == label

    def test_unknown_device_is_not_identified(self):
        """The whole point: no false identification of a stranger."""
        library = self._library()
        observation = burst(seeds=range(900, 904), **STRANGER)
        result = match_signature(observation, library)
        assert result["identified"] is False
        assert result["status"] in {"no-match", "ambiguous"}

    def test_identical_entries_are_reported_ambiguous(self):
        """Two references of the same radio cannot be told apart; say so."""
        vector = burst(seeds=[1], **DEVICES["TX-A"])
        library = FingerprintLibrary()
        library.add(LibraryEntry(label="Unit 1", vector=vector))
        library.add(LibraryEntry(label="Unit 1 (copy)", vector=vector))
        result = match_signature(vector, library)
        assert result["status"] == "ambiguous"
        assert result["identified"] is False

    def test_result_always_reports_distance_and_features(self):
        library = self._library()
        result = match_signature(signature(), library)
        assert result["features"] == list(FEATURE_NAMES)
        assert set(result["values"]) == set(FEATURE_NAMES)
        assert len(result["candidates"]) >= 1
        assert "distance" in result["candidates"][0]

    def test_gain_change_does_not_break_identification(self):
        library = self._library()
        for scale in (0.5, 250.0):
            observation = mean_signature([
                extract_signature(synth(seed=800 + i, **DEVICES["TX-A"]) * scale,
                                  SAMPLE_RATE)
                for i in range(4)
            ])
            result = match_signature(observation, library)
            assert result["identified"] is True, result["reason"]
            assert result["match"]["label"] == "TX-A"

    def test_entries_from_a_different_feature_set_are_skipped_not_mis_scored(self):
        library = FingerprintLibrary()
        library.add(LibraryEntry(
            label="Old format",
            vector=FingerprintVector((0.5, 0.2, 0.1), ("a", "b", "c")),
        ))
        result = match_signature(signature(), library)
        assert result["candidates"] == [] or all(
            c["entry"]["label"] != "Old format" for c in result["candidates"]
        )


class TestLibraryPersistence:
    def test_round_trips_through_disk(self, tmp_path):
        path = str(tmp_path / "fingerprints.json")
        vector = signature()
        library = FingerprintLibrary(path)
        library.add(LibraryEntry(label="TX-A", vector=vector, vendor="Acme",
                                 model="TX-1", frequency_hz=145.5e6, samples=65536))
        reloaded = FingerprintLibrary.load(path)
        assert len(reloaded) == 1
        entry = reloaded.entries[0]
        assert entry.label == "TX-A"
        assert entry.vendor == "Acme"
        assert entry.frequency_hz == pytest.approx(145.5e6)
        assert entry.vector.values == pytest.approx(vector.values)
        assert entry.entry_id == library.entries[0].entry_id

    def test_empty_library_is_still_persisted_when_configured(self, tmp_path):
        """Regression: an empty library is falsy (__len__ == 0).

        Code of the form ``library or FingerprintLibrary()`` therefore threw away
        the caller's configured path, and nothing was ever written to disk.
        """
        path = str(tmp_path / "fingerprints.json")
        library = FingerprintLibrary(path)
        assert not library, "an empty library is falsy, which is the trap"
        assert library.path == path

    def test_corrupt_file_degrades_to_empty_library(self, tmp_path):
        path = tmp_path / "fingerprints.json"
        path.write_text("{not json", encoding="utf-8")
        library = FingerprintLibrary.load(str(path))
        assert len(library) == 0

    def test_malformed_entries_are_dropped_not_fatal(self, tmp_path):
        path = tmp_path / "fingerprints.json"
        path.write_text(json.dumps({
            "entries": [
                {"label": "good", "signature": {
                    "labels": list(FEATURE_NAMES),
                    "values": [0.1] * SIGNATURE_LENGTH}},
                {"label": "bad", "signature": {"labels": ["x"], "values": []}},
            ]
        }), encoding="utf-8")
        library = FingerprintLibrary.load(str(path))
        assert [entry.label for entry in library.entries] == ["good"]

    def test_remove_reports_whether_anything_was_removed(self, tmp_path):
        library = FingerprintLibrary(str(tmp_path / "f.json"))
        entry = library.add(LibraryEntry(label="X", vector=signature()))
        assert library.remove(entry.entry_id) is True
        assert library.remove(entry.entry_id) is False
        assert library.get("missing") is None