import os
import numpy as np

from mirdata.datasets import swd
from mirdata import annotations
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "Schubert_D911-01_QU98"
    data_home = os.path.normpath("tests/resources/mir_datasets/swd")
    dataset = swd.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)

    expected_attributes = {
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/swd/"),
            "01_RawData/audio_wav/Schubert_D911-01_QU98.wav",
        ),
        "measure_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/swd/"),
            "02_Annotations/ann_audio_measure/Schubert_D911-01_QU98.csv",
        ),
        "chord_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/swd/"),
            "02_Annotations/ann_audio_chord/Schubert_D911-01_QU98.csv",
        ),
        "localkey_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/swd/"),
            "02_Annotations/ann_audio_localkey-ann3/Schubert_D911-01_QU98.csv",
        ),
        "note_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/swd/"),
            "02_Annotations/ann_audio_note/Schubert_D911-01_QU98.csv",
        ),
        "structure_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/swd/"),
            "02_Annotations/ann_audio_structure/Schubert_D911-01_QU98.csv",
        ),
        "track_id": "Schubert_D911-01_QU98",
    }

    expected_property_types = {
        "measure": annotations.BeatData,
        "chord": annotations.ChordData,
        "localkey": annotations.KeyData,
        "sections": annotations.SectionData,
        "notes": annotations.NoteData,
        "audio": tuple,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    audio, sr = track.audio
    assert sr == 22050, "sample rate {} is not 22050".format(sr)
    assert audio.shape == (22050 * 2,), "audio shape {} was not (44100,)".format(
        audio.shape
    )


def test_load_measure():
    measure_path = os.path.join(
            os.path.normpath("tests/resources/mir_datasets/swd/"),
            "02_Annotations/ann_audio_measure/Schubert_D911-01_QU98.csv",
        )
    measure_data = swd.load_measure(measure_path)

    assert (
        type(measure_data) == annotations.BeatData
    ), "measure_data is not type annotations.BeatData"
    assert type(measure_data.times) == np.ndarray, "measure_data.times is not an np.ndarray"
    assert (
        type(measure_data.positions) == np.ndarray
    ), "measure_data.positions is not an np.ndarray"

    assert np.array_equal(
        measure_data.times,
        np.array([0.258662132, 2.709886621, 5.830453515]),
    ), "measure_data.times different than expected"
    assert np.array_equal(
        measure_data.positions, np.array([1, 2, 3])
    ), "measure_data.positions different from expected"

    assert swd.load_measure(None) is None, "load_measure(None) should return None"


def test_load_chord():
    chords_path = os.path.join(
            os.path.normpath("tests/resources/mir_datasets/swd/"),
            "02_Annotations/ann_audio_chord/Schubert_D911-01_QU98.csv",
        )
    chord_data = swd.load_chord(chords_path)

    assert type(chord_data) == annotations.ChordData
    assert type(chord_data.intervals) == np.ndarray
    assert type(chord_data.labels) == list

    assert np.array_equal(
        chord_data.intervals[:, 0], np.array([0.26, 5.16])
    )
    assert np.array_equal(
        chord_data.intervals[:, 1], np.array([5.16, 5.84])
    )
    assert np.array_equal(chord_data.labels, np.array(["A#:(b3,5)", "A:(b3,b5,bb7)/A#"]))

    assert swd.load_chord(None) is None


def test_load_localkey():
    key_path = os.path.join(
        os.path.normpath("tests/resources/mir_datasets/swd/"),
        "02_Annotations/ann_audio_localkey-ann3/Schubert_D911-01_QU98.csv",
    )
    key_data = swd.load_localkey(key_path)

    assert type(key_data) == annotations.KeyData
    assert type(key_data.intervals) == np.ndarray

    assert np.array_equal(key_data.intervals[:, 0], np.array([0.26, 37.62]))
    assert np.array_equal(key_data.intervals[:, 1], np.array([37.62, 49.06]))
    assert np.array_equal(key_data.keys, ["A#:minor", "C#:major"])

    assert swd.load_localkey(None) is None


def test_load_sections():
    sections_path = os.path.join(
        os.path.normpath("tests/resources/mir_datasets/swd/"),
        "02_Annotations/ann_audio_structure/Schubert_D911-01_QU98.csv",
    )
    section_data = swd.load_sections(sections_path)

    assert type(section_data) == annotations.SectionData
    assert type(section_data.intervals) == np.ndarray
    assert type(section_data.labels) == list

    assert np.array_equal(section_data.intervals[:, 0], np.array([0.46, 14.76]))
    assert np.array_equal(section_data.intervals[:, 1], np.array([14.76, 36.3]))
    assert np.array_equal(section_data.labels, np.array(["I", "A"]))

    assert swd.load_sections(None) is None

def test_load_notes():
    notes_path = os.path.join(
        os.path.normpath("tests/resources/mir_datasets/swd/"),
        "02_Annotations/ann_audio_note/Schubert_D911-01_QU98.csv",
    )
    note_data = swd.load_notes(notes_path)

    assert type(note_data) == annotations.NoteData
    assert type(note_data.intervals) == np.ndarray
    assert type(note_data.pitches) == np.ndarray

    assert np.array_equal(
        note_data.intervals[:, 0], np.array([0.24, 0.24, 0.24])
    )
    assert np.array_equal(
        note_data.intervals[:, 1], np.array([0.7854, 0.7854, 0.7854])
    )
    assert np.array_equal(note_data.pitches, np.array([46.0, 53.0, 61.0])) # sorted automatically by annotations.NoteData
                                                                           # in the order of start time, end time, pitch

    assert swd.load_notes(None) is None
