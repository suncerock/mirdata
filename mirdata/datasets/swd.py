"""SWD Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    The Schubert Wintereise Dataset (SWD) presents a multimodal dataset comprising Franz Schubert's song cycle Winterreise.
    It includes 9 versions (performances) of the complete 24 song cycle and various annotations including
    measure (downbeat), chord, global and local key, note events, and structure.
    Details can be found in https://dl.acm.org/doi/10.1145/3429743.

"""

import csv
import os
from typing import BinaryIO, Optional, TextIO, Tuple

from deprecated.sphinx import deprecated
import librosa
import numpy as np

from mirdata import download_utils

from mirdata import core
from mirdata import annotations
from mirdata import io


BIBTEX = """@article{weiss2021swd,
    title = {Schubert Winterreise Dataset: A Multimodal Scenario for Music Analysis},
    author = {Wei{\ss}, Christof and Zalkow, Frank and Arifi-M\"{u}ller, Vlora and M\"{u}ller, Meinard
    and Koops, Hendrik Vincent and Volk, Anja and Grohganz, Harald G.},
    year = {2021},
    journal = {Journal on Computing and Cultural Heritage (JOCCH)},
    publisher = {Association for Computing Machinery},
    volume = {14},
    number = {2},
}"""

INDEXES = {
    "default": "2.2",
    "test": "sample",
    "2.2": core.Index(
        filename="swd_index_2.2.json",
        url="to be added",
        checksum="9d08ac3c21f04eb1b71818584402a541",
    ),
    "sample": core.Index(filename="swd_index_2.2_sample.json"),
}

REMOTES = {
    "annotations": download_utils.RemoteFileMetadata(
        filename="Schubert_Winterreise_Dataset_v2-2.zip",
        url="https://zenodo.org/records/10839767/files/Schubert_Winterreise_Dataset_v2-2.zip",
        checksum="TO BE UPDATED",
    )
}
DOWNLOAD_INFO = """
    All the annotations of the SWD dataset are available on Zenodo.
    Two (out of nine) audio versions of the SWD dataset are available for download.
    If you have the complete audio dataset, place the audio into
        > Schubert_Winterreise_Dataset_v2-2/
            > 01_RawData/
                > audio_wav/
"""

LICENSE_INFO = (
    "TODO: License info for SWD dataset to be added."
)


class Track(core.Track):
    """SWD track class

    Args:
        track_id (str): track id of the track
        data_home (str): path where the data lives

    Attributes:
        audio_path (str): track audio path
        chord_path (str): chord annotation path
        localkey_path (str): local key annotation path
        note_path (str): note events annotation path
        structure_path (str): structure annotation path
        measure_path (str): measure annotation path
        track_id (str): track id

    Cached Properties:
        chord (ChordData): chord annotations
        measure (BeatData): measure (downbeat) annotations
        localkey (KeyData): local key annotations

    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)

        self.chord_path = self.get_path("chord")
        self.localkey_path = self.get_path("localkey")
        self.note_path = self.get_path("note")
        self.structure_path = self.get_path("structure")
        self.measure_path = self.get_path("measure")

        self.audio_path = self.get_path("audio")

        # self.title = os.path.basename(self._track_paths["sections"][0]).split(".")[0]

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The track's audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_path)

    @core.cached_property
    def measure(self) -> Optional[annotations.BeatData]:
        return load_measure(self.measure_path)

    @core.cached_property
    def chord(self) -> Optional[annotations.ChordData]:
        return load_chord(self.chord_path)

    @core.cached_property
    def localkey(self) -> Optional[annotations.KeyData]:
        return load_localkey(self.localkey_path)

@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load an SWD audio file.

    Args:
        fhandle (str or file-like): path or file-like object pointing to an audio file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    return librosa.load(fhandle, sr=None, mono=True)


@io.coerce_to_string_io
def load_measure(fhandle: TextIO) -> annotations.BeatData:
    """Load SWD format measure data from a file

    Args:
        fhandle (str or file-like): path or file-like object pointing to a measure annotation file

    Returns:
        BeatData: loaded measure data

    """
    measure_times, measure_positions = [], []
    reader = csv.reader(fhandle, delimiter=';')
    header = next(reader)  # Skip header
    for line in reader:
        measure_times.append(float(line[0]))
        measure_positions.append(line[-1])

    # After fixing New Point labels convert positions to int
    measure_data = annotations.BeatData(
        np.array(measure_times),
        "s",
        np.array([int(b) for b in measure_positions]),
        "global_fraction",
    )

    return measure_data


@io.coerce_to_string_io
def load_chord(fhandle: TextIO) -> annotations.ChordData:
    """Load SWD format chord data from a file

    Args:
        fhandle (str or file-like): path or file-like object pointing to a chord annotation file

    Returns:
        ChordData: loaded chord data

    """
    start_times, end_times, chords = [], [], []
    reader = csv.reader(fhandle, delimiter=';')
    header = next(reader)  # Skip header
    for line in reader:
        start_times.append(float(line[0]))
        end_times.append(float(line[1]))
        chords.append(line[3])

    return annotations.ChordData(
        np.array([start_times, end_times]).T, "s", chords, "open"
    )


@io.coerce_to_string_io
def load_localkey(fhandle: TextIO) -> annotations.KeyData:
    """Load SWD format local key data from a file

    Args:
        fhandle (str or file-like): path or file-like object pointing to a key annotation file

    Returns:
        KeyData: loaded key data

    """
    def format_swd_key_label(label: str) -> str:
        return label.replace("min", "minor").replace("maj", "major")
    
    start_times, end_times, keys = [], [], []
    reader = csv.reader(fhandle, delimiter=';')
    header = next(reader)  # Skip header
    for line in reader:
        start_times.append(float(line[0]))
        end_times.append(float(line[1]))
        keys.append(format_swd_key_label(line[2]))

    return annotations.KeyData(
        np.array([start_times, end_times]).T, "s", keys, "key_mode"
    )


@io.coerce_to_string_io
def load_sections(fhandle: TextIO) -> annotations.SectionData:
    """Load SWD format section data from a file

    Args:
        fhandle (str or file-like): path or file-like object pointing to a section annotation file

    Returns:
        SectionData: loaded section data
    """
    start_times, end_times, sections = [], [], []
    reader = csv.reader(fhandle, delimiter=";")
    header = next(reader)  # Skip header
    for line in reader:
        start_times.append(float(line[0]))
        end_times.append(float(line[1]))
        sections.append(line[2])

    return annotations.SectionData(
        np.array([start_times, end_times]).T, "s", sections, "open"
    )


@io.coerce_to_string_io
def load_notes(fhandle: TextIO) -> annotations.NoteData:
    """Load SWD format note data from a file

    Args:
        fhandle (str or file-like): path or file-like object pointing to a note annotation file

    Returns:
        NoteData: loaded note data
    """
    start_times, end_times, pitches = [], [], []
    reader = csv.reader(fhandle, delimiter=";")
    header = next(reader)  # Skip header
    for line in reader:
        start_times.append(float(line[0]))
        end_times.append(float(line[1]))
        pitches.append(float(line[2]))

    return annotations.NoteData(
        np.array([start_times, end_times]).T, "s", np.array(pitches), "midi"
    )


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The SWD dataset
    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="swd",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )

    @deprecated(reason="Use mirdata.datasets.swd.load_audio", version="0.3.4")
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.swd.load_beats", version="0.3.4")
    def load_measure(self, *args, **kwargs):
        return load_measure(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.swd.load_chord", version="0.3.4")
    def load_chord(self, *args, **kwargs):
        return load_chord(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.swd.load_sections", version="0.3.4")
    def load_sections(self, *args, **kwargs):
        return load_sections(*args, **kwargs)

    @deprecated(reason="Use mirdata.datasets.swd.load_notes", version="0.3.4")
    def load_notes(self, *args, **kwargs):
        return load_notes(*args, **kwargs)