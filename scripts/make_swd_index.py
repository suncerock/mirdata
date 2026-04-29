import argparse
import json
import os
import re
from mirdata.validate import md5


SWD_INDEX_PATH = 'mirdata/datasets/indexes/swd_index_2.2.json'
SWD_ANNOTATION_TYPE_FOLDER = {
    'chord': 'ann_audio_chord',
    'localkey': 'ann_audio_localkey-ann3',
    'note': 'ann_audio_note',
    'structure': 'ann_audio_structure',
    'measure': 'ann_audio_measure'
}


def make_swd_index(data_path):
    annotations_dir = os.path.join(data_path, '02_Annotations')
    audio_dir = os.path.join(data_path, '01_RawData', 'audio_wav')

    track_ids = [re.match(r"(Schubert\_D911\-\d{2}\_\w{2}\d{2})\.wav", filename) for filename in os.listdir(audio_dir)]
    track_ids = sorted([track_id.group(1) for track_id in track_ids if track_id])

    swd_tracks = {track_id: {} for track_id in track_ids}
    for track_id in track_ids:

        # checksum
        audio_path = os.path.join(audio_dir, '{}.wav'.format(track_id))
        audio_checksum = md5(audio_path)
        swd_tracks[track_id]['audio'] = (os.path.relpath(audio_path, data_path), audio_checksum)


        for annot_type, folder in SWD_ANNOTATION_TYPE_FOLDER.items():
            annot_path = os.path.join(annotations_dir, folder, '{}.csv'.format(track_id))
            annot_checksum = md5(annot_path)

            swd_tracks[track_id][annot_type] = (os.path.relpath(annot_path, data_path), annot_checksum)

    swd_index = {
        'version': "2.2",
        'tracks': swd_tracks,
    }

    with open(SWD_INDEX_PATH, 'w') as fhandle:
        json.dump(swd_index, fhandle, indent=2)

    print(md5(SWD_INDEX_PATH))


def main(args):
    make_swd_index(args.swd_data_path)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='Make SWD index file.')
    PARSER.add_argument(
        'swd_data_path', type=str, help='Path to SWD data folder.'
    )

    main(PARSER.parse_args())
