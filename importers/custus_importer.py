from cProfile import label
from email.policy import default
from multiprocessing.sharedctypes import synchronized

import numpy as np
import os
import csv
import sqlite3

import warnings
from typing import Union, Tuple, List
from xml.dom import minidom
import re

from annotationweb.models import TrackingData, SynchronisedTrackingData, VolumetricImage, Dataset, Subject, ImageSequence
from annotationweb.settings import BASE_DIR
from importers.image_sequence_importer import ImageSequenceImporter, ImageSequenceImporterForm
from shutil import copy2, copytree
from common.importer import Importer, importers
from django import forms
import SimpleITK as sitk
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors



class CustusPatientImporterForm(forms.Form):
    path = forms.CharField(label='Data path', max_length=1000)
    create_table = forms.BooleanField(label='Create table', required=False)
    convert_nifti = forms.BooleanField(label='Convert images to Nifti', required=False)
    sync_tracking_data = forms.BooleanField(label='Synchronize images and tracking data', required=True, initial=True)
    image_formats = forms.MultipleChoiceField(label='Accepted 3D image formats', required=True,
                                              initial=['vtk','dcm','nii','nii.gz','mhd'],
                                              choices=[('vtk', 'VTK'),
                                                       ('dcm', 'Dicom'),
                                                       ('nii', 'Nifti'),
                                                       ('nii.gz', 'Compressed nifti'),
                                                       ('mhd', 'Meta Header')],
                                              widget=forms.CheckboxSelectMultiple())

    def __init__(self, data=None):
        super(CustusPatientImporterForm, self).__init__(data)

    def clean(self):
        super(CustusPatientImporterForm, self).clean()
        patient_folder = self.cleaned_data.get('path')
        create_table = self.cleaned_data.get('create_table')
        sync_tracking_data = self.cleaned_data.get('sync_tracking_data')
        convert_nifti = self.cleaned_data.get('convert_nifti')
        image_formats = self.cleaned_data.get('image_formats')

        if not os.path.exists(patient_folder):
            self._errors['path'] = self.error_class([f'Patient folder {patient_folder} does not exist'])

        if create_table is None:
            self.cleaned_data['create_table'] = False

        if sync_tracking_data is None:
           self.cleaned_data['sync_tracking_data'] = False

        if convert_nifti is None:
            self.cleaned_data['convert_nifti'] = False

        if image_formats is None:
            self.cleaned_data['image_formats'] = False
        elif 'mhd' in image_formats:
            self.cleaned_data['image_formats'] += ['zraw', 'raw']

        return self.cleaned_data


class CustusPatientImporter(Importer):
    TRACKING_HEADER = ('Timestamp', 'Branch number', 'Position in branch', 'Branch length', 'Branch generation', 'branchCode', 'Offset [mm]')
    SYNCHRONISED_TRACKING_HEADER = ('Filename', 'Timestamp from FTS', 'Matching Timestamp from TXT', 'Branch number', 'Position in branch', 'Branch length', 'Branch generation', 'branchCode', 'Offset [mm]')
    DELIMITER = ';'
    ALL_IMG_FORMATS = ('vtk', 'dcm', 'nii', 'nii.gz', 'mhd', 'zraw', 'raw')
    DICT_SEQUENCE_TYPES = {'US_Acq': 'US', 'BronchoscopyVideo': 'BV'}
    REG_EXP_ACCEPTED_VOL_IMAGES = f'\.({"|".join(ALL_IMG_FORMATS)})$'
    TRACKING_FIELDNAMES = ['Timestamp',
                           'Branch number',
                           'Position in branch',
                           'Branch length',
                           'Branch generation',
                           'Branch code',
                           'Offset [mm]']
    SYNCHRONISED_TRACKING_FIELDNAMES = ['Filename',
                           'Timestamp from FTS',
                           'Matching Timestamp from TXT',
                           'Tracking ID']
    IMPORTED_TIMESTAMP_FIELDNAMES = ['Timestamp', 'FrameFile']
    TIMESTAMP_FIELDNAMES = ['FrameFile']
    patient_folder = None
    processed_data_folder = None
    create_table = None
    do_sync_tracking_data = False
    dataset = None
    convert_nifti = False
    image_formats = ALL_IMG_FORMATS
    import_vol_images = True

    name = "Custus patient importer"

    def __init__(self, *args, **kwargs):
        """
        Import Custus/Fraxinus patient files. The image sequences will be moves to BASE_DIR/imported_data folder,
        following the structured expected by the ImageSequenceImporter.
        """
        super().__init__(*args, **kwargs)

    def get_form(self, data=None):
        return CustusPatientImporterForm(data)

    def import_data(self, form: forms.Form):
        self.patient_folder = form.cleaned_data['path']
        self.create_table = form.cleaned_data['create_table']
        self.convert_nifti = form.cleaned_data['convert_nifti']
        self.do_sync_tracking_data = form.cleaned_data['sync_tracking_data']
        self.image_formats = form.cleaned_data['image_formats']

        self.processed_data_folder = os.path.join(self.patient_folder, 'AW_processed_data')

        if self.image_formats:
            self.REG_EXP_ACCEPTED_VOL_IMAGES = f'\.({"|".join(self.image_formats)})$'
            self.import_vol_images = True
        else:
            warnings.warn('No volumetric images will be imported. No format was selected!')
            self.import_vol_images = False

        if self.dataset is None:
            raise Exception('Dataset must be given to the importer')

        assert os.path.exists(self.patient_folder), f'Folder {self.patient_folder} does not exist!'
        print("Parsing Custusdoc...")
        patient_name, images, sequences, tracking_files, timestamp_files = self.parse_custusdoc()

        print("Moving files...")
        sequences_paths, images_paths, tracking_files = self.move_files(sequences,
                                                        images,
                                                        tracking_files)

        try:
            subject = Subject.objects.get(name=patient_name, dataset=self.dataset)
        except Subject.DoesNotExist:
            subject = Subject()
            subject.name = patient_name
            subject.dataset = self.dataset
        subject.save()

        # Import the US sequence
        print("Importing image sequences...")
        imported_sequences = self.import_sequences(sequences_paths, subject)

        # Import the volumetric image
        print("Importing volumetric images...")
        self.import_volumetric_image(images_paths, subject)

        # Import the tracking form
        print("Importing tracking information...")
        for tracking_file in tracking_files:
            self.import_tracking_file(tracking_file, subject, imported_sequences)

        # sync tracking data to images:
        if self.do_sync_tracking_data:
            print("Synchronising tracking information with image sequences...")
            sync_tracking_data = self.synchronise_tracking_data(imported_sequences)
            # print(sync_tracking_data)
            for sync_file in sync_tracking_data:
                self.import_synchronised_tracking_file(sync_file)
        return True, self.processed_data_folder

    def import_tracking_file(self, tracking_file: str, subject, image_sequences: List[ImageSequence]):
        """
        Parse a tracking file and populate the table.
        Parameters:
            tracking_file: Path to the location of the file with the tracking records
            subject: ID of te Subject entry
            image_sequences: Information on the imported image sequences linked to the tracking file
        """
        new_entries = list()

        with open(tracking_file, 'r') as csvfile:
            csvreader = csv.DictReader(csvfile, fieldnames=self.TRACKING_FIELDNAMES, delimiter=self.DELIMITER)

            for r_num, row in tqdm(enumerate(csvreader)):
                if r_num > 0: # The first row is the header
                    new_trackingdata_entry, created = TrackingData.objects.update_or_create(
                        timestamp=int(row['Timestamp']),
                        branch_number=int(row['Branch number']) if row['Branch number'] != -1 else -1,
                        position_in_branch=int(row['Position in branch']) if row['Position in branch'] != -1 else -1,
                        branch_length=float(row['Branch length']) if row['Branch length'] != -1 else -1,
                        branch_generation=int(row['Branch generation']) if row['Branch generation'] != -1 else -1,
                        branch_code=row['Branch code'] if row['Branch code'] != -1 else -1,
                        offset=float(row['Offset [mm]']) if row['Offset [mm]'] != -1 else -1,
                        subject=subject,
                    )
                    if created:
                        new_trackingdata_entry.save()
                        new_trackingdata_entry.image_sequence.add(*image_sequences)
                    new_entries.append(new_trackingdata_entry)
        return new_entries

    def import_synchronised_tracking_file(self, sync_tracking_data):
        """
        Parse a tracking file and populate the table.
        Parameters:
            sync_tracking_data: Path to the location of the file with the tracking records
            tracking_data: ID of the Image sequence entry

        """
        with open(sync_tracking_data, 'r') as csvfile:
            csvreader = csv.DictReader(csvfile, fieldnames=self.SYNCHRONISED_TRACKING_FIELDNAMES, delimiter=self.DELIMITER)
            for r_num, row in enumerate(csvreader):
                if r_num > 0: # The first row is the header
                    new_synchtrackdata_entry, created = SynchronisedTrackingData.objects.update_or_create(
                        filename=row['Filename'],
                        image_sequence_timestamp=int(row['Timestamp from FTS']),
                        tracking_system_timestamp=int(row['Matching Timestamp from TXT']),
                        tracking_data=TrackingData.objects.get(id=int(row['Tracking ID'])) if int(
                            row['Tracking ID']) != -1 else None
                    )
                    new_synchtrackdata_entry.save()

    @staticmethod
    def _read_fts_file_failproof(fts_path: str, expected_number_of_entries: int = 0):
        if fts_path is None:
            sequence_ts = [None, ] * expected_number_of_entries
        else:
            with open(fts_path, 'r') as f:
                sequence_ts = f.read().splitlines()
            if expected_number_of_entries > 0:
                assert len(sequence_ts) == expected_number_of_entries, "Mismatch between expected number of timestamps and entries"
        return sequence_ts

    @staticmethod
    def _group_sequences_by_name(image_sequences: list):
        ret_val = {}
        for (sequence_name, sequence_type, img_seq_entry, fts_filename) in image_sequences:
            if sequence_name not in ret_val.keys():
                ret_val[sequence_name] = [[sequence_type, img_seq_entry, fts_filename]]
            else:
                ret_val[sequence_name].append([sequence_type, img_seq_entry, fts_filename])
        return ret_val

    @staticmethod
    def import_volumetric_image(images_paths: list, subject: Subject):
        ret_val = list()
        for f in images_paths:
            new_volimg_entry, created = VolumetricImage.objects.update_or_create(
                format=f,
                subject=subject
            )
            new_volimg_entry.save()
            ret_val.append(new_volimg_entry)
        return ret_val

    @staticmethod
    def import_sequences(sequences: list, subject: Subject) -> List[ImageSequence]:
        ret_val = list()
        for (sequence_name, sequence_dir, sequence_type, ts_file) in tqdm(sequences):
            frames, _, extension  = ImageSequenceImporter._parse_sequence_dir(sequence_dir)
            if len(frames) == 0:
                continue

            filename_format = os.path.join(sequence_dir, f'{sequence_type}_{sequence_name}' + '_#')
            filename_format += extension

            image_sequence, already_imported = ImageSequenceImporter._import_image_sequence(frames, subject, filename_format)
            ret_val.append(image_sequence)
            if already_imported:
                continue

            _ = ImageSequenceImporter._import_metadata(sequence_dir, image_sequence)

        return ret_val

    def move_files(self, sequences: List[List[str]], volumetric_images: list, tracking_data_files: List[str] = None):
        os.makedirs(self.processed_data_folder, exist_ok=True)

        # Move sequences
        sequences_folder = os.path.join(self.processed_data_folder, 'Sequences')
        os.makedirs(sequences_folder, exist_ok=True)
        list_sequences = list()
        sitk_reader = sitk.ImageFileReader()
        sitk_reader.SetImageIO("MetaImageIO")
        for (sequence_name, sequence_files, sequence_type, fts_filename) in sequences:
            r_dest_folder = os.path.join(sequences_folder, f'{sequence_type}_{sequence_name}')
            os.makedirs(r_dest_folder, exist_ok=True)
            mhd_files = [f for f in sequence_files if f.endswith('mhd')]
            sequence_ts = self._read_fts_file_failproof(fts_filename, len(mhd_files))

            new_fts_file = os.path.join(r_dest_folder, f'{sequence_type}_{sequence_name}_timestamps.csv')
            with open(new_fts_file, 'w') as ts_f:
                ts_f.write('Timestamp;FrameFile;\n')
                for (f, ts) in zip(mhd_files, sequence_ts):
                    old_filename, ext = os.path.split(f)[-1].split('.')
                    i = int(old_filename.split('_')[-1])
                    out_filename = f'{sequence_type}_{sequence_name}_{i:d}.{ext}'
                    out_filepath = os.path.join(r_dest_folder, out_filename)
                    if not os.path.exists(out_filepath):
                        sitk_reader.SetFileName(f)
                        sitk.WriteImage(sitk_reader.Execute(), out_filepath, useCompression=True)
                    ts_f.write(f'{ts};{out_filename}\n')

            list_sequences.append((sequence_name, r_dest_folder, sequence_type, new_fts_file))

        # Move images
        images_dest_folder = os.path.join(self.processed_data_folder, 'Images')
        os.makedirs(images_dest_folder, exist_ok=True)
        list_images = list()
        for (f, e) in volumetric_images:
            if os.path.exists(f):
                if self.convert_nifti:
                    try:
                        sitk_reader = sitk.ImageFileReader()
                        sitk_reader.SetFileName(f)

                        nifti_filename = os.path.split(f)[-1].replace(f'.{e}', '.nii.gz')
                        dest_filepath = os.path.join(images_dest_folder, nifti_filename)
                        sitk.WriteImage(sitk_reader.Execute(), dest_filepath, useCompression=True)
                        list_images.append(dest_filepath)
                    except (IOError, RuntimeError) as err:
                        print(f'Failed to convert to Nifti. Saving original file instead: {err}')
                        dest_filepath = os.path.join(images_dest_folder, os.path.split(f)[-1])
                        copy2(f, dest_filepath)
                        list_images.append(dest_filepath)
                else:
                    copy2(f, os.path.join(images_dest_folder, os.path.split(f)[-1]))
                    list_images.append(os.path.join(images_dest_folder, os.path.split(f)[-1]))
            else:
                warnings.warn(f'File not found: {f}')

        # Move tracking data file
        dest_tracking_data_files = []
        if tracking_data_files is not None:
            for i, f in enumerate(tracking_data_files):
                dest = os.path.join(self.processed_data_folder, f'TrackingData_{i:03d}.txt')
                copy2(f, dest)
                dest_tracking_data_files.append(dest)

        return list_sequences, list_images, dest_tracking_data_files

    def parse_custusdoc(self, file_path: str=None):
        if file_path is None:
            file_path = os.path.join(self.patient_folder, 'custusdoc.xml')
            patient_folder = self.patient_folder
        else:
            patient_folder = os.path.split(file_path)[0]

        custusdoc = minidom.parse(file_path)

        patient = custusdoc.getElementsByTagName('patient')[0]
        patient_name = os.path.split(patient.getElementsByTagName('active_patient')[0].childNodes[0].data)[-1].split('.')[0]

        images = custusdoc.getElementsByTagName('data')
        image_paths = list()

        sequences = custusdoc.getElementsByTagName('recordSession')
        list_sequences = list()

        # Fetch 3D images or data
        if self.import_vol_images:
            for i in images:
                if i.getAttribute('type') in ('mesh', 'image'):
                    img_path = os.path.join(patient_folder, i.getElementsByTagName('filePath')[0].childNodes[0].data).replace('/', os.sep)
                    is_valid, img_extension = self._is_valid_file(img_path, True)
                    if is_valid:
                        image_paths.append((img_path, img_extension))

        # Fetch US, video, or other type of sequences
        timestamp_files = list()
        for seq in sequences:
            sequence_type = seq.getElementsByTagName('category')[0].childNodes[0].data
            if sequence_type in self.DICT_SEQUENCE_TYPES.keys():
                sequence_name = seq.getAttribute('uid')
                sequence_folder = os.path.join(patient_folder, 'US_Acq', f'{sequence_type}_{sequence_name.lstrip("0")}')
                sequence_files = [os.path.join(sequence_folder, f) for f in os.listdir(sequence_folder) if self._is_valid_sequence(f, f'{sequence_type}_{sequence_name.lstrip("0")}'.replace('_', '\_'))]
                sequence_ts_file = [os.path.join(sequence_folder, f) for f in os.listdir(sequence_folder) if f.endswith('_openCV.fts')]
                sequence_ts_file = sequence_ts_file[0] if len(sequence_ts_file) > 0 else None
                sequence_files.sort()
                list_sequences.append([sequence_name,
                                       sequence_files,
                                       self.DICT_SEQUENCE_TYPES[sequence_type],
                                       sequence_ts_file,
                                       ])

                # Set the timestamp folder in a similar fashion
                timestamp_folder = os.path.join(patient_folder, 'US_Acq',
                                                f'{sequence_type}_{sequence_name.lstrip("0")}')
                timestamp_files = [[f.strip("_openCV.fts"), os.path.join(timestamp_folder, f)] for f in
                                   os.listdir(timestamp_folder) if f.endswith('_openCV.fts')]
                #print(f"Timestamp Folder: {timestamp_folder}")

        try:
            tracking_folder = os.path.join(patient_folder, 'TrackingInformation')
            tracking_files = [os.path.join(tracking_folder, f) for f in os.listdir(tracking_folder) if f.endswith('TrackingInformation.txt')]
            print(f"tracking file {tracking_files}")
        except FileNotFoundError:
            tracking_files = list()

        return patient_name, image_paths, list_sequences, tracking_files, timestamp_files

    @staticmethod
    def _is_valid_sequence(file_path: str, sequence_name: str, return_extension: bool = False):
        re_match = re.match(f'{sequence_name}_.+\d+\.(mhd|zraw)$', file_path)
        ret_val = False
        if re_match:
            ret_val = True
            if return_extension:
                ret_val = (ret_val, re_match[1])
        return ret_val

    def _is_valid_file(self, file_path: str, return_extension: bool = False):
        re_match = re.match(f'.*{self.REG_EXP_ACCEPTED_VOL_IMAGES}', os.path.split(file_path)[-1])
        ret_val = bool(re_match)
        if return_extension:
            ret_val = (ret_val, re_match[1] if re_match else None)
        return ret_val

    @staticmethod
    def _read_timestamp_files(files):
        content_list = []
        for name, filepath in files:
            with open(filepath, 'r') as file:
                content = file.read()
                content_list.append((name, content))
        return content_list

    @staticmethod
    def _read_tracking_files(tracking_files):
        all_timestamps = []
        all_data = []
        for name, filepath in tracking_files:
            with open(filepath, 'r') as file:
                lines = file.readlines()
            data = [line.strip().split(';') for line in lines[1:]]  # Skip header line and split data
            timestamps = [int(line[0]) for line in data]
            all_timestamps.extend(timestamps)
            all_data.extend(data)
        return all_timestamps, all_data

    def _read_imported_timestamp_file(self, file_path: str):
        """
        Parse the generated file with the timestamps of the imported frames
        """
        ts = list()
        files = list()
        with open(file_path, 'r') as f:
            csv_reader = csv.DictReader(f, fieldnames=self.IMPORTED_TIMESTAMP_FIELDNAMES, delimiter=self.DELIMITER)
            for r_num, row in enumerate(csv_reader):
                if r_num > 0:
                    ts.append(int(row[self.IMPORTED_TIMESTAMP_FIELDNAMES[0]]))
                    files.append(row[self.IMPORTED_TIMESTAMP_FIELDNAMES[1]])
        return ts, files

    def synchronise_tracking_data(self, sequences_entries: List[ImageSequence]) -> List[Union[str, ImageSequence]]:
        # Fetch the timestamp file for the frames and the TrackingData corresponding to the sequence entries
        sync_ts_files = list()
        for image_sequence in sequences_entries:
            # Get a list of the timestamps of the frames
            sequence_folder = os.path.split(image_sequence.format)[0]
            sequence_name = os.path.split(sequence_folder)[-1]
            timestamp_file = next((os.path.join(sequence_folder, f) for f in os.listdir(sequence_folder) if f.endswith('timestamps.csv')), None)

            if timestamp_file is None:
                print(f"Could not find the timestamps file for sequence {image_sequence.id}, in folder {sequence_folder}")
                continue

            # Get a list of the timestamps of the tracking data
            trackingdata = TrackingData.objects.filter(image_sequence=image_sequence)
            timestamps_tracking = [t.timestamp for t in trackingdata]
            timestamps_mhd, frames_mhd = self._read_imported_timestamp_file(timestamp_file)

            # Find exact matches
            exact_matches = set(timestamps_mhd).intersection(timestamps_tracking)

            # Remove exact matches from the lists
            remaining_mhd_timestamps = [ts for ts in timestamps_mhd if ts not in exact_matches]
            remaining_tracking_timestamps = [ts for ts in timestamps_tracking if ts not in exact_matches]

            # Find the closest matches within the range limit
            close_matches = []
            range_limit = 60 #TODO maybe not hardcode value?
            for mhd_ts in remaining_mhd_timestamps:
                closest_match = min((track_ts for track_ts in remaining_tracking_timestamps if abs(track_ts - mhd_ts) <= range_limit),
                                    key=lambda x: abs(x - mhd_ts), default=None)
                if closest_match is not None:
                    close_matches.append((mhd_ts, closest_match))
                    remaining_tracking_timestamps.remove(closest_match)

            sync_timestep_file = os.path.join(self.processed_data_folder, f'sync_timestamp_file_{sequence_name}.csv')
            with open(sync_timestep_file, 'w') as f:
                f.write(f'{";".join(self.SYNCHRONISED_TRACKING_FIELDNAMES)}\n')

                for i, (mhd_ts, mhd_filename) in enumerate(zip(timestamps_mhd, frames_mhd)):
                    mhd_file = os.path.join(sequence_folder, mhd_filename)

                    if os.path.exists(mhd_file):
                        if mhd_ts in exact_matches:
                            f.write(f"{mhd_file}; {mhd_ts}; {mhd_ts}; {trackingdata.get(timestamp=mhd_ts).id}\n")
                        else:
                            close_match = next((match[1] for match in close_matches if match[0] == mhd_ts), None)
                            if close_match:
                                f.write(f"{mhd_file}; {mhd_ts}; {close_match}; {trackingdata.get(timestamp=close_match).id}\n")
                            else:
                                f.write(f"{mhd_file}; {mhd_ts};-1; -1;\n")
                    else:
                        raise FileNotFoundError("Failed to retrieve the MHD sequence files")
            sync_ts_files.append(sync_timestep_file)
            with open(sync_timestep_file, 'r') as file:
                content = file.read()
                print(content)

        return sync_ts_files





