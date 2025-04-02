from cProfile import label
from email.policy import default

import numpy as np
import os
import csv
import sqlite3
from plum.exceptions import ImplementationError
from typing import Union, Tuple, List
from xml.dom import minidom
import re

from annotationweb.models import TrackingData, VolumetricImage, Dataset, Subject, ImageSequence
from annotationweb.settings import BASE_DIR
from importers.image_sequence_importer import ImageSequenceImporter, ImageSequenceImporterForm
from shutil import copy2, copytree
from common.importer import Importer, importers
from django import forms
import SimpleITK as sitk


ROOT_PATH = os.path.join(BASE_DIR, 'imported_data')

class CustusPatientImporterForm(forms.Form):
    path = forms.CharField(label='Data path', max_length=1000)
    create_table = forms.BooleanField(label='Create table', required=False)
    convert_nifti = forms.BooleanField(label='Convert images to Nifti', required=False)

    def __init__(self, data=None):
        super(CustusPatientImporterForm, self).__init__(data)

    def clean(self):
        super(CustusPatientImporterForm, self).clean()
        patient_folder = self.cleaned_data.get('path')
        create_table = self.cleaned_data.get('create_table')
        convert_nifti = self.cleaned_data.get('convert_nifti')

        if not os.path.exists(patient_folder):
            self._errors['path'] = self.error_class([f'Patient folder {patient_folder} does not exist'])

        if create_table is None:
            self.cleaned_data['create_table'] = False

        if convert_nifti is None:
            self.cleaned_data['convert_nifti'] = False

        return self.cleaned_data


class CustusPatientImporter(Importer):
    HEADER = ('Timestamp','Branch number', 'Position in branch', 'Branch length', 'Branch generation', 'branchCode', 'Offset [mm]')
    DELIMITER = ';'
    IMAGE_FORMATS = ('vtk', 'dcm', 'nii', 'nii.gz', 'mhd', 'zraw')
    DICT_SEQUENCE_TYPES = {'US_Acq': 'US', 'BronchoscopyVideo': 'BV'}
    REG_EXP_ACCEPTED_FILES = f'\.({"|".join(IMAGE_FORMATS)})$'# f'\_.+\d+\.({"|".join(IMAGE_FORMATS)})'
    TRACKING_FIELDNAMES = ['Timestamp',
                           'Branch number',
                           'Position in branch',
                           'Branch length',
                           'Branch generation',
                           'Branch code',
                           'Offset [mm]']

    patient_folder = None
    create_table = None
    dataset = None
    convert_nifti = False

    name = "Custus patient importer"

    def __init__(self, *args, **kwargs):
        """
        Import Custus/Fraxinus patient files. The image sequences will be moves to BASE_DIR/imported_data folder,
        following the structured expected by the ImageSequenceImporter.
        """
        super().__init__(*args, **kwargs)
        os.makedirs(ROOT_PATH, exist_ok=True)

    def get_form(self, data=None):
        return CustusPatientImporterForm(data)

    def import_data(self, form: forms.Form):
        self.patient_folder = form.cleaned_data['path']
        self.create_table = form.cleaned_data['create_table']
        self.convert_nifti = form.cleaned_data['convert_nifti']

        if self.dataset is None:
            raise Exception('Dataset must be given to the importer')

        assert os.path.exists(self.patient_folder), f'Folder {self.patient_folder} does not exist!'

        patient_name, images, sequences, tracking_files = self.parse_custusdoc()

        imported_patient_dir, sequences_paths, images_paths = self.move_files(sequences,
                                                                                  images,
                                                                                  patient_name)

        try:
            subject = Subject.objects.get(name=patient_name, dataset=self.dataset)
        except Subject.DoesNotExist:
            subject = Subject()
            subject.name = patient_name
            subject.dataset = self.dataset
        subject.save()

        # Import the US sequence
        imported_sequences = self.import_sequences(sequences_paths, subject)

        # Import the volumetric image
        self.import_volumetric_image(images_paths, subject)

        # Import the tracking form
        self.import_tracking_file(tracking_files, subject, imported_sequences)

        return True, imported_patient_dir

    @staticmethod
    def _populate_trackingdata_entry(tracking_data_obj: TrackingData, data_dict: dict):
        tracking_data_obj.timestamp = int(data_dict['Timestamp'])
        tracking_data_obj.branch_number = int(data_dict['Branch number'])
        tracking_data_obj.position_in_branch = float(data_dict['Position in branch'])
        tracking_data_obj.branch_length = float(data_dict['Branch length'])
        tracking_data_obj.branch_generation = int(data_dict['Branch generation'])
        tracking_data_obj.branch_code = data_dict['Branch code']
        tracking_data_obj.offset = float(data_dict['Offset [mm]'])

    def import_tracking_file(self, tracking_files: list, subject, image_sequences: list):
        """
        Parse a tracking file and populate the table.
        Parameters:
            tracking_files: Path to the location of the file with the tracking records
            subject: ID of te Subject entry
            image_sequences: name and ID of the ImageSequence entries
        """
        seq_dict =self._group_sequences_by_name(image_sequences)
        for (n, f) in tracking_files:
            with open(f, 'r') as csvfile:
                csvreader = csv.DictReader(csvfile, fieldnames=self.TRACKING_FIELDNAMES, delimiter=self.DELIMITER)
                for r_num, row in enumerate(csvreader):
                    if r_num > 0: # The first row is the header
                        new_entry = TrackingData()
                        for seq_type, seq in seq_dict[n]:
                            if seq_type == 'US':
                                new_entry.ultrasound_sequence = seq
                            elif seq_type == 'BV':
                                new_entry.video_sequence = seq
                            else:
                                continue
                        self._populate_trackingdata_entry(new_entry, row)
                        new_entry.subject = subject
                        new_entry.save()

    @staticmethod
    def _group_sequences_by_name(image_sequences: list):
        ret_val = {}
        for (sequence_name, sequence_type, sequence) in image_sequences:
            if sequence_name not in ret_val.keys():
                ret_val[sequence_name] = [[sequence_type, sequence]]
            else:
                ret_val[sequence_name].append([sequence_type, sequence])
        return ret_val

    @staticmethod
    def import_volumetric_image(images_paths: list, subject: Subject):
        ret_val = list()
        for f in images_paths:
            try:
                new_entry = VolumetricImage.objects.get(subject=subject, format=f)
            except VolumetricImage.DoesNotExist:
                new_entry = VolumetricImage()
                new_entry.format = f
                new_entry.subject = subject
            new_entry.save()
            ret_val.append(new_entry)
        return ret_val

    @staticmethod
    def import_sequences(sequences: list, subject: Subject):
        ret_val = list()
        for (sequence_name, sequence_dir, sequence_type) in sequences:
            frames, _, extension  = ImageSequenceImporter()._parse_sequence_dir(sequence_dir)
            if len(frames) == 0:
                continue

            filename_format = os.path.join(sequence_dir, f'{sequence_type}_{sequence_name}' + '_#')
            filename_format += extension

            image_sequence, already_imported = ImageSequenceImporter()._import_image_sequence(frames, subject, filename_format)
            ret_val.append([sequence_name, sequence_type, image_sequence])
            if already_imported:
                continue

            _ = ImageSequenceImporter()._import_metadata(sequence_dir, image_sequence)

        return ret_val

    def move_files(self, sequences: List[List[str]], volumetric_images: list, patient_name: str):
        dest_folder = os.path.join(ROOT_PATH, patient_name)
        os.makedirs(dest_folder, exist_ok=True)

        # Move sequences
        sequences_folder = os.path.join(dest_folder, 'Sequences')
        os.makedirs(sequences_folder, exist_ok=True)
        list_sequences = list()
        sitk_reader = sitk.ImageFileReader()
        sitk_reader.SetImageIO("MetaImageIO")
        for (sequence_name, sequence_files, sequence_type) in sequences:
            r_dest_folder = os.path.join(sequences_folder, f'{sequence_type}_{sequence_name}')
            os.makedirs(r_dest_folder, exist_ok=True)
            # TODO: check for duplicates
            for f in sequence_files:
                old_filename, ext = os.path.split(f)[-1].split('.')
                i = int(old_filename.split('_')[-1])
                if ext == "mhd":
                    out_filename = os.path.join(r_dest_folder, f'{sequence_type}_{sequence_name}_{i:01d}.{ext}')
                    sitk_reader.SetFileName(f)
                    sitk.WriteImage(sitk_reader.Execute(), out_filename, useCompression=True)

            list_sequences.append((sequence_name, r_dest_folder, sequence_type))

        # Move images
        images_dest_folder = os.path.join(dest_folder, 'Images')
        os.makedirs(images_dest_folder, exist_ok=True)
        list_images = list()
        for (f, e) in volumetric_images:
            if self.convert_nifti:
                try:
                    sitk_reader = sitk.ImageFileReader()
                    sitk_reader.SetFileName(f)

                    nifti_filename = os.path.split(f)[-1].replace(f'.{e}', '.nii.gz')
                    sitk.WriteImage(sitk_reader.Execute(), nifti_filename, useCompression=True)
                    list_images.append(os.path.join(images_dest_folder, nifti_filename))
                except IOError as err:
                    print(f'Failed to convert to Nifti. Saving original file instead: {err}')
                    copy2(f, os.path.join(images_dest_folder, os.path.split(f)[-1]))
                    list_images.append(os.path.join(images_dest_folder, os.path.split(f)[-1]))
            else:
                copy2(f, os.path.join(images_dest_folder, os.path.split(f)[-1]))
                list_images.append(os.path.join(images_dest_folder, os.path.split(f)[-1]))

        return dest_folder, list_sequences, list_images

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
        for i in images:
            if i.getAttribute('type') in ('mesh', 'image'):
                img_path = os.path.join(patient_folder, i.getElementsByTagName('filePath')[0].childNodes[0].data).replace('/', '\\')
                is_valid, img_extension = self._is_valid_file(img_path, True)
                if is_valid:
                    image_paths.append((img_path, img_extension))

        # Fetch US, video, or other type of sequences
        for seq in sequences:
            sequence_type = seq.getElementsByTagName('category')[0].childNodes[0].data
            if sequence_type in self.DICT_SEQUENCE_TYPES.keys():
                sequence_name = seq.getAttribute('uid')
                sequence_folder = os.path.join(patient_folder, 'US_Acq', f'{sequence_type}_{sequence_name.lstrip("0")}')
                sequence_files = [os.path.join(sequence_folder, f) for f in os.listdir(sequence_folder) if self._is_valid_sequence(f, f'{sequence_type}_{sequence_name.lstrip("0")}'.replace('_', '\_'))]
                sequence_files.sort()
                list_sequences.append([sequence_name,
                                       sequence_files,
                                       self.DICT_SEQUENCE_TYPES[sequence_type]])

        try:
            tracking_folder = os.path.join(patient_folder, 'TrackingInformation')
            tracking_files = [[f.strip("_TrackingInformation.txt"), os.path.join(tracking_folder, f)] for f in os.listdir(tracking_folder) if f.endswith('TrackingInformation.txt')]
        except FileNotFoundError:
            tracking_files = list()

        return patient_name, image_paths, list_sequences, tracking_files

    def _is_valid_sequence(self, file_path: str, sequence_name: str, return_extension: bool = False):
        re_match = re.match(f'{sequence_name}_.+\d+{self.REG_EXP_ACCEPTED_FILES}', file_path)
        ret_val = False
        if re_match:
            ret_val = True
            if return_extension:
                ret_val = (ret_val, re_match[1])
        return ret_val

    def _is_valid_file(self, file_path: str, return_extension: bool = False):
        re_match = re.match(f'.*{self.REG_EXP_ACCEPTED_FILES}', os.path.split(file_path)[-1])
        ret_val = False
        if re_match:
            ret_val = True
            if return_extension:
                ret_val = (ret_val, re_match[1])
        return ret_val





