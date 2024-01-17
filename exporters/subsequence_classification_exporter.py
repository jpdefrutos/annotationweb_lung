"""
TODO: Write exporter for Subsequence classification task
"""
import os
from os.path import join
import csv
from shutil import rmtree, copyfile

import PIL
import h5py
import numpy as np
from django import forms

from annotationweb.models import *
from common.exporter import Exporter
from common.label import get_all_labels
from common.metaimage import MetaImage
from common.utility import copy_image, create_folder
from subsequence_classification.models import SubsequenceLabel


class SubsequenceClassificationExporterForm(forms.Form):
    path = forms.CharField(label='Storage path', max_length=1000)
    delete_existing_data = forms.BooleanField(label='Delete any existing data at storage path', initial=False, required=False)
    output_image_format = forms.ChoiceField(choices=(
        ('png', 'PNG'),
        ('mhd', 'MetaImage')
    ), initial='png', label='Output image format')

    def __init__(self, task, data=None):
        super().__init__(data)
        self.fields['dataset'] = forms.ModelMultipleChoiceField(queryset=Dataset.objects.filter(task=task))


class SubsequenceClassificationExporter(Exporter):
    """
    This exporter will create a folder at the given path and create 2 files:
    labels.txt      - list of labels
    file_list.txt   - list of files and labels
    A folder is created for each dataset with the actual images
    """

    task_type = Task.SUBSEQUENCE_CLASSIFICATION
    name = 'Default subsequence classification exporter'

    def get_form(self, data=None):
        return SubsequenceClassificationExporterForm(self.task, data=data)

    def export(self, form):
        datasets = form.cleaned_data['dataset']
        delete_existing_data = form.cleaned_data['delete_existing_data']
        # Create dir, delete old if it exists
        path = form.cleaned_data['path']
        if delete_existing_data:
            # Delete path
            try:
                os.stat(path)
                rmtree(path)
            except:
                # Folder does not exist
                pass

            # Create folder again
            try:
                os.mkdir(path)
            except:
                return False, 'Failed to create directory at ' + path
        else:
            # Check that folder exist
            try:
                os.stat(path)
            except:
                return False, 'Path does not exist: ' + path

        # Create label file (mapping label_id to label name)
        label_file = open(os.path.join(path, 'labels.csv'), 'w')
        label_dict = get_all_labels(task=self.task)
        label_file.write(';'.join(('label_id', 'label_name', 'label_path')) + '\n')
        for label in label_dict:
            label_name_full = label['name']
            label_name = label_name_full.split('-')[-1]
            label_file.write(';'.join((str(label['id']), label_name, label_name_full)) + '\n')
        label_file.close()

        # Create file_list.txt file
        file_list = open(os.path.join(path, 'file_list.csv'), 'w')
        file_list.write(';'.join(('file_path', 'label_id')) + '\n')

        labeled_sequences = ImageAnnotation.objects.filter(task=self.task, rejected=False)
        for image_sequence in labeled_sequences:
            keyframes = KeyFrameAnnotation.objects.filter(
                image_annotation=image_sequence
            )
            for keyframe in keyframes:
                # Get filename
                frame_no = keyframe.frame_nr
                filename_format = image_sequence.image.format
                file_path = filename_format.replace('#', str(frame_no))
                # Get image label
                subsequence_label = keyframe.subsequencelabel
                # Write filepath/label_id pair to the file
                file_list.write(';'.join((file_path, str(subsequence_label.label.id))) + '\n')

        file_list.close()

        return True, path


def read_csv_file(filename, has_header=False, include_header=False):
    print(f"Reading {filename} as .csv")

    with open(filename, newline='') as csvfile:
        reader = csv.reader(
            csvfile, delimiter=';', #quotechar='|',
            skipinitialspace=True,
        )

        if has_header:
            if include_header:
                i = 0   # start with first (header) row
            else:
                i = 1   # start with first content row
        else:
            i = 0   # start with first row and first content row

        data = []
        for row in reader:
            data.append([])
            for c in row:
                data[i].append(c)
            i += 1

        return data

def write_dict_to_csv(filename, dictionary: dict):
    print(f"Writing dict to .csv file {filename}")

    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        print(dictionary)
        for k, v in dictionary.items():
            writer.writerow([k, v])
