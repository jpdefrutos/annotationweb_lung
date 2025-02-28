from os import listdir

from common.importer import Importer
from django import forms
from annotationweb.models import ImageSequence, Dataset, Subject, ImageMetadata
import os
from os.path import join, basename
import glob

class ImageSequenceImporterForm(forms.Form):
    path = forms.CharField(label='Data path', max_length=1000)

    # TODO validate path

    def __init__(self, data=None):
        super().__init__(data)


class ImageSequenceImporter(Importer):
    """
    Data should be sorted in the following way in the root folder:
    Subject 1/
        Sequence 1/
            <name>_0.mhd or .png
            <name>_1.mhd or .png
            ...
        Sequence 2/
            ...
    Subject 2/
        ...
    Allow both .mhd and .png sequences from the same root folder.
    Assumes that a single sequence does not mix .mhd and .png

    This importer will create a subject for each subject folder and an image sequence for each subfolder.
    """

    name = 'Image sequence importer'
    dataset = None

    def get_form(self, data=None):
        return ImageSequenceImporterForm(data)

    def import_data(self, form):
        if self.dataset is None:
            raise Exception('Dataset must be given to importer')

        path = form.cleaned_data['path']
        # Go through each subfolder and create a subject for each
        for file in os.listdir(path):
            subject_dir = join(path, file)
            if not os.path.isdir(subject_dir):
                continue

            try:
                # Check if subject exists in this dataset first
                subject = Subject.objects.get(name=file, dataset=self.dataset)
            except Subject.DoesNotExist:
                # Create new subject
                subject = Subject()
                subject.name = file
                subject.dataset = self.dataset
                subject.save()

            for sequence_dir in os.listdir(subject_dir):
                image_sequence_dir = join(subject_dir, sequence_dir)
                if not os.path.isdir(image_sequence_dir):
                    continue

                frames, name, extension = self._parse_sequence_dir(image_sequence_dir)

                if len(frames) == 0:
                    continue

                filename_format = join(image_sequence_dir, name + '_#')
                filename_format += extension

                image_sequence, already_imported = self._import_image_sequence(frames, subject, filename_format)
                if already_imported:
                    continue

                _ = self._import_metadata(image_sequence_dir, image_sequence)

        return True, path

    def _parse_sequence_dir(self, image_sequence_dir):
        # Count nr of frames
        # Handle only monotype sequence: .mhd or .png or .jpg
        name = None
        frames = list()
        extension = None
        for frame_file in os.listdir(image_sequence_dir):
            if frame_file[-4:] == '.mhd':
                image_filename = join(image_sequence_dir, frame_file)
                frames.append(image_filename)
                name = frame_file[:frame_file.rfind('_')]
                if extension is None or extension == '.mhd':
                    extension = '.mhd'
                else:
                    raise Exception('Found both mhd and png images in the same folder.')
            elif frame_file[-4:] == '.png':
                image_filename = join(image_sequence_dir, frame_file)
                frames.append(image_filename)
                name = frame_file[:frame_file.rfind('_')]
                if extension is None or extension == '.png':
                    extension = '.png'
                else:
                    raise Exception('Found both mhd and png images in the same folder.')
            elif frame_file[-4:] == '.jpg':
                image_filename = join(image_sequence_dir, frame_file)
                frames.append(image_filename)
                name = frame_file[:frame_file.rfind('_')]
                if extension is None or extension == '.jpg':
                    extension = '.jpg'
                else:
                    raise Exception('Found both jpg and mhd/png images in the same folder.')
        return frames, name, extension

    def _import_image_sequence(self, frames, subject, filename_format):
        sequence_already_imported = False
        try:
            # Check to see if sequence exist
            image_sequence = ImageSequence.objects.get(format=filename_format, subject=subject)
            # Check to see that nr of sequences is correct
            if image_sequence.nr_of_frames < len(frames):
                # Delete this sequnce, and redo it
                image_sequence.delete()
                # Create new
                image_sequence = ImageSequence()
                image_sequence.format = filename_format
                image_sequence.subject = subject
                image_sequence.nr_of_frames = len(frames)
                image_sequence.save()
            else:
                sequence_already_imported = True
        except ImageSequence.DoesNotExist:
            # Create new image sequence
            image_sequence = ImageSequence()
            image_sequence.format = filename_format
            image_sequence.subject = subject
            image_sequence.nr_of_frames = len(frames)
            image_sequence.save()
        return image_sequence, sequence_already_imported

    def _import_metadata(self, image_sequence_dir, image_sequence):
        metadata = None
        metadata_filename = join(image_sequence_dir, 'metadata.txt')
        if os.path.exists(metadata_filename):
            with open(metadata_filename, 'r') as f:
                for line in f:
                    parts = line.split(':')
                    if len(parts) != 2:
                        raise Exception('Excepted 2 parts when spliting metadata in file ' + metadata_filename)

                    # Save to DB
                    metadata = ImageMetadata()
                    metadata.image = image_sequence
                    metadata.name = parts[0].strip()
                    metadata.value = parts[1].strip()
                    metadata.save()
        return metadata