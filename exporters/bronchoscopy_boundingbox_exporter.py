import csv
import os
from os.path import join
from shutil import rmtree

from django import forms
from django.http import HttpResponse

from annotationweb.models import Task, Subject, KeyFrameAnnotation
from bronchoscopy_boundingbox.models import BronchoscopyBoundingBox
from common.exporter import Exporter
from common.utility import create_folder, copy_image


class BronchoscopyBoundingBoxExporterForm(forms.Form):
    path = forms.CharField(label='Storage path', max_length=1000)
    delete_existing_data = forms.BooleanField(
        label='Delete any existing data at storage path', initial=False, required=False)

    def __init__(self, task, data=None):
        super().__init__(data)
        self.fields['subjects'] = forms.ModelMultipleChoiceField(
            queryset=Subject.objects.filter(dataset__task=task))


class BronchoscopyBoundingBoxExporter(Exporter):

    task_type = Task.BRONCHOSCOPY_BOUNDING_BOX
    name = 'Default bronchoscopy bounding box exporter'

    def get_form(self, data=None):
        return BronchoscopyBoundingBoxExporterForm(self.task, data=data)

    def export(self, form):
        path = form.cleaned_data['path']
        if form.cleaned_data['delete_existing_data']:
            try:
                os.stat(path)
                rmtree(path)
            except:
                pass

        create_folder(path)

        label_colors = {
            label.name: '#%02x%02x%02x' % (label.color_red, label.color_green, label.color_blue)
            for label in self.task.label.all()
        }

        csv_path = join(path, 'annotations.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['subject', 'sequence', 'frame_nr', 'label', 'x', 'y', 'width', 'height', 'color'])

            for subject in form.cleaned_data['subjects']:
                frames = KeyFrameAnnotation.objects.filter(
                    image_annotation__task=self.task,
                    image_annotation__image__subject=subject,
                    image_annotation__rejected=False,
                )
                for frame in frames:
                    image_sequence = frame.image_annotation.image
                    boxes = BronchoscopyBoundingBox.objects.filter(image=frame)
                    for box in boxes:
                        writer.writerow([
                            subject.name,
                            image_sequence.format,
                            frame.frame_nr,
                            box.label,
                            box.x,
                            box.y,
                            box.width,
                            box.height,
                            label_colors.get(box.label, ''),
                        ])

                    # Copy the annotated frame image
                    filename = image_sequence.format.replace('#', str(frame.frame_nr))
                    target_name = os.path.basename(image_sequence.format).replace('#', str(frame.frame_nr))
                    subject_path = join(path, subject.name)
                    create_folder(subject_path)
                    copy_image(filename, join(subject_path, target_name))

        return True, path
