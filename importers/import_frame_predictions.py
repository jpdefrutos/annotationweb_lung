import os
import csv
from django import forms
from annotationweb.models import ImageSequence
from subsequence_classification.models import FramePrediction
from common.importer import Importer


class FramePredictionImporterForm(forms.Form):
    csv_file = forms.CharField(label="CSV file path", max_length=1000)

    def clean_csv_file(self):
        csv_file = self.cleaned_data['csv_file']
        if not os.path.exists(csv_file):
            raise forms.ValidationError(f"CSV file {csv_file} does not exist")
        return csv_file


class FramePredictionImporter(Importer):
    name = "Frame prediction CSV importer"
    dataset = None  # must be set externally before running
    def has_sequences(self):
        if self.dataset is None:
            return False
        return ImageSequence.objects.filter(subject__dataset=self.dataset).exists()

    def is_available(self):
        return self.has_sequences()

    def get_form(self, data=None):
        if not self.has_sequences():
            return None
        return FramePredictionImporterForm(data)

    def import_data(self, form: forms.Form):
        if self.dataset is None:
            raise Exception("Dataset must be given to the importer")

        if not self.has_sequences():
            raise Exception("No ImageSequences found in this dataset. Cannot import.")

        csv_file = form.cleaned_data['csv_file']

        with open(csv_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filepath = row['filepath']
                predicted_class_info = row['predicted_class']

                # Extract frame index from filename
                filename = os.path.basename(filepath)  # e.g. frame_0.png
                try:
                    frame_nr = int(filename.split("_")[-1].split(".")[0])
                except ValueError:
                    print(f"Could not parse frame index from {filename}")
                    continue

                # Build ImageSequence.format string from filepath
                sequence_format = filepath.replace(f"frame_{frame_nr}.png", "frame_#.png")

                # Find sequence in DB
                seq = ImageSequence.objects.filter(
                    subject__dataset=self.dataset,
                    format=sequence_format
                ).first()

                if not seq:
                    print(f"No ImageSequence found for format {sequence_format}") # Log and skip
                    continue

                # Insert or update FramePrediction
                FramePrediction.objects.update_or_create(
                    sequence=seq,
                    frame_nr=frame_nr,
                    defaults={
                        "filepath": filepath,
                        "predicted_class_info": predicted_class_info
                    },
                )

        return True, csv_file