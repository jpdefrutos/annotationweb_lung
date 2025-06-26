import os.path

from django import template

register = template.Library()

@register.filter
def get_sequence_name(image_sequence_format):
    """ Get the sequence name from an ImageSequence format string """
    path = os.path.dirname(image_sequence_format)
    sequence_dir = os.path.basename(path)
    return sequence_dir


@register.filter
def get_sequence_and_filename(image_sequence_format):
    """
    Get sequence folder name and filename from an ImageSequence format string
    """
    path, fn = os.path.split(image_sequence_format)
    sequence_dir = os.path.join(os.path.basename(path), fn)
    return sequence_dir


@register.filter
def get_patient_sequence_and_filename(image_sequence_format):
    """
    Get patient_xx/sequence_xx/filename.png from an ImageSequence format string
    """
    seq_path, fn = os.path.split(image_sequence_format)
    patient_path, seq_dir = os.path.split(seq_path)
    path = os.path.join(os.path.basename(patient_path), seq_dir, fn)
    return path
