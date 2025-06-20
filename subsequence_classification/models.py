from django.db import models
from annotationweb.models import KeyFrameAnnotation, Label


class SubsequenceLabel(models.Model):
    """

    Attributes
    ----------
    image : models.OneToOneField --> KeyFrameAnnotation
        The KeyFrameAnnotation instance that the label is connected to
    label : models.ForeignKey --> Label
        The label of the subsequence/video segment
    """
    image = models.OneToOneField(KeyFrameAnnotation, on_delete=models.CASCADE)
    label = models.ForeignKey(Label, on_delete=models.CASCADE)
