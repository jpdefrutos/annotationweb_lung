from django.db import models
from annotationweb.models import KeyFrameAnnotation, Label, ImageSequence


class SubsequenceLabel(models.Model):
    """
    Attributes
    ----------
    image : models.OneToOneField --> KeyFrameAnnotation
        The KeyFrameAnnotation instance that the label is connected to
    label : models.ForeignKey --> Label
        The label of the subsequence/video segment
    """
    image = models.ForeignKey(KeyFrameAnnotation, on_delete=models.CASCADE)
    label = models.ForeignKey(Label, on_delete=models.CASCADE)


class FramePrediction(models.Model):
    """
    Stores automated model predictions for a frame in an ImageSequence.
    Keeps these separate from human annotations (KeyFrameAnnotation).
    """

    sequence = models.ForeignKey(ImageSequence, on_delete=models.CASCADE)
    frame_nr = models.IntegerField(default=0)
    filepath = models.TextField(default='')
    predicted_class_info = models.CharField(default='', max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.sequence} Frame {self.frame_nr}: {self.predicted_class_info}"

    class Meta:
        ordering = ["created_at"]
        unique_together = ("sequence", "frame_nr")
