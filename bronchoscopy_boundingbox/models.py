from django.db import models
from annotationweb.models import KeyFrameAnnotation


class BronchoscopyBoundingBox(models.Model):
    image = models.ForeignKey(KeyFrameAnnotation, on_delete=models.CASCADE)
    x = models.PositiveIntegerField()
    y = models.PositiveIntegerField()
    width = models.PositiveIntegerField()
    height = models.PositiveIntegerField()
    label = models.CharField(max_length=255, blank=False)
    color = models.CharField(max_length=7)  # hex color, e.g. '#e6194b'
