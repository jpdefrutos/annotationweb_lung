from django.test import TestCase
from django.db import transaction

from annotationweb.models import Task, Label
from common.label import get_or_create_task_label


class GetOrCreateTaskLabelTests(TestCase):
    """Covers common.label.get_or_create_task_label(), shared by
    subsequence_classification and bronchoscopy_boundingbox."""

    def setUp(self):
        self.task = Task.objects.create(name='t', type=Task.SUBSEQUENCE_CLASSIFICATION)

    def test_creates_once_and_reuses(self):
        with transaction.atomic():
            label1 = get_or_create_task_label(self.task, 'foo', (1, 2, 3))
        with transaction.atomic():
            label2 = get_or_create_task_label(self.task, 'foo', (9, 9, 9))

        self.assertEqual(label1.pk, label2.pk)
        self.assertEqual(Label.objects.filter(name='foo').count(), 1)
        self.assertEqual(self.task.label.count(), 1)
        self.assertEqual(label1.get_hex_code(), '#010203')

    def test_scoped_per_task_not_global(self):
        other_task = Task.objects.create(name='t2', type=Task.SUBSEQUENCE_CLASSIFICATION)
        with transaction.atomic():
            label_a = get_or_create_task_label(self.task, 'shared', (0, 0, 0))
        with transaction.atomic():
            label_b = get_or_create_task_label(other_task, 'shared', (0, 0, 0))

        self.assertNotEqual(label_a.pk, label_b.pk)
