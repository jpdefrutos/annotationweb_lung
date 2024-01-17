import json

from django.contrib import messages
from django.http import JsonResponse, HttpResponseRedirect
from django.shortcuts import render, redirect
from django.http import Http404
from django.db import transaction

import common.task
from .models import *
from annotationweb.models import Task, ImageAnnotation


def label_next_image(request, task_id):
    return label_subsequence(request, task_id, None)


def label_subsequence(request, task_id, image_id):
    """
    TODO: From classification/views.py. Adapted to render page, needs further work
    """
    try:
        context = common.task.setup_task_context(request, task_id, Task.SUBSEQUENCE_CLASSIFICATION, image_id)
        context['javascript_files'] = ['subsequence_classification/subsequence_classification.js']

        # Load labels
        #context['labels'] = Label.objects.filter(task=task_id)

        # Get label, if image has been already labeled
        try:
            sequence_annotations = ImageAnnotation.objects.filter(
                task=task_id, image_id=image_id)
            context['subsequence_labels'] = SubsequenceLabel.objects.filter(
                image__image_annotation__in=sequence_annotations)
        except KeyFrameAnnotation.DoesNotExist:
            print('No previous labels found..')
            pass

        return render(request, 'subsequence_classification/label_subsequence.html', context)
    except common.task.NoMoreImages:
        messages.info(request, 'This task is finished, no more images to annotate.')
        return redirect('index')
    except RuntimeError as e:
        messages.error(request, str(e))
        return HttpResponseRedirect(request.META.get('HTTP_REFERER'))


def save_labels(request):
    """
    TODO: From classification/views.py. Adapt to this task
    """
    response = {}  # initialize response
    try:
        rejected = request.POST['rejected'] == 'true'
        if rejected:
            annotations = common.task.save_annotation(request)
            response = {
                'success': 'true',
                'message': 'Completed'
            }
        else:
            with transaction.atomic():
                annotations = common.task.save_annotation(request)
                frame_labels = json.loads(request.POST['frame_labels'])

                for annotation in annotations:
                    labeled_image = SubsequenceLabel()
                    labeled_image.image = annotation
                    label = Label.objects.get(id=frame_labels[str(annotation.frame_nr)])
                    labeled_image.label = label
                    labeled_image.task = annotation.image_annotation.task
                    labeled_image.save()

            response = {
                'success': 'true',
                'message': 'Completed'
            }
        messages.success(request, 'Subsequence classification saved')
    except Exception as e:
        response = {
            'success': 'false',
            'message': str(e)
        }

    return JsonResponse(response)

