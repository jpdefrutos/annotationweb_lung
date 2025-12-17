import json
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from django.http import JsonResponse, HttpResponseRedirect
from django.shortcuts import render, redirect
from django.http import Http404
from django.db import transaction

import common.task
from .models import *
from annotationweb.models import Task, ImageAnnotation, KeyFrameAnnotation, Label
from subsequence_classification.models import FramePrediction

from .models import SubsequenceLabel

def label_next_image(request, task_id):
    return label_subsequence(request, task_id, None)



def get_frame_label_ids(request, frame_id):
    try:
        frame = KeyFrameAnnotation.objects.get(id=frame_id)
        label_ids = list(SubsequenceLabel.objects.filter(image=frame).values_list('label_id', flat=True))
        return JsonResponse({'label_ids': label_ids})
    except KeyFrameAnnotation.DoesNotExist:
        return JsonResponse({'error': 'Frame not found'}, status=404)

def label_subsequence(request, task_id, image_id):
    """
    TODO: From classification/views.py. Adapted to render page, needs further work
    """
    print("label_subsequence view called")
    try:
        context = common.task.setup_task_context(request, task_id, Task.SUBSEQUENCE_CLASSIFICATION, image_id)
        context['javascript_files'] = ['subsequence_classification/subsequence_classification.js']

        # Load labels
        context['labels'] = Label.objects.filter(task=task_id)
        context['toplabels'] = Label.objects.filter(task=task_id, parent=None)
        # Get the sequence
        sequence = context.get("image_sequence")

        if sequence:
            print("Loading frame predictions for sequence:", sequence)
            frame_predictions = FramePrediction.objects.filter(sequence=sequence).order_by("frame_nr")
            print("Loaded", frame_predictions.count(), "frame predictions")
            # Print predicted class info for each frame
            for fp in frame_predictions:
                print(f"Frame {fp.frame_nr}: Predicted class = {fp.predicted_class_info}")
            context["frame_predictions"] = {fp.frame_nr: fp.predicted_class_info for fp in frame_predictions}
            context["frame_predictions_json"] = json.dumps(context["frame_predictions"])

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

"""
def save_labels(request):
    
    #TODO: From classification/views.py. Adapt to this task
    
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
"""


def save_labels(request):
    print("save_labels called")  # Step 1
    print("POST data:", request.POST)  # Step 2
    response = {}
    try:
        rejected = request.POST.get('rejected', 'false') == 'true'
        if rejected:
            print("Annotation rejected")
            annotations = common.task.save_annotation(request)
            response = {
                'success': 'true',
                'message': 'Completed'
            }
        else:
            with transaction.atomic():
                annotations = common.task.save_annotation(request)
                print("Annotations:", annotations)
                frame_labels = json.loads(request.POST['frame_labels'])
                print("Annotations:", annotations)
                custom_frame_labels = json.loads(request.POST.get('custom_frame_labels', '{}')) # Load custom frame labels if provided (from text boxes)
                print("Custom frame labels:", custom_frame_labels)
                # Save standard and custom labels for annotated frames
                for annotation in annotations:
                    SubsequenceLabel.objects.filter(image=annotation).delete()
                    label_ids = frame_labels.get(str(annotation.frame_nr), [])
                    if not isinstance(label_ids, list):
                        label_ids = [label_ids]
                    for label_id in label_ids:
                        print("Saving label_id:", label_id)
                        labeled_image = SubsequenceLabel(
                            image=annotation,
                            label=Label.objects.get(id=label_id)
                        )
                        #
                        labeled_image.save()

                    # Save custom label if present
                    custom_label = custom_frame_labels.get(str(annotation.frame_nr))
                    if custom_label:
                        label, created = Label.objects.get_or_create(
                            name=custom_label,
                            #task=annotation.image_annotation.task,
                            defaults={'color_red': 128, 'color_green': 128, 'color_blue': 128}
                        )
                        sublabel = SubsequenceLabel.objects.create(
                            image=annotation,
                            label=label,
                            #task=annotation.image_annotation.task
                        )
                        print(f"Created SubsequenceLabel for custom label id={sublabel.id}")
                # Save custom labels for frames not in annotations
                task_id = int(request.POST['task_id'])
                image_id = int(request.POST['image_id'])
                for frame_str, custom_label in custom_frame_labels.items():
                    frame_nr = int(frame_str)
                    if not any(a.frame_nr == frame_nr for a in annotations):
                        image_annotation = ImageAnnotation.objects.filter(task=task_id, image_id=image_id).first()
                        if not image_annotation:
                            raise Exception("No ImageAnnotation found for this task")
                        annotation, _ = KeyFrameAnnotation.objects.get_or_create(
                            image_annotation=image_annotation,
                            frame_nr=frame_nr
                        )
                        label, created = Label.objects.get_or_create(
                            name=custom_label,
                            #task=annotation.image_annotation.task,
                            defaults={'color_red': 0, 'color_green': 255, 'color_blue': 0}
                        )
                        sublabel = SubsequenceLabel.objects.create(
                            image=annotation,
                            label=label,
                            #task=annotation.image_annotation.task
                        )
                        print(f"Created SubsequenceLabel for custom label id={sublabel.id}")
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
