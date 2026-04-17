import json
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from django.http import JsonResponse, HttpResponseRedirect
from django.shortcuts import render, redirect
from django.http import Http404
from django.db import transaction

import common.task
from .models import *
from annotationweb.models import Task, ImageAnnotation, KeyFrameAnnotation, Label, TrackingData


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


        # Load frame predictions if available
        if sequence:
            print("Loading frame predictions for sequence:", sequence)
            frame_predictions_qs = FramePrediction.objects.filter(sequence=sequence).order_by("frame_nr")
            print("Loaded", frame_predictions_qs.count(), "frame predictions")
            for fp in frame_predictions_qs:
                print(f"Frame {fp.frame_nr}: Predicted class = {fp.predicted_class_info}")
            frame_predictions = {fp.frame_nr: fp.predicted_class_info for fp in frame_predictions_qs}

            # Load tracking data sync if available (obs! (td.id -1) corresponds to frame_nr)
            tracking_data = TrackingData.objects.filter(subject_id=sequence.subject,
                                                        synchronisedtrackingdata__isnull=False).order_by("id")
            branch_code = {td.id -1: td.branch_code for td in tracking_data}

            context["branch_codes_json"] = json.dumps(branch_code)
            #print("Frame predictions:", frame_predictions)
            #print("Branch codes:", branch_code)

            # Choose which source to expose, e.g.:
            # prefer frame predictions if they exist, otherwise tracking data
            if frame_predictions:
                chosen_codes = frame_predictions
                use_predictions = True
                print("Using frame predictions")
                #context["frame_predictions"] = frame_predictions
                #context["frame_predictions_json"] = json.dumps(frame_predictions)
            elif branch_code:
                chosen_codes = branch_code
                use_predictions = False
                print("Using branch codes")
                #context["tracking_data_sync"] = branch_code
                #context["tracking_data_sync_json"] = json.dumps(branch_code)
            else:
                chosen_codes = {}
                use_predictions = True
                print("No frame predictions or branch codes available")

            context["frame_predictions"] = frame_predictions
            context["frame_predictions_json"] = json.dumps(context["frame_predictions"])
            #context["tracking_data_sync"] = branch_code
            #context["tracking_data_sync_json"] = json.dumps(context["branch_code"])

            context["branch_codes"] = chosen_codes
            context["branch_codes_json"] = json.dumps(chosen_codes)
            context["use_predictions"] = use_predictions

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
            with (transaction.atomic()):
                annotations = common.task.save_annotation(request)
                print("Annotations:", annotations)
                frame_labels = json.loads(request.POST['frame_labels'])
                print("Annotations:", annotations)
                custom_frame_labels = json.loads(request.POST.get('custom_frame_labels', '{}')) # Load custom frame labels if provided (from text boxes)
                print("Custom frame labels:", custom_frame_labels)
                # NEW: parse custom label colors dict
                custom_label_colors = json.loads(request.POST.get('custom_frame_label_colors', '{}'))

                # Save standard labels for annotated frames
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

                    # Save custom texbox labels if present
                    custom_label = custom_frame_labels.get(str(annotation.frame_nr))
                    if custom_label:
                        # get or create Label *without* overwriting existing colors
                        label, created = Label.objects.get_or_create(
                            name=custom_label,
                            #task=annotation.image_annotation.task,
                            #defaults={'color_red': 128, 'color_green': 128, 'color_blue': 128}
                        )
                        print(f"Custom label: {custom_label}, Created: {created}, Colors: {custom_label_colors.get(custom_label)}")
                        #if created:
                        color = custom_label_colors.get(custom_label)

                        def _coerce_color_channel(value, default=128):
                            """
                            Coerce a single color channel to an int between 0 and 255.
                            Falls back to `default` if the value is missing or invalid.
                            """
                            try:
                                ivalue = int(value)
                            except (TypeError, ValueError):
                                return default
                            return max(0, min(255, ivalue))

                        if isinstance(color, dict):
                            label.color_red = _coerce_color_channel(color.get('red'), 128)
                            label.color_green = _coerce_color_channel(color.get('green'), 128)
                            label.color_blue = _coerce_color_channel(color.get('blue'), 128)
                        else:  # assign some default color only once
                            label.color_red = 128
                            label.color_green = 128
                            label.color_blue = 128
                        label.save()

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
                            #defaults={'color_red': 0, 'color_green': 255, 'color_blue': 0}
                        )
                        # Only set the label color when the label is newly created
                        # or when its color fields are not yet set, to avoid
                        # changing colors for existing shared labels.
                        if created or label.color_red is None or label.color_green is None or label.color_blue is None:
                            color = custom_label_colors.get(custom_label)
                            if color is not None:
                                label.color_red = color.get('red', 128)
                                label.color_green = color.get('green', 128)
                                label.color_blue = color.get('blue', 128)
                            else:
                                label.color_red = 128
                                label.color_green = 128
                                label.color_blue = 128
                            label.save()

                        sublabel =  SubsequenceLabel.objects.create(
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

