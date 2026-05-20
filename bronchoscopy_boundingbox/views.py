import json
from django.shortcuts import render, redirect
from django.http import JsonResponse
from annotationweb.models import Task
from common.task import setup_task_context, save_annotation, NoMoreImages
from .models import BronchoscopyBoundingBox
from subsequence_classification.models import SubsequenceLabel


def process_next_image(request, task_id):
    return process_image(request, task_id, None)


def process_image(request, task_id, image_id):
    try:
        context = setup_task_context(request, task_id, Task.BRONCHOSCOPY_BOUNDING_BOX, image_id)
        context['javascript_files'] = ['bronchoscopy_boundingbox/bronchoscopy_boundingbox.js']
        context['boxes'] = BronchoscopyBoundingBox.objects.filter(image__in=context['frames'])

        frame_labels = {}
        if context.get('image_sequence'):
            for sl in SubsequenceLabel.objects.filter(
                image__image_annotation__image=context['image_sequence']
            ).select_related('image', 'label'):
                frame_labels[sl.image.frame_nr] = sl.label.name
        context['frame_labels_json'] = json.dumps(frame_labels)

        return render(request, 'bronchoscopy_boundingbox/process_image.html', context)
    except NoMoreImages:
        return redirect('index')
    except RuntimeError:
        return redirect(request.META.get('HTTP_REFERER', '/'))


def save_boxes(request):
    try:
        annotations = save_annotation(request)
        boxes_data = json.loads(request.POST['boxes'])
        for annotation in annotations:
            frame_nr = str(annotation.frame_nr)
            for box in boxes_data[frame_nr]:
                BronchoscopyBoundingBox.objects.create(
                    image=annotation,
                    x=int(box['x']),
                    y=int(box['y']),
                    width=int(box['width']),
                    height=int(box['height']),
                    label=box.get('label', ''),
                    color=box.get('color', '#e6194b'),
                )
        return JsonResponse({'success': 'true', 'message': 'Completed'})
    except Exception as e:
        return JsonResponse({'success': 'false', 'message': str(e)})
