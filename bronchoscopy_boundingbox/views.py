import json
import os
from django.conf import settings
from django.db import transaction
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from annotationweb.models import Task, ImageSequence
from common.task import setup_task_context, save_annotation, NoMoreImages
from common.label import get_or_create_task_label
from .models import BronchoscopyBoundingBox
from . import lumen
from subsequence_classification.models import SubsequenceLabel


def _hex_to_rgb(hex_color):
    h = hex_color.lstrip('#')
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def process_next_image(request, task_id):
    return process_image(request, task_id, None)


def process_image(request, task_id, image_id):
    try:
        context = setup_task_context(request, task_id, Task.BRONCHOSCOPY_BOUNDING_BOX, image_id)
        context['javascript_files'] = ['bronchoscopy_boundingbox/bronchoscopy_boundingbox.js',
                                       'bronchoscopy_boundingbox/proposals.js']
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
        with transaction.atomic():
            annotations = save_annotation(request)
            boxes_data = json.loads(request.POST['boxes'])

            task = Task.objects.get(pk=int(request.POST['task_id']))
            label_colors = {}
            for frame_boxes in boxes_data.values():
                for box in frame_boxes:
                    name = box.get('label', '').strip()
                    if name and name not in label_colors:
                        label_colors[name] = box.get('color', '#e6194b')
            for name, hex_color in label_colors.items():
                get_or_create_task_label(task, name, _hex_to_rgb(hex_color))

            for annotation in annotations:
                frame_nr = str(annotation.frame_nr)
                if frame_nr not in boxes_data:
                    continue  # key frame exists but user drew no boxes on it
                for box in boxes_data[frame_nr]:
                    BronchoscopyBoundingBox.objects.create(
                        image=annotation,
                        x=int(box['x']),
                        y=int(box['y']),
                        width=int(box['width']),
                        height=int(box['height']),
                        label=box.get('label', ''),
                    )
        return JsonResponse({'success': 'true', 'message': 'Completed'})
    except Exception as e:
        return JsonResponse({'success': 'false', 'message': str(e)})


# Max frames per suggestion request; reading the .mhd frames dominates the
# time (~0.1 s per 1920x1080 frame), so this keeps a request well under a minute.
MAX_PROPOSAL_FRAMES = 300


def _lumen_cache_dir():
    return getattr(settings, 'LUMEN_CACHE_DIR',
                   os.path.join(settings.BASE_DIR, 'lumen_cache'))


@require_POST
def propose_boxes(request):
    """
    Lumen box proposals for frames [start, start+count) of an image sequence.

    POST: task_id, image_sequence_id, start, count, params (JSON object),
    state (JSON object from the previous response, or empty). Returns
    {success, rows, state, frames: [first, last]}. When the params are
    unchanged and start follows the previous chunk, the tracks continue.
    Nothing is saved: proposals only become boxes when the annotator accepts
    them and saves the annotation.
    """
    try:
        task = Task.objects.get(pk=int(request.POST['task_id']),
                                type=Task.BRONCHOSCOPY_BOUNDING_BOX)
        sequence = ImageSequence.objects.select_related('subject').get(
            pk=int(request.POST['image_sequence_id']))
    except (KeyError, ValueError, Task.DoesNotExist, ImageSequence.DoesNotExist):
        return JsonResponse({'success': False, 'message': 'Unknown task or image sequence'}, status=404)
    if not request.user.is_staff and not task.user.filter(pk=request.user.pk).exists():
        return JsonResponse({'success': False, 'message': 'No access to this task'}, status=403)
    if not task.dataset.filter(pk=sequence.subject.dataset_id).exists():
        return JsonResponse({'success': False, 'message': 'Image sequence is not part of this task'}, status=404)

    try:
        start = int(request.POST['start'])
        count = int(request.POST.get('count', 30))
        params = lumen.params_from_dict(json.loads(request.POST.get('params') or '{}'))
        state = json.loads(request.POST.get('state') or 'null')
    except (KeyError, ValueError, TypeError, lumen.LumenError) as e:
        return JsonResponse({'success': False, 'message': 'Invalid request: %s' % e}, status=400)

    try:
        frames = lumen.sequence_frames(sequence)
        first, last = min(frames), max(frames)
        if not first <= start <= last:
            raise lumen.LumenError('Frame %d is outside the video (%d-%d)' % (start, first, last))
        count = max(1, min(count, MAX_PROPOSAL_FRAMES))
        end = min(start + count - 1, last)

        roi, mask = lumen.roi_and_mask(frames, _lumen_cache_dir(),
                                       lumen.cache_key_for(sequence), params.erode)
        runner = lumen.Runner(frames, roi, mask, params, state)
        rows = runner.process(range(start, end + 1))
    except lumen.LumenError as e:
        return JsonResponse({'success': False, 'message': str(e)}, status=400)

    return JsonResponse({'success': True, 'rows': rows, 'state': runner.state(),
                         'frames': [start, end], 'dropped': runner.dropped})
