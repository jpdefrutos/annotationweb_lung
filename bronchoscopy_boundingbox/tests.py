import json
import os
import shutil
import tempfile
import zlib

import numpy as np
from django.contrib.auth.models import User
from django.test import TestCase, override_settings

from annotationweb.models import Dataset, ImageSequence, Subject, Task
from . import lumen

W, H = 320, 240


def _write_mhd(path, img):
    """Compressed 3-channel uint8 MetaImage, like the bronchoscopy frames."""
    raw = zlib.compress(np.ascontiguousarray(img).tobytes())
    data_name = os.path.basename(path)[:-4] + '.zraw'
    with open(os.path.join(os.path.dirname(path), data_name), 'wb') as f:
        f.write(raw)
    with open(path, 'w') as f:
        f.write('ObjectType = Image\nNDims = 2\nBinaryData = True\n'
                'CompressedData = True\nCompressedDataSize = %d\n'
                'ElementSpacing = 1 1\nDimSize = %d %d\n'
                'ElementNumberOfChannels = 3\nElementType = MET_UCHAR\n'
                'ElementDataFile = %s\n' % (len(raw), W, H, data_name))


def _frame(dx):
    """Bright scope image (black margins) with two dark 'lumens', moved by dx."""
    img = np.zeros((H, W, 3), np.uint8)
    yy, xx = np.mgrid[:H, :W]
    img[:, 40:300] = (150 + xx[:, 40:300] % 7)[..., None]   # textured bright area
    img[(xx - (100 + dx)) ** 2 + (yy - 120) ** 2 < 30 ** 2] = 10
    img[(xx - (220 + dx)) ** 2 + (yy - 100) ** 2 < 25 ** 2] = 12
    return img


class LumenProposalTests(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tmp = tempfile.mkdtemp()
        cls.frame_dir = os.path.join(cls.tmp, 'frames')
        os.makedirs(cls.frame_dir)
        for f in range(10):
            _write_mhd(os.path.join(cls.frame_dir, 'seq_%d.mhd' % f), _frame(f))
        cls.cache_dir = os.path.join(cls.tmp, 'cache')
        cls.settings_override = override_settings(LUMEN_CACHE_DIR=cls.cache_dir)
        cls.settings_override.enable()

    @classmethod
    def tearDownClass(cls):
        cls.settings_override.disable()
        shutil.rmtree(cls.tmp)
        super().tearDownClass()

    def setUp(self):
        dataset = Dataset.objects.create(name='d')
        subject = Subject.objects.create(name='s', dataset=dataset)
        self.sequence = ImageSequence.objects.create(
            format=os.path.join(self.frame_dir, 'seq_#.mhd'), subject=subject,
            nr_of_frames=10, start_frame_nr=0)
        self.task = Task.objects.create(name='t', type=Task.BRONCHOSCOPY_BOUNDING_BOX)
        self.task.dataset.add(dataset)
        self.user = User.objects.create_user('annotator', password='x')
        self.task.user.add(self.user)
        self.client.force_login(self.user)

    def _propose(self, start, count, params=None, state=None, **extra):
        data = {'task_id': self.task.pk, 'image_sequence_id': self.sequence.pk,
                'start': start, 'count': count,
                'params': json.dumps(params or {}),
                'state': json.dumps(state) if state else ''}
        data.update(extra)
        return self.client.post('/bronchoscopy-bounding-box/propose/', data)

    def test_detects_both_lumens_and_tracks_them_across_chunks(self):
        r1 = self._propose(0, 5).json()
        self.assertTrue(r1['success'], r1)
        self.assertEqual(r1['frames'], [0, 4])
        self.assertEqual(len(r1['rows']), 10)                 # 2 lumens x 5 frames
        self.assertEqual({r['box_id'] for r in r1['rows']}, {1, 2})

        r2 = self._propose(5, 5, state=r1['state']).json()    # next chunk continues
        self.assertEqual(r2['state']['run_id'], r1['state']['run_id'])
        self.assertEqual({r['box_id'] for r in r2['rows']}, {1, 2})
        left = [r for r in r1['rows'] + r2['rows'] if r['box_id'] == 1]
        self.assertEqual([r['frame_nr'] for r in left], list(range(10)))
        self.assertTrue(all(r['label'] == 'T1' and r['level'] == 1 for r in left))

    def test_chunked_equals_single_run(self):
        whole = self._propose(0, 10).json()['rows']
        a = self._propose(0, 4).json()
        b = self._propose(4, 6, state=a['state']).json()
        strip = lambda rows: [{k: v for k, v in r.items() if k != 'run_id'} for r in rows]
        self.assertEqual(strip(a['rows'] + b['rows']), strip(whole))

    def test_changed_params_start_a_new_run(self):
        r1 = self._propose(0, 3).json()
        r2 = self._propose(3, 3, params={'track_iou': 0.5}, state=r1['state']).json()
        self.assertNotEqual(r2['state']['run_id'], r1['state']['run_id'])
        self.assertEqual(min(r['box_id'] for r in r2['rows']), 1)

    def test_no_tracking_uses_detection_names(self):
        rows = self._propose(0, 2, params={'track': False}).json()['rows']
        self.assertTrue(all(r['label'].startswith('D') for r in rows))

    def test_count_is_clipped_to_the_video(self):
        r = self._propose(8, 30).json()
        self.assertEqual(r['frames'], [8, 9])

    def test_rejects_out_of_range_frame_and_bad_params(self):
        self.assertEqual(self._propose(10, 1).status_code, 400)
        self.assertEqual(self._propose(0, 1, params={'levels': 3}).status_code, 400)
        self.assertEqual(self._propose(0, 1, params={'min_area': 'abc'}).status_code, 400)

    def test_garbage_state_starts_a_new_run(self):
        r = self._propose(0, 2, state={'run_id': 1, 'tracks': 'x'}).json()
        self.assertTrue(r['success'], r)

    def test_user_not_on_task_is_refused(self):
        other = User.objects.create_user('other', password='x')
        self.client.force_login(other)
        self.assertEqual(self._propose(0, 1).status_code, 403)

    def test_sequence_outside_task_datasets_is_refused(self):
        other_ds = Dataset.objects.create(name='other')
        self.sequence.subject = Subject.objects.create(name='s2', dataset=other_ds)
        self.sequence.save()
        self.assertEqual(self._propose(0, 1).status_code, 404)

    def test_get_not_allowed(self):
        self.assertEqual(self.client.get('/bronchoscopy-bounding-box/propose/').status_code, 405)


class TrackerStateTests(TestCase):
    def test_state_roundtrip_through_json_continues_ids(self):
        ids = lumen._IdCounter()
        t = lumen.IoUTracker(0.3, 5, ids)
        self.assertEqual(t.update([(0, 0, 10, 10), (50, 50, 10, 10)]), [1, 2])
        tracks = lumen.IoUTracker.load_tracks(json.loads(json.dumps(t.dump())))
        t2 = lumen.IoUTracker(0.3, 5, lumen._IdCounter(ids.next_id), tracks)
        self.assertEqual(t2.update([(1, 0, 10, 10), (200, 200, 10, 10)]), [1, 3])
