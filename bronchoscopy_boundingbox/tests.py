import json
import os
import shutil
import time
from unittest import mock
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


def _frame(dx, extras=()):
    """Bright scope image (black margins) with two dark 'lumens', moved by dx,
    plus extra dark disks (cx, cy, r)."""
    img = np.zeros((H, W, 3), np.uint8)
    yy, xx = np.mgrid[:H, :W]
    img[:, 40:300] = (150 + xx[:, 40:300] % 7)[..., None]   # textured bright area
    img[(xx - (100 + dx)) ** 2 + (yy - 120) ** 2 < 30 ** 2] = 10
    img[(xx - (220 + dx)) ** 2 + (yy - 100) ** 2 < 25 ** 2] = 12
    for cx, cy, r in extras:
        img[(xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2] = 11
    return img


class SyntheticVideoTestCase(TestCase):
    """A 10-frame fake video in a temp folder, a task and an annotator on it.
    Subclasses set extras(f) to add dark disks to frame f."""

    @staticmethod
    def extras(f):
        return ()

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tmp = tempfile.mkdtemp()
        cls.frame_dir = os.path.join(cls.tmp, 'frames')
        os.makedirs(cls.frame_dir)
        for f in range(10):
            _write_mhd(os.path.join(cls.frame_dir, 'seq_%d.mhd' % f), _frame(f, cls.extras(f)))
        cls.cache_dir = os.path.join(cls.tmp, 'cache')
        cls.settings_override = override_settings(LUMEN_CACHE_DIR=cls.cache_dir)
        cls.settings_override.enable()

    @classmethod
    def tearDownClass(cls):
        cls.settings_override.disable()
        shutil.rmtree(cls.tmp)
        super().tearDownClass()

    def setUp(self):
        # Each test computes its suggestions (the scope-area cache may stay)
        if os.path.isdir(self.cache_dir):
            for name in os.listdir(self.cache_dir):
                if name.endswith('_rows'):
                    shutil.rmtree(os.path.join(self.cache_dir, name))
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

    @staticmethod
    def strip_run_id(rows):
        return [{k: v for k, v in r.items() if k != 'run_id'} for r in rows]


class LumenProposalTests(SyntheticVideoTestCase):
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
        self.assertEqual(self.strip_run_id(a['rows'] + b['rows']), self.strip_run_id(whole))

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
        # min_area 0 would make every speck of noise a box (minutes per request)
        self.assertEqual(self._propose(0, 1, params={'min_area': 0}).status_code, 400)
        # track_iou 0 would link boxes that do not overlap at all
        self.assertEqual(self._propose(0, 1, params={'track_iou': 0}).status_code, 400)
        self.assertEqual(self._propose(0, 1, params={'track_iou2': 0}).status_code, 400)

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

    def test_look_ahead_settings_are_limited(self):
        self.assertEqual(self._propose(0, 1, params={'max_lost': 11}).status_code, 400)
        self.assertEqual(self._propose(0, 1, params={'min_track_len': 6}).status_code, 400)
        self.assertEqual(self._propose(0, 1, params={'max_lost': 10, 'min_track_len': 5}).status_code, 200)

    def test_malformed_state_starts_a_new_run(self):
        good = self._propose(0, 2).json()['state']
        for bad in ({'run_id': 1}, {'prev': 'a'}, {'tracks': {'1': {'1': {'box': [1, 2], 'lost': 0}}}},
                    {'tracks': 'x'}, {'next_id': 'x'}):
            r = self._propose(2, 2, state=dict(good, **bad))
            self.assertEqual(r.status_code, 200, bad)
            self.assertTrue(r.json()['success'], bad)

    def test_unsupported_frame_is_skipped_with_reason(self):
        path = os.path.join(self.frame_dir, 'seq_5.mhd')
        original = open(path).read()
        try:
            with open(path, 'w') as f:
                f.write(original.replace('MET_UCHAR', 'MET_USHORT'))
            r = self._propose(5, 1)
            self.assertEqual(r.status_code, 200)            # skipped, not fatal
            self.assertEqual(r.json()['skipped'], [5])
            self.assertIn('unsupported pixel type MET_USHORT', r.json()['skip_message'])
        finally:
            with open(path, 'w') as f:
                f.write(original)

    def test_header_without_element_type_is_skipped(self):
        path = os.path.join(self.frame_dir, 'seq_3.mhd')
        original = open(path).read()
        self.addCleanup(lambda: open(path, 'w').write(original))
        with open(path, 'w') as f:
            f.write('\n'.join(l for l in original.splitlines() if not l.startswith('ElementType')))
        r = self._propose(3, 1)
        self.assertEqual(r.status_code, 200, r.content)
        self.assertEqual(r.json()['skipped'], [3])

    def test_damaged_frame_is_skipped_with_reason(self):
        raw = os.path.join(self.frame_dir, 'seq_6.zraw')
        original = open(raw, 'rb').read()
        try:
            with open(raw, 'wb') as f:
                f.write(b'not zlib')
            r = self._propose(6, 1)
            self.assertEqual(r.json()['skipped'], [6])
            self.assertIn('Could not read seq_6.mhd', r.json()['skip_message'])
        finally:
            with open(raw, 'wb') as f:
                f.write(original)

    def test_frame_of_other_size_is_skipped_with_reason(self):
        self._propose(0, 1)                       # builds the scope-mask cache
        path = os.path.join(self.frame_dir, 'seq_7.mhd')
        original = open(path).read()
        raw = os.path.join(self.frame_dir, 'seq_7.zraw')
        original_raw = open(raw, 'rb').read()
        try:
            small = np.zeros((100, 120, 3), np.uint8)
            data = zlib.compress(small.tobytes())
            with open(raw, 'wb') as f:
                f.write(data)
            with open(path, 'w') as f:
                f.write(original.replace('DimSize = %d %d' % (W, H), 'DimSize = 120 100'))
            r = self._propose(7, 1)
            self.assertEqual(r.json()['skipped'], [7])
            self.assertIn('different size', r.json()['skip_message'])
        finally:
            with open(path, 'w') as f:
                f.write(original)
            with open(raw, 'wb') as f:
                f.write(original_raw)

    def test_unexpected_error_returns_json(self):
        from unittest import mock
        with mock.patch.object(lumen, 'detect', side_effect=RuntimeError('boom')), \
                self.assertLogs('bronchoscopy_boundingbox.views', level='ERROR'):
            r = self._propose(0, 1)
        self.assertEqual(r.status_code, 500)
        self.assertEqual(r.json()['message'], 'Box suggestions failed: RuntimeError: boom')


class SingleFrameTrackTests(SyntheticVideoTestCase):
    """A lumen in frames 3-4 only (kept, also when a chunk ends at frame 3) and a
    blip in frame 6 only (dropped as noise)."""
    PAIR = (170, 200, 22)       # frames 3 and 4
    BLIP = (270, 200, 22)       # frame 6 only

    @staticmethod
    def extras(f):
        return ([SingleFrameTrackTests.PAIR] if f in (3, 4) else []) + \
               ([SingleFrameTrackTests.BLIP] if f == 6 else [])

    def _boxes_at(self, rows, cx, cy):
        return [r for r in rows if r['x'] <= cx <= r['x'] + r['width']
                and r['y'] <= cy <= r['y'] + r['height']]

    def test_single_frame_blip_is_dropped_two_frame_track_kept(self):
        r = self._propose(0, 10).json()
        self.assertEqual(self._boxes_at(r['rows'], 270, 200), [])
        pair = self._boxes_at(r['rows'], 170, 200)
        self.assertEqual(sorted(b['frame_nr'] for b in pair), [3, 4])
        self.assertEqual(r['dropped'], 1)

    def test_chunked_equals_single_run_with_short_tracks(self):
        # the look-ahead past each chunk must give the same result as one run
        whole = self._propose(0, 10).json()['rows']
        a = self._propose(0, 4).json()
        b = self._propose(4, 3, state=a['state']).json()
        c = self._propose(7, 3, state=b['state']).json()
        self.assertEqual(self.strip_run_id(a['rows'] + b['rows'] + c['rows']),
                         self.strip_run_id(whole))

    def test_track_split_over_chunk_boundary_is_kept(self):
        a = self._propose(0, 4).json()          # frames 0-3: pair only in frame 3 here
        self.assertEqual([b['frame_nr'] for b in self._boxes_at(a['rows'], 170, 200)], [3])
        b = self._propose(4, 6, state=a['state']).json()
        self.assertEqual([x['frame_nr'] for x in self._boxes_at(b['rows'], 170, 200)], [4])

    def test_min_track_len_one_keeps_blip(self):
        r = self._propose(0, 10, params={'min_track_len': 1}).json()
        self.assertEqual(len(self._boxes_at(r['rows'], 270, 200)), 1)

    def test_no_tracking_keeps_blip(self):
        r = self._propose(0, 10, params={'track': False}).json()
        self.assertEqual(len(self._boxes_at(r['rows'], 270, 200)), 1)


class MainBoxOnlyFilterTests(TestCase):
    """min_track_len filters main boxes (with their children), never nested tracks."""

    def _run(self, per_frame, **params):
        # per_frame: {frame: {"level1": [box], "level2": [(box, parent_index)]}}
        from unittest import mock
        frames = {f: '/x/%d.mhd' % f for f in per_frame}
        with mock.patch.object(lumen, 'read_mhd', side_effect=lambda path: path), \
                mock.patch.object(lumen, 'detect',
                                  side_effect=lambda img, roi, mask, p: per_frame[int(img[3:-4])]), \
                mock.patch.object(lumen.os.path, 'isfile', return_value=True), \
                mock.patch.object(lumen.Runner, '_check_size'):
            r = lumen.Runner(frames, None, None, lumen.Params(levels=2, **params))
            return r.process(sorted(frames)), r

    BIG, OTHER = (0, 0, 100, 100), (300, 300, 100, 100)
    KID1, KID2 = (10, 10, 20, 20), (60, 60, 20, 20)

    def test_nested_tracks_are_not_filtered(self):
        # parent in both frames; its two children jump around, so each child
        # track is one frame long, but they are kept
        per_frame = {0: {"level1": [self.BIG], "level2": [(self.KID1, 0), (self.KID2, 0)]},
                     1: {"level1": [self.BIG], "level2": [((40, 10, 20, 20), 0), ((10, 60, 20, 20), 0)]}}
        rows, r = self._run(per_frame)
        self.assertEqual(sum(x['level'] == 2 for x in rows), 4)
        self.assertEqual(r.dropped, 0)

    def test_short_main_track_dropped_with_children(self):
        per_frame = {0: {"level1": [self.BIG], "level2": []},
                     1: {"level1": [self.BIG, self.OTHER], "level2": [((310, 310, 20, 20), 1), ((360, 360, 20, 20), 1)]},
                     2: {"level1": [self.BIG], "level2": []}}
        rows, r = self._run(per_frame)
        self.assertEqual([(x['frame_nr'], x['level']) for x in rows], [(0, 1), (1, 1), (2, 1)])
        self.assertEqual(r.dropped, 3)

    def test_min_track_len_one_keeps_everything(self):
        per_frame = {0: {"level1": [self.BIG], "level2": []},
                     1: {"level1": [self.BIG, self.OTHER], "level2": []}}
        rows, r = self._run(per_frame, min_track_len=1)
        self.assertEqual(len(rows), 3)


class MinChildrenTests(TestCase):
    """A parent's level-2 children are only kept if there are at least min_children."""

    def _detect(self, n_kids, **params):
        from unittest import mock
        kids = [(10 + 20 * i, 10, 15, 15) for i in range(n_kids)]
        level1 = {"gray": None, "labels": np.zeros((2, 2)), "kept": [1], "boxes": [(0, 0, 100, 100)]}
        with mock.patch.object(lumen, 'detect_level1', return_value=level1), \
                mock.patch.object(lumen, 'split_blob', return_value=kids):
            return lumen.detect(None, None, None, lumen.Params(levels=2, **params))

    def test_single_child_is_dropped(self):
        self.assertEqual(self._detect(1)["level2"], [])

    def test_two_children_are_kept(self):
        self.assertEqual(len(self._detect(2)["level2"]), 2)

    def test_min_children_one_keeps_single_child(self):
        self.assertEqual(len(self._detect(1, min_children=1)["level2"]), 1)


class TrackerStateTests(TestCase):
    def test_state_roundtrip_through_json_continues_ids(self):
        ids = lumen._IdCounter()
        t = lumen.IoUTracker(0.3, 5, ids)
        self.assertEqual(t.update([(0, 0, 10, 10), (50, 50, 10, 10)]), [1, 2])
        tracks = lumen.IoUTracker.load_tracks(json.loads(json.dumps(t.dump())))
        t2 = lumen.IoUTracker(0.3, 5, lumen._IdCounter(ids.next_id), tracks)
        self.assertEqual(t2.update([(1, 0, 10, 10), (200, 200, 10, 10)]), [1, 3])


class HandoffTests(TestCase):
    """A nested box that becomes a main box keeps its track (Params.handoff)."""

    def _runner(self, nested):
        r = lumen.Runner({}, None, None, lumen.Params(levels=2))
        r._ids.next_id = 100
        r.trackers[2].tracks = {t: {"box": b, "lost": 0, "n": 3} for t, b in nested.items()}
        return r

    def test_each_child_continues_as_main_box(self):
        r = self._runner({7: (100, 100, 50, 50), 8: (300, 100, 50, 50)})
        ids = r.trackers[1].update([(90, 90, 80, 80), (290, 90, 80, 80)], r._adopt_nested)
        self.assertEqual(ids, [7, 8])
        self.assertEqual(r.trackers[1].tracks[7]["n"], 4)
        self.assertEqual(r.trackers[2].tracks, {})
        # the flag survives matching and the JSON round trip
        r.trackers[1].update([(92, 90, 80, 80), (290, 92, 80, 80)])
        tracks = lumen.IoUTracker.load_tracks(json.loads(json.dumps(r.trackers[1].dump())))
        self.assertEqual(tracks[7].get("adopted"), 1)

    def test_box_covering_both_children_is_new(self):
        r = self._runner({7: (100, 100, 50, 50), 8: (200, 100, 50, 50)})
        ids = r.trackers[1].update([(80, 80, 200, 100)], r._adopt_nested)
        self.assertEqual(ids, [100])
        self.assertEqual(set(r.trackers[2].tracks), {7, 8})

    def test_child_mostly_outside_is_not_taken(self):
        r = self._runner({7: (100, 100, 50, 50)})
        ids = r.trackers[1].update([(140, 100, 80, 80)], r._adopt_nested)  # 20% inside
        self.assertEqual(ids, [100])


class CoverMatchTests(TestCase):
    """Main-box tracks continue when the box grows/shrinks too much for IoU."""

    def _tracker(self, cover):
        t = lumen.IoUTracker(0.45, 1, lumen._IdCounter(), cover_thr=cover, max_ratio=3.0)
        t.update([(100, 100, 400, 400)])
        return t

    def test_shrinking_box_keeps_track(self):
        # IoU 0.39, but the new box (2.6x smaller) lies completely inside the old one
        self.assertEqual(self._tracker(0.6).update([(150, 150, 250, 250)]), [1])

    def test_off_gives_new_track(self):
        self.assertEqual(self._tracker(0.0).update([(150, 150, 250, 250)]), [2])

    def test_much_smaller_box_is_not_taken(self):
        # inside, but 16x smaller: more likely a different opening
        self.assertEqual(self._tracker(0.6).update([(200, 200, 100, 100)]), [2])

    def test_params_defaults(self):
        p = lumen.Params()
        self.assertEqual((p.track_cover, p.track_max_ratio), (0.6, 3.0))


class BadFrameAfterChunkTests(SyntheticVideoTestCase):
    """A missing frame just past the requested frames must not fail the request
    (the look-ahead for min_track_len stops there instead)."""

    def setUp(self):
        super().setUp()
        self._propose(0, 1)     # scope area estimated and cached while frame 5 exists
        path = os.path.join(self.frame_dir, 'seq_5.mhd')
        moved = path + '.away'
        os.rename(path, moved)
        self.addCleanup(os.rename, moved, path)

    def test_lookahead_stops_at_missing_frame(self):
        r = self._propose(0, 4)
        self.assertEqual(r.status_code, 200, r.content)
        self.assertEqual(r.json()['frames'], [0, 3])

    def test_missing_frame_inside_chunk_is_skipped(self):
        # one bad frame must not stop the whole-video load
        r = self._propose(4, 3, params={'min_track_len': 1})
        self.assertEqual(r.status_code, 200, r.content)
        d = r.json()
        self.assertEqual(d['frames'], [4, 6])
        self.assertEqual(d['skipped'], [5])
        self.assertEqual(sorted({row['frame_nr'] for row in d['rows']}), [4, 6])
        # tracks start again after the gap
        ids4 = {row['box_id'] for row in d['rows'] if row['frame_nr'] == 4}
        ids6 = {row['box_id'] for row in d['rows'] if row['frame_nr'] == 6}
        self.assertFalse(ids4 & ids6)
        # kept (else a bad frame in the first chunk means a new run on every
        # open), but Reload with unchanged settings (refresh) computes again
        self.assertTrue(self._propose(4, 3, params={'min_track_len': 1}).json().get('cached'))
        self.assertNotIn('cached', self._propose(4, 3, params={'min_track_len': 1}, refresh=1).json())


class DamagedScopeCacheTests(SyntheticVideoTestCase):
    """A damaged cached scope area is estimated again instead of failing every
    request for the video until someone deletes the file."""

    def test_damaged_cache_is_rebuilt(self):
        self.assertEqual(self._propose(0, 1).status_code, 200)
        caches = [f for f in os.listdir(self.cache_dir) if f.endswith('_scope.npz')]
        self.assertEqual(len(caches), 1)
        path = os.path.join(self.cache_dir, caches[0])
        with open(path, 'wb') as f:
            f.write(b'not an npz file')
        r = self._propose(0, 2)      # not the cached chunk: the scope area is needed
        self.assertEqual(r.status_code, 200, r.content)
        with np.load(path) as z:
            self.assertIn('roi', z.files)


class ProposalCacheTests(SyntheticVideoTestCase):
    """Suggestion chunks are kept on disk, so opening a video again is fast."""

    def test_same_request_is_served_from_the_cache(self):
        r1 = self._propose(0, 5).json()
        self.assertNotIn('cached', r1)
        with mock.patch.object(lumen, 'detect', side_effect=AssertionError('not cached')):
            r2 = self._propose(0, 5).json()
            self.assertTrue(r2.get('cached'), r2)
            self.assertEqual(r2['rows'], r1['rows'])            # same run_id too
            self.assertEqual(r2['state'], r1['state'])
            # the next chunk, continuing from that state, is computed once ...
        r3 = self._propose(5, 5, state=r1['state']).json()
        self.assertNotIn('cached', r3)
        with mock.patch.object(lumen, 'detect', side_effect=AssertionError('not cached')):
            r4 = self._propose(5, 5, state=r2['state']).json()   # ... then cached
        self.assertEqual(r4['rows'], r3['rows'])

    def test_other_params_or_frames_are_computed(self):
        self._propose(0, 5)
        self.assertNotIn('cached', self._propose(0, 4).json())
        self.assertNotIn('cached', self._propose(0, 5, params={'levels': 2}).json())

    def test_damaged_cache_file_is_computed_again(self):
        self._propose(0, 5)
        folder = [d for d in os.listdir(self.cache_dir) if d.endswith('_rows')][0]
        for name in os.listdir(os.path.join(self.cache_dir, folder)):
            with open(os.path.join(self.cache_dir, folder, name), 'w') as f:
                f.write('{broken')
        r = self._propose(0, 5)
        self.assertEqual(r.status_code, 200, r.content)
        self.assertNotIn('cached', r.json())


class CacheRobustnessTests(SyntheticVideoTestCase):
    def _clear_scope_cache(self):
        for n in os.listdir(self.cache_dir) if os.path.isdir(self.cache_dir) else []:
            if n.endswith('_scope.npz'):
                os.remove(os.path.join(self.cache_dir, n))

    def test_replaced_frame_is_computed_again(self):
        self._propose(0, 3)
        self.assertTrue(self._propose(0, 3).json().get('cached'))
        path = os.path.join(self.frame_dir, 'seq_1.zraw')
        st = os.stat(path)
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
        self.addCleanup(os.utime, path, ns=(st.st_atime_ns, st.st_mtime_ns))
        self.assertNotIn('cached', self._propose(0, 3).json())

    def test_no_write_access_to_cache_still_works(self):
        self._clear_scope_cache()
        with mock.patch.object(lumen.np, 'savez_compressed', side_effect=PermissionError('no')), \
                mock.patch('bronchoscopy_boundingbox.views.json.dump', side_effect=PermissionError('no')):
            r = self._propose(0, 2)
        self.assertEqual(r.status_code, 200, r.content)
        self.assertTrue(r.json()['rows'])

    def test_damaged_sample_frame_does_not_stop_the_video(self):
        self._clear_scope_cache()
        raw = os.path.join(self.frame_dir, 'seq_0.zraw')      # always a sample
        original = open(raw, 'rb').read()
        self.addCleanup(lambda: open(raw, 'wb').write(original))
        with open(raw, 'wb') as f:
            f.write(b'not zlib')
        r = self._propose(1, 3)
        self.assertEqual(r.status_code, 200, r.content)
        self.assertTrue(r.json()['rows'])


class CacheKeyAndPruneTests(SyntheticVideoTestCase):
    def _touch(self, name):
        path = os.path.join(self.frame_dir, name)
        st = os.stat(path)
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
        self.addCleanup(os.utime, path, ns=(st.st_atime_ns, st.st_mtime_ns))

    def test_frame_read_past_the_chunk_is_part_of_the_key(self):
        params = {'min_track_len': 2, 'max_lost': 1}    # reads 2 frames past the chunk
        self._propose(0, 3, params=params)
        self.assertTrue(self._propose(0, 3, params=params).json().get('cached'))
        self._touch('seq_4.zraw')                        # frame 4: look-ahead only
        self.assertNotIn('cached', self._propose(0, 3, params=params).json())

    def test_scope_sample_frame_is_part_of_the_key(self):
        self._propose(0, 3)
        self._touch('seq_9.zraw')                        # last frame: always a sample
        self.assertNotIn('cached', self._propose(0, 3).json())

    def test_unused_stored_suggestions_are_pruned(self):
        from bronchoscopy_boundingbox import views
        self._propose(0, 3)
        folder = [os.path.join(self.cache_dir, d) for d in os.listdir(self.cache_dir)
                  if d.endswith('_rows')][0]
        old = os.path.join(folder, 'old.json')
        with open(old, 'w') as f:
            f.write('{}')
        long_ago = time.time() - views.PROPOSAL_CACHE_MAX_AGE - 3600
        os.utime(old, (long_ago, long_ago))
        with mock.patch.object(views, '_last_prune', 0.0):
            self._propose(3, 3)
        self.assertFalse(os.path.exists(old))
        self.assertTrue(os.listdir(folder))              # the ones in use stay


class CacheFilesTests(SyntheticVideoTestCase):
    def test_data_file_named_in_header_is_part_of_the_signature(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        mhd = os.path.join(tmp, 'frame_1.mhd')
        data = os.path.join(tmp, 'pixels_0001.zraw')       # not the header's name
        with open(mhd, 'w') as f:
            f.write('NDims = 2\nElementDataFile = pixels_0001.zraw\n')
        with open(data, 'wb') as f:
            f.write(b'x')
        self.assertEqual(lumen._data_file(mhd), data)
        before = lumen.files_signature([mhd])
        st = os.stat(data)
        os.utime(data, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
        self.assertNotEqual(lumen.files_signature([mhd]), before)

    def test_unused_scope_files_are_pruned_and_used_ones_kept(self):
        from bronchoscopy_boundingbox import views
        self._propose(0, 2)
        old = os.path.join(self.cache_dir, 'seq0_old_scope.npz')
        with open(old, 'wb') as f:
            f.write(b'x')
        long_ago = time.time() - views.PROPOSAL_CACHE_MAX_AGE - 3600
        os.utime(old, (long_ago, long_ago))
        used = [os.path.join(self.cache_dir, n) for n in os.listdir(self.cache_dir)
                if n.endswith('_scope.npz') and n != 'seq0_old_scope.npz']
        for u in used:
            os.utime(u, (long_ago, long_ago))
        with mock.patch.object(views, '_last_prune', 0.0):
            self._propose(2, 2)          # reads the scope file (refreshes it), then prunes
        self.assertFalse(os.path.exists(old))
        self.assertTrue(all(os.path.exists(u) for u in used))


class ScopeAndEmptyVideoTests(SyntheticVideoTestCase):
    def test_sample_frames_are_read_once(self):
        for n in os.listdir(self.cache_dir) if os.path.isdir(self.cache_dir) else []:
            if n.endswith('_scope.npz'):
                os.remove(os.path.join(self.cache_dir, n))
        frames = {f: os.path.join(self.frame_dir, 'seq_%d.mhd' % f) for f in range(10)}
        real = lumen.read_mhd
        with mock.patch.object(lumen, 'read_mhd', side_effect=real) as m:
            lumen.roi_and_mask(frames, self.cache_dir, 'once')
        self.assertEqual(m.call_count, len(lumen.scope_sample_paths(frames)))

    def test_video_without_frames_gives_clear_message(self):
        self.sequence.nr_of_frames = 0
        self.sequence.save()
        r = self._propose(0, 1)
        self.assertEqual(r.status_code, 400)
        self.assertIn('no frames', r.json()['message'])


class FailedCacheWriteTests(SyntheticVideoTestCase):
    def test_no_temp_files_left_when_writing_fails(self):
        for n in os.listdir(self.cache_dir) if os.path.isdir(self.cache_dir) else []:
            if n.endswith('_scope.npz'):
                os.remove(os.path.join(self.cache_dir, n))

        def fail_after_partial_write(*args, **kwargs):
            raise OSError('disk full')
        with mock.patch.object(lumen.np, 'savez_compressed', side_effect=fail_after_partial_write), \
                mock.patch('bronchoscopy_boundingbox.views.json.dump', side_effect=fail_after_partial_write):
            r = self._propose(0, 2)
        self.assertEqual(r.status_code, 200, r.content)
        leftovers = [os.path.join(d, n) for d, _, names in os.walk(self.cache_dir)
                     for n in names if '.tmp' in n]
        self.assertEqual(leftovers, [])

    def test_frame_reader_code_is_part_of_the_key(self):
        from bronchoscopy_boundingbox import views
        self._propose(0, 2)
        self.assertTrue(self._propose(0, 2).json().get('cached'))
        with mock.patch.object(views, '_lumen_code_hash', 'other-code'):
            self.assertNotIn('cached', self._propose(0, 2).json())
