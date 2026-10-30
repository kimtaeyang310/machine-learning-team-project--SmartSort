"""함수 분리 후에도 외부 결과와 데이터 분할·확인 규칙이 유지되는지 검사합니다."""
import contextlib
import copy
import csv
import io
import json
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from manual_mask import polygon_mask
from scripts.register_mandarin_diversity import build_registration_rows
from src.assistant import tools
# 요청 분기 테스트에서는 Ollama 서버나 선택 의존성이 필요하지 않습니다.
with patch.dict(sys.modules, {'ollama': Mock()}):
    from src.assistant.ollama_client import OllamaAssistant
from src.predict import save_prediction_artifacts
from src.tracking import FruitTracker, TemporalVote
from src.video_tracking import _classify_tracked_frame, summarize_tracks


class PredictionOutputTests(unittest.TestCase):
    def test_export_and_recent_summary_agree_with_new_analysis(self):
        rows = [
            {'status': 'ok', 'fruit': 'banana', 'ripeness': 'ripe', 'total_ms': 10,
             'image_path': 'banana.jpg', 'probabilities': {'ripe': .8},
             'agent_decision': {'action': 'auto_classify'}},
            {'status': 'unknown', 'candidate_fruit': 'mandarin', 'total_ms': 20,
             'candidate_class': 'mandarin_ripe',
             'image_path': 'unknown.jpg', 'fruit_identity': {'candidate_fruit': 'mandarin',
                                                          'model_sha256': 'test'}},
            {'status': 'failed', 'total_ms': 70, 'error': 'unreadable'},
        ]
        original = copy.deepcopy(rows)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # 인위적인 모델 파일을 사용하여 실제 모델·학습 자료에 쓰지 않습니다.
            model = root / 'model.pt'
            model.write_bytes(b'test model')
            saved = save_prediction_artifacts(rows, root / 'results.csv', model_path=model,
                                              source_folder=root)
            with saved[0].open(encoding='utf-8-sig', newline='') as stream:
                exported = list(csv.DictReader(stream))
            self.assertEqual(json.loads(exported[0]['probabilities_json']), {'ripe': .8})
            self.assertEqual(exported[2]['error'], 'unreadable')
            manifest = json.loads(saved[1].read_text(encoding='utf-8'))
            self.assertEqual((manifest['result_count'], manifest['unknown_count'], manifest['failure_count']),
                             (3, 1, 1))
            timing = {'mean_ms': 100 / 3, 'p95_ms': 65., 'fps': 20., 'processing_ms': 100.}
            for fruit in (None, 'banana', 'mandarin', 'tomato'):
                with self.subTest(fruit=fruit), \
                     patch.object(tools, '_image_paths', return_value=[root / 'photo.jpg']), \
                     patch.object(tools, 'predict_folder', return_value=(rows, timing)), \
                     patch.object(tools, 'save_prediction_artifacts', return_value=saved), \
                     patch.object(tools, '_new_csv_path', return_value=saved[0]), \
                     patch.object(tools, '_read_latest_rows', return_value=(saved[0], exported)):
                    live = tools.analyze_folder(fruit, selected_folder=root)
                    recent = tools.summarize_recent(fruit)
                    self.assertTrue(live['ok'])
                    self.assertTrue(recent['ok'])
                    for key in ('ripeness_counts', 'unknown_count', 'unknown_candidate_count',
                                'failure_count', 'retake_count', 'image_count'):
                        self.assertEqual(live[key], recent[key], key)
                    self.assertEqual(recent['performance']['fps'], 20.)
        self.assertEqual(rows, original)

    def test_saved_timing_skips_bad_values_but_counts_failures_in_time(self):
        result = tools._summarize_saved_timing([
            {'status': 'ok', 'total_ms': '10'},
            {'status': 'unknown', 'total_ms': '10'},
            {'status': 'failed', 'total_ms': '80'},
            {'status': 'failed', 'total_ms': 'bad'},
            {'status': 'failed', 'total_ms': '-5'},
        ])
        self.assertEqual(result['processing_ms'], 100)
        self.assertEqual(result['fps'], 20)

    def test_cli_uses_explicit_arguments_and_preserves_error_exit(self):
        from src import predict
        with patch.object(predict, 'predict_folder', side_effect=ValueError('invalid folder')), \
             contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as error:
            predict.main(['missing', '--results-csv', 'unused.csv'])
        self.assertEqual(error.exception.code, 1)


class AssistantScopeTests(unittest.TestCase):
    def intent(self, fruits, scope='specific'):
        return {'action': 'analyze_folder', 'fruit_mentions': fruits, 'scope': scope, 'answer': ''}

    def test_unsupported_fruit_and_partial_multiple_scope_never_execute(self):
        for fruits in (['lemon'], ['banana', 'tomato']):
            with self.subTest(fruits=fruits), \
                 patch('src.assistant.ollama_client._classify_intent', return_value=self.intent(fruits)):
                assistant = OllamaAssistant()
                assistant._run_task = Mock()
                answer = assistant.ask('분석해줘', selected_folder='photos')
                assistant._run_task.assert_not_called()
                self.assertTrue(answer)

    def test_explicit_all_waits_for_folder_confirmation(self):
        assistant = OllamaAssistant()
        assistant._run_task = Mock(return_value='done')
        with patch('src.assistant.ollama_client._classify_intent', return_value=self.intent([], 'all')):
            assistant.ask('전체 과일 분석해줘')
        assistant._run_task.assert_not_called()
        self.assertEqual(assistant.pending_intent, ('analyze_folder', None, True))
        self.assertEqual(assistant.ask('네', default_folder_confirmed=True), 'done')
        assistant._run_task.assert_called_once_with('analyze_folder', None, selected_folder=None, all_fruits=True)
        self.assertIsNone(assistant.pending_intent)

    def test_model_all_scope_alone_does_not_authorize_whole_folder(self):
        assistant = OllamaAssistant()
        assistant._run_task = Mock()
        with patch('src.assistant.ollama_client._classify_intent', return_value=self.intent([], 'all')):
            assistant.ask('분석해줘', selected_folder='photos')
        assistant._run_task.assert_not_called()


class ImageAndGroupingTests(unittest.TestCase):
    def test_mask_rescales_and_clips_without_mutating_clicks(self):
        points = [[-1, -1], [50, -1], [50, 40], [-1, 40]]
        original = copy.deepcopy(points)
        mask = polygon_mask(points, 100, 80, 50, 40)
        self.assertEqual(mask.shape, (80, 100))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertTrue(np.all(mask == 255))
        self.assertEqual(points, original)

    def test_registration_keeps_source_families_together_and_is_repeatable(self):
        groups = [[f'source_{i}', f'variant_{i}'] for i in range(20)]
        original = copy.deepcopy(groups)
        first = build_registration_rows(groups)
        self.assertEqual(first, build_registration_rows(groups))
        self.assertEqual(groups, original)
        mapping, rows, records = first
        for a, b in groups:
            self.assertEqual(mapping[a], mapping[b])
        phases = [record['phase'] for record in records]
        self.assertEqual({phase: phases.count(phase) for phase in set(phases)},
                         {'train': 12, 'calibration': 3, 'selection': 3, 'test': 2})
        for row in rows:
            self.assertEqual(row['split'] == 'test', row['phase'] == 'test')


class TrackingPipelineTests(unittest.TestCase):
    def test_comparison_inference_does_not_make_edge_observations_vote(self):
        frame = np.full((200, 300, 3), 255, np.uint8)
        frame[60:100, :40] = (0, 0, 200)
        classifier = Mock()
        classifier.predict_frame.return_value = {'status': 'unknown', 'fruit': 'unknown', 'ripeness': None}
        tracker, vote = FruitTracker(), TemporalVote(30)
        for index in range(5):
            detections, calls = _classify_tracked_frame(frame, index, classifier, tracker, vote, 100, True)
            self.assertEqual(calls, 1)
            self.assertFalse(detections[0]['used_for_vote'])
            self.assertEqual(detections[0]['prediction']['status'], 'pending')

    def test_blank_frame_and_empty_track_summary(self):
        frame = np.full((100, 100, 3), 255, np.uint8)
        classifier = Mock()
        rows, calls = _classify_tracked_frame(frame, 0, classifier, FruitTracker(), TemporalVote(30), 100, False)
        self.assertEqual((rows, calls), ([], 0))
        classifier.predict_frame.assert_not_called()
        summaries, metrics = summarize_tracks({}, 30)
        self.assertEqual(summaries, [])
        self.assertEqual(metrics['track_count'], 0)
        self.assertIsNone(metrics['raw_label_changes'])


if __name__ == '__main__':
    unittest.main()
