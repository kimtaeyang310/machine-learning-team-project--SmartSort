import tempfile
import contextlib
import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from src.yolo_compare import materialize, probabilities, fit_yolo


class YoloComparisonTests(unittest.TestCase):
    def test_unified_cli_enables_yolo_by_default(self):
        from src.train import main
        with patch('src.train.train') as train:
            main([])
            self.assertTrue(train.call_args.kwargs['include_yolo'])
            main(['--without-yolo'])
            self.assertFalse(train.call_args.kwargs['include_yolo'])

    def test_yolo_receives_same_folds_and_can_win(self):
        from sklearn.dummy import DummyClassifier
        from src.train import fit_fruit_models
        from src.model_bundle import FRUITS
        samples = [dict(fruit=fruit, ripeness_label=label, group_id=f'{fruit}_{label}_{g}')
                   for fruit in FRUITS for label in range(3) for g in range(6)]
        folds_seen = []
        def cv(model, X, y, **kwargs):
            folds_seen.append(kwargs['cv'])
            return np.array([0.2, 0.3, 0.4])
        candidate = SimpleNamespace(input_kind='image')
        def compare(samples, splits, root, config):
            self.assertIs(splits, folds_seen[-1])
            self.assertEqual(len({s['fruit'] for s in samples}), 1)
            return dict(model='yolov8n_cls', cv_macro_f1_mean=0.9,
                        cv_macro_f1_std=0.0), candidate
        with patch('src.train.get_models', return_value={'dummy': DummyClassifier()}), \
             patch('src.train.cross_val_score', side_effect=cv), \
             patch('src.yolo_compare.compare_fruit', side_effect=compare), \
             contextlib.redirect_stdout(io.StringIO()):
            models, tables = fit_fruit_models(pd.DataFrame({'x': range(len(samples))}), samples,
                                              yolo_config={}, yolo_root=Path('unused'))
        self.assertTrue(all(entry['model'] is candidate for entry in models.values()))
        self.assertTrue(all(len(table) == 2 for table in tables.values()))

    def test_prediction_image_model_skips_feature_extraction(self):
        from src.predict import predict_loaded
        model = SimpleNamespace(input_kind='image', classes_=np.array([0,1,2]),
                                predict_proba=Mock(return_value=np.array([[0.1,0.2,0.7]])))
        with patch('src.predict.extract_image_features') as extract:
            result = predict_loaded(Path('tomato.jpg'),
                                    {'label_mapping': {'unripe':0,'ripe':1,'overripe':2}}, model)
        extract.assert_not_called()
        self.assertEqual(result['ripeness'], 'overripe')
        self.assertEqual(model.predict_proba.call_args.args[0], [Path('tomato.jpg')])

    def test_evaluation_routes_mixed_image_and_feature_models(self):
        from src.evaluate import evaluate
        image_model = SimpleNamespace(input_kind='image', predict=Mock(return_value=np.array([2])))
        feature_model = SimpleNamespace(predict=Mock(return_value=np.array([1,0])))
        bundle = {'models': {'tomato': {'model': image_model, 'model_name': 'yolov8n_cls'},
                             'banana': {'model': feature_model, 'model_name': 'dummy'}},
                  'label_mapping': {'unripe':0,'ripe':1,'overripe':2}}
        frame = pd.DataFrame({'fruit':['banana','tomato','banana'], 'image_path':['a.jpg','b.jpg','c.jpg'],
                              'group_id':['a','b','c'], 'ripeness_label':[1,2,0]})
        with tempfile.TemporaryDirectory() as directory, \
             patch('src.evaluate.load_inputs', return_value=(bundle,frame)), \
             patch('src.evaluate.extract_test_features', return_value=pd.DataFrame({'x':[10,20]})) as extract, \
             contextlib.redirect_stdout(io.StringIO()):
            summary, _ = evaluate(output_root=directory)
        self.assertEqual(extract.call_args.args[0].image_path.tolist(), ['a.jpg','c.jpg'])
        self.assertEqual(feature_model.predict.call_args.args[0].x.tolist(), [10,20])
        self.assertEqual(image_model.predict.call_args.args[0][0].name, 'b.jpg')
        self.assertTrue(summary.accuracy.eq(1).all())

    def test_class_names_are_mapped_not_assumed_in_project_order(self):
        tensor = Mock()
        tensor.cpu.return_value.numpy.return_value = np.array([0.7, 0.2, 0.1])
        result = SimpleNamespace(names={0: 'overripe', 1: 'ripe', 2: 'unripe'},
                                 probs=SimpleNamespace(data=tensor))
        model = Mock()
        model.predict.return_value = [result]
        values = probabilities(model, [Path('fruit.jpg')], {'imgsz': 224, 'device': 'cpu'})
        np.testing.assert_allclose(values, [[0.1, 0.2, 0.7]])

    def test_pretrained_imagenet_classes_are_rejected(self):
        model = Mock()
        model.predict.return_value = [SimpleNamespace(names={0: 'apple'})]
        with self.assertRaises(ValueError):
            probabilities(model, ['fruit.jpg'], {'imgsz': 224, 'device': 'cpu'})

    def test_explicit_split_and_same_basename_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source.jpg'
            source.write_bytes(b'test')
            train = pd.DataFrame([{'image_path': 'a/x.jpg', 'ripeness': 'ripe'},
                                  {'image_path': 'b/x.jpg', 'ripeness': 'ripe'}])
            valid = pd.DataFrame([{'image_path': 'c/x.jpg', 'ripeness': 'overripe'}])
            with patch('src.yolo_compare.image_path', return_value=source):
                destination = materialize(train, valid, root / 'dataset')
                self.assertEqual(len(list((destination / 'train' / 'ripe').glob('*.jpg'))), 2)
                self.assertEqual(len(list((destination / 'val' / 'overripe').glob('*.jpg'))), 1)
                self.assertFalse((destination / 'test').exists())
                self.assertTrue((destination / 'val' / 'unripe').is_dir())
                with self.assertRaises(FileExistsError):
                    materialize(train, valid, destination)

    def test_fixed_epoch_last_checkpoint_is_used(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / 'weights' / 'last.pt'
            checkpoint.parent.mkdir()
            checkpoint.touch()
            model = Mock()
            model.trainer.save_dir = root
            factory = Mock(return_value=model)
            config = dict(weights='yolov8n-cls.pt', epochs=2, imgsz=224, batch=2, device='cpu')
            with patch('src.yolo_compare.yolo_class', return_value=factory):
                self.assertEqual(fit_yolo(root / 'data', root / 'training', config), checkpoint)
            self.assertEqual(model.train.call_args.kwargs['patience'], 0)
            self.assertFalse(model.train.call_args.kwargs['val'])
            factory.assert_called_once_with('yolov8n-cls.pt')


if __name__ == '__main__':
    unittest.main()
