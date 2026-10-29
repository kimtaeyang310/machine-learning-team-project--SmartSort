"""Video regressions: blank frames, foreground boxes and ndarray inference."""
import unittest
from unittest.mock import Mock
import numpy as np
from src.video import detect_regions, crop_region
from src.joint import JointClassifier


class VideoTests(unittest.TestCase):
    def test_square_padding_preserves_crop_pixels_and_background(self):
        frame = np.full((80,140,3),150,np.uint8)
        frame[30:50,30:100] = (0,0,220)
        box = (30,30,70,20)
        original = crop_region(frame,box)
        square = crop_region(frame,box,square=True)
        self.assertEqual(square.shape[0],square.shape[1])
        top = (square.shape[0]-original.shape[0])//2
        np.testing.assert_array_equal(square[top:top+original.shape[0]],original)
        np.testing.assert_array_equal(square[0,0],np.array([150,150,150]))
        white = crop_region(frame,box,square=True,pad_color='white')
        np.testing.assert_array_equal(white[0,0],np.array([255,255,255]))
        with self.assertRaises(ValueError):
            crop_region(frame,box,padding=-.1)

    def test_blank_frame_does_not_reuse_an_old_box(self):
        fruit = np.full((200,300,3),255,np.uint8)
        fruit[60:140,100:180] = (0,0,220)
        self.assertEqual(len(detect_regions(fruit)),1)
        self.assertEqual(detect_regions(np.full_like(fruit,255)),[])

    def test_separate_fruits_and_crop_clipping(self):
        frame = np.full((200,300,3),255,np.uint8)
        frame[20:80,0:50] = (0,180,0)
        frame[100:170,200:270] = (0,0,200)
        boxes = detect_regions(frame)
        self.assertEqual(len(boxes),2)
        crop = crop_region(frame,boxes[0])
        self.assertGreater(crop.size,0)
        self.assertLessEqual(crop.shape[0],200)

    def test_frame_passes_to_model_without_becoming_a_path(self):
        classifier = object.__new__(JointClassifier)
        classifier.sha = 'test'
        classifier.model_name = 'test'
        classifier.engine = Mock()
        classifier.engine.predict.return_value = {
            'status':'unknown','candidate_fruit':'banana','candidate_ripeness':'ripe',
            'candidate_class':'banana_ripe','score':.5,'threshold':.2,'method':'embedding',
            'probabilities':{'banana_ripe':.5}}
        frame = np.zeros((40,40,3),np.uint8)
        result = classifier.predict_frame(frame)
        self.assertIs(classifier.engine.predict.call_args.args[0],frame)
        self.assertEqual(result['image_path'],'')
        self.assertIsNone(result['ripeness'])
        self.assertEqual(result['status'],'unknown')
        with self.assertRaises(ValueError):
            classifier.predict_frame(np.zeros((0,0,3),np.uint8))


if __name__ == '__main__':
    unittest.main()
