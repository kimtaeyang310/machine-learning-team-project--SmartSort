"""New-source registration and grouping tests."""
import json
import tempfile
import unittest
from pathlib import Path
from src.joint_labels import merge_registered_samples,refresh_labels
from scripts.register_mandarin_diversity import canonical_name


class DiversityRegistrationTests(unittest.TestCase):
    def test_augmentation_variants_share_the_same_source_key(self):
        source='Screen Shot 2018-06-12 at 11.50.14 PM.png'
        for prefix in ('','rotated_by_15_','vertical_flip_','translation_','saltandpepper_'):
            self.assertEqual(canonical_name(prefix+source),source)

    def test_registry_is_idempotent_and_quarantine_is_not_added(self):
        sample={'path':'old.png','sha256':'old','fruit':'mandarin','group':'mandarin:M01','source_group':'M01_date'}
        added={'path':'new.png','sha256':'new','fruit':'mandarin','group':'mandarin:N01','source_group':'N01_date','active':True}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'registry.json'
            path.write_text(json.dumps([added,{**added,'path':'quarantine.png','active':False}]),encoding='utf-8')
            merged=merge_registered_samples([sample],path)
            self.assertEqual(len(merged),2)
            self.assertEqual(merge_registered_samples(merged,path),merged)
            path.write_text(json.dumps([{**added,'sha256':'conflict'}]),encoding='utf-8')
            with self.assertRaises(ValueError):merge_registered_samples(merged,path)
            self.assertEqual(sample['path'],'old.png')

    def test_explicit_new_calibration_group_is_not_moved_to_training(self):
        sample={'path':'new.png','fruit':'mandarin','group':'mandarin:N01','source_group':'N01_20261029','phase':'train','ripeness':'unripe'}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'labels.csv'
            path.write_text('fruit,group_id,ripeness,split,phase\nmandarin,N01_20261029,ripe,train,calibration\n',encoding='utf-8')
            updated,_=refresh_labels([sample],path)
            self.assertEqual(updated[0]['phase'],'calibration')
            self.assertEqual(updated[0]['ripeness'],'ripe')


if __name__=='__main__':unittest.main()
