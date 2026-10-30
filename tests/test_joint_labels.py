# CSV 반영 시 검증 분할을 유지하고 같은 개체의 분할 충돌을 막는지 검사합니다.
import tempfile
import unittest
from pathlib import Path

from src.joint_labels import refresh_labels


class JointLabelTests(unittest.TestCase):
    def test_explicit_selection_persists_and_conflicting_split_is_rejected(self):
        sample = {"fruit":"mandarin", "source_group":"M09_20261008", "ripeness":"ripe",
                  "group":"mandarin:M09", "phase":"test", "path":"photo.jpg"}
        with tempfile.TemporaryDirectory() as directory:
            csv = Path(directory)/"labels.csv"
            csv.write_text("fruit,group_id,ripeness,split,phase\nmandarin,M09_20261008,ripe,train,selection\n",encoding="utf-8")
            updated,changes = refresh_labels([sample],csv)
            self.assertEqual(updated[0]["phase"],"selection")
            self.assertEqual(updated[0]["ripeness"],"ripe")
            self.assertEqual(refresh_labels(updated,csv)[0][0]["phase"],"selection")
            csv.write_text("fruit,group_id,ripeness,split,phase\nmandarin,M09_20261008,ripe,test,selection\n",encoding="utf-8")
            with self.assertRaises(ValueError):
                refresh_labels([sample],csv)

    def test_refresh_applies_split_and_does_not_mutate_input(self):
        original = [{"fruit": "mandarin", "source_group": "M09_20261008",
                     "ripeness": "unripe", "phase": "test", "path": "photo.jpg",
                     "group": "mandarin:M09", "sha256": "image-hash"}]
        with tempfile.TemporaryDirectory() as directory:
            csv = Path(directory) / "labels.csv"
            csv.write_text("fruit,group_id,ripeness,split\nmandarin,M09_20261008,ripe,train\n", encoding="utf-8")
            updated, changes = refresh_labels(original, csv)
            self.assertEqual(updated[0], {**original[0], "ripeness": "ripe", "phase": "train"})
            self.assertEqual(original[0]["ripeness"], "unripe")
            self.assertEqual(len(changes), 1)
            self.assertEqual(refresh_labels(updated, csv)[1], [])
            holdout = [{**original[0], "phase": "calibration"}]
            self.assertEqual(refresh_labels(holdout, csv)[0][0]["phase"], "calibration")
            csv.write_text("fruit,group_id,ripeness,split\nmandarin,M09_20261008,ripe,test\n", encoding="utf-8")
            self.assertEqual(refresh_labels(updated, csv)[0][0]["phase"], "test")
            conflict = [original[0], {**original[0], "source_group": "other"}]
            csv.write_text("fruit,group_id,ripeness,split\nmandarin,M09_20261008,ripe,test\nmandarin,other,ripe,train\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                refresh_labels(conflict, csv)
            csv.write_text("fruit,group_id,ripeness\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                refresh_labels(original, csv)


if __name__ == "__main__":
    unittest.main()
