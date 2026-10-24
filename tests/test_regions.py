import importlib.util
import unittest
import numpy as np


class RegionTests(unittest.TestCase):
    def api(self):
        if importlib.util.find_spec('vdcp.regions') is None:
            self.fail('Region metric implementation is missing')
        import vdcp.regions as regions
        return regions

    def test_diagonally_touching_pixels_use_eight_connectivity(self):
        r = self.api()
        masks = np.zeros((4, 4, 4), dtype=np.uint8)
        masks[0, 0, 0] = masks[0, 1, 1] = 1
        self.assertEqual(len(r.extract_regions(masks)), 1)

    def test_wrong_class_detected_but_not_correctly_classified(self):
        r = self.api()
        truth = np.zeros((4, 8, 8), dtype=np.uint8)
        pred = truth.copy()
        truth[0, 2:5, 2:5] = 1
        pred[1, 2:5, 2:5] = 1
        result = r.region_counts(truth, pred)
        self.assertEqual((result['tp'], result['fp'], result['fn'], result['correct_class']), (1, 0, 0, 0))
        self.assertEqual(r.summarize_counts(result)['classification_accuracy'], 0.0)

    def test_empty_prediction_and_zero_denominator(self):
        r = self.api()
        truth = np.zeros((4, 8, 8), dtype=np.uint8)
        truth[0, 1:3, 1:3] = 1
        result = r.region_counts(truth, np.zeros_like(truth))
        self.assertEqual(result['fn'], 1)
        self.assertIsNone(r.summarize_counts(result)['classification_accuracy'])
        self.assertIsNone(r.summarize_counts(r.region_counts(np.zeros_like(truth), np.zeros_like(truth)))['recall'])

    def test_maximum_cardinality_prevents_greedy_undercount(self):
        r = self.api()
        matches = r.match_iou_matrix(np.array([[0.9, 0.6], [0.6, 0.0]]), 0.5)
        self.assertEqual(set(matches), {(0, 1), (1, 0)})

    def test_duplicate_predictions_cannot_share_one_truth(self):
        r = self.api()
        matches = r.match_iou_matrix(np.array([[1.0, 1.0]]), 0.5)
        self.assertEqual(len(matches), 1)


if __name__ == '__main__':
    unittest.main()
