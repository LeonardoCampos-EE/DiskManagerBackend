import unittest
from src.preprocessor import *


class PreprocessorTests(unittest.TestCase):
    def setUp(self):
        self.image = cv2.imread("tests/disk_test.jpeg")
        self.preprocessor = Preprocessor()

    def test_detect_edges(self):
        edges = self.preprocessor.detect_edges(self.image)
        len_edges = len(edges)

        self.assertEqual(len_edges, 1280, "len(edges) should be 1280")

        return

    def test_get_largest_contour(self):
        edges = self.preprocessor.detect_edges(self.image)
        largest_contour = self.preprocessor.get_largest_contour(edges)

        len_largest_contour = len(largest_contour)

        self.assertEqual(len_largest_contour, 204, "len(largest_contour) should be 204")


if __name__ == "__main__":

    unittest.main()
