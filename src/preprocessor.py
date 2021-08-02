import cv2
import numpy as np
import imutils
import pytesseract


class Preprocessor:
    def __init__(self):

        # List of Post-It colors
        self.colors = {
            "pink": [
                np.array([167, 50, 50]),  # F383A7
                np.array([173, 255, 255]),
            ]
        }

        return

    def crop_postit(self, image_path: str):

        image = cv2.imread(image_path, 1)

        edges = self.detect_edges(image.copy())

        print(len(edges))

        largest_contour = self.get_largest_contour(edges)

        x, y, w, h = cv2.boundingRect(largest_contour)

        new = np.zeros_like(image.copy())
        cv2.drawContours(new, [largest_contour], -1, (255, 255, 255), -1)

        new_mask = cv2.inRange(new, (255, 255, 255), (255, 255, 255))
        result = cv2.bitwise_and(image.copy(), image.copy(), mask=new_mask)

        print(len(largest_contour))

        cropped_image = result[y : y + h, x : x + w]

        return cropped_image

    def detect_edges(self, image: np.ndarray):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv_image, self.colors["pink"][0], self.colors["pink"][1])

        # Paint all pixels not equal to the postit color as black
        hsv_image = cv2.bitwise_and(hsv_image, hsv_image, mask=mask)

        rgb = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        canny = imutils.auto_canny(gray)

        return canny

    def get_largest_contour(self, edges: np.ndarray):

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        max_len = 0
        largest_contour = None
        for contour in contours:
            if len(contour) > max_len:
                largest_contour = contour
                max_len = len(contour)

        return largest_contour

    def get_prediction(self, cropped_image: np.ndarray):

        cv2.namedWindow("T", 0)
        cv2.imshow("T", cropped_image)
        cv2.waitKey(0)

        pred = pytesseract.image_to_string(
            cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        )

        print(pred)

        return pred


if __name__ == "__main__":

    image = cv2.imread("tests/disk_test.jpeg", 1)
    preprocessor = Preprocessor()
    cropped_image = preprocessor.crop_postit("tests/disk_test.jpeg")
    pred = preprocessor.get_prediction(cropped_image)
