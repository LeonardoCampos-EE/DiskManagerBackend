import cv2
import numpy as np
import imutils


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

    def crop_postit(self, image_path):

        image = cv2.imread(image_path, 1)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv_image, self.colors["pink"][0], self.colors["pink"][1])
        
        # Paint all pixels not equal to the postit color as black
        hsv_image = cv2.bitwise_and(hsv_image, hsv_image, mask=mask)

        rgb = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        canny = imutils.auto_canny(gray)

        contours, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = self.find_largest_contour(contours)

        new = np.zeros_like(image)
        cv2.drawContours(new, [largest_contour], -1, (255, 255, 255), -1)
        new_mask = cv2.inRange(new, (255, 255, 255), (255, 255, 255))

        result = cv2.bitwise_and(image, image, mask=new_mask)

        print(len(largest_contour))

        cv2.namedWindow("T", 0)
        cv2.imshow("T", result)
        cv2.waitKey(0)

        return

    def find_largest_contour(self, contours):

        max_len = 0
        largest_contour = None
        for contour in contours:
            if len(contour) > max_len:
                largest_contour = contour
                max_len = len(contour)

        return largest_contour


# Color = (341, 46, 95)

if __name__ == "__main__":

    # image = cv2.imread("tests/disk_test.jpeg", 1)
    # cv2.namedWindow("T", 0)
    # cv2.imshow("T", image)
    # cv2.waitKey(0)

    preprocessor = Preprocessor()
    preprocessor.crop_postit("tests/disk_test.jpeg")
