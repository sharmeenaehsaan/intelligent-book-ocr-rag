import cv2
import numpy as np
import matplotlib.pyplot as plt


class ImagePreprocessor:

    def __init__(self, resize_width=1800):
        self.resize_width = resize_width
        
    def load_image(self, image_path):
        img = cv2.imread(image_path)

        if img is None:
            raise FileNotFoundError(image_path)

        return img

    def resize(self, img):
        height, width = img.shape[:2]

        if width == self.resize_width:
            return img

        scale = self.resize_width / width
        new_height = int(height * scale)

        if scale > 1:
            interpolation = cv2.INTER_CUBIC
        else:
            interpolation = cv2.INTER_AREA

        return cv2.resize(
            img,
            (self.resize_width, new_height),
            interpolation=interpolation
        )

    def grayscale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def remove_shadows(self, gray):

        dilated = cv2.dilate(gray, np.ones((7,7), np.uint8))

        background = cv2.medianBlur(dilated, 21)

        diff = 255 - cv2.absdiff(gray, background)

        normalized = cv2.normalize(diff, None, 0, 255,
                                   cv2.NORM_MINMAX)

        return normalized

    def enhance_contrast(self, gray):

        clahe = cv2.createCLAHE(
            clipLimit=2.5,
            tileGridSize=(8,8)
        )

        return clahe.apply(gray)

    def denoise(self, gray):

        return cv2.fastNlMeansDenoising(
            gray,
            None,
            h=10,
            templateWindowSize=7,
            searchWindowSize=21
        )

    def threshold(self, gray):

        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11
        )

    def morphology(self, binary):

        kernel = np.ones((2,2), np.uint8)

        cleaned = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel
        )

        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_CLOSE,
            kernel
        )

        return cleaned

    def preprocess(self, image_path):

        img = self.load_image(image_path)

        img = self.resize(img)

        gray = self.grayscale(img)

        shadow_free = self.remove_shadows(gray)

        contrast = self.enhance_contrast(shadow_free)

        denoised = self.denoise(contrast)

        binary = self.threshold(denoised)

        final = self.morphology(binary)

        return {
            "original": img,
            "gray": gray,
            "shadow_removed": shadow_free,
            "contrast": contrast,
            "denoised": denoised,
            "binary": binary,
            "final": final
        }


def show(results):

    plt.figure(figsize=(15,12))

    titles = [
        "Original",
        "Gray",
        "Shadow Removed",
        "CLAHE",
        "Denoised",
        "Threshold",
        "Final"
    ]

    keys = [
        "original",
        "gray",
        "shadow_removed",
        "contrast",
        "denoised",
        "binary",
        "final"
    ]

    for i, key in enumerate(keys):

        plt.subplot(3,3,i+1)

        img = results[key]

        if len(img.shape)==3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.imshow(img,cmap="gray")

        plt.title(titles[i])

        plt.axis("off")

    plt.tight_layout()
    plt.show()