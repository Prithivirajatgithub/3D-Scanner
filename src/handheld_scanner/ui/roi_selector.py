import cv2
import numpy as np


class ObjectROISelector:
    @staticmethod
    def select_bounding_box(color_image):
        """
        Allows the user to drag a bounding box around the target object on frame 0.
        Returns a binary uint8 mask (0 = background, 255 = target object).
        """
        # Convert RGB to BGR for OpenCV display
        bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        window_name = "BundleSDF - Drag Box Around Target Object (Press SPACE/ENTER to Confirm)"
        
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        bbox = cv2.selectROI(window_name, bgr, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(window_name)

        x, y, w, h = bbox
        mask = np.zeros(color_image.shape[:2], dtype=np.uint8)

        if w > 10 and h > 10:
            # GrabCut initial foreground segmentation inside the bounding box
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            grabcut_mask = np.zeros(color_image.shape[:2], np.uint8)
            cv2.grabCut(bgr, grabcut_mask, (x, y, w, h), bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            mask = np.where((grabcut_mask == 2) | (grabcut_mask == 0), 0, 255).astype(np.uint8)
        else:
            print("[WARN] No bounding box selected. Using center fallback region.")
            h, w = mask.shape
            mask[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)] = 255

        return mask
