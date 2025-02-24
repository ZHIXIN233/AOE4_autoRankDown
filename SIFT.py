import pyautogui
import numpy as np
import cv2
import warnings

ratio_threshold = 0.3  # e.g., require at least 30% of target keypoints to be matched well

def find_target_image_in_screen(target_image, screenshot=None, region=None, tuning=False, ratio_threshold=0.5, response_threshold=0.02, good_match_threshold=0.7, return_full_detail=False):
    '''
    Find a target image's location within a screenshot of the screen using the
    SIFT (Scale-Invariant Feature Transform) algorithm for keypoint detection and matching.

    :param target_image: The target image to be found on the screen. It can be either:
                         - A file path of the image (str)
                         - A numpy array of the image
    :param region: The region on the screen to take the screenshot from,
                   provided as a tuple (left, top, width, height). If not provided, the whole screen is used.
    :param tuning: If True, will print the relative intermidiate results for tuning the algorithm.

    :return: A tuple (found, bounding_box), where:
             - found (bool): Whether the target image was found on the screen or not.
             - bounding_box (tuple): The bounding box of the target image on the screen in the format:
                                    (x_min, y_min, width, height). If not found, returns None.
    '''
    # Capture the screen or the specified region
    if screenshot is not None:
        screenshot = np.array(screenshot)
        if region is not None:
            screenshot = screenshot[region[1]:region[1] + region[3], region[0]:region[0] + region[2]]
        else:
            pass
    if screenshot is None:
        if region is not None:
            screenshot = pyautogui.screenshot(region=region)
        else:
            screenshot = pyautogui.screenshot()
    screenshot = np.array(screenshot)
    screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)

    # Read the target image or use it directly if already a numpy array
    if isinstance(target_image, str):  # If the target image is a file path
        template = cv2.imread(target_image, cv2.IMREAD_GRAYSCALE)
    else:
        template = cv2.cvtColor(target_image, cv2.COLOR_RGB2GRAY)

    if template is None:
        raise ValueError("Target image could not be read. Ensure the file path is correct or the input is valid.")

    # Initialize the SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors in both images
    kp1, des1 = sift.detectAndCompute(screenshot_gray, None)
    kp2, des2 = sift.detectAndCompute(template, None)

    filtered_kp2 = []
    filtered_des2 = []
    for kp, desc in zip(kp2, des2):
        if kp.response >= response_threshold:
            filtered_kp2.append(kp)
            filtered_des2.append(desc)

    # If too few keypoints, return fail with a warning directly
    if len(filtered_kp2) < 5:
        warnings.warn("Too few keypoints on target image, try find better target image.")
        return False, None

    des2 = np.array(filtered_des2)
    kp2 = filtered_kp2

    # Use FLANN matcher to find matches between descriptors
    index_params = dict(algorithm=1, trees=5)  # FLANN parameters
    search_params = dict(checks=50)  # Number of checks
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Perform feature matching
    matches = flann.knnMatch(des2, des1, k=2)

    # Filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < good_match_threshold * n.distance:  # Distance ratio for filtering good matches
            good_matches.append(m)

    # Calculate the ratio of good matches to the target image's keypoints
    good_match_ratio = len(good_matches) / len(kp2)

    if tuning:
        if isinstance(target_image, str):
            print(f"Target image: {target_image}")
        print(f"Number of good matches: {len(good_matches)}")
        print(f"Number of keypoints in target image: {len(kp2)}")
        print(f"Number of keypoints in screenshot: {len(kp1)}")
        print(f"Number of matches in both images: {len(matches)}")
        print(f"Number of matches in both images after filtering: {len(good_matches)}")
        print(f"Ratio of good match in target images: {good_match_ratio} (good_matches / kp2)")

    # If not enough good matches, return not found
    if good_match_ratio < ratio_threshold:
        if return_full_detail:
            return False, None, len(good_matches), len(kp2), len(kp1), len(matches), len(good_matches), good_match_ratio
        return False, None

    # Extract matched keypoints' coordinates
    src_pts = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute the homography to map the target image to the screen
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if M is None:
        return False, None

    # Get the dimensions of the target image
    h, w = template.shape

    # Map the corners of the target image to the screenshot space
    corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners, M)

    # Compute bounding box from the transformed corners
    x_min, y_min = np.int32(transformed_corners.min(axis=0).ravel())
    x_max, y_max = np.int32(transformed_corners.max(axis=0).ravel())
    bounding_box = (x_min, y_min, x_max - x_min, y_max - y_min)

    if tuning:
        # Show the image in the bounding box in the screenshot
        import matplotlib.pyplot as plt
        screenshot_with_box = screenshot.copy()
        cv2.rectangle(screenshot_with_box, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        plt.imshow(cv2.cvtColor(screenshot_with_box, cv2.COLOR_BGR2RGB))
        plt.title("Screenshot with Bounding Box")
        plt.axis("off")
        plt.show()

    if return_full_detail:
        return True, bounding_box, len(good_matches), len(kp2), len(kp1), len(matches), len(good_matches), good_match_ratio
    return True, bounding_box

