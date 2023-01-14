import cv2
import numpy as np
import glob
import os.path

CAPTCHA_IMAGE_FOLDER = "images"
OUTPUT_FOLDER = "letter_images"


captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))
counts = {}

for (i, captcha_image_file) in enumerate(captcha_image_files):
    print("[INFO] processing image {} {}/{}".format(captcha_image_file, i + 1, len(captcha_image_files)))
    filename = os.path.basename(captcha_image_file)
    captcha_correct_text = os.path.splitext(filename)[0][0:5]

    # open file
    img = cv2.imread(captcha_image_file)

    lower = np.array([0, 0, 0])  # -- Lower range --
    upper = np.array([30, 30, 30])  # -- Upper range --
    mask = cv2.inRange(img, lower, upper)
    img = cv2.bitwise_not(img, img, mask=mask)
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # convert to grayscale
    gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gry', gry)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # upscale
    (h, w) = gry.shape[:2]
    gry = cv2.resize(gry, (w * 2, h * 2))
    # cv2.imshow('resize', gry)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # morph close
    cls = cv2.morphologyEx(gry, cv2.MORPH_CLOSE, None)
    # cv2.imshow('cls', cls)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # remove background
    thr = cv2.threshold(cls, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # cv2.imshow('image', thr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # remove small objects
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
    cls2 = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, element)
    # cv2.imshow('cls2', cls2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    blur = cv2.blur(cls2, (10, 10))
    # cv2.imshow('blur', blur)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # downscale
    (h, w) = blur.shape[:2]
    blur2 = cv2.resize(blur, (w // 2, h // 2))
    # cv2.imshow('blur2', blur2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    inverted = 255*(blur2 < 128).astype(np.uint8)

    coords = cv2.findNonZero(inverted) # Find all non-zero points (text)
    x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
    rect = blur2[y:y+h, x:x+w]
    # cv2.imshow('rect', rect)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    (h, w) = rect.shape[:2]
    letter_image_regions = []
    letter_image_regions.append((0, 0, w // 5, h))
    letter_image_regions.append((w // 5, 0, w // 5, h))
    letter_image_regions.append((2 * w // 5, 0, w // 5, h))
    letter_image_regions.append((3 * w // 5, 0, w // 5, h))
    letter_image_regions.append((4 * w // 5, 0, w // 5, h))

    # Save out each letter as a single image
    for letter_bounding_box, letter_text in zip(letter_image_regions, captcha_correct_text):
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box

        # Extract the letter from the original image with a 2-pixel margin around the edge
        letter_image = rect[y:y + h, x:x + w]

        # Get the folder to save the image in
        save_path = os.path.join(OUTPUT_FOLDER, letter_text)

        # if the output directory does not exist, create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # write the letter image to a file
        count = counts.get(letter_text, 1)
        p = os.path.join(save_path, "{}.jpg".format(str(count).zfill(6)))
        cv2.imwrite(p, letter_image)

        # increment the count for the current key
        counts[letter_text] = count + 1