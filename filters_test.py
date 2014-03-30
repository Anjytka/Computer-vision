import numpy as np
import cv2
import sys

if __name__ == '__main__':
    img = cv2.imread(sys.argv[1])
    imgFormat = sys.argv[1][len(sys.argv[1]) - 4:]  # parse img format

    for matsize in range(3, 8, 2):
        cv2.imwrite('blur' + str(matsize) +
                    imgFormat, cv2.blur(img, (matsize, matsize)))
        cv2.imwrite('gaus' + str(matsize) +
                    imgFormat, cv2.GaussianBlur(img, (matsize, matsize), 0))
        cv2.imwrite('median' + str(matsize) +
                    imgFormat, cv2.medianBlur(img, matsize))
        cv2.imwrite('bilateral' + str(matsize) +
                    imgFormat, cv2.bilateralFilter(img, matsize, 75, 75))

    cv2.imwrite('nonlocal' + imgFormat,
                cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21))
