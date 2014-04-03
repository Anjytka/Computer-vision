import numpy as np
import cv2
import sys

if __name__ == '__main__':
    img = cv2.imread(sys.argv[1])
    imgFormat = sys.argv[1][len(sys.argv[1]) - 4:]  # parse img format

    for matsize in [3, 5, 9, 15, 21]:
        cv2.imwrite('blur' + str(matsize) +
                    imgFormat, cv2.blur(img, (matsize, matsize)))
        cv2.imwrite('gaus' + str(matsize) +
                    imgFormat, cv2.GaussianBlur(img, (matsize, matsize), 0))
        cv2.imwrite('median' + str(matsize) +
                    imgFormat, cv2.medianBlur(img, matsize))
        cv2.imwrite('bilateral' + str(matsize) +
                    imgFormat, cv2.bilateralFilter(img, matsize, 75, 75))
        cv2.imwrite('nonlocal' + str(matsize) + imgFormat,
                    cv2.fastNlMeansDenoisingColored(img, None, 20, 20,
                                                    matsize / 3, matsize))
