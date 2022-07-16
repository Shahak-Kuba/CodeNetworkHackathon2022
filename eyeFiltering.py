import cv2

# original image 
original_img = cv2.imread("Photos/IMG_2301.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Image 1", original_img)

#eye_cutdown = original_img[]

_, thresh = cv2.threshold(original_img, 30, 255, cv2.THRESH_BINARY_INV)

cv2.imshow("thresh_img", thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()

