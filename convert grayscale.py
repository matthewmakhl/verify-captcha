import cv2
import numpy as np
from pathlib import Path

data_dir = Path("./Images/")
images = sorted(list(map(str, list(data_dir.glob("*.jfif")))))

max_height = 0
max_width = 0
max_channel = 1

for img_path in images:
  temp_img = cv2.imread(img_path, 0)
  height, width = temp_img.shape
  if height > max_height:
    max_height = height
  if width > max_width:
    max_width = width

print("Max height: ",max_height)
print("Max width: ",max_width)

for img_path in images:
  img = cv2.imread(img_path)

  lower =(170, 170, 170) # lower bound for each channel
  upper = (255, 255, 255) # upper bound for each channel

  # create the mask and use it to change the colors
  mask = cv2.inRange(img, lower, upper)
  img[mask != 0] = [255,255,255]

  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  (thresh, gray) = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
  height, width = gray.shape[:2]

  blank_image = np.zeros((max_height,max_width), np.uint8)
  blank_image[:,:] = (255)

  l_img = blank_image.copy()

  x_offset = y_offset = 0
  # Here, y_offset+height <= blank_image.shape[0] and x_offset+width <= blank_image.shape[1]
  l_img[y_offset:y_offset+height, x_offset:x_offset+width] = gray.copy()


  

  ## Remove horizontal
  ## horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
  ## detected_lines = cv2.morphologyEx(l_img, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
  ## cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  ## cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  ## for c in cnts:
  ##   cv2.drawContours(l_img, [c], -1, (255,255,255), 2)


  # se=cv2.getStructuringElement(cv2.MORPH_RECT , (1,3))
  # bg=cv2.morphologyEx(masked, cv2.MORPH_DILATE, se)
  # out_gray=cv2.divide(masked, bg, scale=255)
  # out_binary=cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU)[1] 

  thresh = cv2.threshold(l_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

  cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  for c in cnts:
      if cv2.contourArea(c) < 10:
          cv2.drawContours(thresh, [c], -1, (0,0,0), -1)

  result = 255 - thresh



  # # Find lines
  # edges = cv2.Canny(result,50,150,apertureSize = 5)
  # lines = cv2.HoughLines(edges,1,np.pi/180,100)

  # mask = blank_image.copy()

  # if not(lines is None):
  #   for line in lines:
  #     rho,theta = line[0]
  #     a = np.cos(theta)
  #     b = np.sin(theta)
  #     x0 = a*rho
  #     y0 = b*rho
  #     x1 = int(x0 + 1000*(-b))
  #     y1 = int(y0 + 1000*(a))
  #     x2 = int(x0 - 1000*(-b))
  #     y2 = int(y0 - 1000*(a))

  #     cv2.line(mask,(x1,y1),(x2,y2),(0,0,0),2)

  # masked = cv2.bitwise_not(result, result)
  # cv2.imshow('Gray', gray)
  # cv2.imshow('Masked', mask)
  # cv2.waitKey(0)

  # masked = cv2.bitwise_and(masked, masked, mask=mask)
  # result = cv2.bitwise_not(masked, masked)



  # cv2.imshow('Img', result)
  # cv2.waitKey(0)

  # morph = cv2.resize(result, (200,50)) 

  file_to_rem = Path(img_path)
  file_to_rem.unlink()
  # cv2.imwrite(img_path.replace(".jfif","2.jpg"), detected_lines)
  cv2.imwrite(img_path.replace(".jfif",".png"), result)