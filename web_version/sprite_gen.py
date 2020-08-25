import cv2
import sys
from sprite_helper import *

if (len(sys.argv)<2):
    print("Missing arguments")
    exit()
im_path = sys.argv[1]
img = cv2.imread(im_path)
masked_img = get_masked_image(img)
croped_image = crop_image(masked_img)
resized = cv2.resize(croped_image, (150,450))
board = cv2.imread("sprites/b3.png",cv2.IMREAD_UNCHANGED)
#dst = cv2.add(resized,board)

#dst = np.concatenate((resized, board), axis=0)
dst = cv2.addWeighted(board ,1,resized,1,0)

imgs = []
imgs.append(dst)
#cv2.imwrite("sprites/frame.png",dst)
angles = [7,-7,5,-5,10,-10,3,-3]
for a in angles:
  i = rotate_image(dst,a)
  print(i.shape)
  #cv2.imwrite(f"sprites/f{a}.png",i)
  imgs.append(i)
vis = np.concatenate(imgs, axis=1)
cv2.imwrite("sprites.png",vis)


exit()
keypoints = get_key_points(croped_image)

for k in keypoints.items():
    print(k)
