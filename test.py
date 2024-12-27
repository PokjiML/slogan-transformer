# Load images
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)
img3 = cv2.imread(img3_path)

# Turn them to grayscale
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

# Initialize SIFT algorithm
sift = cv2.SIFT_create()
# sift = cv2.ORB_create()

# Find the keypoint and descriptors for every image
kps1, desc1 = sift.detectAndCompute(img1, None)
kps2, desc2 = sift.detectAndCompute(img2, None)
kps3, desc3 = sift.detectAndCompute(img3, None)

def find_best_matches(descriptor1, descriptor2):
  """ Using BruteForce algorithm to find the closest matches between
      two given images, return matches list """

  # Finding the best matches between the images
  brute_force = cv2.BFMatcher(cv2.NORM_L2,crossCheck=True)
  matches = brute_force.match(descriptor1, descriptor2)

  # sort the matches from most similar
  matches = sorted(matches, key=lambda x: x.distance)

  # Return the list of matches and an image of them

  matches = matches[:20]

  return matches


# Finding best matches between img1 and img2
matches_12 = find_best_matches(desc1, desc2)

# Finding best matches between img2 and img3
matches_23 = find_best_matches(desc2, desc3)

# Get the keypoints source points
src_pts = np.float32([kps2[m.queryIdx].pt for m in matches_23]).reshape(-1, 1, 2)
dst_pts = np.float32([kps3[m.trainIdx].pt for m in matches_23]).reshape(-1, 1, 2)

# Find homography matrix with RANSAC
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
# H, mask = cv2.findHomography(src_pts, dst_pts)

print(H)

height, width = img2.shape

im_dst = cv2.warpPerspective(img3, H, (width, height))

