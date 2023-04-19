from enum import Enum
import numpy as np
import imageio.v2 as imageio

def rank_transform(left_image, right_image, window_width):
  (H,W) = left_image.shape
  # pad image
  window_radius = int(((window_width-1)/2))
  left_image_padded = np.pad(left_image, window_radius, mode='constant', constant_values=0)
  right_image_padded = np.pad(right_image, window_radius, mode='constant', constant_values=0)

  rt_left = np.zeros(left_image_padded.shape)
  rt_right = np.zeros(left_image_padded.shape)
  # Loop over internal image in steps of 5
  for y in range(window_radius, H - window_radius, 5):
    print("y: ", y)
    for x in range(window_radius, W - window_radius, 5):  
      # Loop over window
      pixel_vals_left = []
      pixel_vals_right = []
      # make list of all pixel values in window
      for v in range(-window_radius, window_radius + 1):
        for u in range(-window_radius, window_radius + 1):
          pixel_vals_left.append(left_image_padded[y+v, x+u])
          pixel_vals_right.append(right_image_padded[y+v, x+u])
  
      # create a sorted set from pixel values retrieved above
      unique_ranked_pixels_left = sorted(set(pixel_vals_left))
      unique_ranked_pixels_right = sorted(set(pixel_vals_right))

      # step through window again, replacing pixel values with index ("rank") of pixel val from above list
      for v in range(-window_radius, window_radius + 1):
        for u in range(-window_radius, window_radius + 1):
          rt_left[y+v, x+u] = unique_ranked_pixels_left.index(left_image_padded[y+v, x+u])
          rt_right[y+v, x+u] = unique_ranked_pixels_right.index(right_image_padded[y+v, x+u])
  
  return rt_left, rt_right


def SAD(left_image: np.ndarray, right_image: np.ndarray, max_disparity: int, filter_width: int) -> np.ndarray:
  # Compute a cost volume with maximum disparity D considering a neighbourhood R with Sum of Absolute Differences (SAD)
  #   @param[in] left_image: The left image to be used for stereo matching (H,W) 
  #   @param[in] right_image: The right image to be used for stereo matching (H,W)
  #   @param[in] max_disparity: The maximum disparity to consider
  #   @param[in] filter_width: The filter width to be considered for matching
  #   @return: The best matching pixel inside the cost volume according to the pre-defined criterion (H,W,D) 
  (H,W) = left_image.shape
  cost_volume = np.zeros((H,W,max_disparity))
  filter_radius = int(((filter_width-1)/2))

  # Loop over internal image
  for y in range(filter_radius, H - filter_radius):
    print("y: ", y)
    for x in range(filter_radius, W - filter_radius):  
      # Loop over window
      for v in range(-filter_radius, filter_radius + 1):
        for u in range(-filter_radius, filter_radius + 1):
          # Loop over all possible disparities
          for d in range(0, max_disparity):
            # need to cast to int here because the numbers are uint8 by default which cannot allow negatives
            cost_volume[y,x,d] += np.absolute(int(left_image[y+v, x+u]) - int(right_image[y+v, x+u-d]))
        
  return cost_volume


def winner_takes_all(cost_volume: np.ndarray) -> np.ndarray:
  # Function for matching the best suiting pixels for the disparity image
  #   @param[in] cost_volume: The three-dimensional cost volume to be searched for the best matching pixel (H,W,D)
  #   @return: The two-dimensional disparity image resulting from the best matching pixel inside the cost volume (H,W)
    
  print(np.argmin(cost_volume, axis=2)) 
  return np.argmin(cost_volume, axis=2)  


# Class constructor
#   @param[in] left_image: The left stereo image (H,W)
#   @param[in] right_image: The right stereo image (H,W)
#   @param[in] matching_cost: The class implementing the matching cost
#   @param[in] matching_algorithm: The class implementing the matching algorithm
#   @param[in] max_disparity: The maximum disparity to consider
#   @param[in] filter_width: The width of the filter

left_image = np.asarray(imageio.imread("teddy/teddyL.pgm"))
right_image = np.asarray(imageio.imread("teddy/teddyR.pgm"))
ground_truth = np.asarray(imageio.imread("teddy/disp2.pgm"))
max_disparity = 63
filter_width = 3

# if (left_image.ndim != 2):
#   raise ValueError("The left image has to be a grey-scale image with a single channel as its last dimension.")
# if (right_image.ndim != 2):
#   raise ValueError("The right image has to be a grey-scale image with a single channel as its last dimension.")
# if (left_image.shape != right_image.shape):
#   raise ValueError("Dimensions of left (" + left_image.shape + ") and right image (" + right_image.shape + ") do not match.")
# if (max_disparity <= 0):
#   raise ValueError("Maximum disparity (" + max_disparity + ") has to be greater than zero.")
# if (filter_width <= 0):
#   raise ValueError("width (" + filter_width + ") has to be greater than zero.")

rt_left, rt_right = rank_transform(left_image, right_image, 5)
#imageio.imsave("rt_left.pgm", rt_left)
#imageio.imsave("rt_right.pgm", rt_right)
np.savetxt('rt_left values.csv', rt_left, fmt = '%d', delimiter=",")

cost_volume = SAD(rt_left, rt_right, max_disparity, filter_width)
result = winner_takes_all(cost_volume)

np.savetxt('3x3 disparity map.csv', result, fmt = '%d', delimiter=",")
#result = np.loadtxt("3x3 disparity map.csv", delimiter=",")
no_pad_result = result[2:377, 2:452]
fourth_gt = np.around(ground_truth / 4)
#np.savetxt('ground truth values.csv', ground_truth, fmt = '%d', delimiter=",")
#np.savetxt('fourth_gt values.csv', fourth_gt, fmt = '%d', delimiter=",")
errors = np.absolute(np.subtract(no_pad_result, fourth_gt))

print((errors > 1).sum() / errors.size)

filter_width = 15

cost_volume = SAD(rt_left, rt_right, max_disparity, filter_width)
result = winner_takes_all(cost_volume)

np.savetxt('15x15 disparity map.csv', result, fmt = '%d', delimiter=",")

no_pad_result = result[2:377, 2:452]
errors_2 = np.absolute(np.subtract(no_pad_result, fourth_gt))

print("3x3: ", (errors > 1).sum() / errors.size)
print("15x15: ", (errors_2 > 1).sum() / errors_2.size)