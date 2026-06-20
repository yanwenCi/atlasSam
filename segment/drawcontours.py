# Extract images
import cv2
def draw_contours(concat_image):
    base_image = concat_image[0].astype(np.uint8)  # Original grayscale image
    pred_mask = concat_image[1].astype(np.uint8)  # Predicted mask
    gt_mask = concat_image[2].astype(np.uint8)  # Ground truth mask

        # Convert masks to binary
    _, pred_binary = cv2.threshold(pred_mask, 50, 255, cv2.THRESH_BINARY)
    _, gt_binary = cv2.threshold(gt_mask, 50, 255, cv2.THRESH_BINARY)

    # Find contours
    contours_pred, _ = cv2.findContours(pred_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_gt, _ = cv2.findContours(gt_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert base image to BGR for colored contours
    contoured_image = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    # Draw predicted contours (green & blue)
    draw_dual_contours(contoured_image, contours_pred, [(0, 255, 0), (255, 0, 0)])  # Green & Blue

    # Draw ground truth contours (red & yellow)
    draw_dual_contours(contoured_image, contours_gt, [(0, 0, 255), (0, 255, 255)])  # Red & Yellow
    return contoured_image

    # Function to draw contours with different colors for two regions
def draw_dual_contours(image, contours, colors):
    for i, contour in enumerate(contours):
        color = colors[i % len(colors)]  # Cycle through colors
        cv2.drawContours(image, [contour], -1, color, 2)

