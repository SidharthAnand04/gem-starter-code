import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Header, Bool
from cv_bridge import CvBridge, CvBridgeError
import cv2
import torch
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import time

###############################################################################
# Lane Detection Node
# 
# This module implements deep learning-based lane detection using YOLOPv2.
# It processes images from a camera, identifies lane markings, and publishes
# waypoints for autonomous navigation.
###############################################################################

class LaneNetDetector:
    """
    Main class for lane detection using YOLOPv2 neural network.
    
    This class handles:
    1. Image preprocessing and enhancement
    2. Deep learning model inference
    3. Lane detection and boundary identification
    4. Waypoint generation for vehicle navigation
    5. Visual feedback through annotated images
    """
    
    def __init__(self):
        """
        Initialize the lane detection node with model, parameters and ROS connections.
        
        Sets up:
        - Frame buffering for stable detection
        - Deep learning model (YOLOPv2)
        - ROS publishers and subscribers
        - Image processing parameters
        """
        # Frame buffer for batch processing to increase efficiency
        self.frame_buffer = []
        self.buffer_size = 4  # Process 4 frames at once for better throughput
        
        # Initialize ROS node
        rospy.init_node('lane_detection_node', anonymous=True)
        
        # Image processing utilities and state variables
        self.bridge = CvBridge()  # Converts between ROS Image messages and OpenCV images
        self.prev_left_boundary = None  # Store previous lane boundary for smoothing
        self.estimated_lane_width_pixels = 200  # Approximate lane width in image pixels
        self.prev_waypoints = None  # Previous waypoints for temporal consistency
        self.endgoal = None  # Target point for navigation
        
        ###############################################################################
        # Deep Learning Model Setup
        ###############################################################################
        
        # Set up compute device (GPU if available, otherwise CPU)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load pre-trained YOLOPv2 model for lane detection
        self.model = torch.jit.load('weights/yolopv2.pt')
        self.half = self.device != 'cpu'  # Use half precision for faster inference on GPU
        
        # Configure model for inference
        if self.half:
            self.model.half()  # Convert model to half precision
            
        self.model.to(self.device).eval()  # Move model to device and set to evaluation mode
        
        # Commented out stop sign detection model code
        # self.stop_model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True, trust_repo=True)
        # self.stop_model.to(self.device).eval()
        
        ###############################################################################
        # Camera and Control Parameters
        ###############################################################################
        
        self.Focal_Length = 800  # Camera focal length in pixels
        self.Real_Height_SS = .75  # Height of stop sign in meters (not used currently)
        self.Brake_Distance = 5  # Distance at which to apply brakes (not used currently)
        self.Brake_Duration = 3  # Duration to hold brakes (not used currently)
        
        ###############################################################################
        # ROS Communication Setup
        ###############################################################################
        
        # Subscribe to camera feed
        self.sub_image = rospy.Subscriber('oak/rgb/image_raw', Image, self.img_callback, queue_size=1)
        
        # Publishers for visualization and control
        self.pub_contrasted_image = rospy.Publisher("lane_detection/contrasted_image", Image, queue_size=1)
        self.pub_annotated = rospy.Publisher("lane_detection/annotate", Image, queue_size=1)
        self.pub_waypoints = rospy.Publisher("lane_detection/waypoints", Path, queue_size=1)
        self.pub_endgoal = rospy.Publisher("lane_detection/endgoal", PoseStamped, queue_size=1)

    def img_callback(self, img):
        """
        Process incoming camera images to detect lanes and generate waypoints.
        
        This function:
        1. Converts ROS image to OpenCV format
        2. Enhances image using color filtering and contrast
        3. Preprocesses image for neural network
        4. Adds image to buffer for batch processing
        5. Performs inference when buffer is full
        6. Generates and publishes waypoints and visualizations
        
        Args:
            img: ROS Image message from camera
        """
        try:
            # Convert ROS Image to OpenCV format
            img = self.bridge.imgmsg_to_cv2(img, "bgr8")
            
            ###############################################################################
            # Image Enhancement Pipeline
            ###############################################################################
            
            # Convert to HSV color space for better color segmentation
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Define yellow color range for lane markings
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([30, 255, 255])
            
            # Create masks to enhance yellow lane markings
            yellow_mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
            non_yellow_mask = cv2.bitwise_not(yellow_mask)
            
            # Remove non-yellow areas and convert to grayscale
            img_no_yellow = cv2.bitwise_and(img, img, mask=non_yellow_mask)
            img_gray = cv2.cvtColor(img_no_yellow, cv2.COLOR_BGR2GRAY)
            
            # Apply contrast enhancement using thresholding
            threshold = 180
            mask = img_gray >= threshold
            dimmed_gray = (img_gray * 0.5).astype(np.uint8)  # Reduce brightness of non-lane areas
            dimmed_gray[mask] = img_gray[mask]  # Keep bright areas at original intensity
            
            # Convert back to BGR for visualization
            contrasted_img = cv2.cvtColor(dimmed_gray, cv2.COLOR_GRAY2BGR)
            
            # Publish enhanced image for debugging
            contrasted_image_msg = self.bridge.cv2_to_imgmsg(contrasted_img, "bgr8")
            self.pub_contrasted_image.publish(contrasted_image_msg)
            
            ###############################################################################
            # Model Inference Pipeline
            ###############################################################################
            
            # Preprocess image for neural network
            img_tensor = self.preprocess_frame(contrasted_img)
            
            # Add to buffer for batch processing
            self.frame_buffer.append((contrasted_img, img_tensor))
            
            # When buffer is full, process batch for efficiency
            if len(self.frame_buffer) >= self.buffer_size:
                # Separate original images and tensors
                original_images, tensors = zip(*self.frame_buffer)
                
                # Stack tensors into a batch
                batch = torch.stack(tensors).to(self.device)
                
                # Clear buffer after processing
                self.frame_buffer.clear()
                
                # Run inference on batch
                with torch.no_grad():
                    [pred, anchor_grid], seg, ll = self.model(batch)
                
                # Process each result in the batch
                for i, contrasted_img in enumerate(original_images):
                    # Generate waypoints and annotate image
                    annotated_img = self.detect_lanes(seg[i], ll[i], contrasted_img)
                    
                    # Publish annotated image
                    annotated_image_msg = self.bridge.cv2_to_imgmsg(annotated_img, "bgr8")
                    self.pub_annotated.publish(annotated_image_msg)
                    
        except CvBridgeError as e:
            print(e)

    def preprocess_frame(self, img):
        """
        Preprocess image for neural network input.
        
        Steps:
        1. Resize image with letterboxing to maintain aspect ratio
        2. Convert BGR to RGB and change channel order
        3. Convert to tensor and normalize
        
        Args:
            img: OpenCV image (BGR format)
            
        Returns:
            PyTorch tensor ready for model inference
        """
        # Resize with letterboxing to model input size (384x640)
        img_resized, _, _ = self.letterbox(img, new_shape=(384, 640))
        
        # Convert BGR to RGB and change to channel-first format
        img_resized = img_resized[:, :, ::-1].transpose(2, 0, 1)
        
        # Convert to PyTorch tensor
        img_tensor = torch.from_numpy(np.ascontiguousarray(img_resized))
        
        # Convert to half precision if using GPU
        img_tensor = img_tensor.half() if self.half else img_tensor.float()
        
        # Normalize pixel values to 0-1
        img_tensor /= 255.0
        
        return img_tensor

    def image_to_world(self, u, v, camera_matrix, camera_height):
        """
        Convert image coordinates to world coordinates using pinhole camera model.
        
        This function projects a point from the image plane to 3D world coordinates
        assuming a flat ground plane.
        
        Args:
            u, v: Image coordinates (pixels)
            camera_matrix: Camera intrinsic parameters
            camera_height: Height of camera above ground plane
            
        Returns:
            X, Y, Z: World coordinates (meters)
        """
        # Extract camera intrinsic parameters
        fx = camera_matrix[0, 0]  # Focal length in x direction
        fy = camera_matrix[1, 1]  # Focal length in y direction
        cx = camera_matrix[0, 2]  # Principal point x coordinate
        cy = camera_matrix[1, 2]  # Principal point y coordinate
        
        # Assume ground plane is at z = -camera_height
        Z = -camera_height
        
        # Project using pinhole camera model
        X = Z * (u - cx) / fx
        Y = Z * (v - cy) / fy
        
        return X, Y, Z

    def detect_lanes(self, seg, ll, img):
        """
        Process neural network output to detect lanes and generate waypoints.
        
        Steps:
        1. Extract drivable area mask from segmentation output
        2. Extract lane line mask from lane detection output
        3. Generate waypoints based on detected lanes
        4. Draw waypoints on image for visualization
        
        Args:
            seg: Segmentation output from neural network
            ll: Lane line output from neural network
            img: Original image for annotation
            
        Returns:
            Annotated image with waypoints and lane boundaries
        """
        # Extract drivable area mask from segmentation output
        da_seg_mask = driving_area_mask(seg)
        
        # Extract lane line mask with confidence threshold
        ll_seg_mask = lane_line_mask(ll, threshold=0.2)
        
        # Generate waypoints from lane line mask
        waypoints, left_boundary = self.generate_waypoints(ll_seg_mask)
        
        # Draw waypoints on image for visualization
        img_with_waypoints = self.draw_waypoints(img.copy(), waypoints, left_boundary)
        
        # Publish waypoints for vehicle control
        self.publish_waypoints(waypoints)
        
        return img_with_waypoints

    def region_of_interest(self, img):
        """
        Apply a region of interest mask to focus on relevant portion of image.
        
        This function creates a polygon mask to focus processing on the left
        half of the image where the lane is expected to be.
        
        Args:
            img: Input image
            
        Returns:
            Masked image with only the region of interest visible
        """
        # Get image dimensions
        height, width = img.shape[:2]
        
        # Create empty mask
        mask = np.zeros_like(img)
        
        # Define polygon for left half of image
        polygon = np.array([[(0, height), (0, 0), (width // 2, 0), (width // 2, height)]], np.int32)
        
        # Fill polygon with white
        cv2.fillPoly(mask, polygon, 255)
        
        # Apply mask to image
        masked_image = cv2.bitwise_and(img, mask)
        
        return masked_image

    def generate_waypoints(self, lane_mask):
        """
        Generate navigation waypoints from lane mask.
        
        This function:
        1. Scans the lane mask from bottom to top
        2. Identifies the left lane boundary
        3. Estimates center of lane using offset from boundary
        4. Creates waypoints for vehicle to follow
        
        Args:
            lane_mask: Binary mask of detected lane markings
            
        Returns:
            path: ROS Path message with waypoints
            left_boundary: List of points representing left lane boundary
        """
        # Create ROS Path message
        path = Path()
        path.header.frame_id = "map"
        
        # Get dimensions of lane mask
        height, width = lane_mask.shape
        
        # Sampling parameters
        sampling_step = 10  # Sample every 10 pixels vertically
        left_boundary = []  # Store left boundary points
        offset_pixels = 200  # Estimated distance from left boundary to lane center
        
        ###############################################################################
        # Scan Lane Mask and Extract Boundaries
        ###############################################################################
        
        # Scan from bottom to top of image
        for y in range(height - 1, 0, -sampling_step):
            # Find all x coordinates where lane marking exists at this y
            x_indices = np.where(lane_mask[y, :] > 0)[0]
            
            if len(x_indices) > 0:
                # Use leftmost point as left boundary
                x_left = x_indices[0]
                # Store with slight offset for better tracking
                left_boundary.append((x_left - 40, y))
            else:
                # No lane marking found at this y
                left_boundary.append(None)
                
        # Filter boundaries to get continuous line segments
        left_boundary = self.filter_continuous_boundary(left_boundary)
        
        ###############################################################################
        # Generate Waypoints from Boundaries
        ###############################################################################
        
        # Create waypoints from left boundary with offset
        for lb in left_boundary:
            if lb:  # If boundary point exists
                # Calculate lane center by adding offset to left boundary
                x_center = lb[0] + offset_pixels
                y = lb[1]
                
                # Create ROS PoseStamped message for waypoint
                point = PoseStamped()
                point.pose.position.x = x_center
                point.pose.position.y = y
                path.poses.append(point)
                
        # Use only the first 7 waypoints (closest to vehicle)
        path.poses = path.poses[:7]
        
        ###############################################################################
        # Calculate End Goal (Target Point)
        ###############################################################################
        
        if len(path.poses) > 0:
            # Extract x and y coordinates from all waypoints
            xs = [p.pose.position.x for p in path.poses]
            ys = [p.pose.position.y for p in path.poses]
            
            # Use median for stability
            median_x = np.median(xs)
            median_y = np.median(ys)
            
            # Create end goal pose
            self.endgoal = PoseStamped()
            self.endgoal.header = path.header
            self.endgoal.pose.position.x = median_x
            self.endgoal.pose.position.y = median_y
        else:
            # No waypoints found
            self.endgoal = None
            
        return path, left_boundary

    def filter_continuous_boundary(self, boundary):
        """
        Filter boundary points to get continuous line segments.
        
        This helps remove noise and discontinuities in the detected lane boundary.
        Points with large horizontal gaps are considered discontinuities.
        
        Args:
            boundary: List of boundary points (x,y) or None
            
        Returns:
            List of filtered boundary points
        """
        max_gap = 60  # Maximum allowed horizontal gap between consecutive points
        continuous_boundary = []
        previous_point = None
        
        for point in boundary:
            if point is not None:
                if previous_point is None or abs(point[0] - previous_point[0]) <= max_gap:
                    # Point is continuous with previous point
                    continuous_boundary.append(point)
                    previous_point = point
                else:
                    # Gap too large, start new segment
                    continuous_boundary.append(None)
                    previous_point = None
            else:
                # No point detected
                continuous_boundary.append(None)
                previous_point = None
                
        return continuous_boundary

    def publish_waypoints(self, waypoints):
        """
        Publish waypoints and end goal for vehicle control.
        
        Args:
            waypoints: ROS Path message with waypoints
        """
        # Publish waypoints
        self.pub_waypoints.publish(waypoints)
        
        # Publish end goal if available
        if self.endgoal is not None:
            self.pub_endgoal.publish(self.endgoal)

    def draw_waypoints(self, img, waypoints, left_boundary):
        """
        Draw waypoints and lane boundaries on image for visualization.
        
        Args:
            img: Image to draw on
            waypoints: Path message containing waypoints
            left_boundary: List of points representing left lane boundary
            
        Returns:
            Annotated image
        """
        # Draw waypoints as yellow circles
        for pose in waypoints.poses:
            x, y = int(pose.pose.position.x), int(pose.pose.position.y)
            cv2.circle(img, (x, y), radius=5, color=(0, 255, 255), thickness=-1)
        
        # Draw left boundary as blue circles
        for lb in left_boundary:
            if lb is not None:
                cv2.circle(img, (lb[0], lb[1]), radius=3, color=(255, 0, 0), thickness=-1)
        
        # Draw end goal as red circle with label
        if self.endgoal is not None:
            ex = int(self.endgoal.pose.position.x)
            ey = int(self.endgoal.pose.position.y)
            cv2.circle(img, (ex, ey), radius=10, color=(0, 0, 255), thickness=-1)
            cv2.putText(img, "Endgoal", (ex + 15, ey), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
        return img

    def letterbox(self, img, new_shape=(384, 640), color=(114, 114, 114)):
        """
        Resize image with letterboxing to maintain aspect ratio.
        
        This is important for neural network input to prevent distortion.
        
        Args:
            img: Input image
            new_shape: Target shape (height, width)
            color: Padding color
            
        Returns:
            resized_img: Resized and padded image
            ratio: Scale ratio (used for inverse mapping)
            padding: Padding values (dw, dh)
        """
        # Original shape
        shape = img.shape[:2]
        
        # Handle single dimension input
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        
        # Calculate scale ratio
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        
        # Calculate new unpadded dimensions
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        
        # Calculate padding
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        
        # Resize image
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        # Add padding
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        
        return img, ratio, (dw, dh)

###############################################################################
# Neural Network Output Processing Functions
###############################################################################

def driving_area_mask(seg):
    """
    Extract drivable area mask from segmentation output.
    
    Args:
        seg: Segmentation output tensor from neural network
        
    Returns:
        Binary mask of drivable area
    """
    # Handle different tensor shapes
    if len(seg.shape) == 4:
        # Batch of images
        da_predict = seg[:, :, 12:372, :]
    elif len(seg.shape) == 3:
        # Single image
        seg = seg.unsqueeze(0)  # Add batch dimension
        da_predict = seg[:, :, 12:372, :]
    else:
        raise ValueError(f"Unexpected tensor shape in driving_area_mask: {seg.shape}")
    
    # Upscale mask to original resolution
    da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=2, mode='bilinear', align_corners=False)
    
    # Convert to binary mask (argmax across channels)
    _, da_seg_mask = torch.max(da_seg_mask, 1)
    
    # Convert to numpy array
    da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
    
    return da_seg_mask

def lane_line_mask(ll, threshold):
    """
    Extract lane line mask from lane detection output.
    
    Args:
        ll: Lane line detection tensor from neural network
        threshold: Confidence threshold for lane detection
        
    Returns:
        Binary mask of lane lines
    """
    # Handle different tensor shapes
    if len(ll.shape) == 4:
        # Batch of images
        ll_predict = ll[:, :, 12:372, :]
    elif len(ll.shape) == 3:
        # Single image
        ll = ll.unsqueeze(0)  # Add batch dimension
        ll_predict = ll[:, :, 12:372, :]
    else:
        raise ValueError(f"Unexpected tensor shape in lane_line_mask: {ll.shape}")
    
    # Upscale mask to original resolution
    ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=2, mode='bilinear', align_corners=False)
    
    # Apply threshold to get binary mask
    ll_seg_mask = (ll_seg_mask > threshold).int().squeeze(1)
    
    # Convert to numpy array
    ll_seg_mask = ll_seg_mask.squeeze().cpu().numpy().astype(np.uint8)
    
    # Dilate mask to fill gaps and strengthen lane line detection
    kernel = np.ones((2, 2), np.uint8)
    ll_seg_mask = cv2.dilate(ll_seg_mask, kernel, iterations=1)
    
    return ll_seg_mask

###############################################################################
# Main Entry Point
###############################################################################

if __name__ == "__main__":
    try:
        # Create detector instance
        detector = LaneNetDetector()
        
        # Keep node running until shutdown
        rospy.spin()
    except rospy.ROSInterruptException:
        pass