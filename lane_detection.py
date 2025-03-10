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

class LaneNetDetector:
    """
    Neural network-based lane detection and stop sign recognition system.
    Uses YOLOPv2 for lane detection and YOLOv5 for stop sign detection.
    Implements a complete perception pipeline including image preprocessing,
    lane detection, and stop sign recognition.
    """
    def __init__(self):
        # Frame buffer for batch processing
        self.frame_buffer = []
        self.buffer_size = 4  # Process 4 frames at once for efficiency
        
        rospy.init_node('lane_detection_node', anonymous=True)
        
        # OpenCV bridge for ROS image conversion
        self.bridge = CvBridge()
        
        # Lane tracking state variables
        self.prev_left_boundary = None  # Previous frame's left lane boundary
        self.estimated_lane_width_pixels = 200  # Expected lane width in image space
        self.prev_waypoints = None  # Previous frame's waypoints for smoothing
        self.endgoal = None  # Target point for vehicle control
        
        # Neural network setup
        # Use GPU if available for faster inference
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Load YOLOPv2 model for lane detection
        self.model = torch.jit.load('weights/yolopv2.pt')
        # Enable half-precision for GPU inference
        self.half = self.device != 'cpu'
        
        if self.half:
            self.model.half()
            
        self.model.to(self.device).eval()
        # Load YOLOv5 model for stop sign detection
        self.stop_model = torch.hub.load('weights/yolov5', 'yolov5n', pretrained=True, trust_repo=True)
        self.stop_model.to(self.device).eval()
        
        # Stop sign detection parameters
        self.Focal_Length = 800  # Camera focal length in pixels
        self.Real_Height_SS = .75  # Real-world stop sign height in meters
        self.Brake_Distance = 5  # Distance threshold for braking (meters)
        self.Brake_Duration = 3  # Duration to maintain stop (seconds)
        self.stop_flag = False  # Current stop state
        self.stop_start_time = None  # Timestamp of stop initiation
        
        # ROS Topics Setup
        # Subscribers
        self.sub_image = rospy.Subscriber('oak/rgb/image_raw', Image, self.img_callback, queue_size=1)
        
        # Publishers for visualization and control
        self._setup_publishers()

    def _setup_publishers(self):
        """Initialize all ROS publishers with appropriate topics and message types"""
        self.pub_contrasted_image = rospy.Publisher("lane_detection/contrasted_image", Image, queue_size=1)
        self.pub_annotated = rospy.Publisher("lane_detection/annotate", Image, queue_size=1)
        self.pub_waypoints = rospy.Publisher("lane_detection/waypoints", Path, queue_size=1)
        self.pub_stop_signal = rospy.Publisher("stop_signal/signal", Bool, queue_size=1)
        self.pub_endgoal = rospy.Publisher("lane_detection/endgoal", PoseStamped, queue_size=1)
        self.pub_stop_annotated = rospy.Publisher("stop_signal/stop_sign_annotate", Image, queue_size=1)

    def img_callback(self, img):
        """
        Main image processing pipeline callback
        Implements the following steps:
        1. Image preprocessing (color filtering, contrast enhancement)
        2. Neural network inference for lane detection
        3. Stop sign detection
        4. Waypoint generation
        5. Visualization
        """
        try:
            # Convert ROS image to OpenCV format
            img = self.bridge.imgmsg_to_cv2(img, "bgr8")
            
            # Color-based lane marking enhancement
            # Convert to HSV for better color segmentation
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Yellow lane marking detection parameters
            lower_yellow = np.array([20, 100, 100])  # HSV lower bound
            upper_yellow = np.array([30, 255, 255])  # HSV upper bound
            
            # Create mask for yellow lane markings
            yellow_mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
            non_yellow_mask = cv2.bitwise_not(yellow_mask)
            img_no_yellow = cv2.bitwise_and(img, img, mask=non_yellow_mask)
            
            # Convert to grayscale for general lane marking detection
            img_gray = cv2.cvtColor(img_no_yellow, cv2.COLOR_BGR2GRAY)
            
            # Contrast enhancement
            threshold = 180  # Brightness threshold
            mask = img_gray >= threshold
            dimmed_gray = (img_gray * 0.5).astype(np.uint8)  # Reduce intensity of non-lane regions
            dimmed_gray[mask] = img_gray[mask]  # Preserve bright regions (potential lane markings)
            contrasted_img = cv2.cvtColor(dimmed_gray, cv2.COLOR_GRAY2BGR)
            
            # Publish enhanced image for visualization
            contrasted_image_msg = self.bridge.cv2_to_imgmsg(contrasted_img, "bgr8")
            self.pub_contrasted_image.publish(contrasted_image_msg)
            
            # Prepare image for neural network inference
            img_tensor = self.preprocess_frame(contrasted_img)
            self.frame_buffer.append((contrasted_img, img_tensor))
            
            # Batch processing when buffer is full
            if len(self.frame_buffer) >= self.buffer_size:
                self._process_batch()
                    
        except CvBridgeError as e:
            print(e)

    def _process_batch(self):
        """Process a batch of frames through the neural networks"""
        original_images, tensors = zip(*self.frame_buffer)
        batch = torch.stack(tensors).to(self.device)
        self.frame_buffer.clear()
        
        # Neural network inference
        with torch.no_grad():
            [pred, anchor_grid], seg, ll = self.model(batch)
            
        # Process each frame's results
        for i, contrasted_img in enumerate(original_images):
            # Detect lanes and generate waypoints
            annotated_img = self.detect_lanes(seg[i], ll[i], contrasted_img)
            annotated_image_msg = self.bridge.cv2_to_imgmsg(annotated_img, "bgr8")
            self.pub_annotated.publish(annotated_image_msg)
            
        # Stop sign detection on most recent frame
        brake = self.detect_stop(original_images[0])
        if brake:
            print("Stop sign detected. Stopping...")
    def preprocess_frame(self, img):
        """
        Preprocess an image frame for neural network inference.
        
        Args:
            img: Input image in BGR format
            
        Returns:
            torch.Tensor: Preprocessed image tensor ready for model input
        """
        # Resize image to model input dimensions while maintaining aspect ratio
        img_resized, _, _ = self.letterbox(img, new_shape=(384, 640))
        
        # Convert BGR to RGB and change dimensions to (C,H,W) format
        img_resized = img_resized[:, :, ::-1].transpose(2, 0, 1)
        
        # Convert to contiguous numpy array and then to torch tensor
        img_tensor = torch.from_numpy(np.ascontiguousarray(img_resized))
        
        # Convert to half precision if model uses half precision, otherwise use float
        img_tensor = img_tensor.half() if self.half else img_tensor.float()
        
        # Normalize pixel values to [0,1] range
        img_tensor /= 255.0
        
        return img_tensor
    
    def detect_stop(self, img):
        """
        Detect stop signs in an image and determine if vehicle should brake.
        
        Uses object detection to find stop signs, estimates their distance based on apparent size,
        and determines if they are facing the vehicle based on aspect ratio. Manages stop sign 
        state and publishes stop signals.
        
        Args:
            img: Input image in BGR format
            
        Returns:
            bool: True if vehicle should brake for stop sign, False otherwise
        """
        brake = False
        stop_signs = self.detect_objects(img, 'stop_sign')
        
        if stop_signs:
            for (xcenter, ycenter, width, height) in stop_signs:
                # Calculate bounding box coordinates
                xmin = int(xcenter - width / 2)
                ymin = int(ycenter - height / 2) 
                xmax = int(xcenter + width / 2)
                ymax = int(ycenter + height / 2)
                
                if height > 0:
                    # Estimate distance using apparent height
                    estimated_distance = (self.Real_Height_SS * self.Focal_Length) / height
                    
                    # Check if stop sign is roughly square (facing vehicle)
                    aspect_ratio = width / height
                    facing_us = 0.7 < aspect_ratio < 1.3
                    
                    # Set brake flag if stop sign is close enough and facing us
                    if estimated_distance <= self.Brake_Distance and facing_us:
                        brake = True
                        
        if brake and not self.stop_flag:
            # Start new stop sequence
            self.stop_flag = True
            self.stop_start_time = time.time()
            
            # Publish annotated image and stop signal
            detected_image_msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
            self.pub_stop_annotated.publish(detected_image_msg)
            stop_msg = Bool()
            stop_msg.data = True
            self.pub_stop_signal.publish(stop_msg)
            
        elif not brake and self.stop_flag:
            # Check if minimum stop duration has elapsed
            stopped_time = time.time() - self.stop_start_time
            if stopped_time > self.Brake_Duration:
                # End stop sequence
                self.stop_flag = False
                stop_msg = Bool()
                stop_msg.data = False
                self.pub_stop_signal.publish(stop_msg)
                
        return brake
    def image_to_world(self, u, v, camera_matrix, camera_height):
        """
        Convert image coordinates to world coordinates using pinhole camera model.
        
        Args:
            u (float): Image x-coordinate in pixels
            v (float): Image y-coordinate in pixels  
            camera_matrix (np.ndarray): 3x3 camera intrinsic matrix
            camera_height (float): Height of camera above ground plane in meters
            
        Returns:
            tuple: (X, Y, Z) world coordinates in meters
                  X: Forward distance
                  Y: Lateral distance 
                  Z: Height (negative since camera points down)
        """
        # Extract camera intrinsic parameters
        fx = camera_matrix[0, 0]  # Focal length x
        fy = camera_matrix[1, 1]  # Focal length y
        cx = camera_matrix[0, 2]  # Principal point x
        cy = camera_matrix[1, 2]  # Principal point y
        
        # Set Z as negative camera height since camera points down
        Z = -camera_height
        
        # Project image coordinates to world using pinhole model
        X = Z * (u - cx) / fx  # Forward distance
        Y = Z * (v - cy) / fy  # Lateral distance
        
        return X, Y, Z

    def detect_lanes(self, seg, ll, img):
        """
        Detect lanes using segmentation masks and generate waypoints
        
        Args:
            seg: Segmentation output from neural network
            ll: Lane line detection output from neural network 
            img: Original RGB image
            
        Returns:
            Image with visualized waypoints overlaid
        """
        # Generate drivable area and lane line masks
        da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll, threshold=0.2)
        
        # Generate waypoints from lane line mask
        waypoints, left_boundary = self.generate_waypoints(ll_seg_mask)
        
        # Draw and publish waypoints
        img_with_waypoints = self.draw_waypoints(img.copy(), waypoints, left_boundary)
        self.publish_waypoints(waypoints)
        
        return img_with_waypoints

    def region_of_interest(self, img):
        """
        Create a mask to focus on the relevant region of the image
        
        Args:
            img: Input image to mask
            
        Returns:
            Masked image showing only left half of frame
        """
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
        Generate waypoints for vehicle path from lane mask image.
        
        Args:
            lane_mask: Binary image with detected lane markings
            
        Returns:
            tuple: (path, left_boundary)
                path: Path message containing waypoint poses
                left_boundary: List of detected left boundary points
        """
        # Initialize path message and variables
        path = Path()
        path.header.frame_id = "map"
        height, width = lane_mask.shape
        sampling_step = 10  # Step size for sampling points along y-axis
        left_boundary = []
        offset_pixels = 200  # Offset from left boundary to lane center
        
        # Sample points along image y-axis from bottom to top
        for y in range(height - 1, 0, -sampling_step):
            x_indices = np.where(lane_mask[y, :] > 0)[0]
            if len(x_indices) > 0:
                x_left = x_indices[0]  # Leftmost lane marking point
                left_boundary.append((x_left - 40, y))  # Add point with margin
            else:
                left_boundary.append(None)  # No lane marking detected
                
        # Filter boundary points to remove discontinuities
        left_boundary = self.filter_continuous_boundary(left_boundary)
        
        # Generate path poses from filtered boundary points
        for lb in left_boundary:
            if lb:
                x_center = lb[0] + offset_pixels  # Estimate lane center
                y = lb[1]
                point = PoseStamped()
                point.pose.position.x = x_center
                point.pose.position.y = y
                path.poses.append(point)
                
        # Limit number of waypoints for efficiency
        path.poses = path.poses[:7]
        
        # Calculate endgoal as median of waypoints
        if len(path.poses) > 0:
            xs = [p.pose.position.x for p in path.poses]
            ys = [p.pose.position.y for p in path.poses]
            median_x = np.median(xs)
            median_y = np.median(ys)
            self.endgoal = PoseStamped()
            self.endgoal.header = path.header
            self.endgoal.pose.position.x = median_x
            self.endgoal.pose.position.y = median_y
        else:
            self.endgoal = None
            
        return path, left_boundary

    def filter_continuous_boundary(self, boundary):
        """
        Filter boundary points to ensure smooth lane detection by removing discontinuities.
        
        Args:
            boundary: List of (x,y) coordinate tuples representing detected lane boundary points.
                     Points may be None if no boundary was detected at that y-coordinate.
                     
        Returns:
            continuous_boundary: Filtered list of boundary points with discontinuities removed.
                               Points that create gaps larger than max_gap are replaced with None.
        """
        max_gap = 60  # Maximum allowed x-distance between consecutive boundary points
        continuous_boundary = []  # Output list of filtered boundary points
        previous_point = None     # Track previous valid point for gap checking
        
        for point in boundary:
            if point is not None:
                # Check if point creates acceptable gap with previous point
                if previous_point is None or abs(point[0] - previous_point[0]) <= max_gap:
                    continuous_boundary.append(point)  # Add point if gap is acceptable
                    previous_point = point
                else:
                    continuous_boundary.append(None)   # Replace point with None if gap too large
                    previous_point = None  # Reset previous point tracking
            else:
                continuous_boundary.append(None)  # Maintain None values from input
                previous_point = None  # Reset previous point tracking
                
        return continuous_boundary

    def publish_waypoints(self, waypoints):
        """Publish waypoints and endgoal to ROS topics"""
        self.pub_waypoints.publish(waypoints)
        if self.endgoal is not None:
            self.pub_endgoal.publish(self.endgoal)

    def draw_waypoints(self, img, waypoints, left_boundary):
        """
        Draw detected waypoints, boundaries and endgoal on the image
        
        Args:
            img: Input image to draw on
            waypoints: Path message containing waypoint poses
            left_boundary: List of detected left boundary points
            
        Returns:
            img: Image with visualizations drawn
        """
        # Draw waypoints as yellow circles
        for pose in waypoints.poses:
            x, y = int(pose.pose.position.x), int(pose.pose.position.y)
            cv2.circle(img, (x, y), radius=5, color=(0, 255, 255), thickness=-1)
            
        # Draw left boundary points as blue circles
        for lb in left_boundary:
            if lb is not None:
                cv2.circle(img, (lb[0], lb[1]), radius=3, color=(255, 0, 0), thickness=-1)
                
        # Draw endgoal point as red circle with label
        if self.endgoal is not None:
            ex = int(self.endgoal.pose.position.x)
            ey = int(self.endgoal.pose.position.y)
            cv2.circle(img, (ex, ey), radius=10, color=(0, 0, 255), thickness=-1)
            cv2.putText(img, "Endgoal", (ex + 15, ey), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
        return img
    
    
    def letterbox(self, img, new_shape=(384, 640), color=(114, 114, 114)):
        """
        Resize and pad image to target shape while maintaining aspect ratio.
        
        Args:
            img: Input image array
            new_shape: Target shape as (height, width) tuple or single int
            color: Border color as RGB tuple
            
        Returns:
            img: Resized and padded image
            ratio: Scale ratio (width_ratio, height_ratio)
            (dw, dh): Padding amounts for width and height
        """
        # Get original image dimensions
        shape = img.shape[:2]
        
        # Handle case where new_shape is single int
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
            
        # Calculate scale ratio to maintain aspect ratio
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        
        # Calculate new unpadded dimensions
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        
        # Calculate padding on each side
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        
        # Resize image if dimensions don't match
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        # Calculate padding coordinates
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        # Add padding
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        
        return img, ratio, (dw, dh)
    
    
def driving_area_mask(seg):
    """
    Generate a binary mask for the drivable area from segmentation output.
    
    Args:
        seg (torch.Tensor): Segmentation tensor of shape [B,C,H,W] or [C,H,W]
        
    Returns:
        numpy.ndarray: Binary mask indicating drivable area
    """
    # Handle both batched and unbatched inputs
    if len(seg.shape) == 3:
        seg = seg.unsqueeze(0)  # Add batch dimension
    elif len(seg.shape) != 4:
        raise ValueError(f"Unexpected tensor shape in driving_area_mask: {seg.shape}")
        
    # Extract and crop relevant segmentation region
    da_predict = seg[:, :, 12:372, :]  # Crop to road region
    
    # Upsample prediction by 2x using bilinear interpolation
    da_seg_mask = torch.nn.functional.interpolate(
        da_predict, 
        scale_factor=2,
        mode='bilinear',
        align_corners=False
    )
    
    # Convert to binary mask
    _, da_seg_mask = torch.max(da_seg_mask, 1)  # Get class with highest probability
    da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
    
    return da_seg_mask

def lane_line_mask(ll, threshold):
    """
    Generate a binary mask for lane lines from model output.
    
    Args:
        ll (torch.Tensor): Lane line tensor of shape [B,C,H,W] or [C,H,W]
        threshold (float): Confidence threshold for lane line detection
        
    Returns:
        numpy.ndarray: Binary mask indicating lane lines
    """
    # Handle both batched and unbatched inputs
    if len(ll.shape) == 3:
        ll = ll.unsqueeze(0)  # Add batch dimension
    elif len(ll.shape) != 4:
        raise ValueError(f"Unexpected tensor shape in lane_line_mask: {ll.shape}")
        
    # Extract and crop relevant region
    ll_predict = ll[:, :, 12:372, :]  # Crop to road region
    
    # Upsample prediction by 2x using bilinear interpolation
    ll_seg_mask = torch.nn.functional.interpolate(
        ll_predict,
        scale_factor=2, 
        mode='bilinear',
        align_corners=False
    )
    
    # Convert to binary mask using threshold
    ll_seg_mask = (ll_seg_mask > threshold).int().squeeze(1)
    ll_seg_mask = ll_seg_mask.squeeze().cpu().numpy().astype(np.uint8)
    
    # Dilate mask slightly to connect nearby points
    kernel = np.ones((2, 2), np.uint8)
    ll_seg_mask = cv2.dilate(ll_seg_mask, kernel, iterations=1)
    
    return ll_seg_mask

if __name__ == "__main__":
    try:
        detector = LaneNetDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
