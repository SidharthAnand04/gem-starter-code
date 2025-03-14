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
    def __init__(self):
        self.frame_buffer = []
        self.buffer_size = 4
        
        rospy.init_node('lane_detection_node', anonymous=True)
        
        self.bridge = CvBridge()
        self.prev_left_boundary = None
        self.estimated_lane_width_pixels = 200
        self.prev_waypoints = None
        self.endgoal = None
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = torch.jit.load('weights/yolopv2.pt')
        self.half = self.device != 'cpu'
        
        if self.half:
            self.model.half()
            
        self.model.to(self.device).eval()
        
        # Download the object detection model from the Ultralytics YOLOv5 repository
        # self.stop_model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True, trust_repo=True)
        # self.stop_model.to(self.device).eval()
        
        self.Focal_Length = 800
        self.Real_Height_SS = .75
        self.Brake_Distance = 5
        self.Brake_Duration = 3
        
        self.sub_image = rospy.Subscriber('oak/rgb/image_raw', Image, self.img_callback, queue_size=1)
        
        self.pub_contrasted_image = rospy.Publisher("lane_detection/contrasted_image", Image, queue_size=1)
        self.pub_annotated = rospy.Publisher("lane_detection/annotate", Image, queue_size=1)
        self.pub_waypoints = rospy.Publisher("lane_detection/waypoints", Path, queue_size=1)
        self.pub_endgoal = rospy.Publisher("lane_detection/endgoal", PoseStamped, queue_size=1)

    def img_callback(self, img):
        try:
            img = self.bridge.imgmsg_to_cv2(img, "bgr8")
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([30, 255, 255])
            
            yellow_mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
            non_yellow_mask = cv2.bitwise_not(yellow_mask)
            img_no_yellow = cv2.bitwise_and(img, img, mask=non_yellow_mask)
            img_gray = cv2.cvtColor(img_no_yellow, cv2.COLOR_BGR2GRAY)
            
            threshold = 180
            mask = img_gray >= threshold
            dimmed_gray = (img_gray * 0.5).astype(np.uint8)
            dimmed_gray[mask] = img_gray[mask]
            contrasted_img = cv2.cvtColor(dimmed_gray, cv2.COLOR_GRAY2BGR)
            contrasted_image_msg = self.bridge.cv2_to_imgmsg(contrasted_img, "bgr8")
            
            self.pub_contrasted_image.publish(contrasted_image_msg)
            img_tensor = self.preprocess_frame(contrasted_img)
            self.frame_buffer.append((contrasted_img, img_tensor))
            
            if len(self.frame_buffer) >= self.buffer_size:
                original_images, tensors = zip(*self.frame_buffer)
                batch = torch.stack(tensors).to(self.device)
                self.frame_buffer.clear()
                with torch.no_grad():
                    [pred, anchor_grid], seg, ll = self.model(batch)
                for i, contrasted_img in enumerate(original_images):
                    annotated_img = self.detect_lanes(seg[i], ll[i], contrasted_img)
                    annotated_image_msg = self.bridge.cv2_to_imgmsg(annotated_img, "bgr8")
                    self.pub_annotated.publish(annotated_image_msg)
                    
        except CvBridgeError as e:
            print(e)

    def preprocess_frame(self, img):
        img_resized, _, _ = self.letterbox(img, new_shape=(384, 640))
        img_resized = img_resized[:, :, ::-1].transpose(2, 0, 1)
        img_tensor = torch.from_numpy(np.ascontiguousarray(img_resized))
        img_tensor = img_tensor.half() if self.half else img_tensor.float()
        img_tensor /= 255.0
        return img_tensor

    def image_to_world(self, u, v, camera_matrix, camera_height):
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        Z = -camera_height
        X = Z * (u - cx) / fx
        Y = Z * (v - cy) / fy
        return X, Y, Z

    def detect_lanes(self, seg, ll, img):
        da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll, threshold=0.2)
        waypoints, left_boundary = self.generate_waypoints(ll_seg_mask)
        img_with_waypoints = self.draw_waypoints(img.copy(), waypoints, left_boundary)
        self.publish_waypoints(waypoints)
        return img_with_waypoints

    def region_of_interest(self, img):
        height, width = img.shape[:2]
        mask = np.zeros_like(img)
        polygon = np.array([[(0, height), (0, 0), (width // 2, 0), (width // 2, height)]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def generate_waypoints(self, lane_mask):
        path = Path()
        path.header.frame_id = "map"
        height, width = lane_mask.shape
        sampling_step = 10
        left_boundary = []
        offset_pixels = 200
        
        for y in range(height - 1, 0, -sampling_step):
            x_indices = np.where(lane_mask[y, :] > 0)[0]
            if len(x_indices) > 0:
                x_left = x_indices[0]
                left_boundary.append((x_left - 40, y))
            else:
                left_boundary.append(None)
                
        left_boundary = self.filter_continuous_boundary(left_boundary)
        for lb in left_boundary:
            if lb:
                x_center = lb[0] + offset_pixels
                y = lb[1]
                point = PoseStamped()
                point.pose.position.x = x_center
                point.pose.position.y = y
                path.poses.append(point)
                
        path.poses = path.poses[:7]
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
        max_gap = 60
        continuous_boundary = []
        previous_point = None
        for point in boundary:
            if point is not None:
                if previous_point is None or abs(point[0] - previous_point[0]) <= max_gap:
                    continuous_boundary.append(point)
                    previous_point = point
                else:
                    continuous_boundary.append(None)
                    previous_point = None
            else:
                continuous_boundary.append(None)
                previous_point = None
        return continuous_boundary

    def publish_waypoints(self, waypoints):
        self.pub_waypoints.publish(waypoints)
        if self.endgoal is not None:
            self.pub_endgoal.publish(self.endgoal)

    def draw_waypoints(self, img, waypoints, left_boundary):
        for pose in waypoints.poses:
            x, y = int(pose.pose.position.x), int(pose.pose.position.y)
            cv2.circle(img, (x, y), radius=5, color=(0, 255, 255), thickness=-1)
        for lb in left_boundary:
            if lb is not None:
                cv2.circle(img, (lb[0], lb[1]), radius=3, color=(255, 0, 0), thickness=-1)
        if self.endgoal is not None:
            ex = int(self.endgoal.pose.position.x)
            ey = int(self.endgoal.pose.position.y)
            cv2.circle(img, (ex, ey), radius=10, color=(0, 0, 255), thickness=-1)
            cv2.putText(img, "Endgoal", (ex + 15, ey), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return img

    def letterbox(self, img, new_shape=(384, 640), color=(114, 114, 114)):
        shape = img.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, ratio, (dw, dh)

def driving_area_mask(seg):
    if len(seg.shape) == 4:
        da_predict = seg[:, :, 12:372, :]
    elif len(seg.shape) == 3:
        seg = seg.unsqueeze(0)
        da_predict = seg[:, :, 12:372, :]
    else:
        raise ValueError(f"Unexpected tensor shape in driving_area_mask: {seg.shape}")
    da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=2, mode='bilinear', align_corners=False)
    _, da_seg_mask = torch.max(da_seg_mask, 1)
    da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
    return da_seg_mask

def lane_line_mask(ll, threshold):
    if len(ll.shape) == 4:
        ll_predict = ll[:, :, 12:372, :]
    elif len(ll.shape) == 3:
        ll = ll.unsqueeze(0)
        ll_predict = ll[:, :, 12:372, :]
    else:
        raise ValueError(f"Unexpected tensor shape in lane_line_mask: {ll.shape}")
    ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=2, mode='bilinear', align_corners=False)
    ll_seg_mask = (ll_seg_mask > threshold).int().squeeze(1)
    ll_seg_mask = ll_seg_mask.squeeze().cpu().numpy().astype(np.uint8)
    kernel = np.ones((2, 2), np.uint8)
    ll_seg_mask = cv2.dilate(ll_seg_mask, kernel, iterations=1)
    return ll_seg_mask

if __name__ == "__main__":
    try:
        detector = LaneNetDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
