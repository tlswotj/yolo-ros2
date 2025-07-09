import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header, Float32MultiArray, MultiArrayLayout, MultiArrayDimension
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2

class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector_node')

        self.model = YOLO('best.pt')  # 또는 'custom.pt'
        self.bridge = CvBridge()

        # 이미지 구독
        self.sub_img = self.create_subscription(
            Image,
            '/camera/image',  # 이미지 토픽 이름
            self.img_callback,
            10
        )

        # 탐지 결과 발행
        self.pub_result = self.create_publisher(
            Float32MultiArray,
            '/detection_result',  # 결과 토픽 이름
            10
        )

        self.get_logger().info("YOLOv8 detector node started!")

    def img_callback(self, msg = Image()):
        # ROS 이미지 → OpenCV 이미지
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # YOLO 객체 탐지 수행
        results = self.model.predict(frame, imgsz=640, conf=0.5)

        detection_result = Float32MultiArray()

        for result in results:
            dim = MultiArrayDimension()
            dim.size = 5
            dim.stride = len(result.boxes)
            detection_result.layout.dim.append(dim)

            for box in result.boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = float(box.cls[0])
                detection_result.data.append(cls_id)
                detection_result.data.append((x1+x2)/2)
                detection_result.data.append((y1+y2)/2)
                detection_result.data.append(x2-x1)
                detection_result.data.append(y2+y1)

        # 탐지 결과 발행
        self.pub_result.publish(detection_result)
        self.get_logger().info(f"Published detection result (seq: {msg.header.stamp.nanosec})")

def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()