import cv2
import rospy
import sensor_msgs.point_cloud2 as pcl2
import time
from numba import jit
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import PointCloud2
import message_filters

import torch

from loss import CustomLoss
from utils import get_model_name, load_config, get_logger, plot_bev, plot_label_map, plot_pr_curve, get_bev
from model import PIXOR
from datagen import KITTI
from postprocess import filter_pred, compute_matches, compute_ap

def build_model(config, device, train=True):
    net = PIXOR(config['geometry'], config['use_bn'])
    loss_fn = CustomLoss(device, config, num_classes=1)

    if torch.cuda.device_count() <= 1:
        config['mGPUs'] = False
    if config['mGPUs']:
        print("using multi gpu")
        net = nn.DataParallel(net)

    net = net.to(device)
    loss_fn = loss_fn.to(device)
    if not train:
        return net, loss_fn

    optimizer = torch.optim.SGD(net.parameters(), lr=config['learning_rate'], momentum=config['momentum'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['lr_decay_at'], gamma=0.1)

    return net, loss_fn, optimizer, scheduler


class synchronizer:
    def __init__(self):
        # self.pub_Image = rospy.Publisher('image_raw_sync', SesnorImage, queue_size=1)
        # self.pub_Cam_Info = rospy.Publisher('camera_info_sync', CameraInfo, queue_size=1)
        # self.pub_Lidar = rospy.Publisher('rslidar_points_sync', PointCloud2, queue_size=1)
        config, _, _, _ = load_config("default")
        self.net, _ = build_model(config, "cuda", train=False)
        self.net.load_state_dict(torch.load(get_model_name(config), map_location="cuda"))
        self.net.set_decode(True)
        self.net.eval()

        self.dataset = KITTI(1000)
        self.dataset.load_velo()


        self.imageInput = message_filters.Subscriber("/pointgrey/image_raw", Image)
        self.lidar = message_filters.Subscriber('/velo/pointcloud', PointCloud2)

        self.ts = message_filters.TimeSynchronizer([self.imageInput
                                                    #,self.cameraInfo
                                                    , self.lidar
                                                    ], 10)
        self.ts.registerCallback(self.general_callback)

        self._image_raw = Image()
        self._camera_info = CameraInfo()
        self._lidar_points = PointCloud2()

        
    def general_callback(self, image_raw, lidar_points):
        self._image_raw = image_raw
        self._lidar_points = lidar_points
        # print (msg)
        date = time.time()
        points = pcl2.read_points_list(self._lidar_points)
        print ('1: ', time.time()-date)
        points = [[point.x, point.y, point.z, 1] for point in points]
        points = np.array(points)
        print ('2: ', time.time()-date)
        cv_image = CvBridge().imgmsg_to_cv2(self._image_raw, "bgr8")
        print ('3: ', time.time()-date)
        print ("points: ", type(points))
        print ("cv_image: ", type(cv_image))
        # TODO
        cv_image = np.resize(cv_image,(370,1240,3))
        self.one_test(cv_image, points)

        print ('4: ', time.time()-date)

    
    def one_test(self, image_raw, lidar_points):
        
        device = "cuda"
        bev, image, bev2image, pc_diff = self.dataset.raw_to_tensor(lidar_points, image_raw)
        bev = bev.to(device)
        image = image.to(device)
        bev2image = bev2image.to(device)
        pc_diff = pc_diff.to(device)
        print ("bev: ",bev.shape)
        print ("image: ",image.shape)
        print ("bev2image: ",bev2image.shape)
        print ("pc_diff: ",pc_diff.shape)
        # bev = torch.ones(33,512,448).cuda()
        # image = torch.ones(3,370,1240).cuda()
        # bev2image = torch.ones(256, 224, 16, 1, 2).cuda()
        # pc_diff = torch.ones(256, 224, 16, 1, 3).cuda()

        pred = self.net(bev.unsqueeze(0), image.unsqueeze(0), bev2image.unsqueeze(0), pc_diff.unsqueeze(0))

        # print (pred)


    # def publisher(self):
    #     while True:
    #         self.pub_Image.publish(self._image_raw)
    #         self.pub_Cam_Info.publish(self._camera_info)
    #         # self.pub_Lidar.publish(self._rslidar_points)


if __name__ == '__main__':
    rospy.init_node('synchronizer')
    synchronizer = synchronizer()
    # synchronizer.publisher()
    rospy.spin()