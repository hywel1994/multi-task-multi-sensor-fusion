'''
Load pointcloud/labels from the KITTI dataset folder
'''
import os.path
import numpy as np
import time
import math
import cv2
import ctypes

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils import plot_bev, get_points_in_a_rotated_box, plot_label_map, trasform_label2metric
from utils2d import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian, draw_gaussian
import calibration as calibration

#KITTI_PATH = '/home/hqxie/0249/Data/kitti'
KITTI_PATH = '/home/hywel/DataDisk/Kitti/object'
# KITTI_PATH = '/home/hywel/DataDisk/lidar-image-gps-ahrs'
#KITTI_PATH = 'KITTI'

class KITTI(Dataset):

    geometry = {
        'L1': -40.0,
        'L2': 40.0,
        'W1': 0.0,
        'W2': 70.0,
        'H1': -2.5,
        'H2': 1.1,
        'interval': 0.1,
        'image_input_shape': (384, 1248),
        'image_label_shape': (96, 312),
        'lidar_input_shape': (800, 700, 37),
        'knn_shape': (400, 350, 18),
        'lidar_label_shape': (200, 175, 7)
    }

    # height, width
    target_mean = np.array([0.008, 0.001, 0.202, 0.2, 0.43, 1.368])
    target_std_dev = np.array([0.866, 0.5, 0.954, 0.668, 0.09, 0.111])


    def __init__(self, frame_range = 10000, use_npy=False, train=True):
        self.frame_range = frame_range
        self.velo = []
        self.use_npy = use_npy
        self.LidarLib = ctypes.cdll.LoadLibrary('preprocess/LidarPreprocess.so')
        self.image_sets = self.load_imageset(train) # names
        self.image = []
        self.calib = []
        self.max_objs = 50

    def __len__(self):
        return len(self.image_sets)

    def __getitem__(self, item):
        image = self.load_image(item)
        calib = self.load_calib(item)
        point = self.load_velo_origin(item)
        scan = self.load_velo_scan(item)
        bev2image, pc_diff = self.find_knn_image(calib, scan, point, k=1)

        label_map, _ , image_label= self.get_label(item)
        self.reg_target_transform(label_map)
    
        image = torch.from_numpy(image)
        scan = torch.from_numpy(scan)
        bev2image = torch.from_numpy(bev2image)
        label_map = torch.from_numpy(label_map)
        pc_diff = torch.from_numpy(pc_diff)
        image = image.float()
        bev2image = bev2image.float()
        pc_diff = pc_diff.float()
        image = image.permute(2, 0, 1)
        scan = scan.permute(2, 0, 1)
        label_map = label_map.permute(2, 0, 1)

        for key in image_label.keys():
            image_label[key] = torch.from_numpy(image_label[key])
            image_label[key] = image_label[key].float()

        return scan, image, bev2image, pc_diff, label_map, image_label, item

    def reg_target_transform(self, label_map):
        '''
        Inputs are numpy arrays (not tensors!)
        :param label_map: [200 * 175 * 7] label tensor
        :return: normalized regression map for all non_zero classification locations
        '''
        cls_map = label_map[..., 0]
        reg_map = label_map[..., 1:]

        index = np.nonzero(cls_map)
        reg_map[index] = (reg_map[index] - self.target_mean)/self.target_std_dev


    def load_imageset(self, train):
        path = KITTI_PATH
        if train:
            path = os.path.join(path, "train.txt")
        else:
            path = os.path.join(path, "val.txt")

        with open(path, 'r') as f:
            lines = f.readlines() # get rid of \n symbol
            names = []
            for line in lines[:-1]:
                if int(line[:-1]) < self.frame_range:
                    names.append(line[:-1])

            # Last line does not have a \n symbol
            last = lines[-1][:6]
            if int(last) < self.frame_range:
                names.append(last)
            # print(names[-1])
            print("There are {} images in txt file".format(len(names)))

            return names
    
    def load_image(self, item):
        img_file = self.image[item]
        assert os.path.exists(img_file)
        image = cv2.imread(img_file)
        # TODO need check, image_width and image_height
        print ("before", image.size)
        image = cv2.resize(image, self.geometry['image_shape'], interpolation=cv2.INTER_CUBIC)
        print ("after", image.size)
        return image
    
    def load_calib(self, item):
        calib_file = self.calib[item]
        assert os.path.exists(calib_file)
        return calibration.Calibration(calib_file)
    
    # Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01
    def interpret_kitti_label(self, bbox):
        w, h, l, y, z, x, yaw = bbox[8:15]
        y = -y
        yaw = - (yaw + np.pi / 2)
        
        return x, y, w, l, yaw
    
    def interpret_custom_label(self, bbox):
        w, l, x, y, yaw = bbox
        return x, y, w, l, yaw

    def interpret_kitti_2D_label(self, bbox):
        y1, x1, y2, x2 = bbox[4:8]
        return y1, x1, y2, x2
    
    def get_corners_2d(self, bbox):
        y1, x1, y2, x2 = bbox[4:8]
        xc = (x1+x2)/2
        yc = (y1+y2)/2
        height = x2-x1
        width = y2-y1

        reg_target = [xc, yc, height, width]
        return reg_target

    def get_corners(self, bbox):
        # base velo cord
        w, h, l, y, z, x, yaw = bbox[8:15]
        y = -y
        # manually take a negative s. t. it's a right-hand system, with
        # x facing in the front windshield of the car
        # z facing up
        # y facing to the left of driver

        yaw = -(yaw + np.pi / 2)
        
        #x, y, w, l, yaw = self.interpret_kitti_label(bbox)
        
        bev_corners = np.zeros((4, 2), dtype=np.float32)
        # rear left
        bev_corners[0, 0] = x - l/2 * np.cos(yaw) - w/2 * np.sin(yaw)
        bev_corners[0, 1] = y - l/2 * np.sin(yaw) + w/2 * np.cos(yaw)

        # rear right
        bev_corners[1, 0] = x - l/2 * np.cos(yaw) + w/2 * np.sin(yaw)
        bev_corners[1, 1] = y - l/2 * np.sin(yaw) - w/2 * np.cos(yaw)

        # front right
        bev_corners[2, 0] = x + l/2 * np.cos(yaw) + w/2 * np.sin(yaw)
        bev_corners[2, 1] = y + l/2 * np.sin(yaw) - w/2 * np.cos(yaw)

        # front left
        bev_corners[3, 0] = x + l/2 * np.cos(yaw) - w/2 * np.sin(yaw)
        bev_corners[3, 1] = y + l/2 * np.sin(yaw) + w/2 * np.cos(yaw)

        reg_target = [np.cos(yaw), np.sin(yaw), x, y, w, l]

        return bev_corners, reg_target
    
    # def update_image_label_map(self, map, reg_target):
    #     label_x = reg_target[0]//4
    #     label_y = reg_target[1]//4
    #     map[label_x, label_y, 0] = 1.0
    #     reg_x = reg_target[0] - reg_target[0]//4*4
    #     reg_y = reg_target[1] - reg_target[1]//4*4
    #     map[label_x, label_y, 1:5] = [reg_target[2], reg_target[3], reg_x, reg_y]

    def update_label_map(self, map, bev_corners, reg_target):
        label_corners = (bev_corners / 4 ) / self.geometry['interval']
        label_corners[:, 1] += self.geometry['lidar_label_shape'][0] / 2

        points = get_points_in_a_rotated_box(label_corners, self.geometry['lidar_label_shape'])

        for p in points:
            label_x = p[0]
            label_y = p[1]
            # TODO can be better. output ans is in metric space. But in label map space is better
            metric_x, metric_y = trasform_label2metric(np.array(p))
            actual_reg_target = np.copy(reg_target)
            actual_reg_target[2] = reg_target[2] - metric_x
            actual_reg_target[3] = reg_target[3] - metric_y
            actual_reg_target[4] = np.log(reg_target[4])
            actual_reg_target[5] = np.log(reg_target[5])

            map[label_y, label_x, 0] = 1.0
            map[label_y, label_x, 1:7] = actual_reg_target
    
    def update_image_label(self, hm, wh, reg, ind, reg_mask, gt_det, k, cls_id, reg_target):
        xc, yc, h, w = reg_target
        if h > 0 and w > 0:
            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = max(0, int(radius))

            ct = np.array([xc, yc], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            draw_gaussian(hm[cls_id], ct_int, radius)
            wh[k] = 1. * w, 1. * h
            ind[k] = ct_int[1] * self.geometry['image_label_shape'][1] + ct_int[0]
            reg[k] = ct - ct_int
            reg_mask[k] = 1
            
            gt_det.append([ct[0] - w / 2, ct[1] - h / 2, 
                        ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])
    

    def get_label(self, index):
        '''
        :param i: the ith velodyne scan in the train/val set
        :return: label map: <--- This is the learning target
                a tensor of shape 800 * 700 * 7 representing the expected output


                label_list: <--- Intended for evaluation metrics & visualization
                a list of length n; n =  number of cars + (truck+van+tram+dontcare) in the frame
                each entry is another list, where the first element of this list indicates if the object
                is a car or one of the 'dontcare' (truck,van,etc) object

        '''
        index = self.image_sets[index]
        f_name = (6-len(index)) * '0' + index + '.txt'
        label_path = os.path.join(KITTI_PATH, 'training', 'label_2', f_name)

        object_list = {'Car': 1, 'Truck':0, 'DontCare':0, 'Van':0, 'Tram':0}
        label_map = np.zeros(self.geometry['lidar_label_shape'], dtype=np.float32)
        label_list = []

        hm = np.zeros((1, *self.geometry['image_label_shape'][0:2]), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        #dense_wh = np.zeros((2, *self.geometry['image_label_shape'][0:2]), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        gt_det = []

        with open(label_path, 'r') as f:
            lines = f.readlines() # get rid of \n symbol
            for k, line in enumerate(lines):
                bbox = []
                entry = line.split(' ')
                name = entry[0]
                if name in list(object_list.keys()):
                    bbox.append(object_list[name])
                    bbox.extend([float(e) for e in entry[1:]])
                    if name == 'Car':
                        corners, reg_target = self.get_corners(bbox)
                        self.update_label_map(label_map, corners, reg_target)
                        label_list.append(corners)

                        reg_target_2d = self.get_corners_2d(bbox)
                        self.update_image_label(hm, wh, reg, ind, reg_mask, gt_det, k, 0, reg_target_2d)
        image_label = {'hm': hm, 'wh': wh, 'reg':reg, 'ind': ind, 'reg_mask': reg_mask, 'gt_det': gt_det}
        return label_map, label_list, image_label

    def get_rand_velo(self):
        import random
        rand_v = random.choice(self.velo)
        print("A Velodyne Scan has shape ", rand_v.shape)
        return random.choice(self.velo)

    def load_velo_scan(self, item):
        """Helper method to parse velodyne binary files into a list of scans."""
        filename = self.velo[item]

        if self.use_npy:
            scan = np.load(filename[:-4]+'.npy')
        else:
            c_name = bytes(filename, 'utf-8')
            scan = np.zeros(self.geometry['lidar_input_shape'], dtype=np.float32)
            c_data = ctypes.c_void_p(scan.ctypes.data)
            self.LidarLib.createTopViewMaps(c_data, c_name)
            #scan = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
            
        return scan

    def load_velo_origin(self, item):
        filename = self.velo[item]
        scan = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
        return scan

    def load_velo(self):
        """Load velodyne [x,y,z,reflectance] scan data from binary files."""
        # Find all the Velodyne files

        velo_files = []
        image_files = []
        calib_files = []

        for file in self.image_sets:
            velo_file = '{}.bin'.format(file)
            velo_files.append(os.path.join(KITTI_PATH, 'training', 'velodyne', velo_file))
            image_file = '{}.png'.format(file)
            image_files.append(os.path.join(KITTI_PATH, 'training', 'image_2', image_file))
            calib_file = '{}.txt'.format(file)
            calib_files.append(os.path.join(KITTI_PATH, 'training', 'calib', calib_file))

        print('Found ' + str(len(velo_files)) + ' Velodyne scans...')
        # Read the Velodyne scans. Each point is [x,y,z,reflectance]
        self.velo = velo_files
        self.image = image_files
        self.calib = calib_files

        print('done.')

    def point_in_roi(self, point):
        if (point[0] - self.geometry['W1']) < 0.01 or (self.geometry['W2'] - point[0]) < 0.01:
            return False
        if (point[1] - self.geometry['L1']) < 0.01 or (self.geometry['L2'] - point[1]) < 0.01:
            return False
        if (point[2] - self.geometry['H1']) < 0.01 or (self.geometry['H2'] - point[2]) < 0.01:
            return False
        return True

    def passthrough(self, velo):
        geom = self.geometry
        q = (geom['W1'] < velo[:, 0]) * (velo[:, 0] < geom['W2']) * \
            (geom['L1'] < velo[:, 1]) * (velo[:, 1] < geom['L2']) * \
            (geom['H1'] < velo[:, 2]) * (velo[:, 2] < geom['H2'])
        indices = np.where(q)[0]
        return velo[indices, :]

    def lidar_preprocess(self, scan):
        # TODO 
        velo_processed = np.zeros(self.geometry['lidar_input_shape'], dtype=np.float32)
        intensity_map_count = np.zeros((velo_processed.shape[0], velo_processed.shape[1]))
        velo = self.passthrough(scan)
        for i in range(velo.shape[0]):
            x = int((velo[i, 1]-self.geometry['L1']) / self.geometry['interval'])
            y = int((velo[i, 0]-self.geometry['W1']) / self.geometry['interval'])
            z = int((velo[i, 2]-self.geometry['H1']) / self.geometry['interval'])
            velo_processed[x, y, z] = 1
            velo_processed[x, y, -1] += velo[i, 3]
            intensity_map_count[x, y] += 1
        velo_processed[:, :, -1] = np.divide(velo_processed[:, :, -1],  intensity_map_count,
                                             where=intensity_map_count != 0)
        return velo_processed
    
    def bev_to_velo(self, x, y, z):
        scales = self.geometry['lidar_input_shape'][0]/self.geometry['knn_shape'][0]
        l = (scales*x+0.5)*self.geometry['interval'] + self.geometry['L1']
        w = (scales*y+0.5)*self.geometry['interval'] + self.geometry['W1']
        h = (scales*z+0.5)*self.geometry['interval'] + self.geometry['H1']
        return w, l, h
    
    def cal_index_bev(self, x, y, z):
        return y*(self.geometry['knn_shape'][0]*self.geometry['knn_shape'][2])+x*self.geometry['knn_shape'][2]+z
    
    def cal_index_velo(self, w, l ,h):
        scales = self.geometry['lidar_input_shape'][0]/self.geometry['knn_shape'][0]
        x = round(((l-self.geometry['L1']) / self.geometry['interval']-0.5)/scales)
        y = round(((w-self.geometry['W1']) / self.geometry['interval']-0.5)/scales)
        z = round(((h-self.geometry['H1']) / self.geometry['interval']-0.5)/scales)
        return self.cal_index_bev(x, y, z)
    
    def find_knn_image(self, calib, scan, point, k=1):
        point = point[:, 0:3]
        
        center = np.zeros([self.geometry['knn_shape'][0],self.geometry['knn_shape'][1],self.geometry['knn_shape'][2]])

        itemindex = np.argwhere(center==0)
        itemindex = itemindex.astype(np.float32)
        
        scales = self.geometry['lidar_input_shape'][0]/self.geometry['knn_shape'][0]
        itemindex[:,0] = (scales*itemindex[:,0]+0.5)*self.geometry['interval'] + self.geometry['L1']
        itemindex[:,1] = (scales*itemindex[:,1]+0.5)*self.geometry['interval'] + self.geometry['W1']
        itemindex[:,2] = (scales*itemindex[:,2]+0.5)*self.geometry['interval'] + self.geometry['H1']
        itemindex = itemindex[:,[1,0,2]]
        #t = itemindex[:,0]
        #itemindex[:,0] = itemindex[:,1]
        #itemindex[:,1] = t
        center = np.reshape(itemindex, (self.geometry['knn_shape'][0],self.geometry['knn_shape'][1],self.geometry['knn_shape'][2],3))
        size = center.shape
        
        try:
            import pcl
            # print ('itemindex', itemindex)
            itemindex = itemindex.astype(np.float32)    
            pc_point = pcl.PointCloud(point)
            pc_center = pcl.PointCloud(itemindex)
            kd = pc_point.make_kdtree_flann()
            # find the single closest points to each point in point cloud 2
            # (and the sqr distances)
            indices, _ = kd.nearest_k_search_for_cloud(pc_center, k)
            # print ('indices', indices.shape)
            # print ('point', point.shape)
            # print ('center', center.shape)
            indices = np.reshape(indices, (-1))
            k_nearest = point[indices]
            
            k_nearest = np.reshape(k_nearest, (size[0], size[1], size[2],k,size[3]))
            k_nearest_image = self.velo_to_image(calib, k_nearest)
            k_dif = k_nearest - center[:,:,:,np.newaxis,:]
        except:
            print ('uninstall pcl')
            center = np.reshape(center, (size[0], size[1], size[2],1,size[3]))
            k_nearest_image = self.velo_to_image(calib, center)
            k_dif = center - center
        return k_nearest_image, k_dif


    def velo_to_image(self, calib, point):
        size = point.shape
        point = np.reshape(point, (-1,3))
        image, dis = calib.lidar_to_img(point)
        image = np.reshape(image, (size[0], size[1], size[2],size[3], 2))
        return image
    
    def raw_to_tensor(self, point, image):
        calib = self.load_calib(0)
        scan = self.lidar_preprocess(point)
        bev2image, pc_diff = self.find_knn_image(calib, scan, point, k=1)

        image = torch.from_numpy(image)
        scan = torch.from_numpy(scan)
        bev2image = torch.from_numpy(bev2image)
        pc_diff = torch.from_numpy(pc_diff)
        image = image.float()
        bev2image = bev2image.float()
        pc_diff = pc_diff.float()
        image = image.permute(2, 0, 1)
        scan = scan.permute(2, 0, 1)
        return scan, image, bev2image, pc_diff



def get_data_loader(batch_size, use_npy, geometry=None, frame_range=10000):
    train_dataset = KITTI(frame_range, use_npy=use_npy, train=True)
    if geometry is not None:
        train_dataset.geometry = geometry
    train_dataset.load_velo()
    train_data_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    val_dataset = KITTI(frame_range, use_npy=use_npy, train=False)
    if geometry is not None:
        val_dataset.geometry = geometry
    val_dataset.load_velo()
    val_data_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size * 4, num_workers=0)

    print("------------------------------------------------------------------")
    return train_data_loader, val_data_loader


def test0():
    k = KITTI()

    id = 25
    k.load_velo()
    tstart = time.time()
    scan = k.load_velo_scan(id)
    processed_v = k.lidar_preprocess(scan)
    label_map, label_list = k.get_label(id)
    print('time taken: %gs' %(time.time()-tstart))
    plot_bev(processed_v, label_list)
    plot_label_map(label_map[:, :, 6])


def find_reg_target_var_and_mean():
    k = KITTI()
    reg_targets = [[] for _ in range(6)]
    for i in range(len(k)):
        label_map, _ = k.get_label(i)
        car_locs = np.where(label_map[:, :, 0] == 1)
        for j in range(1, 7):
            map = label_map[:, :, j]
            reg_targets[j-1].extend(list(map[car_locs]))

    reg_targets = np.array(reg_targets)
    means = reg_targets.mean(axis=1)
    stds = reg_targets.std(axis=1)

    np.set_printoptions(precision=3, suppress=True)
    print("Means", means)
    print("Stds", stds)
    return means, stds

def preprocess_to_npy(train=True):
    k = KITTI(train=train)
    k.load_velo()
    for item, name in enumerate(k.velo):
        scan = k.load_velo_scan(item)
        scan = k.lidar_preprocess(scan)
        path = name[:-4] + '.npy'
        np.save(path, scan)
        print('Saved ', path)
    return

def preprocess_to_knn(train=True):
    k = KITTI(train=train)
    k.load_velo()
    for item, name in enumerate(k.velo):
        scan = k.load_velo_scan(item)
        scan = k.lidar_preprocess(scan)
        path = name[:-4] + '.npy'
        np.save(path, scan)
        print('Saved ', path)
    return

def test():
    # npy average time 0.31s
    # c++ average time 0.08s 4 workers
    batch_size = 2
    train_data_loader, val_data_loader = get_data_loader(batch_size, False)
    times = []
    tic = time.time()
    for i, (scan, image, bev2image, pc_diff, label_map, item) in enumerate(train_data_loader):
        toc = time.time()
        print(toc - tic)
        times.append(toc-tic)
        tic = time.time()
        print("Entry", i)
        print("Input shape:", scan.shape)
        print("bev2image shape", bev2image.shape)
        print("pc_diff shape", pc_diff.shape)
        print("image shape", image.shape)
        print("Label Map shape", label_map.shape)
        print("Input type():", scan.type())
        print("bev2image type()", bev2image.type())
        print("pc_diff type()", pc_diff.type())
        print("image type()", image.type())
        print("Label Map type()", label_map.type())
        if i == 20:
            break
    print("average preprocess time per image", np.mean(times)/batch_size)    

    print("Finish testing train dataloader")


if __name__=="__main__":
    test()
    # preprocess_to_npy(True)
    # preprocess_to_npy(False)
    # test0()
