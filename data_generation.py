import numpy as np

# generate test objects
class Shape():
    def __init__(self, type=0, sample_points=1, scale=1, theta=0, translation=np.array([0,0])):
        self.nums = sample_points
        self.xy = []
        self.key_points = []
        theta = theta/180.*np.pi
        self.rotate_matrix = np.array([[np.cos(theta), - np.sin(theta)],
                                       [np.sin(theta),  np.cos(theta)]])
        if sample_points == 1:
            self.xy = translation.reshape(1,-1)
#             self.key_points = 
        else:
            if type==0:
#                 print "circle nums=", sample_points
                theta = np.linspace(0, 2*np.pi, sample_points)
                x = np.cos(theta)
                y = np.sin(theta)
                xy = np.concatenate([x.reshape(-1,1), y.reshape(-1,1)],axis=1)
                self.xy = xy*scale + translation.reshape(1,-1)
                tmp = np.linspace(0, 2*np.pi, 100)
                xy = np.concatenate([np.cos(tmp).reshape(-1,1), np.sin(tmp).reshape(-1,1)],axis=1)
                self.key_points = xy * scale + translation.reshape(1,-1)
            elif type == 1:
#                 print type,"-triangle, nums=", sample_points
                initial_xy = np.array([[0.5,0],
                                       [0, 2],
                                       [-0.5,0]])
                self.xy = self.sample(initial_xy)
                self.key_points = np.concatenate([initial_xy, initial_xy[0,None,:]],axis=0)
            elif type ==2:
#                 print type,"-rectangle, nums=", sample_points
                initial_xy = np.array([[0,0],
                                       [2,0],
                                       [2,1],
                                       [0,1]])
                self.xy = self.sample(initial_xy)
                self.key_points = np.concatenate([initial_xy, initial_xy[0,None,:]],axis=0)
            elif type ==3:
#                 print type,"-star, nums=", sample_points
                initial_xy = np.array([[0,1],
                                    [np.cos((72*2+90)/180.*np.pi), np.sin((72*2+90)/180.*np.pi)],
                                    [np.cos(18./180.*np.pi), np.sin(18/180.*np.pi)],
                                    [np.cos((72+90)/180.*np.pi), np.sin((72+90)/180.*np.pi)],
                                    [np.cos(-54/180.*np.pi),-np.sin(54/180.*np.pi)]])
                self.xy = self.sample(initial_xy)
                self.key_points = np.concatenate([initial_xy, initial_xy[0,None,:]],axis=0)
            elif type ==4:
#                 print type,"-container, nums=", sample_points
                thickness = 0.1
                initial_xy = np.array([[0,1],[0,0],[1,0],[1,1],
                                       [1-thickness,1],[1-thickness,thickness],
                                       [thickness,thickness],[thickness,1]])
#                 initial_xy = np.array([[0,1],[0,0],[1,0],[1,1]])
                self.xy = self.sample(initial_xy)
                self.key_points = np.concatenate([initial_xy, initial_xy[0,None,:]],axis=0)
            else:
                raise NotImplemented
            if type !=0:
                self.xy = np.dot(self.xy * scale, self.rotate_matrix) + translation.reshape(1,-1) 
                self.key_points = np.dot(self.key_points * scale, self.rotate_matrix) + translation.reshape(1,-1) 

    def sample(self, key_points):
        dis = [0]
        dis_data = 0
        for i in range(key_points.shape[0]):
            dis_data = dis_data + np.linalg.norm(key_points[i,:]-key_points[i-1,:])
            dis.append(dis_data)
        dis_sample = np.linspace(0, dis_data, self.nums)
        xy_samples = np.zeros([self.nums,2])
        for i in range(self.nums):
            dis_tmp = np.copy(dis_sample[i])
            for j in range(key_points.shape[0]):
                if dis_tmp>=dis[j] and dis_tmp<=dis[j+1]:
                    dis_tmp = dis_tmp - dis[j]
                    break

            p1 = key_points[j-1,:]
            p2 = key_points[j,:]
            rate = dis_tmp / np.linalg.norm(p1-p2)
#             print rate
            xy_samples[i,:] = (p2-p1) *rate + p1
        return xy_samples

