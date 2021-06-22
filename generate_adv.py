import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from matlab_cp2tform import get_similarity_transform_for_cv2
import torch.nn.functional as F
import cv2
import pdb
import net_sphere
from torch.autograd import Variable
import torchvision.utils
import matplotlib.pyplot as plt



def alignment(src_img,src_pts):
    ref_pts = [ [30.2946, 51.6963],[65.5318, 51.5014],
        [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]
    crop_size = (96, 112)
    src_pts = np.array(src_pts).reshape(5,2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)

    face_img = cv2.warpAffine(src_img, tfm, crop_size)


    return face_img

def criterion(net,x,x_adv):
    # x, x_adv : torch.Tensor with shape [1,3,112,96]
    best_thresh = 0.3
 
    imglist = [x,x_adv]
    img = torch.vstack(imglist)
    img = Variable(img.float(),volatile=True).cuda()
    output = net(img)
    f1,f2 = output.data
    cosdistance = f1.dot(f2)/(f1.norm()*f2.norm()+1e-5)

 
    D = torch.norm(x-x_adv)

    C = 0 if cosdistance < best_thresh else float('inf')
    return D+C


 
def generate_adversarial_face(net,x,T):
    # input
    #    L : the attack objective function
    #    x : torch tensor img with shape (1,3,112,96)
    # output
    #    : adversarial face image x
    _,_,H,W = x.shape
    m = 3 * 45 * 45 
    k = m//20
    C = torch.eye(m)
    p_c = torch.zeros(m)
    c_c = 0.01
    c_cov = 0.001
    sigma = 0.01
    success_rate = 0
    mu = 1
    x_adv = torch.randn_like(x)
    criterion(net,x,x_adv)

    for t in range(T):
        z = MultivariateNormal(loc=torch.zeros([m]), covariance_matrix=(sigma**2) * C).rsample()
        # z = np.random.normal(loc=0.0, scale=(sigma**2) * C)
        
        zeroIdx = np.argsort(-C.diagonal())[k:]
        z[zeroIdx] = 0
        
        z = z.reshape([1,3,45,45]) 
        z_ = F.interpolate(z,(H,W),mode = 'bilinear')
        z_ = z_ + mu * (x - x_adv)
        L_after = criterion(net,x,x_adv + z_)
        L_before = criterion(net,x,x_adv)
        
        if L_after < L_before:
            x_adv = x_adv + z_
            p_c = (1 - c_c) * p_c + np.sqrt(2*(2-c_c)) * z.reshape(-1)/sigma
            C[range(m),range(m)] = (1 - c_cov)*C.diagonal() + c_cov *(p_c)**2
            print (L_after)
            saveImg(x_adv,'iter_' + str(t))
            success_rate += 1
            

    
        if t % 10 == 0 :
    
            mu = mu* np.exp(success_rate/10 - 1/5)
            success_rate = 0

            

        print (t)

    return x_adv



def saveImg(x,name):
    # input
    # x : torch tensor with normalization
    
    x = (x * 128 + 127.5).type(torch.int)
    x = x[0].permute(1,2,0)
    cv2.imwrite('./fig/' + name + '.png',np.array(x))


def randomly_select_image(landmark,pairs_lines,data_path):

    idx = np.random.randint(6000)
    pair = pairs_lines[idx]
    line = pair.replace('\n','').split('\t')
    name = line[0]+'/'+line[0]+'_'+'{:04}.jpg'.format(int(line[1]))
    img = cv2.imread(data_path + '/lfw/'+name)
    img = alignment(cv2.imread(data_path + '/lfw/'+name),landmark[name])
    img = img.transpose(2, 0, 1).reshape((1,3,112,96))
    img = (img-127.5)/128.0
    img = torch.tensor(img)
    
    return img


def main():

    net = getattr(net_sphere,'sphere20a')()
# net.load_state_dict(torch.load(args.model))
    net.load_state_dict(torch.load('./model/sphere20a_20171020.pth'))
    net.cuda()
    net.eval()
    net.feature = True

    data_path = '/mnt/server9_hard1/seungju/dataset/LFW'

    landmark = {}
    with open(data_path + '/lfw_landmark.txt') as f:
        landmark_lines = f.readlines()

    for line in landmark_lines:
        l = line.replace('\n','').split('\t')
        landmark[l[0]] = [int(k) for k in l[1:]]

    # randomly select image from lfw data set
    
    with open(data_path + '/pairs.txt') as f :
        pairs_lines = f.readlines()[1:]

    x = randomly_select_image(landmark,pairs_lines,data_path)
    saveImg(x,'original')
    x_adv = generate_adversarial_face(net,x,T = 10000)
    pdb.set_trace()
    saveImg(x,'final')

main()