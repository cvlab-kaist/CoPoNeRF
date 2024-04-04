import cv2
import numpy as np
import os
import json


def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def two_view_geometry(intrinsics1, intrinsics2, rel_pose_est, rel_pose_gt):
    relative_pose = rel_pose_est
    R = relative_pose[:,:3, :3]
    T = relative_pose[:, :3, 3]
   
    tx = []
    for i in range(len(T)):
        tx.append(skew(T[i].cpu().numpy()))
    tx = np.stack(tx)
 
    E = np.matmul(tx, R.cpu().numpy())
    F = []
    for i in range(len(T)):
        F.append(np.linalg.inv(intrinsics2[ i, :3, :3].cpu().numpy()).T.dot(E[i]).dot(np.linalg.inv(intrinsics1[i, :3, :3].cpu().numpy())))

    relative_pose_gt = rel_pose_gt
    R_gt = relative_pose_gt[:,:3, :3]
    T_gt = relative_pose_gt[:, :3, 3]
    tx_gt = []
    for i in range(len(T)):
        tx_gt.append( skew(T_gt[i].cpu().numpy()))
    tx_gt = np.stack(tx_gt)
    E_gt = np.matmul(tx_gt, R_gt.cpu().numpy())
    F_gt = []
    for i in range(len(T)):
        F_gt.append(np.linalg.inv(intrinsics2[ i, :3, :3].cpu().numpy()).T.dot(E_gt[i]).dot(np.linalg.inv(intrinsics1[i, :3, :3].cpu().numpy())))
   



    return E, F, relative_pose,  E_gt, F_gt, relative_pose_gt


def drawpointslines(img1, pts1, img2, lines2, colors):
    
    h, w = img2.shape[:2]

    for p, l, c in zip(pts1, lines2, colors):
        c = tuple(c.tolist())
        img1 = cv2.circle((img1)  , tuple(p), 5, c, -1)

        x0, y0 = map(int, [0, -l[2]/l[1]])
        x1, y1 = map(int, [w, -(l[2]+l[0]*w)/l[1]])
        img2 = cv2.line((img2  ) , (x0, y0), (x1, y1), c,10)
    return img1, img2

def drawpoint(img1, pts1, img2, pts2, colors):
        # 
        for p2, color in zip(pts2, colors):
            
            color = tuple(color.tolist())
            img1 = cv2.circle((img1)  , tuple(p2), 5, color, -1)
            
            img2 = cv2.circle((img2  ) , (int(pts1[p2[0] + (p2[1] * 256)].round()[0]),int(pts1[p2[0] + (p2[1] * 256)].round()[1]) ), 5, color, -1)

 
        return img1, img2


        


def inspect(img1, K1, img2, K2, rel_pose_est, rel_pose_gt):
    E, F, relative_pose, E_gt, F_gt, relative_pose_gt = two_view_geometry(K1, K2, rel_pose_est, rel_pose_gt)
    
    try:
        orb = cv2.ORB_create()
        img = []
        img_gt = []
       
        pts1 = np.array([[64,64], [64,128], [64,192], [128,64], [128,128], [128,192], [192,64], [192,128], [192,192]])
        colors = np.array([[63,228,92], [222, 155, 167], [ 56, 220, 130],  [216,  43, 206], [ 47, 172,  72],  [198, 181,   0], [137,  99, 246],  [ 22, 160,  10], [ 23, 240, 252]  ]   )
    
        for i in range(len(E)):
            
            lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1, F[i])
            lines2 = lines2.reshape(-1, 3)

            lines2_gt = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1, F_gt[i])
            lines2_gt = lines2_gt.reshape(-1, 3)
    
  
            im1, im2 = drawpointslines((img1[i].cpu().numpy() + 1 )*127.5, pts1, (img2[i].cpu().numpy() + 1 )*127.5, lines2, colors)

            im1_gt, im2_gt = drawpointslines((img1[i].cpu().numpy() + 1 )*127.5, pts1, (img2[i].cpu().numpy() + 1 )*127.5, lines2_gt, colors)
            im2_copy = ((img2[i].cpu().numpy() + 1 )*127.5).copy()
            im2_gt_copy =((img2[i].cpu().numpy() + 1 )*127.5).copy()

            # Define the alpha value to control line intensity reduction (0.0 to 1.0)
            alpha = 0.5  # You can adjust this value to control the intensity

            # Blend the lines with im2 and im2_gt using addWeighted
            im2 = cv2.addWeighted(im2,  0.4, im2_copy, 0.6, 0)
            im2_gt = cv2.addWeighted(im2_gt, 0.4, im2_gt_copy, 0.6, 0)


            
            im_to_show = np.concatenate((im1, im2), axis=1)

            im_to_show_gt = np.concatenate((im1_gt, im2_gt), axis=1)
            img.append(im_to_show)
            img_gt.append(im_to_show_gt)
        
        img= np.stack(img)
        img_gt = np.stack(img_gt)

    
        return img, img_gt
    except:
        return None, None
        

