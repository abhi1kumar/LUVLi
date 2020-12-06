
import numpy as np

pts_input_res = np.array([[6.34294886,132.05269824],[20.87406229,157.60049578],[38.99187277,176.13985941], [ 62.27546107, 192.18773347], [ 85.8890913,  209.18643975], [108.16105101, 220.95283187], [129.26127439, 233.22994298], [152.53731307, 241.38972608], [174.39532546, 232.77884778], [192.21166362, 220.34130611], [210.3582922, 189.12917841], [224.85046338, 160.13956334], [236.30818646, 128.13631072], [241.6503922,   95.64582636], [232.45405994,  59.73830235], [210.65362424, 29.12836917], [194.09325736,   5.78956872], [-15.0831065,  109.97133933], [-10.97797956,  96.01129899]])
inp_res = 256

# https://stackoverflow.com/a/3252222
# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# line segment intersection using vectors
# see Computer Graphics by F.S. Hill
def perp(a) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def seg_intersect_pt(a1,a2, b1,b2) :
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    return (num / denom.astype(float))*db + b1

def get_ideal_scale(pts_input_res, inp_res):
    orig = np.array([inp_res/2. -1, inp_res/2. -1])
    outside_index = np.where(np.logical_or(np.logical_or(pts_input_res[:,0] < 0, pts_input_res[:,0] >= inp_res), np.logical_or(pts_input_res[:,0] < 0, pts_input_res[:,0] >= inp_res)))[0]
    dist = np.sum(np.abs(pts_input_res[outside_index] - orig)**2, axis=-1)**0.5
    dist_max = np.max(dist)
    temp = dist.argmax(axis=0)
    index = outside_index[temp]

    p1 = orig
    p2 = pts_input_res[index,:]

    if p2[0] < 0:
        p3 = np.array([0., 0])
        p4 = np.array([0., 1])
    if p2[0] >= inp_res:
        p3 = np.array([inp_res, 0.])
        p4 = np.array([inp_res, 0.])

    if p2[1] < 0:
        p3 = np.array([0., 0])
        p4 = np.array([1., 0])
    if p2[1] >= inp_res:
        p3 = np.array([0., inp_res])
        p4 = np.array([0., inp_res])

    intersection_pt = seg_intersect_pt(p1, p2, p3, p4)
    dist_ideal = np.linalg.norm(orig-intersection_pt)

    scale_down = dist_ideal/dist_max
    return scale_down

scale_down = get_ideal_scale(pts_input_res, inp_res)
print(scale_down)
