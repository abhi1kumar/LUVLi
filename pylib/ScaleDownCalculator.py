

"""
    Version 1 2019-07-26 Abhinav Kumar
"""
import numpy as np

def get_coeff(p1, p2):
    """
        Coefficients of a straight line passing through two points p1 and p2
        p1 and p2 - numpy arrays
        The coefficients have to be of the form ax + by + c = 0    
    """
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]

    coeff_x = y2 - y1
    coeff_y = -(x2 - x1)
    constant = y1*(x2 - x1) - x1*(y2-y1)

    return np.array([coeff_x, coeff_y, constant])

def get_intersection_bw_lines(line1, line2):
    """
        Gets the intersection point between two lines
    """
    a1 = line1[0]
    b1 = line1[1]
    c1 = line1[2]

    a2 = line2[0]
    b2 = line2[1]
    c2 = line2[2]

    deno = a1*b2 - a2*b1
    if np.abs(deno) < 0.0001:
        print("Lines do not intersect")
        x0 = 0.
        y0 = 0.
    else:
        x0 = (b1*c2 - b2*c1)/deno
        y0 = (c1*a2 - c2*a1)/deno     
    
    return np.array([x0, y0])

def get_intersection(p1, p2, p3, p4):
    """
        Gets the intersection between two lines - one by p1 and p2 and other by
        p3 and p4
    """
    line1 = get_coeff(p1, p2)
    line2 = get_coeff(p3, p4)

    return get_intersection_bw_lines(line1, line2)

def get_ideal_scale_euclidean(pts_input_res, inp_res, img_path):
    """
        Calculates the additional scale required to bring the points inside based
        on Euclidean distance of points from the origin.
    """
    print("\n" + img_path)
    orig = np.array([inp_res/2. -1, inp_res/2. -1])
    outside_index = np.where(np.logical_or(np.logical_or(pts_input_res[:,0] < 0, pts_input_res[:,0] >= inp_res), np.logical_or(pts_input_res[:,1] < 0, pts_input_res[:,1] >= inp_res)))[0]
    if len(outside_index)>0:
        print(outside_index)
        print(pts_input_res[outside_index])
        dist = np.sum(np.abs(pts_input_res[outside_index] - orig)**2, axis=-1)**0.5
        dist_max = np.max(dist)
        temp = dist.argmax(axis=0)
        index = outside_index[temp]
        print("Max point= {}".format(index))

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

        intersection_pt = get_intersection(p1, p2, p3, p4)
        dist_ideal = np.linalg.norm(orig-intersection_pt)

        scale_down = dist_ideal/dist_max
    else:
        print("No points outside")
        scale_down = 1.

    return scale_down


def get_ideal_scale(pts_input_res, inp_res, img_path, visible= None):
    """
        Calculates the additional scale required to bring the points inside based
        on maximum perturbation from the origin.
        Use visibility into account
    """
    # Origin is the center of the image with zero-indexing.
    orig = np.array([inp_res/2. -1, inp_res/2. -1])

    if visible is not None:
        # Create a masked array depending on visibility
        # visible is 1 for visible and 0 for not
        # mask is 1 for to be masked and 0 for not to be masked
        pts_input_res = np.ma.array(pts_input_res, mask = np.column_stack((1-visible, 1-visible)))
    
        if np.sum(1-visible) == pts_input_res.shape[0]:
            # All points are invisible
            return 1.
        
    # Calculate the outer bounds first
    xmin = np.min(pts_input_res[:,0])
    xmax = np.max(pts_input_res[:,0])
    ymin = np.min(pts_input_res[:,1])
    ymax = np.max(pts_input_res[:,1])    

    # Calculate distances
    dist = np.zeros((4,))
    dist[0] = orig[0]-xmin
    dist[1] = xmax - orig[0]
    dist[2] = orig[1]-ymin
    dist[3] = ymax - orig[1]

    max_dist = np.max(dist)

    # If max distance is more than half of the image, this is weird.
    if max_dist > orig[0]:
        scale_down = orig[0]/max_dist
    else:
        scale_down = 1.

    return scale_down
