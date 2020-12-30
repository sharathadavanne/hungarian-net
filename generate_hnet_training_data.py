import numpy as np
import random
import pickle
from IPython import  embed
eps = np.finfo(np.float).eps
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance

def sph2cart(azimuth, elevation, r):
    '''
    Convert spherical to cartesian coordinates
    :param azimuth: in radians
    :param elevation: in radians
    :param r: in meters
    :return: cartesian coordinates
    '''

    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return np.array([x, y, z])


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def main():
    #### MAIN ALGO starts here
    pickle_filename = 'hung_data'
    sample_range = np.array([500, 10000, 20000])
    
    max_doas = 2
    doa_ind_range = range(max_doas)

    # Generate training data
    data_dict = {}
    cnt = 0
    for resolution in [1, 2, 3, 4, 5, 10, 15, 20, 30]:
        azi_range = range(-180, 180, resolution)
        ele_range = range(-90, 91, resolution)

        for nb_ref in range(max_doas+1):
            for nb_pred in range(max_doas+1):
                total_samples = sample_range[min(nb_ref, nb_pred)]
                for nb_cnt in range(total_samples):
                    # Generate random azimuth elevation vectors
                    ref_ang = np.array((random.sample(azi_range, nb_ref), random.sample(ele_range, nb_ref))).T
                    pred_ang = np.array((random.sample(azi_range, nb_pred), random.sample(ele_range, nb_pred))).T

                    # initialize fixed length vector
                    if random.random()>0.5:
                        ref_cart, pred_cart = np.random.uniform(low=-100, high=100, size=(max_doas, 3)), np.random.uniform(low=-100, high=100, size=(max_doas, 3))
                        ref_cart[(ref_cart<=1) & (ref_cart>=-1)], pred_cart[(pred_cart<=1) & (pred_cart>=-1)] = 10, 10
                    else:
                        ref_cart, pred_cart = 10*np.ones((max_doas, 3)), 10*np.ones((max_doas, 3))

                    # Convert to cartesian vectors
                    ref_ang_rad, pred_ang_rad = ref_ang * np.pi / 180., pred_ang * np.pi / 180.
                    ref_cart[:nb_ref, :] = sph2cart(ref_ang_rad[:, 0], ref_ang_rad[:, 1], np.ones(nb_ref)).T
                    pred_cart[:nb_pred, :] = sph2cart(pred_ang_rad[:, 0], pred_ang_rad[:, 1], np.ones(nb_pred)).T

                    # Compute distance matrix
                    dist_mat = distance.cdist(ref_cart, pred_cart, 'minkowski', p=2.)

                    # Compute data association matrix
                    act_dist_mat = dist_mat[:nb_ref, :nb_pred]
                    row_ind, col_ind = linear_sum_assignment(act_dist_mat)
                    da_mat = np.zeros((max_doas, max_doas))
                    da_mat[row_ind, col_ind] = 1

                    #randomly shuffle dist and da matrices
                    rand_ind = random.sample(range(max_doas), max_doas)
                    if random.random()>0.5:
                        dist_mat = dist_mat[rand_ind, :]
                        da_mat = da_mat[rand_ind, :]
                    else:
                        dist_mat = dist_mat[:, rand_ind]
                        da_mat = da_mat[:, rand_ind]
                    data_dict[cnt] = [nb_ref, nb_pred, dist_mat, da_mat, ref_cart, pred_cart]
                    cnt += 1
    out_filename = 'data/{}_train'.format(pickle_filename)
    print('Saving data in: {}, #examples: {}'.format(out_filename, len(data_dict)))
    save_obj(data_dict, out_filename)

    # Generate testing data
    data_dict = {}
    cnt = 0
    for resolution in [1, 2, 3, 4, 5, 10, 15, 20, 30]:
        azi_range = range(-180, 180, resolution)
        ele_range = range(-90, 91, resolution)
        for nb_ref in range(max_doas+1):
            for nb_pred in range(max_doas+1):
                total_samples = int(0.1*sample_range[min(nb_ref, nb_pred)])
                for nb_cnt in range(total_samples):
                    # Generate random azimuth elevation vectors
                    ref_ang = np.array((random.sample(azi_range, nb_ref), random.sample(ele_range, nb_ref))).T
                    pred_ang = np.array((random.sample(azi_range, nb_pred), random.sample(ele_range, nb_pred))).T

                    # initialize fixed length vector
                    if random.random()>0.5:
                        ref_cart, pred_cart = np.random.uniform(low=-100, high=100, size=(max_doas, 3)), np.random.uniform(low=-100, high=100, size=(max_doas, 3))
                        ref_cart[(ref_cart<=1) & (ref_cart>=-1)], pred_cart[(pred_cart<=1) & (pred_cart>=-1)] = 10, 10
                    else:
                        ref_cart, pred_cart = 10*np.ones((max_doas, 3)), 10*np.ones((max_doas, 3))

                    # Convert to cartesian vectors
                    ref_ang_rad, pred_ang_rad = ref_ang * np.pi / 180., pred_ang * np.pi / 180.
                    ref_cart[:nb_ref, :] = sph2cart(ref_ang_rad[:, 0], ref_ang_rad[:, 1], np.ones(nb_ref)).T
                    pred_cart[:nb_pred, :] = sph2cart(pred_ang_rad[:, 0], pred_ang_rad[:, 1], np.ones(nb_pred)).T

                    # Compute distance matrix
                    dist_mat = distance.cdist(ref_cart, pred_cart, 'minkowski', p=2.)

                    # Compute data association matrix
                    act_dist_mat = dist_mat[:nb_ref, :nb_pred]
                    row_ind, col_ind = linear_sum_assignment(act_dist_mat)
                    da_mat = np.zeros((max_doas, max_doas))
                    da_mat[row_ind, col_ind] = 1

                    #randomly shuffle dist and da matrices
                    rand_ind = random.sample(range(max_doas), max_doas)
                    if random.random()>0.5:
                        dist_mat = dist_mat[rand_ind, :]
                        da_mat = da_mat[rand_ind, :]
                    else:
                        dist_mat = dist_mat[:, rand_ind]
                        da_mat = da_mat[:, rand_ind]
                    data_dict[cnt] = [nb_ref, nb_pred, dist_mat, da_mat, ref_cart, pred_cart]
                    cnt += 1
    out_filename = 'data/{}_test'.format(pickle_filename)
    print('Saving data in: {}, #examples: {}'.format(out_filename, len(data_dict)))
    save_obj(data_dict, out_filename)


if __name__ == "__main__":
    main()
