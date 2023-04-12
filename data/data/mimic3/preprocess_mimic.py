"""
Script for preprocessing MIMIC III data for DP-PVI testing.
Assume MIMIC data has been extracted with the code from Harutyunyan et al. 2019 in-hospital mortality prediction: https://github.com/YerevaNN/mimic3-benchmarks
This code uses data readers from the above paper.

After preprocessing:
i) combine train and validation data,
ii) dimensionality reduction: learn logistic regression model on train data, only keep dims with largest abs value weights
iii) divide to 5 clients: do 10 cluster kmeans, combine smallest categories to arrive at 5 partitions with roughly same order of samples, uneven data balance
"""

import os
import sys

import argparse
from copy import copy
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

import importlib


#### mimic3 data readers from Harutyunyan et al. 2019 ####
def read_and_extract_features(reader, period, features):
    ret = common_utils.read_chunk(reader, reader.get_number_of_examples())
    X = common_utils.extract_features_from_rawdata(ret['X'], ret['header'], period, features)
    return (X, ret['y'], ret['name'])

def get_readers_mortality(input_dir):
    '''
    Args:
        input_dir : path to the in-hospital mortality task subfolder
    '''

    train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(input_dir, 'train'),
                                                 listfile=os.path.join(input_dir, 'train_listfile.csv'),
                                                 period_length=48.0)
    # note: validation data separated from train, no separate folder
    val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(input_dir, 'train'),
                                                 listfile=os.path.join(input_dir, 'val_listfile.csv'),
                                                 period_length=48.0)
    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(input_dir, 'test'),
                                                 listfile=os.path.join(input_dir, 'test_listfile.csv'),
                                                 period_length=48.0)

    return train_reader, val_reader, test_reader


def get_readers_LOS(input_dir):
    '''
    Args:
        input_dir : path to the length of stay task subfolder
        note: not clear if these work properly
    '''
    print('init LOS train reader')
    train_reader = LengthOfStayReader(dataset_dir=os.path.join(input_dir, 'train'),
                                                 listfile=os.path.join(input_dir+'train/', 'listfile.csv'))
    # note: validation data separated from train, no separate folder
    #val_reader = LengthOfStayReader(dataset_dir=os.path.join(input_dir+'train/', 'train'),
    #                                             listfile=os.path.join(input_dir, 'val_listfile.csv'))
    print('init LOS test reader')
    test_reader = LengthOfStayReader(dataset_dir=os.path.join(input_dir, 'test'),
                                                 listfile=os.path.join(input_dir+'test/', 'listfile.csv'))
    print('Done')
    return train_reader, test_reader

##########################################################


def do_data_split_LOS(args):
    """
    main script for data preprocessing for length of stay data
    """
    #/scratch/project_2003275/dp-pvi/pvi/data/data/mimic_raw

    # fix random seeds
    np.random.seed(args.random_seed)

    #folder = 'length_of_stay/'
    #tmp = np.load(folder+'test_ts.npy')
    #tmp2 = np.load(folder+'test_y.npy')
    #print(tmp.shape, tmp2.shape)
    #print(np.allclose(tmp,tmp2))
    #sys.exit()
    try:
        # read numpy mimic
        filename = args.output_dir+'numpy_mimic_LOS.npz'
        tmp = np.load(filename)
        train_X = tmp['train_X']
        val_X = tmp['val_X']
        test_X = tmp['test_X']
        train_y = tmp['train_y']
        val_y = tmp['val_y']
        test_y = tmp['test_y']

    except:
        """
        # read mimic data
        train_reader, test_reader = get_readers_LOS(args.input_dir_LOS)
        period, features = 'all', 'all'
        print('Reading data and extracting features ...')
        tmp = read_and_extract_features(train_reader, period, features)
        (train_X, train_y, _) = read_and_extract_features(train_reader, period, features)
        #(val_X, val_y, _) = read_and_extract_features(val_reader, period, features)
        (test_X, test_y, _) = read_and_extract_features(test_reader, period, features)
        """
        # read LOS data from numpy arrays and save as one single
        folder = 'length_of_stay/'
        print(f'reading from {folder}')
        train_X = np.load(folder+'train_X.npy')
        train_y = np.load(folder+'train_y.npy')
        val_X = np.load(folder+'val_X.npy')
        val_y = np.load(folder+'val_y.npy')
        test_X = np.load(folder+'test_X.npy')
        test_y = np.load(folder+'test_y.npy')
        #print(train_X.shape)
        #sys.exit()

        # save intermediate results
        if args.save_intermediate:
            np.savez_compressed(args.output_dir+f'numpy_mimic_LOS', 
                        #**{'train_X':train_X, 'test_X':test_X,'train_y':train_y, 'test_y':test_y})
                        **{'train_X':train_X, 'val_X':val_X, 'test_X':test_X,'train_y':train_y, 'val_y':val_y, 'test_y':test_y})

    print('Done reading data')
    print('  train data shape = {}'.format(train_X.shape))
    print('  valid data shape = {}'.format(val_X.shape))
    print('  test data shape = {}'.format(test_X.shape))

    # balance data: keep same numebr of positive and negative samples?
    #print(f'train number of pos samples: {np.sum(train_y == 1)}, train fraction of pos samples: { np.sum(train_y == 1)/(len(train_y))}')
    #print(f'val number of pos samples: {np.sum(val_y == 1)}, val fraction of pos samples: { np.sum(val_y == 1)/(len(val_y))}')
    #print(f'test number of pos samples: {np.sum(test_y == 1)}, test fraction of pos samples: { np.sum(test_y == 1)/(len(test_y))}')


def do_data_split_mortality(args):
    """
    main script for data preprocessing for mortality prediction
    """

    # fix random seeds
    np.random.seed(args.random_seed)

    try:
        # read numpy mimic
        filename = args.output_dir+'numpy_mimic.npz'
        tmp = np.load(filename)
        train_X = tmp['train_X']
        val_X = tmp['val_X']
        test_X = tmp['test_X']
        train_y = tmp['train_y']
        val_y = tmp['val_y']
        test_y = tmp['test_y']
    except:
        # read mimic data
        train_reader, val_reader, test_reader = get_readers_mortality(args.input_dir_mortality)
        period, features = 'all', 'all'
        print('Reading data and extracting features ...')
        (train_X, train_y, _) = read_and_extract_features(train_reader, period, features)
        (val_X, val_y, _) = read_and_extract_features(val_reader, period, features)
        (test_X, test_y, _) = read_and_extract_features(test_reader, period, features)
        # save intermediate results
        if args.save_intermediate:
            np.savez_compressed(args.output_dir+f'numpy_mimic', 
                        **{'train_X':train_X, 'val_X':val_X, 'test_X':test_X,'train_y':train_y, 'val_y':val_y, 'test_y':test_y})

    print('Done reading data')
    print('  train data shape = {}'.format(train_X.shape))
    print('  valid data shape = {}'.format(val_X.shape))
    print('  test data shape = {}'.format(test_X.shape))

    if not args.balance_data:
        # inbalanced data: combine train and validation to train set, test set as is

        print('Combining train and validation data, change all to np.arrays')
        # concat train and val to create larger train set
        train_X = np.concatenate((train_X,val_X))
        train_y = np.concatenate((train_y,val_y))
        #print('train data shapes: {}, {}'.format(train_X.shape, train_y.shape)) # should be (17903, 714), (17903,)
        assert train_X.shape == (17903,714) and train_y.shape == (17903,)
        # change from lists to np.arrays
        test_X, test_y = np.array(test_X), np.array(test_y)
        #print('test data shapes: {}, {}'.format(test_X.shape, test_y.shape)) # should be (3236,714), (3236,)
        assert test_X.shape == (3236,714) and test_y.shape == (3236,)
    else:
        # balanced data: keep only as many samples from the majority class as there are in the minority class

        print('Combining train, validation, and test data, change to np.arrays and balance data')
        # concat train and val to create larger train set
        train_X = np.concatenate((train_X, val_X, test_X))
        train_y = np.concatenate((train_y, val_y, test_y))
        del val_X, val_y, test_X, test_y
        assert train_X.shape == (21139,714) and train_y.shape == (21139,)
        # shuffle full data
        inds = np.linspace(0,len(train_y)-1,len(train_y),dtype=int)
        np.random.shuffle(inds)
        train_X = train_X[inds,:]
        train_y = train_y[inds]
        pos_inds = np.linspace(0,len(train_y)-1,len(train_y),dtype=int)[train_y == 1]
        neg_inds = (np.linspace(0,len(train_y)-1,len(train_y),dtype=int)[train_y == 0])[:len(pos_inds)]
        #print(len(pos_inds), len(neg_inds))
        assert len(pos_inds) == len(neg_inds)
        inds = np.concatenate((pos_inds,neg_inds))
        train_X = train_X[inds,:]
        train_y = train_y[inds]
        assert np.allclose(np.sum(train_y==1)/len(train_y),.5) and train_X.shape[0] == 5594 and train_y.shape[0] == 5594
        #print('train data shapes: {}, {}'.format(train_X.shape, train_y.shape)) # should be (5594, 714), (5594,)
    print('Done.')

    # impute missing data
    print('Imputing missing data')
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean', verbose=0, copy=False)
    imputer.fit(train_X)
    train_X = np.array(imputer.transform(train_X), dtype=np.float32)
    if not args.balance_data:
        imputer.fit(test_X)
        test_X = np.array(imputer.transform(test_X), dtype=np.float32)
    print('Done.')

    # Normalize data to have zero mean and unit variance; note: not fully normalized after dim reduction
    print('Normalizing data')
    scaler = StandardScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    if not args.balance_data:
        scaler.fit(test_X)
        test_X = scaler.transform(test_X)
    print('Done.')

    print('Doing dimensinality reduction via logistic regression modelling')
    model = LogisticRegression(max_iter=1000, random_state=42).fit(X=train_X,y=train_y)
    # find indices for largest weights in abs value
    largest_inds = np.argsort(-np.abs(model.coef_[0]))[:args.target_dim]
    train_X = train_X[:,largest_inds]
    print('New train_X shape: {}'.format(train_X.shape))
    if not args.balance_data:
        test_X = test_X[:,largest_inds]
        print('New test_X shape: {}'.format(test_X.shape))
    print('Done.')

    if not args.balance_data:
        assert train_X.shape == (17903,args.target_dim) and test_X.shape == (3236,args.target_dim)
    else:
        assert train_X.shape == (5594,args.target_dim)

    # Renormalize data to have zero mean and unit variance; note
    print('Renormalizing data')
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    if not args.balance_data:
        scaler.fit(test_X)
        test_X = scaler.transform(test_X)
    print('Done.')

    sys.exit('aborting before writing')

    # note: with balanced data train-test split is done by utils-script when initing clients
    if not args.balance_data:
        print('Dividing training data into client partitions via K-means')
        kmeans = KMeans(n_clusters=args.n_clusters, max_iter=300, tol=0.0001, verbose=0, random_state=2303, copy_x=True, algorithm='auto')
        kmeans_res = kmeans.fit(train_X)
        cur_clusters = np.linspace(0,args.n_clusters-1, args.n_clusters, dtype=int)
        labels = copy(kmeans_res.labels_)
        for i in range(args.n_clusters):
            if len(cur_clusters) <= args.n_clients:
                print(f'Finished combining clusters with {len(cur_clusters)} clusters remaining')
                break
            # combine 2 smallest clusters
            tmp = np.array([np.sum(labels==i_client) for i_client in cur_clusters])
            smallest_inds = np.argsort(tmp)[:2]
            labels[labels == cur_clusters[smallest_inds[0]]] = cur_clusters[smallest_inds[1]]
            cur_clusters = np.delete(cur_clusters, smallest_inds[0])

        print('Final client data splits:')
        for i_client, client in enumerate(cur_clusters):
            print(f'client {i_client}: N={np.sum(labels==client)}, pos.frac.={ (np.sum(train_y[labels==client]==1) )/np.sum(labels==client):.5f}')

        filename = args.output_dir + f'mimic_in-hospital_unbal_split_{args.n_clients}clients.npz'
        print(f'Saving all partitions to {filename}')
        partitions = {}
        partitions['x_test'] = test_X
        partitions['y_test'] = test_y
        for i_client in range(args.n_clients):
            partitions[f'x_{i_client}'] = train_X[labels==cur_clusters[i_client],:]
            partitions[f'y_{i_client}'] = train_y[labels==cur_clusters[i_client]]
        np.savez_compressed(filename, **partitions)
            
    else:
        # balanced data
        filename = args.output_dir + f'mimic_in-hospital_bal_split.npz'
        print(f'Saving combined data to {filename}')
        np.savez_compressed(filename, **{'train_X': train_X, 'train_y':train_y})

    print('All done.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--input_dir_mortality', default='/Users/mixheikk/Documents/mimic3-benchmarks/data/in-hospital-mortality/', type=str, help="Path to in-hospital mortality subfolder created by MIMIC preprocessing scripts")
    #parser.add_argument('--input_dir_LOS', default='/Users/mixheikk/Documents/mimic3-benchmarks/data/length-of-stay/', type=str, help="Path to lenght of stay subfolder created by MIMIC preprocessing scripts")
    parser.add_argument('--input_dir_LOS_numpy', default='/Users/mixheikk/Documents/mimic3-benchmarks/data/length-of-stay/', type=str, help="Path to lenght of stay subfolder created by MIMIC preprocessing scripts in numpy format")

    parser.add_argument('--output_dir', default='/Users/mixheikk/Documents/git/DP-PVI/pytorch_pvi/pvi/data/data/mimic3/', type=str, help="Path to output dir")
    parser.add_argument('--target_dim', default=50, type=int, help="Target data dimensionality <= 714, use logistic regression abs weight valeus to determine most important dims")
    parser.add_argument('--n_clusters', default=10, type=int, help="Initial number of clusters for K-means; used in partitioning data to clients; >= n_clients")
    parser.add_argument('--n_clients', default=5, type=int, help="Number of clients; smallest K-means clusters will be combined until n_clients clusters remain")
    parser.add_argument('--balance_data', default=False, action='store_true', help="Do data balancing by only keeping as many samples from majority as are in minority class")
    parser.add_argument('--save_intermediate', default=True, action='store_false', help="Save data as numpy arrays before preprocessing. This speeds up further preprocessing runs significantly")
    parser.add_argument('--random_seed', default=2303, type=int, help="Random seed")


    args = parser.parse_args()

    # a bit of workaround
    sys.path.append('/Users/mixheikk/Documents/mimic3-benchmarks')
    from mimic3benchmark.readers import InHospitalMortalityReader
    #from mimic3benchmark.readers import LengthOfStayReader
    from mimic3models import common_utils

    #do_data_split_mortality(args)
    do_data_split_LOS(args)






