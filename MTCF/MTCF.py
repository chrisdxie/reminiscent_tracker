import numpy as np
import cv2
import tensorflow as tf
import os, sys
import skimage
import sklearn
from sklearn.decomposition import PCA
from time import time
from . import util
from scipy.stats import multivariate_normal

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import AveragePooling2D
from keras.backend import clear_session, get_session
from keras.models import Model

tf.logging.set_verbosity(tf.logging.WARN) # [DEBUG, INFO, WARN, ERROR, FATAL]

def compute_datapoint_weights(l, m):
    """ Computes unnormalized datapoint weights in the same way as CCOT

        alpha_j = alpha_{j-1} / (1-l)
    """
    alpha = [1]
    for i in range(m-1):
        new_alpha = alpha[-1] / (1 - l) 
        alpha.append(new_alpha)
    alpha = np.array(alpha)

    return alpha


class MTCF:
    """ Multi-Template Correlation Filter Tracker (MTCF)

        self.trackers is a dictionary of dictionaries. Each dictionary represents a tracker and contains these keys:

            - correlation_filter:       TensorFlow Variable for the filter itself
            - correlation_maps:         TensorFlow Tensor for calculating correlation response
            - datapoint_weights:        TensorFlow Placeholder for datapoint weights for training filter
            - image_input:              TensorFlow Placeholder for input image to calculate response or loss
            - images:                   Numpy array of training images
            - lbfgsb_train:             TensorFlow object for training the filter using L-BFGS-B
            - loss:                     TensorFlow Tensor for loss function
            - num_images:               Number of images the tracker has been trained on
            - weight:                   Float of tracker weight
    """

    def __init__(self, first_frame, first_gt_bbox, params):

        self.first_frame = first_frame.astype(np.float32)
        self.first_gt_bbox = first_gt_bbox # centered rectangle format
        self.params = params.copy()

        # Get some parameters from this
        self.im_height, self.im_width, self.im_channels = self.first_frame.shape

        # Initialize tracker dictionary
        self.trackers = {}

        # Load feature lookup tables if specified 
        if 'CN' in self.params['im_rep']:
            self.CN_lookup = np.loadtxt(params['color_names_filepath'])[:, 3:]

        # Load pre-trained network if specified
        if self.params['im_rep'] == 'VGG16':
            clear_session() # Clear the session so Keras doesn't raise an error if you try to load the model twice
            base_model = VGG16(weights='imagenet', include_top=False)
            x = base_model.get_layer(self.params['VGG_layer_name']).output
            self.deep_model = Model(inputs=base_model.input, outputs=x)

        # Get Keras session and use that
        self.sess = get_session()

        # Set some initial settings
        self.initial_setup(self.first_frame, self.first_gt_bbox)

        # Initialize tracker 1
        stime = time()
        self.tracker_initialization(1)
        if self.params['verbosity'] >= 1:
            util.stderr_print("Tracker 1 initialization took {0} seconds".format(round(time() - stime, 3)))




    ##### START: Useful Utilities #####

    def construct_label_map(self, wh):
        """ Create a 2D Gaussian label map in range [0,1]. label_center is (x,y) coordinates
        """

        width = wh[0]
        height = wh[1]
        label_center = (np.array([width, height]) - 1) / 2. # -1 for 0-based python indexing

        t_height, t_width = self.template_shape[:2]

        # Create labels for correlation map
        indices = util.build_matrix_of_indices(width, height)
        mn = multivariate_normal(mean=label_center[::-1], cov = np.power(self.params['label_sigma_factor'],2) * t_width*t_height * np.eye(2))
        label_map = util.normalize(mn.pdf(indices))

        return np.expand_dims(label_map, axis=0)

    def construct_datapoint_weights(self, k):
        """ Construct normalized datapoint weights for tracker k
        """

        alpha = self.params['datapoint_weights'][-self.trackers[k]['num_images'] : ]
        alpha = alpha / np.sum(alpha)

        return np.expand_dims(np.expand_dims(alpha, axis=-1), axis=-1)

    ##### END: Useful Utilities #####




    def setup_PCA(self):
        """ Compute PCA for features
        """

        # Turn use_PCA to false and call get_image_patch. This returns the full number of channels to compute the principle components
        self.params['use_PCA'] = False
        img_cut_rep = self.get_image_patch(self.first_frame, self.position, self.search_window_base, 1.0, self.resized_search_window)
        self.params['use_PCA'] = True

        if self.params['im_rep'] == 'HOG+CN':

            # Extract the HOG and Color Name representations
            hog_img = img_cut_rep[..., :31] # Shape: [h, w, 31]
            cn_img = img_cut_rep[..., 31:]  # Shape: [h, w, 11]

            # Instantiate PCA objects
            hog_pca = PCA(n_components=self.params['HOG_PCA_num_components'])
            cn_pca = PCA(n_components=self.params['CN_PCA_num_components'])

            # Fit both PCAs to frame 1
            hog_pca.fit(hog_img.reshape(np.prod(hog_img.shape[:2]), 31))
            cn_pca.fit(cn_img.reshape(np.prod(cn_img.shape[:2]), 11))

            # Save the PCA objects
            self.hog_pca = hog_pca
            self.cn_pca = cn_pca

        elif self.params['im_rep'] == 'VGG16':

            # Instantiate PCA object
            deep_pca = PCA(n_components=self.params['VGG_PCA_num_components'])

            # Fit PCA to frame 1
            deep_pca.fit(img_cut_rep.reshape(np.prod(img_cut_rep.shape[:2]), img_cut_rep.shape[2]))

            # Save the PCA object
            self.deep_pca = deep_pca

        else:
            raise Exception("PCA not implemented for this image representation: {0}".format(self.params['im_rep']))


    def get_image_patch(self, img, position, window_size, scale_factor, target_size, force_feature=None):
        """ Extract an image patch from the full image.
            Then call get_feature_representation() to extract the features.

            Note: The following are equivalent
                Matlab: A([1,3,5],[1,3,5])      # returns a view of the data
                Numpy: A[[0,2,4],:][:,[0,2,4]]  # returns a copy of the data

            @param img: original image, original scale
            @param position: position of image crop in (x,y) coordinates
            @param window_size: (width, height) of desired search window
            @param scale_factor: scale of object currently
            @param target_size: (width, height) to resize cropped image to
        """

        scaled_window = np.round(scale_factor * window_size)
        # Make sure the window isn't too small..
        if scaled_window[0] < 2:
            scaled_window[0] = 2.
            util.stderr_print('Scaled_window width is too small...')
        if scaled_window[1] < 2:
            scaled_window[1] = 2.
            util.stderr_print('Scaled_window height is too small...')

        right_boundary = img.shape[1] - 1
        bottom_boundary = img.shape[0] - 1

        # The following biases the window towards the bottom right about half a pixel
        left = int(np.floor(position[0]) - np.floor(scaled_window[0] / 2.))
        top = int(np.floor(position[1]) - np.floor(scaled_window[1] / 2.))
        right = int(np.floor(position[0]) + np.ceil(scaled_window[0] / 2.))
        bottom = int(np.floor(position[1]) + np.ceil(scaled_window[1] / 2.))

        width_slice = np.arange(left, right)
        height_slice = np.arange(top, bottom)

        # Check for out-of-bounds coordinates, set them to values at borders
        width_slice[width_slice < 0] = 0
        height_slice[height_slice < 0] = 0
        width_slice[width_slice > right_boundary] = right_boundary
        height_slice[height_slice > bottom_boundary] = bottom_boundary

        if np.prod(scaled_window)/np.prod(target_size) > 1.:
            interp = cv2.INTER_AREA # Shrinking
        else:
            interp = cv2.INTER_LINEAR # Zooming

        target_size = np.round(target_size).astype(int)
        img_cut = util.extract_patch_from_slices(img, width_slice, height_slice)
        img_cut = cv2.resize(img_cut, tuple(target_size), interpolation=interp)

        return self.get_feature_representation(img_cut, force_feature=force_feature)


    def get_feature_representation(self, img_cut, force_feature=None):
        """ Extract a feature representation from the image patch

            @param img_cut: a Numpy array of size [H x W x C] to be processed
        """

        # Mechanism for overriding feature choice in self.params
        if force_feature is not None:
            feature = force_feature
        else:
            feature = self.params['im_rep']


        # HOG features
        if feature == 'HOG':
            img_rep = util.hog(img_cut, self.params['HOG_cell_size'])                


        # HOG features with cell size 4
        elif feature == 'HOG4':
            img_rep = util.hog(img_cut, 4)


        # HOG and ColorName features
        elif feature == 'HOG+CN':

            # Extract HOG features
            hog_rep = util.hog(img_cut, self.params['HOG_cell_size']) # Shape: [h/4, w/4, 31]

            # Extract ColorName features
            red = img_cut[..., 0]; green = img_cut[..., 1]; blue = img_cut[..., 2]
            index_im = np.floor(red.flatten(order='F') / 8) + 32*np.floor(green.flatten(order='F') / 8) + 32*32*np.floor(blue.flatten(order='F') / 8)
            index_im = index_im.astype(int)
            color_rep = self.CN_lookup[index_im, :].reshape(img_cut.shape[0], img_cut.shape[1], self.CN_lookup.shape[1], order='F')

            # Average ColorName features over patches
            color_rep = skimage.util.view_as_blocks(color_rep, block_shape=(self.params['HOG_cell_size'], self.params['HOG_cell_size'], color_rep.shape[2])) # Shape: [h/4, w/4, 1, 4, 4, 11]
            color_rep = np.mean(color_rep[:,:,0,:,:,:], axis=(2,3)) # Shape: [h/4, w/4, 11]

            # Reduce dimensionality
            if self.params['use_PCA']:
                hog_rep_pca = self.hog_pca.transform(hog_rep.reshape(np.prod(hog_rep.shape[:2]), 31))
                hog_rep = hog_rep_pca.reshape(hog_rep.shape[0], hog_rep.shape[1], self.hog_pca.n_components)
                color_rep_pca = self.cn_pca.transform(color_rep.reshape(np.prod(color_rep.shape[:2]), 11))
                color_rep = color_rep_pca.reshape(color_rep.shape[0], color_rep.shape[1], self.cn_pca.n_components)

            # Concatenate the HOG and CN representations
            img_rep = np.concatenate([hog_rep, color_rep], axis=2)


        # VGG16 features
        elif feature  == 'VGG16':

            # Preprocess 
            img_cut = preprocess_input(np.expand_dims(img_cut.astype(np.float32), axis=0))

            # Extract deep network features
            img_rep = self.deep_model.predict(img_cut)[0, ...] # Shape: [H, W, channels]

            # Reduce dimensionality
            if self.params['use_PCA']:
                img_rep_pca = self.deep_pca.transform(img_rep.reshape(np.prod(img_rep.shape[:2]), img_rep.shape[2]))
                img_rep = img_rep_pca.reshape(img_rep.shape[0], img_rep.shape[1], self.deep_pca.n_components)
                img_rep / np.linalg.norm(img_rep, axis=2, keepdims=True)



        else:
            raise Exception('Feature representation: {0} is not implemented'.format(self.params['im_rep']))



        return img_rep


    def initial_setup(self, initial_img, centered_bbox):
        """ Set up some initial settings

            @param initial_img: The initial RGB image. Original size/scale
            @param centered_bbox: The bounding box of the object template in 
                                  centered rectangle coordinates (center x, center y, width, height)
        """
        img = initial_img.astype(np.float32)

        center = centered_bbox[:2]-1 # (x,y) center, -1 for 0-based Python indexing
        bbox_width, bbox_height = centered_bbox[2:]


        # Save base template shape
        self.bbox_width = int(round(bbox_width))
        self.bbox_height = int(round(bbox_height))
        self.base_template_size = np.array([self.bbox_width, self.bbox_height])

        # Max/Min template size
        self.resized_template_size = np.minimum(self.base_template_size, self.params['max_template_sidelength'])
        self.resized_template_size = np.maximum(self.resized_template_size, self.params['min_template_sidelength'])

        # Make sure the resized template size is divisible by the reduction factor
        if 'HOG' in self.params['im_rep']: 
            reduction_factor = self.params['HOG_cell_size']
        elif self.params['im_rep'] == 'VGG16':
            exponent = int((self.params['VGG_layer_name'])[5]) - (1 if 'conv' in self.params['VGG_layer_name'] else 0) # Assumes name of output layer looks like "blockX_conv/pool"
            reduction_factor = np.power(2, exponent) 
            if self.params['verbosity'] >= 1:
                util.stderr_print("VGG reduction factor: {0}".format(reduction_factor))
                util.stderr_print("VGG output channels: {0}".format(int(self.deep_model.output.shape[3])))
                util.stderr_print("PCA output channels: {0}".format(self.params['VGG_PCA_num_components']))
        self.resized_template_size = np.ceil(self.resized_template_size / float(reduction_factor)) * float(reduction_factor)

        # Keep track of how we changed aspect ratio of object
        self.aspect_ratio_factor = self.resized_template_size / self.base_template_size.astype(np.float64) 

        # Compute search window size
        if self.params['search_shape'] == 'proportional':
            self.search_window_base = (self.base_template_size * self.params['search_window_factor'])
            self.resized_search_window = np.round(self.search_window_base * self.aspect_ratio_factor)
        elif self.params['search_shape'] == 'square':
            if self.params['search_window_factor'] < 4:
                raise Exception("Search window factor needs to be >= 4 for square search area...")
            sw_sidelength = np.round(np.sqrt(np.prod(self.base_template_size * self.params['search_window_factor'])))
            self.search_window_base = np.array([sw_sidelength, sw_sidelength])
            self.resized_search_window = np.round(self.search_window_base * self.aspect_ratio_factor)
        else:
            raise Exception('Search shape: {0} is not implemented'.format(self.params['search_shape']))

        # adjust sizes for feature representation
        self.resized_search_window = np.round(self.resized_search_window / reduction_factor) * reduction_factor # Make sure the resized search window is divisible by reduction_factor
        self.search_window_base = np.round(self.resized_search_window / self.aspect_ratio_factor) # Adjust search window base to reflect
        self.rep_resized_search_window = np.round(self.resized_search_window / reduction_factor).astype(int)  # Dimensions of resized image patch after computing feature map (e.g. HOG4 reduces spatial resolution by 4)

        # Hanning window
        self.hann_window = np.outer(np.hanning(self.rep_resized_search_window[1]), np.hanning(self.rep_resized_search_window[0]))
        self.hann_window = np.expand_dims(self.hann_window, axis=-1)

        # Get position
        self.position = center

        # Setup PCA if need be
        if self.params['use_PCA']:
            self.setup_PCA()

        # Extract the template shape
        self.template_shape = self.get_image_patch(img, center, self.base_template_size, 1.0, self.resized_template_size).shape
        if self.params['verbosity'] >= 0:
            util.stderr_print("Template size: {0}".format(self.template_shape))

        # Initial scale stuff
        self.params['max_scale_factor'] = np.power(self.params['scale_step'], np.floor( np.log(np.min( np.array([self.im_width, self.im_height], dtype=np.float32)/self.first_gt_bbox[2:] )) / np.log(self.params['scale_step']) ))
        self.params['min_scale_factor'] = np.power(self.params['scale_step'], np.ceil(  np.log(np.max(5/(2*self.first_gt_bbox[2:]))) / np.log(self.params['scale_step']) ))
        self.current_scale_factor = 1.0
        ss = np.arange(1, self.params['num_scales']+1) - np.ceil(self.params['num_scales']/2.)
        self.scale_factors = np.power(self.params['scale_step'], ss)

        # Get first training image in feature representation
        img_cut_rep = self.get_image_patch(self.first_frame, self.position, self.search_window_base, self.current_scale_factor, self.resized_search_window)
        img_cut_rep *= self.hann_window # Apply a Hann window
        self.image_patches = np.array([img_cut_rep])

        # Precompute datapoint weights
        self.params['datapoint_weights'] = compute_datapoint_weights(self.params['image_learning_rate'], self.params['max_tracker_images'])

        # Keep track of frame number
        self.frame_num = 1


    def normalized_tracker_weights(self):
        """ Compute normalized tracker weights
        """
        tracker_weights = {k : self.trackers[k]['weight'] * self.trackers[k]['num_images'] for k in list(self.trackers.keys())}
        total_weight = np.sum(list(tracker_weights.values()))
        for k in list(tracker_weights.keys()):
            tracker_weights[k] /= total_weight
        return tracker_weights

    def update_tracker_weights(self):
        """ Update and assign tracker weights based on decay factor
        """
        num_trackers = len(list(self.trackers.keys()))
        weights = np.power(1 - self.params['tracker_weight_decay'], np.arange(num_trackers)) # this looks like: [1, 1-\gamma, (1-\gamma)^2, ...]
        for index, k in enumerate(sorted(list(self.trackers.keys()), reverse=True)):
            self.trackers[k]['weight'] = weights[index]

    def build_filter_network(self, k):
        """ Build the correlation filter network for a new tracker in TensorFlow.

            @param k: Tracker identifier
        """

        t_channels = self.template_shape[2]

        # Initialize correlation filter to zeros. Store in TF Variable for learning
        correlation_filter = tf.Variable(np.expand_dims(np.zeros(self.template_shape), axis=-1).astype(np.float32)) # Shape: [height, width, channels, 1]

        # Initialize the correlation filter variable in TensorFlow 
        init_new_var_op = tf.variables_initializer([correlation_filter])
        self.sess.run(init_new_var_op)

        # Compute correlation heatmap TF Tensor
        image_input = tf.placeholder(tf.float32, [None, self.rep_resized_search_window[1], self.rep_resized_search_window[0], t_channels]) # None for batch size
        correlation_map = tf.nn.conv2d(image_input, correlation_filter, [1,1,1,1], "SAME", name='correlation' + str(k)) # Shape: [None, image_height, image_width, 1]
        correlation_maps = correlation_map[..., 0] # Shape: [None, image_height, image_width]

        # Create loss function and minimization stuff
        labels = self.construct_label_map(self.rep_resized_search_window) # Shape: [1, height, width]
        datapoint_weights = tf.placeholder(tf.float32, [None, 1, 1]) # Weights for the images
        loss = (tf.reduce_sum( datapoint_weights * tf.square(correlation_maps - labels) ) + \
                    self.params['reg_lambda'] * tf.reduce_sum(tf.square(correlation_filter)) ) / 2.

        lbfgsb_train = tf.contrib.opt.ScipyOptimizerInterface(loss,
                                                              var_list=[correlation_filter],
                                                              method='L-BFGS-B', 
                                                              options={'maxiter' : self.params['LBFGSB_max_learning_iters'],
                                                                       # 'disp' : 1,     # display 
                                                                       # 'gtol' : 1e-8,  # tolerance for norm of projected gradient
                                                                       # 'ftol' : 1e-15, # tolerance of change in function value
                                                                       'maxcor' : 10     # memory for L-BFGS-B approximation to Hessian
                                                                       })

        # Save the important stuff
        self.trackers[k].update( {'correlation_filter' : correlation_filter,
                                  'correlation_maps' : correlation_maps,
                                  'datapoint_weights' : datapoint_weights,
                                  'image_input' : image_input,
                                  'lbfgsb_train' : lbfgsb_train,
                                  'loss' : loss} )


    def tracker_initialization(self, k):
        """ Initialize tracker k
        """

        if self.params['verbosity'] >= 1:
            util.stderr_print("Creating new tracker, with id: {0}".format(k))

        # Initialize tracker as an empty dictionary
        self.trackers[k] = {}

        # Build the filter network
        self.build_filter_network(k)

        # Get initial training images
        self.trackers[k]['images'] = np.empty(((self.params['max_tracker_images'],) + self.image_patches.shape[1:]))
        if k == 1:
            self.trackers[k]['images'][:1,...] = self.image_patches[:1, ...] # just first image
            self.trackers[k]['num_images'] = 1
        else:
            self.trackers[k]['images'][:self.params['tracker_num_initial_images'], ...] = self.image_patches[-self.params['tracker_num_initial_images']:, ...] # last _ image patches
            self.trackers[k]['num_images'] = self.params['tracker_num_initial_images']

        # Perform initial learning
        self.trackers[k]['lbfgsb_train'].optimizer_kwargs['options']['maxiter'] = self.params['LBFGSB_max_initial_learning_iters']
        self.learn(k)
        self.trackers[k]['lbfgsb_train'].optimizer_kwargs['options']['maxiter'] = self.params['LBFGSB_max_learning_iters']

        # Set number of trackers
        self.K = k # Assume k is always the newest tracker. This increments the total number of created trackers
        self.current_tracker = self.K

        # Update tracker weights for all trackers
        self.update_tracker_weights()


    def learn(self, k):
        """ Run the learning (L-BFGS) for tracker k
        """
        stime = time()
        self.trackers[k]['lbfgsb_train'].minimize(self.sess, feed_dict = {self.trackers[k]['image_input'] : self.trackers[k]['images'][:self.trackers[k]['num_images'],...],
                                                                          self.trackers[k]['datapoint_weights'] : self.construct_datapoint_weights(k)})
        if self.params['verbosity'] >= 2:
            util.stderr_print("L-BFGS-B training took {0} seconds".format(round(time() - stime, 3)))


    def extract_scale_patches(self, img, scale_factors):
        """ Extract features from each scale patch

            Return a [num_scales x H x W x C] numpy array
        """

        # multi-resolution translation filter

        scale_patches = []
        for scale in scale_factors:
            img_cut_rep = self.get_image_patch(img, self.position, self.search_window_base, scale, self.resized_search_window)
            img_cut_rep = img_cut_rep * self.hann_window # Apply a Hann window
            scale_patches.append(img_cut_rep)
        return np.array(scale_patches)


    def track(self, img):
        """ Run the MTCF tracker on the image.
            Return a bounding box in centered rectangle format

            @param img: a Numpy array of [H x W x C]. Original image, original scale
        """
        img = img.astype(np.float32)
        interp = cv2.INTER_CUBIC




        ##### START: Compute translation and scale updates #####

        # Extract patches at each scale
        scaled_img_patches = self.extract_scale_patches(img, self.current_scale_factor * self.scale_factors)

        # Run each tracker at each scale
        final_corr_maps = {}
        for k in list(self.trackers.keys()):

            # Get the correlation maps for each scale for filter k
            final_corr_maps[k] = self.sess.run(self.trackers[k]['correlation_maps'], feed_dict={self.trackers[k]['image_input'] : scaled_img_patches})

            # Resize them to the right size
            original_size_final_corr_maps = []
            for ind in range(self.params['num_scales']):
                temp = cv2.resize(final_corr_maps[k][ind], tuple((self.search_window_base * self.current_scale_factor * self.scale_factors[ind]).astype(int)), 
                                            interpolation=interp) # resize to the right size
                original_size_final_corr_maps.append(temp)

            # Save the resized correlation maps
            final_corr_maps[k] = original_size_final_corr_maps

        tracker_weights = self.normalized_tracker_weights()

        # At each scale, aggregate the correlation maps over trackers with a linear combination using weights
        aggregated_heatmaps = []
        for ind in range(self.params['num_scales']):
            combined_heatmap = np.sum([tracker_weights[k] * final_corr_maps[k][ind] for k in list(self.trackers.keys())], axis=0)
            aggregated_heatmaps.append(combined_heatmap)

        # Find the translation and scale update
        scale_index = np.argmax(list(map(np.max, aggregated_heatmaps)))
        argmax = util.multidim_argmax_avg(aggregated_heatmaps[scale_index])
        origin = (np.array(aggregated_heatmaps[scale_index].shape)-1)/2.
        translation = (argmax - origin)[::-1] # Keep position in (x, y) format
        self.position = self.position + translation
        self.current_scale_factor *= self.scale_factors[scale_index]

        ##### END: Compute translation and scale updates #####




        ##### START: Tracker updates #####

        # Get image patch for learning
        img_cut_rep = self.get_image_patch(img, self.position, self.search_window_base, self.current_scale_factor, self.resized_search_window)
        img_cut_rep *= self.hann_window # Apply a Hann window
        self.image_patches = np.append(self.image_patches, np.array([img_cut_rep]), axis=0) # add image patch to set of images
        if self.image_patches.shape[0] > self.params['tracker_num_initial_images']: # only keep around enough to instantiate the next tracker
            self.image_patches = self.image_patches[-self.params['tracker_num_initial_images']:, ...]

        # See if we need a new tracker
        if self.trackers[self.current_tracker]['num_images'] >= self.params['max_tracker_images']:
            self.tracker_initialization(self.K+1)

            # Get rid of old trackers if we go over budget
            if len(list(self.trackers.keys())) > self.params['max_trackers']:
                oldest_tracker = min(self.trackers.keys())
                del self.trackers[oldest_tracker]

                if self.params['verbosity'] >= 1:
                    util.stderr_print("Removing Tracker {0}...".format(oldest_tracker))

        else: # Add image to current tracker

            # Update set of images. Don't go over budget on number of examples
            self.trackers[self.current_tracker]['images'][self.trackers[self.current_tracker]['num_images'], ...] = img_cut_rep
            self.trackers[self.current_tracker]['num_images'] += 1

            # Update the filter for the most recent tracker
            if self.frame_num % self.params['num_frames_between_training'] == 0:
                self.learn(self.current_tracker)

        ##### END: Tracker updates #####




        self.frame_num += 1
        predicted_bbox = np.concatenate([self.position+1, [self.bbox_width*self.current_scale_factor, self.bbox_height*self.current_scale_factor]]) # +1 for 0-based Python indexing
        return predicted_bbox


