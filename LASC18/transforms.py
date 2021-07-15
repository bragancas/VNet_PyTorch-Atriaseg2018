import numpy as np 
import torch
import scipy 
from scipy.ndimage import rotate, map_coordinates, gaussian_filter, shift

class Normalise:
    """
    Apply Z-score normalization to a given input array based on specified mean and std values. If the provided image is a label no normalisation 
    is performed.
    """
    def __init__(self, mean, std, eps = 1e-6, **kwargs):
        """
        Args-
        mean(float): Mean value used for normalising the input array.
        std(float): Standard deviation used for normalising the input array.
        eps(float, optional): Minimum std. value to avoid zero division.

        Kwargs-
        image_type(str): image type 'raw', 'label' of the input image to decide transformation implementation.
        """
        self.mean = mean
        self.std = std
        self.eps = eps
        self.image_type = 'label' if kwargs.get('image_type') == 'label' else None
    
    def __call__(self, input_array):
        if self.image_type != 'label':
            return (input_array - self.mean) / np.clip(self.std, a_min = self.eps, a_max = None)
        
        return input_array
    
class HorizontalFlip:
    """
    Flips the given input array along the height axis.
    """
    def __init__(self, random_state, execution_probability = 0.4):
        """
        Args-
        random_state(np.random.mtrand): Random state object to decide on transformation execution.
        self.execution_probability = Probability of transformation execution. 
        """
        self.random_state = random_state
        self.execution_probability = execution_probability

    def __call__(self, input_array):
        if self.random_state.uniform() <= self.execution_probability:
            flipped_array = np.flip(input_array, axis = 1)
            return flipped_array
        
        return input_array
    
# order 0 performs a nearest neighbour interpolation, 1 is for bilinear
# reflect mode (4 3 2 1 | 1 2 3 4 | 4 3 2 1), constant default zeros
class RotateImage:
    """
    Rotates the given input array along a given axis using a rotation angle chosen at random from a range of -angle to +angle.
    """
    def __init__(self, random_state, angle = 3, axes = None, mode = 'constant', order = 0, execution_probability = 0.4, **kwargs):
        """
        Args-
        random_state(np.random.mtrand): Random state object to decide on transformation execution.
        angle(int, optional): Range from -angle to +angle to be selected from as rotation angle.
        axes(tuple, optional): List of tuples to select rotation axis
        mode(str, optional): Mode to fill in values after performing translation.
        order(int, optional): Order to Interpolate values if needed after translation (Range 0-5).
        execution_probability(float, optional): Probability of transformation execution. 
        
        Kwargs-
        image_type(str): image type 'raw', 'label' of the input image to decide interpolation order to be used.
        """
        if axes is None:
            #axes = [(1, 0), (2, 1), (2, 0)]
            axes = [(2, 1)] #Rotate Image axially
        else:
            assert isinstance(axes, list) and len(axes) > 0
        
        self.random_state = random_state
        self.angle = angle
        self.axes = axes
        self.mode = mode
        self.order = 0 if kwargs.get('image_type') == 'label' else order
        self.execution_probability = execution_probability
        
    def __call__(self, input_array):
        if self.random_state.uniform() <= self.execution_probability:
            axis = self.axes[self.random_state.randint(len(self.axes))]
            angle = self.random_state.randint(-self.angle, self.angle)
            rotated_array = rotate(input_array, angle, axes = axis, reshape = False, order = self.order, mode = self.mode)
            return rotated_array
        
        return input_array

class TranslateImage:
    """
    The provided input array is translated by an amount specified for each dimesnion in the shift parameter.
    """
    def __init__(self, random_state, shift = None, mode = 'constant', order = 0, execution_probability = 0.4, **kwargs):
        """
        Args-
        random_state(np.random.mtrand): Random state object to decide on transformation execution.
        shift(tuple, optional): List of tuple values to translate the given input array by.
        mode(str, optional): Mode to fill in values after performing translation.
        order(int, optional): Order to Interpolate values if needed after translation (Range 0-5).
        execution_probability(float, optional): Probability of transformation execution.
        
        Kwargs-
        image_type(str): Image type 'raw', 'label' of the input image to decide interpolation order to be used.
        """
        if shift is None:
            shift = [(0,5,0), (0,0,5), (0,-5,0), (0,0,-5), (0,5,5), (0,-5,-5), (0,5,-5), (0,-5,5)]
        else:
            assert isinstance(shift, tuple) and len(shift) > 0

        self.random_state = random_state
        self.shift = shift
        self.mode = mode
        self.order = 0 if kwargs.get('image_type') == 'label' else order 
        self.execution_probability = execution_probability
        
    def __call__(self, input_array):
        if self.random_state.uniform() <= self.execution_probability:
            translation = self.shift[self.random_state.randint(len(self.shift))]
            translated_array = scipy.ndimage.shift(input_array, shift = translation, order = self.order, mode = self.mode)
            return translated_array
        
        return input_array

class GaussianNoise:
    def __init__(self, random_state, scale = (0.0, 1.0), execution_probability = 0.4, **kwargs):
        """
        Args-
        random_state(np.random.mtrand): Random state object to decide on transformation execution.
        scale(tuple, optional): Range to select std from for the gaussian noise generation. 
        execution_probability(float, optional): Probability of transformation execution.

        Kwargs-
        image_type(str): Image type 'raw', 'label' of the input image to avoid carrying out transformation execution for label image.
        """
        self.random_state = random_state
        self.scale = scale
        self.kwargs = 'label' if kwargs.get('image_type') == 'label' else None
        self.execution_probability = execution_probability
        
    def __call__(self, input_array):
        if self.random_state.uniform() <= self.execution_probability:
            if self.kwargs != 'label':
                std = self.random_state.uniform(self.scale[0], self.scale[1])
                gaussian_noise = self.random_state.normal(0, std, size = input_array.shape)
                return input_array + gaussian_noise
            else:
                return input_array
        
        return input_array    
    
class ElasticDeformation:
    """
    Apply elastic deformations on a per-voxel mesh. Assumes ZYX axis order.
    Based on https://github.com/fcalvet/image_tools/blob/master/image_augmentation.py#L62
    """
    def __init__(self, random_state, order = 3, alpha = 2000, sigma = 50, execution_probability = 0.4, **kwargs):
        """
        Args-
        random_state(np.random.mtrand): Random state object to decide on transformation execution.
        order(int), optional: The order for interpolation (use 0 for labeled images).
        alpha(int, optional): Scaling factor for deformations.
        sigma(float, optional): Std deviation for the Gaussian filter.
        execution_probability(float, optional): Probability of transformation execution.

        Kwargs-
        image_type(str): Image type 'raw', 'label' of the input image to use suitable 'order'.
        """
        self.random_state = random_state
        self.order = 0 if kwargs.get('image_type') == 'label' else order 
        self.alpha = alpha
        self.sigma = sigma
        self.execution_probability = execution_probability
    
    def __call__(self, input_array):
        if self.random_state.uniform() <= self.execution_probability:
            z_dim, y_dim, x_dim = input_array.shape    
            dz, dy, dx = [gaussian_filter(self.random_state.randn(*input_array.shape),
                                          self.sigma, mode = "constant") * self.alpha for _ in range(3)]
            z, y, x = np.meshgrid(np.arange(z_dim), np.arange(y_dim), np.arange(x_dim), indexing = 'ij')
            indices = z + dz, y + dy, x + dx
            return map_coordinates(input_array, indices, order = self.order, mode = 'constant')
        
        return input_array            
    
class TorchTensor:
    """
    Adds additional 'channel' axis to the input and converts the given input numpy.ndarray into torch.Tensor.
    """
    def __init__(self, **kwargs):
        self.dtype = torch.uint8 if kwargs.get('image_type') == 'label' else torch.float32

    def __call__(self, input_array):
        input_array = np.expand_dims(input_array, axis = 0)
        
        return torch.Tensor(input_array.astype(np.float32)).to(dtype = self.dtype)