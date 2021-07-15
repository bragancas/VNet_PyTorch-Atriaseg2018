from scipy.ndimage import _ni_support 
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, generate_binary_structure
import numpy

# Copyright (C) 2013 Oskar Maier
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# author Oskar Maier
# version r0.1.1
# since 2014-03-13
# status Release

def Distance(predicted_result, reference_ground, voxelspacing = None, connectivity = 2):
    """
    Computes the Hausdorff Distance defined as the maximum surface distance between the binary objects in 
    predicted and reference images across the batch.
    """
    assert predicted_result.shape == reference_ground.shape, 'Provided binary images must have same dimensions'
    #N, _, D, _, _ = predicted_result.shape
    
    #haussdorf_dist_slices = []    
    #for n in range(N):
    #    for d in range(D):
    hd1 = surface_distances(predicted_result[n,:,d,:,:], reference_ground[n,:,d,:,:], voxelspacing, connectivity).max()
    hd2 = surface_distances(reference_ground[n,:,d,:,:], predicted_result[n,:,d,:,:], voxelspacing, connectivity).max()
    haussdorf_dist += [max(hd1, hd2)]
    
    return haussdorf_dist
    #return numpy.mean(haussdorf_dist_slices)
    
def surface_distances(predicted_result, reference_ground_truth, voxelspacing = 0.625, connectivity = 2):
    """
    The distances between the surface voxel of binary objects in predicted_result and their
    nearest partner in the binary objects in reference_ground_truth.
    """
    result = numpy.atleast_1d(predicted_result.astype(numpy.bool))
    reference = numpy.atleast_1d(reference_ground_truth.astype(numpy.bool))
    
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = numpy.asarray(voxelspacing, dtype = numpy.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()
    
    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)
    
    # validate for binary objects in the image arrays
    if 0 == numpy.count_nonzero(result):
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == numpy.count_nonzero(reference):
        raise RuntimeError('The second supplied array does not contain any binary object.')
    
    # extract 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure = footprint, iterations = 1)
    reference_border = reference ^ binary_erosion(reference, structure = footprint, iterations = 1)
    
    # compute surface distance from result border to reference border
    distance_values = distance_transform_edt(~reference_border, sampling = voxelspacing)
    surface_distance = distance_values[result_border]
    
    return surface_distance
    
def JaccardCoefficient(predicted_result, reference_ground):
    """
        The Jaccard coefficient between the object(s) in `result` and the object(s) in `reference`. It ranges from 0 (no overlap) to 1 (perfect overlap).    
    """
    result = predicted_result.astype(numpy.bool)
    reference = reference_ground.astype(numpy.bool)
    
    intersection = numpy.count_nonzero(result & reference)
    union = numpy.count_nonzero(result | reference)
    
    jaccard_similarity = float(intersection) / float(union)
    
    return jaccard_similarity
    
def Precision(predicted_result, reference_ground):
    """
        The precision between two binary datasets, here mostly binary objects in images,
        which is defined as the fraction of retrieved instances that are relevant. The
        precision is not symmetric.
    """
    result = predicted_result.astype(numpy.bool)
    reference = reference_ground.astype(numpy.bool)
    
    tp = numpy.count_nonzero(result & reference)
    fp = numpy.count_nonzero(result & ~reference)
    
    try:
        precision = tp / float(tp + fp)
    except ZeroDivisionError:
        precision = 0.0
    
    return precision
    
def Recall(predicted_result, reference_ground):
    """
        The recall defined as the fraction of relevant instances between two binary datasets 
        across a batch of predicted and reference images.
    """
    result = predicted_result.astype(numpy.bool)
    reference = reference_ground.astype(numpy.bool)
    
    tp = numpy.count_nonzero(result & reference)
    fn = numpy.count_nonzero(~result & reference)
    
    try:
        recall = tp / float(tp + fn)
    except ZeroDivisionError:
        recall = 0.0
    
    return recall
    
def Specificity(predicted_result, reference_ground):
    """
        The specificity defined as the fraction of correctly returned negatives between two binary datasets
        across a batch of predicted and reference images.
    """
    result = predicted_result.astype(numpy.bool)
    reference = reference_ground.astype(numpy.bool)
    
    tn = numpy.count_nonzero(~result & ~reference)
    fp = numpy.count_nonzero(result & ~reference)
    
    try:
        specificity = tn / float(tn + fp)
    except ZeroDivisionError:
        specificity = 0.0
    
    return specificity