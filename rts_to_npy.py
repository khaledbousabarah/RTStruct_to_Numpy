import os
import pydicom
import logging
from dicom2nifti import compressed_dicom
from rts_helpers import load_rts
import glob
import numpy as np
from matplotlib import path as PolyPath
from visualization_helpers import rts_study
import multiprocessing
from joblib import Parallel, delayed

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def load(path, max_number_of_coords=50000, parallel=True):
    def modality_check(x):
        try:
            dcm = pydicom.read_file(x)
            if dcm.Modality.lower() == "rtstruct":
                return True
            else:
                return False
        except pydicom.errors.InvalidDicomError:
            return False

    files = [x for x in glob.glob(os.path.join(path, '*')) if os.path.isfile(x)]
    logging.info('Found %s Files in path' % len(files))
    rtstruct = [x for x in files if modality_check(x)]

    if len(rtstruct) > 1:
        raise Exception('Found multiple RTStructs')
    elif len(rtstruct) == 0:
        raise Exception('Found no RTStructs')
    else:
        logging.info('Identified RTStruct')

    file = rtstruct[0]

    if compressed_dicom._is_compressed(file):
        compressed_dicom._decompress_dicom(file, output_file=file)

    rts_data = load_rts(pydicom.read_file(file), max_contour_points=max_number_of_coords, logger=logging)

    if len(rts_data) == 0:
        raise Exception('Found no contours in RTStruct')
    else:
        logging.info('RTStruct contains %s VOIs' % len(rts_data))

    reference_id = rts_data[0][1][0]

    def load_dicom(x, ref):
        try:
            if compressed_dicom._is_compressed(file):
                logging.info('Decompressing %s' % file)
                compressed_dicom._decompress_dicom(file, output_file=file)
                print('decompressing')

            dcm = pydicom.read_file(x)

            if dcm.FrameOfReferenceUID == ref:
                print('True')
                return dcm
            else:
                print('Wrong')
                return False
        except (pydicom.errors.InvalidDicomError, AttributeError):
            return False

    if parallel:
        num_cores = multiprocessing.cpu_count()
        dicoms = Parallel(n_jobs=num_cores, verbose=0)(delayed(load_dicom, check_pickle=True)(
            x, reference_id) for x in files)
    else:
        dicoms = [load_dicom(x, reference_id) for x in files]

    dicoms = [x for x in dicoms if x is not False]

    if len(dicoms) == 0:
        raise Exception('Found no Dicoms in specified path')
    else:
        logging.info('Found %s Dicoms ' % len(dicoms))

    def sort_dicoms(x):
        return sorted(x, key=lambda y: float(y[('0020', '0032')][2]))

    dicoms = sort_dicoms(dicoms)
    pixel_data = np.stack([x.pixel_array for x in dicoms])

    image_position_patient = [float(x) for x in dicoms[0][('0020', '0032')]]
    pixel_spacing = [float(x) for x in dicoms[0][('0028', '0030')]]
    slice_thickness = float(dicoms[0][('0018', '0050')].value)
    pixel_spacing.append(slice_thickness)

    def physical_to_pixels(x, image_position_patient, pixel_spacing):
        metadata = x[1]
        name = metadata[1]
        contour_data = x[0]

        coordinate_array = []
        for contours in contour_data:
            points = [float(x) for x in contours[0]]
            points = np.array(points).reshape(-1, 3)
            coordinate_array.append(points)

        coordinate_array = np.concatenate(coordinate_array, axis=0)
        coordinate_array = (coordinate_array - image_position_patient) / pixel_spacing

        return name, coordinate_array

    contour_data = {}

    for x in rts_data:
        name, data = physical_to_pixels(x, image_position_patient, pixel_spacing)
        contour_data[name] = data

    def contour_to_binary(contour, image_shape):
        mask = np.zeros(image_shape)
        x_dim, y_dim = image_shape[1], image_shape[2]
        non_empty_slices = [x for x in np.unique(contour[:, 2])]
        X, Y = np.meshgrid(np.linspace(1, x_dim, x_dim), np.linspace(1, y_dim, y_dim))
        XY = np.dstack((X, Y))
        XY_flat = XY.reshape((-1, 2))
        for z in non_empty_slices:
            pos = np.where(contour[:, 2] == z)[0]
            polygon = PolyPath.Path(contour[pos, :2])
            mask[int(z), ...] = polygon.contains_points(XY_flat).reshape(x_dim, y_dim)
        return mask

    masks = {}

    logging.info('Converting Contours to binary Masks')

    if parallel:
        tmp = Parallel(n_jobs=num_cores, verbose=0)(delayed(contour_to_binary, check_pickle=True)(
            val, pixel_data.shape) for key, val in contour_data.items())
        for i, (key, val) in enumerate(contour_data.items()):
            masks[key] = tmp[i]
    else:
        for key, val in contour_data.items():
            masks[key] = contour_to_binary(val, pixel_data.shape)

    logging.info('Done')

    return rts_study(pixel_data, masks)
