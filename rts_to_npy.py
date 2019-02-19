import os
import pydicom
import logging
import dicom_decompresser
from rts_helpers import load_rts
import glob
import numpy as np
from matplotlib import path as PolyPath
from visualization_helpers import rts_study
import multiprocessing
from joblib import Parallel, delayed
from scipy.ndimage import zoom

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def load(path, max_number_of_coords=50000, parallel=False, downscaling=0):
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

    dicom_decompresser.run(file)

    rts_data = load_rts(pydicom.read_file(file), max_contour_points=max_number_of_coords, logger=logging)

    if len(rts_data) == 0:
        raise Exception('Found no contours in RTStruct')
    else:
        logging.info('RTStruct contains %s VOIs' % len(rts_data))

    reference_id = rts_data[0][1][0]

    def load_dicom(x, ref):
        try:
            dicom_decompresser.run(x)

            dcm = pydicom.read_file(x)

            if dcm.FrameOfReferenceUID == ref:
                return dcm
            else:
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
    if downscaling > 0:
        pixel_data = zoom(pixel_data, [downscaling] * 3)


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
        if downscaling > 0:
            data *= downscaling
        contour_data[name] = data

    def contour_to_binary(contour, image_shape):
        mask = np.zeros(image_shape)
        non_empty_slices = [x for x in np.unique(contour[:, 2])]
        for z in non_empty_slices:
            pos = np.where(contour[:, 2] == z)[0]
            contour_slice = contour[pos, :2]

            x0, y0 = int(np.floor(np.min(contour_slice[:, 0]))), int(np.floor(np.min(contour_slice[:, 1])))
            x1, y1 = int(np.ceil(np.max(contour_slice[:, 0]) )) + 1, int(np.ceil(np.max(contour_slice[:, 1]))) + 1
            x_shape, y_shape = x1-x0, y1-y0
            X, Y = np.meshgrid(np.linspace(x0, x1, x_shape), np.linspace(y0, y1, y_shape))
            XY = np.dstack((X, Y))
            XY_flat = XY.reshape((-1, 2))

            polygon = PolyPath.Path(contour_slice)
            mask[int(z), y0:y1, x0:x1] = polygon.contains_points(XY_flat).reshape(y_shape, x_shape)
        return mask

    masks = {}

    logging.info('Converting Contours to binary Masks')


    if parallel:
        tmp = Parallel(n_jobs=num_cores, verbose=0)(delayed(contour_to_binary, check_pickle=True)(
            val, pixel_data.shape) for key, val in contour_data.items())
        for i, (key, val) in enumerate(contour_data.items()):
            if np.sum(tmp[i]) == 0:
                logging.warning('%s is no valid Mask: Skipping' % key)
            else:
                masks[key] = tmp[i]
    else:
        for key, val in contour_data.items():
            data = contour_to_binary(val, pixel_data.shape)
            if np.sum(data) == 0:
                logging.warning('%s is no valid Mask: Skipping' % key)
            else:
                masks[key] = data

    logging.info('Done')

    return rts_study(pixel_data, masks)
