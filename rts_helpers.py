import numpy as np
import pydicom

def load_rts(rts_dicom, max_contour_points=None, logger=None):
    header_data = rts_dicom.StructureSetROISequence
    contour_data = rts_dicom.ROIContourSequence
    rts_metadata = []
    all_contours =[]
    for i in range(len(contour_data)):
        header = header_data[i]
        roi_name = header.ROIName
        roi_number = header.ROINumber
        frame_of_reference = header.ReferencedFrameOfReferenceUID

        number_o_contours = sum([int(x[('3006', '0046')].value) for x in contour_data[i].ContourSequence])

        if max_contour_points is None:
            pass
        elif number_o_contours > max_contour_points:
            if logger is not None:
                logger.info('Skipped %s [%s Contour Points]' % (roi_name, number_o_contours))
            continue

        single_contour = load_contour_points(contour_data[i].ContourSequence)

        all_contours.append([single_contour, contour_data[i].ReferencedROINumber])

        rts_metadata.append([roi_number, frame_of_reference, roi_name])

    rts_data = contour_to_roi(rts_metadata, all_contours)

    return rts_data


def load_contour_points(contours):
    contour_data = []
    for polygon in contours:
        referenced_image = []
        if len(polygon.ContourImageSequence) > 1:
            print('WARNING: Multiple Referenced Images!')

        for contour_image in polygon.ContourImageSequence:
            referenced_image.append(contour_image.ReferencedSOPInstanceUID)

        contour_data.append([polygon.ContourData, referenced_image])

    return contour_data


def contour_to_roi(rts_export,all_contours):
    final_list = []
    for contour in all_contours:
        index = (np.where(np.array(rts_export)[:,0] == str(contour[1]))[0][0])
        final_list.append([contour[0], rts_export[index][1:]])

    return final_list
