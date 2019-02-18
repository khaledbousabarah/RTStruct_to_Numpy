import numpy as np
from matplotlib import pyplot as plt

class rts_study():
    def __init__(self, image, masks):
        self.image = image
        self.items = [key.replace(' ', '_') for key, val in masks.items()]
        for key, val in masks.items():
            attr = key.replace(' ', '_')
            setattr(self, attr, rts_mask(val, key, self))

    def __repr__(self):
        VOIs = ''. join(['\t - %s\n' % x for x in self.items])
        return 'Study cotains %s Structures: \n %s ' % (len(self.items), VOIs)

    def __str__(self):
        VOIs = ''. join(['\t - %s\n' % x for x in self.items])
        return 'Study cotains %s Structures: \n %s ' % (len(self.items), VOIs)

class rts_mask():
    def __init__(self, mask, name, study):
        slice_counts = mask.sum(axis=2).sum(axis=1)
        max_slice = np.where((slice_counts == max(slice_counts)) & (slice_counts > 0))[0][0]
        self.max_slice = max_slice
        self.array = mask
        self.image = study.image
        self.name = name

    def __repr__(self):
        return self.name

    def npy(self):
        return self.array

    def plot(self, slice_to_plot=None, zoom=.5):
        total  = self.image.shape[0]
        if slice_to_plot is None:
            selected_slice = self.max_slice
        else:
            selected_slice = slice_to_plot

        image_to_plot = self.image[selected_slice, ...]
        mask_to_plot = self.array[selected_slice, ...]

        if np.sum(mask_to_plot) > 0:
            X, Y = np.where(mask_to_plot)
            X_range = image_to_plot.shape[0] - (np.max(X) - np.min(X))
            Y_range = image_to_plot.shape[1] - (np.max(Y) - np.min(Y))

            if zoom > 0:
                x_diff = int((image_to_plot.shape[0] - X_range * zoom) / 2)
                y_diff = int((image_to_plot.shape[0] - Y_range * zoom) / 2)
                image_to_plot = np.pad(image_to_plot, [(x_diff, x_diff) , (y_diff, y_diff)], mode='constant')
                mask_to_plot = np.pad(mask_to_plot, [(x_diff, x_diff) , (y_diff, y_diff)], mode='constant')

                X, Y = np.where(mask_to_plot)

                image_to_plot = image_to_plot[np.min(X)-x_diff:np.max(X)+x_diff, np.min(Y)-y_diff:np.max(Y)+y_diff]
                mask_to_plot = mask_to_plot[np.min(X)-x_diff:np.max(X)+x_diff, np.min(Y)-y_diff:np.max(Y)+y_diff]


        plt.imshow(image_to_plot, cmap='gray')
        plt.contour(mask_to_plot)
        plt.axis('off')
        plt.title('%s [%s / %s]' % (self.name, selected_slice, total))
        plt.show()
