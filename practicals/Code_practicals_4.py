import numpy as np
import gryds

def data_augmentation(X_train, y_train, brightness, bspline):
    """
    Input:
    ---
    X_train (image patches) -> output of the extract_patches function.
    y_train (segmentations) -> output of the extract_patches function.
    ---
    Output:
    augmented_image     -> image patches including data augmentations
    augmented_segment   -> image segmentations including data augmentations
    """
    
    X_train_copy = X_train.copy()
    y_train_copy = y_train.copy()
    
    brightness_range = [-0.2, 0.2]

    for idx, (im, seg) in enumerate(zip(X_train_copy, y_train_copy)): 
        if brightness:
            # Apply random brightness adjustment
            brightness_offset = np.random.uniform(low=brightness_range[0], high=brightness_range[1])
            im = np.clip(im + brightness_offset, 0, 1)  # Ensuring pixel values remain between 0 and 1

        if bspline:
            # Define interpolator objects
            image_interpolator_r = gryds.Interpolator(im[..., 0], order=3)
            image_interpolator_g = gryds.Interpolator(im[..., 1], order=3)
            image_interpolator_b = gryds.Interpolator(im[..., 2], order=3)
            segment_interpolator = gryds.Interpolator(seg.squeeze(), order=3)

            # Define random displacement
            disp_x = np.random.uniform(low=-0.05, high=0.05, size=(3, 3))
            disp_y = np.random.uniform(low=-0.05, high=0.05, size=(3, 3))

            # Define bspline transformation object
            bspline_transformation = gryds.BSplineTransformation([disp_x, disp_y])

            # Obtain the augmented images
            augmented_image_r = image_interpolator_r.transform(bspline_transformation)
            augmented_image_g = image_interpolator_g.transform(bspline_transformation)
            augmented_image_b = image_interpolator_b.transform(bspline_transformation)
            augmented_segment = segment_interpolator.transform(bspline_transformation)

            # Construct the images correctly
            im = np.concatenate((np.expand_dims(augmented_image_r, -1),
                                 np.expand_dims(augmented_image_g, -1),
                                 np.expand_dims(augmented_image_b, -1)),
                                 axis=-1)
            augmented_segment = np.round(augmented_segment)
            seg = np.expand_dims(augmented_segment, -1)

        # Place augmented image back into patch array
        X_train_copy[idx, ...] = im
        y_train_copy[idx, ...] = seg

    return X_train_copy, y_train_copy