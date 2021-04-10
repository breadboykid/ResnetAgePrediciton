from HelperFunctions import *
from DataProcessing import *
import SimpleITK as sitk
import random

# custom to tensor transform
class ToTensor(object):
    def __call__(self, data):
        img, label = data[DataEnum.Image], data[DataEnum.Label]

        data = {DataEnum.Image: torch.Tensor(img), DataEnum.Label: torch.Tensor(label)}
        return data


# permute the datasets into tensor compatible format
class PermutateTransform(object):
    def __call__(self, data):
        img, label = data[DataEnum.Image], data[DataEnum.Label]

        data = {DataEnum.Image: img.permute(2, 0, 1), DataEnum.Label: label}
        return data


class ElasticDeformationTransform(object):
    def __init__(self, num_controlpoints=5, sigma=1, plot_chart=False):
        self.num_controlpoints = num_controlpoints
        self.sigma = sigma
        self.plot_chart = plot_chart

    def create_elastic_deformation(self, image, num_controlpoints, sigma, seed):
        itkimg = sitk.GetImageFromArray(np.zeros(image.shape))

        trans_from_domain_mesh_size = [self.num_controlpoints] * itkimg.GetDimension()

        bspline_transformation = sitk.BSplineTransformInitializer(itkimg, trans_from_domain_mesh_size)

        params = np.asarray(bspline_transformation.GetParameters(), dtype=float)

        # maintain elastic deformation across channels, same brain will have the same elastic deformation
        np.random.seed(seed)
        params = params + np.random.randn(params.shape[0]) * self.sigma

        bspline_transformation.SetParameters(tuple(params))

        return bspline_transformation

    def create_grid(self, image):
        grid = sitk.GridSource(
            outputPixelType=sitk.sitkUInt16,
            size=(image.shape[1], image.shape[0]),
            sigma=(0.0001, 0.0001),
            gridSpacing=(int(image.shape[1] / 50), int(image.shape[0] / 50))
        )

        return grid

    def apply_elastic_deformation(self, image, seed):
        # We need to choose an interpolation method for our transformed image, let's just go with b-spline
        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(sitk.sitkBSpline)

        # Let's convert our image to an sitk image
        sitk_image = sitk.GetImageFromArray(image)
        sitk_grid = self.create_grid(image)

        # Specify the image to be transformed: This is the reference image
        resampler.SetReferenceImage(sitk_image)
        resampler.SetDefaultPixelValue(0)

        # Initialise the transform
        bspline_transform = self.create_elastic_deformation(image, self.num_controlpoints, self.sigma, seed)

        # Set the transform in the initialiser
        resampler.SetTransform(bspline_transform)

        # Carry out the resampling according to the transform and the resampling method
        out_img_sitk = resampler.Execute(sitk_image)
        out_grid_sitk = resampler.Execute(sitk_grid)

        # Convert the image back into a python array
        out_img = sitk.GetArrayFromImage(out_img_sitk)

        return out_img.reshape(image.shape), out_grid_sitk

    def elastic_deformation_handler(self, data):
        # Deform each channel separately
        data = data.permute(1, 2, 0).numpy()
        deformed_data = np.zeros_like(data)
        deformed_data_grid = []

        # seed to maintain same elastic deform across channels.
        seed = random.getrandbits(32)

        for i in range(data.shape[2]):
            img = data[:, :, i]
            trans_img, trans_grid = self.apply_elastic_deformation(img, seed)

            if self.plot_chart:
                self.plot_charts(img, trans_img, trans_grid, i)

            deformed_data[:, :, i] = trans_img
            deformed_data_grid.append(trans_grid)

        deformed_data = torch.from_numpy(deformed_data).permute(2, 0, 1)

        return deformed_data

    def plot_charts(self, img, def_img, grid, channel):
        rows = 2
        cols = 2

        fig, ax = plt.subplots(rows, cols, figsize=(20, 15))

        plot_dict = {
            0: {'title': 'Original Image', 'item': img},
            1: {'title': 'Deformed Image', 'item': def_img},
            2: {'title': 'Deformed Image with Grid Overlay', 'item': self.grid_and_image(grid, def_img)},
            3: {'title': 'Grid Deformation', 'item': sitk.GetArrayFromImage(grid)},
        }

        it_count = 0

        for row in range(rows):
            for col in range(cols):
                axis_title = plot_dict[it_count]['title']
                ax[row, col].axis('off')
                ax[row, col].set_title(f'{channel_dict[channel]} {axis_title}')
                ax[row, col].imshow(plot_dict[it_count]['item'])
                it_count += 1

    def grid_and_image(self, grid, image):
        grid_array = sitk.GetArrayViewFromImage(grid)
        return grid_array / np.max(grid_array) * image / np.max(image)
        # no need to flip image on my machine appears in correct orientation
        # return np.flip(grid_array/np.max(grid_array)*image/np.max(image), axis=0)

    def __call__(self, data):
        img = data[DataEnum.Image]

        result = {
            DataEnum.Image: self.elastic_deformation_handler(img),
            DataEnum.Label: data[DataEnum.Label]}

        return result


class GuassianNoiseTransform(object):
    def __init__(self, noise_level):
        self.noise_level = noise_level

    def noise_augment_handler(self, img):
        img = img.numpy()
        shift = abs(np.min(img)) if np.min(img) < 0 else 0

        # shift values to positive only for fourier transform manipulation
        img += shift

        inverse_img = np.fft.fftn(img)

        inverse_img_noisy = np.zeros_like(inverse_img)

        # ensure same noise behaviour applied across channels for the same brain (assumes different channel image are taken at the same time and environment)
        for i in range(inverse_img.shape[0]):
            inverse_img_noisy[i] = self.add_complex_noise(inverse_img[i])

        complex_img_noisy = np.fft.ifftn(inverse_img_noisy)

        magnitude_img_noisy = np.sqrt(np.real(complex_img_noisy) ** 2 + np.imag(complex_img_noisy) ** 2)

        # reverse shift
        magnitude_img_noisy -= shift

        result = torch.from_numpy(magnitude_img_noisy)
        return result

    def add_complex_noise(self, inverse_image):
        noise_level_linear = 10 ** (self.noise_level / 10)

        rand_arr = np.random.randn(inverse_image.shape[0], inverse_image.shape[1])

        # add complex noise into each channel
        real_noise = np.sqrt(noise_level_linear / 2) * rand_arr

        imaginary_noise = np.sqrt(noise_level_linear / 2) * 1j * rand_arr

        noise_addition = inverse_image + real_noise + imaginary_noise
        noisy_inverse_image = noise_addition

        print(noisy_inverse_image.shape)
        return noisy_inverse_image

    def __call__(self, data):
        img = data[DataEnum.Image]

        result = {
            DataEnum.Image: self.noise_augment_handler(img),
            DataEnum.Label: data[DataEnum.Label]}

        return result