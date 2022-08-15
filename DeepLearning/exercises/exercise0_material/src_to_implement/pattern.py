import numpy as np
import matplotlib.pyplot as plt

class Checker:

    def __init__(self, resolution: int, tile_size: int):
        if resolution % (2 * tile_size) != 0:
            raise TypeError('resolution must be dividable by 2 * tile_size')

        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None
        self.number_of_half_titles_in_dim = int(self.resolution / (2 * self.tile_size))
    
    def draw(self) -> np.ndarray:
        white_title = 1
        black_title = 0
        field = [
            [black_title, white_title] * self.number_of_half_titles_in_dim, [white_title, black_title] * self.number_of_half_titles_in_dim
            ] * self.number_of_half_titles_in_dim
        checkboard = np.kron(field, np.ones((self.tile_size, self.tile_size)))
        self.output = checkboard.copy()
        return checkboard

    def show(self) -> None:
        if self.output is not None:
            plt.imshow(self.output, cmap='gray')
            plt.axis('off')
            plt.show()
        else:
            print('Use draw method before show')


class Circle:

    def __init__(self, resolution: int, radius: int, position: tuple):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = None
    
    def draw(self) -> np.ndarray:
        X, Y = np.meshgrid(np.arange(self.resolution), np.arange(self.resolution))
        mask = np.sqrt((X - self.position[0]) ** 2 + (Y - self.position[1]) ** 2) <= self.radius
        self.output = mask.copy()
        return mask

    def show(self) -> None:
        if self.output is not None:
            plt.imshow(self.output, cmap='gray')
            plt.axis('off')
            plt.show()
        else:
            print('Use draw method before show')


class Spectrum:
    def __init__(self, resolution: int):
        self.resolution = resolution
        self.output = None
    
    def draw(self):
        spectrum = np.zeros([self.resolution, self.resolution, 3])
        colors = np.linspace(0, 1, self.resolution)
        
        red = colors
        green = colors.reshape(self.resolution, 1)
        blue = np.flip(colors)

        spectrum[:, :, 0] = red
        spectrum[:, :, 1] = green
        spectrum[:, :, 2] = blue

        self.output = spectrum.copy()
        return spectrum

    def show(self) -> None:
        if self.output is not None:
            plt.imshow(self.output)
            plt.axis('off')
            plt.show()
        else:
            print('Use draw method before show')
