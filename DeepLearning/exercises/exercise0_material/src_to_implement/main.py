from pattern import Checker, Circle, Spectrum
from generator import ImageGenerator

if __name__ == '__main__':
    tile_size = 20
    resolution_board = 160
    checker = Checker(resolution_board, tile_size)
    board = checker.draw()
    board[:tile_size, :tile_size] = 1
    checker.show()

    resolution_image = 1000
    radius = 150
    position = (200, 400)
    circle = Circle(resolution_image, radius, position)
    circle_image = circle.draw()
    circle.show()

    resolution_spectrum = 1000
    spectrum = Spectrum(resolution_spectrum)
    spectrum_image = spectrum.draw()
    spectrum.show()

    label_path = './Labels.json'
    file_path = './exercise_data/'
    batch_size = 8
    image_size = [100, 100, 3]
    gen = ImageGenerator(file_path, label_path, batch_size, image_size, rotation=True, mirroring=True, shuffle=True)
    gen.next()
    gen.show()
    gen.next()
    gen.show()
    gen.next()
    gen.show()
