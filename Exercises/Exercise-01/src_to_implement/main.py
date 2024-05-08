from pattern import Checker, Circle, Spectrum
from generator import ImageGenerator


def main():

    # 1.2 Checkerboard Test
    checkerboard = Checker(16, 2)
    checkerboard.show()

    # 1.3 Circle Test
    circle = Circle(resolution=1000, radius=100, position=(500, 500))
    circle.show()

    # 1.4 Color Spectrum Test
    spectrum = Spectrum(200)
    spectrum.show()

    # 2.1 Image Generator Test
    image_generator = ImageGenerator("exercise_data", "Labels.json", 12, (32, 32, 3), True, True, True)
    image_generator.show()

if __name__ == "__main__":
    main()