
import argparse

from fourier import Fourier_serie
from path_to_points import path_to_points


def main():
    parser = argparse.ArgumentParser(description='Fourier drawing')
    parser.add_argument('--svg', type=str, required=True,
                        help='svg file')
    parser.add_argument('--ppp', type=float, required=True,
                        help='number of points per pixel')
    
    args = parser.parse_args()

    
    print('Convert SVG file to points ...')
    points, t_valid = path_to_points(args.svg, args.ppp)

    print('Deconpose Fourier series ...')
    P = points.shape[0]
    N = P//3
    FS = Fourier_serie(points, P)
    FS.decompose(N)
    FS.inference()

    print('Drawing ...')
    FS.draw(t_valid)

if __name__ == "__main__":
    main()
