import argparse
import os
from icarus_diffusion.input_checks import check_input_file, check_input_range

def measure_diffusion():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, #nargs=1,
                        help="Input .root file from WaveformStudy")
    parser.add_argument("--angle_range", 
                        type=int, nargs=3, default=[26,80,2], help='''\
                        Three ints corresponding to (min, max, step) of angle scan range in degrees
                        Default values: [26,80,2]''')
    parser.add_argument("--long_range", 
                        type=float, nargs=3, default=[3.0,5.0,0.25], help='''\
                        Three floats corresponding to (min, max, step) DL scan range in cm^2/s
                        Default values: [3.0,5.0,0.25]''')
    parser.add_argument("--trans_range", 
                        type=float, nargs=3, default=[7.0,10.0,0.25], help='''\
                        Three floats corresponding to (min, max, step) DT scan range in cm^2/s
                        Default values: [7.0,10.0,0.25]''')
    args = parser.parse_args()

    # Input checking 
    try:
        args.input_file = check_input_file(args.input_file)
    except argparse.ArgumentTypeError as e:
        parser.error(str(e))
    
    try:
        check_input_range(args.angle_range)
        check_input_range(args.long_range)
        check_input_range(args.trans_range)
    except argparse.ArgumentTypeError as e:
        parser.error(str(e))

    measure_diffusion()














