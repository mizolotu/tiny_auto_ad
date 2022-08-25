import os

import serial, pandas, argparse
import numpy as np
import os.path as osp

from time import sleep

def receive_vector(start_marker, end_marker):

    msg = ''
    x = 'z'
    while ord(x) != start_marker:
        x = ser.read()

    while ord(x) != end_marker:
        if ord(x) != start_marker:
            msg = f'{msg}{x.decode("utf-8")}'
        x = ser.read()

    try:
        v = msg.split(',')
        v = [float(item) for item in v]
    except Exception as e:
        print(e)
        v = None

    return v, msg

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parse args')
    parser.add_argument('-p', '--port', help='Serial port', default='/dev/ttyACM1')
    parser.add_argument('-r', '--rate', help='Baud rate', default=115200, type=int)
    parser.add_argument('-s', '--start', help='Start marker', default=60, type=int)
    parser.add_argument('-e', '--end', help='End marker', default=62, type=int)
    parser.add_argument('-b', '--baseline', help='Number of vectors to record as a baseline', default=0, type=int)
    parser.add_argument('-n', '--nvectors', help='Number of vectors to record', default=2000, type=int)
    parser.add_argument('-f', '--fnames', help='File names', nargs='+', default=['500_500'])
    parser.add_argument('-l', '--label', help='File path', default='crack')
    parser.add_argument('-d', '--directory', help='Directory to store the dataset', default='data/bearing_fft_std')
    args = parser.parse_args()


    ser = serial.Serial(args.port, args.rate)

    for fname in args.fnames:

        input('Press Enter to record the next file...')

        # record the baseline

        if args.baseline is not None and args.baseline > 0:
            data = []
            n = 0
            while n < args.baseline:
                x, msg = receive_vector(args.start, args.end)
                if x is not None:
                    print(n, x)
                    data.append(x)
                    n += 1
                else:
                    print(msg)
            B = np.array(data)
            b_mean = np.mean(B, 0)

            print('Baseline data samples have been recorded!')
            input('Press Enter to continue...')

        else:
            b_mean = None

        # record the data

        data = []

        n = 0
        while n < args.nvectors:
            x, msg = receive_vector(args.start, args.end)
            if x is not None:
                print(n, x)
                data.append(x)
                n += 1
            else:
                print(msg)
        X = np.array(data)

        if b_mean is not None:
            X -= b_mean[None, :]

        # save the data

        fpath = osp.join(args.directory, args.label, f'{fname}.csv')
        dpath = osp.dirname(fpath)
        dirs = []
        while dpath != '':
            dirs.append(dpath)
            dpath = osp.dirname(dpath)
        for dir in dirs[::-1]:
            if not osp.isdir(dir):
                os.mkdir(dir)
        pandas.DataFrame(X).to_csv(fpath, header=None, index=None)

    ser.close()