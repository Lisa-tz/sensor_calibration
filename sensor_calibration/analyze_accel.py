import argparse
import json
import math
import os

import numpy as np
import yaml

ORDER = ['+X', '-X', '+Y', '-Y', '+Z', '-Z']


def local_gravity(lat_deg, alt_m=0.0):
    """WGS84 normal gravity (Somigliana) with a free-air altitude correction, m/s^2."""
    phi = math.radians(lat_deg)
    s, s2 = math.sin(phi), math.sin(2 * phi)
    return 9.780327 * (1 + 0.0053024 * s * s - 0.0000058 * s2 * s2) - 3.086e-6 * alt_m


def calibrate(means, g):
    """Model: reading = M @ a_true + b. means: label -> 3-vector (m/s^2)."""
    r = {k: np.asarray(v, float) for k, v in means.items()}
    b = np.mean([r[k] for k in ORDER], axis=0)                      # bias
    M = np.zeros((3, 3))
    for j, ax in enumerate('XYZ'):
        M[:, j] = (r['+' + ax] - r['-' + ax]) / (2 * g)             # scale + cross-axis
    b_pairs = np.array([(r['+' + ax] + r['-' + ax]) / 2 for ax in 'XYZ'])
    return b, M, b_pairs


def main():
    ap = argparse.ArgumentParser(description='Six-position accelerometer calibration')
    ap.add_argument('json')
    ap.add_argument('--name', default='imu')
    ap.add_argument('--out', default='.')
    ap.add_argument('--lat', type=float, help='latitude in degrees, for local gravity')
    ap.add_argument('--alt', type=float, default=0.0, help='altitude in metres')
    ap.add_argument('--g', type=float, help='override gravity in m/s^2')
    args = ap.parse_args()

    if args.g:
        g = args.g
    elif args.lat is not None:
        g = local_gravity(args.lat, args.alt)
    else:
        g = 9.80665
        print('No --lat or --g given: using standard gravity 9.80665 (error up to ~0.3%).')

    with open(os.path.expanduser(args.json)) as f:
        poses = json.load(f)['poses']
    means = {k: v['mean'] for k, v in poses.items()}
    b, M, b_pairs = calibrate(means, g)
    K = np.linalg.inv(M)

    scale = np.diag(M)
    cross = M / scale[None, :]            # cross[i][j]: reading on axis i per unit gravity on axis j
    np.fill_diagonal(cross, 0.0)

    print(f'gravity used: {g:.5f} m/s^2')
    print('bias (m/s^2):     ', np.round(b, 5).tolist(), ' = ', np.round(b / 9.80665 * 1000, 2).tolist(), 'mg')
    print('scale factor:     ', np.round(scale, 5).tolist(),
          ' = error (%):', np.round((scale - 1) * 100, 3).tolist())
    print('cross-axis (%):   ', np.round(cross * 100, 3).tolist())
    print('bias estimates per face pair (X,Y,Z pairs), spread shows setup quality:')
    for ax, row in zip('XYZ', b_pairs):
        print(f'   {ax} pair: {np.round(row, 4).tolist()}')

    norms_before, norms_after = {}, {}
    for k in ORDER:
        raw = np.asarray(means[k])
        norms_before[k] = float(np.linalg.norm(raw))
        norms_after[k] = float(np.linalg.norm(K @ (raw - b)))
    print('|a| per pose before:', {k: round(v, 4) for k, v in norms_before.items()})
    print('|a| per pose after: ', {k: round(v, 4) for k, v in norms_after.items()})

    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    res = {
        'sensor': args.name,
        'gravity_mps2': float(g),
        'model': 'reading = M @ a_true + bias ; a_true = K @ (reading - bias)',
        'bias_mps2': [float(x) for x in b],
        'bias_mg': [float(x) for x in b / 9.80665 * 1000],
        'scale_factor': [float(x) for x in scale],
        'cross_axis_percent': (cross * 100).tolist(),
        'M': M.tolist(),
        'K': K.tolist(),
        'norm_before': norms_before,
        'norm_after': norms_after,
        'pose_means': means,
    }
    path = os.path.join(out, f'{args.name}_accel_calibration.yaml')
    with open(path, 'w') as f:
        yaml.safe_dump(res, f, sort_keys=False)
    print('Wrote', path)


if __name__ == '__main__':
    main()
