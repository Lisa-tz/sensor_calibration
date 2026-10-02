import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml

R2D = 180.0 / np.pi


def allan_deviation(rate, fs, n_points=60):
    """Overlapping Allan deviation of rate data (same units as input)."""
    theta = np.concatenate(([0.0], np.cumsum(rate) / fs))
    n = len(theta)
    ms = np.unique(np.logspace(0, np.log10((n - 1) // 2), n_points).astype(int))
    taus, adev = [], []
    for m in ms:
        d = theta[2 * m:] - 2 * theta[m:-m] + theta[:-2 * m]
        tau = m / fs
        avar = np.sum(d ** 2) / (2.0 * tau ** 2 * len(d))
        taus.append(tau)
        adev.append(np.sqrt(avar))
    return np.array(taus), np.array(adev)


def main():
    p = argparse.ArgumentParser(description='Static gyro analysis from imu_recorder CSV')
    p.add_argument('csv')
    p.add_argument('--name', default='imu', help='sensor name, used for output files')
    p.add_argument('--out', default='.', help='output directory')
    args = p.parse_args()

    d = np.genfromtxt(os.path.expanduser(args.csv), delimiter=',', names=True)
    t = d['t']
    fs = 1.0 / np.median(np.diff(t))
    os.makedirs(os.path.expanduser(args.out), exist_ok=True)

    res = {
        'sensor': args.name,
        'sample_rate_hz': float(fs),
        'duration_s': float(t[-1] - t[0]),
        'gyro': {},
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    for axis in 'xyz':
        r = d['g' + axis]
        if np.allclose(r, 0.0):
            print(f'gyro {axis}: all zeros, skipping')
            continue
        bias, std = float(np.mean(r)), float(np.std(r))
        taus, adev = allan_deviation(r, fs)
        arw = float(np.interp(1.0, taus, adev)) * R2D * 60.0  # deg/sqrt(h)
        i = int(np.argmin(adev))
        min_adev_dph = float(adev[i]) * R2D * 3600.0
        res['gyro'][axis] = {
            'bias_rad_s': bias,
            'bias_deg_h': bias * R2D * 3600.0,
            'noise_std_rad_s': std,
            'arw_deg_per_sqrt_h': arw,
            'min_adev_deg_h': min_adev_dph,
            'bias_instability_deg_h': min_adev_dph / 0.664,
            'min_adev_tau_s': float(taus[i]),
        }
        ax.loglog(taus, adev * R2D * 3600.0, label=f'gyro {axis}')
        print(f'gyro {axis}: bias={bias*R2D:.5f} deg/s ({bias*R2D*3600:.1f} deg/h), '
              f'ARW={arw:.3f} deg/sqrt(h), min ADEV={min_adev_dph:.2f} deg/h @ {taus[i]:.0f} s')

    acc = np.stack([d['ax'], d['ay'], d['az']], axis=1)
    if not np.allclose(acc, 0.0):
        mean = acc.mean(axis=0)
        res['accel_mean_mps2'] = [float(v) for v in mean]
        res['accel_mean_norm_mps2'] = float(np.linalg.norm(mean))
        print(f'accel mean = {mean}, |a| = {np.linalg.norm(mean):.4f} m/s^2')

    ax.set_xlabel('tau (s)')
    ax.set_ylabel('Allan deviation (deg/h)')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    ax.set_title(f'{args.name} gyro Allan deviation')
    out = os.path.expanduser(args.out)
    fig.savefig(os.path.join(out, f'{args.name}_allan.png'), dpi=150)
    with open(os.path.join(out, f'{args.name}_gyro_calibration.yaml'), 'w') as f:
        yaml.safe_dump(res, f, sort_keys=False)
    print(f'Wrote results to {out}')


if __name__ == '__main__':
    main()
