# -*- coding: utf-8 -*-
import sys, io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

"""Find EAST shots that can run the full ONETWO pipeline and produce TEIN/ENEIN/TIIN.

Uses readMDS_onetwo.readmds() to test data completeness — this handles:
- Signal name variations across campaigns (e.g. \ne_Refl vs \ne_ReflJ)
- RZ-to-rho mapping via geqdsk
- Data format conversion and NaN cleanup

Usage:
  python scan_compare_shots.py --start 80000 --end 160000
  python scan_compare_shots.py --start 140000 --end 150000 --min-txcs 10
  python scan_compare_shots.py --shots 81481 124092 140660 156005
"""

import sys, os, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from readMDS_onetwo import readmds
except ImportError as e:
    print(f"Cannot import readMDS_onetwo: {e}")
    print("This script requires the core pipeline modules to be available.")
    sys.exit(1)


def check_shot(shot, sample_times=None, min_txcs_channels=3, timeout=30):
    """Check if a shot has all diagnostics needed for the ONETWO pipeline.

    Uses readmds() which internally handles:
    - efit_east (gfile equilibrium, RZ mapping)
    - TS_EAST (Thomson scattering Te, ne)
    - TXCS_EAST (X-ray crystal spectrometer Ti)
    - ReflJ_EAST (Reflectometer ne)
    - Brem_EAST (Zeff, optional)

    Args:
        shot: Shot number
        sample_times: List of time points to try (defaults to TS time points)
        min_txcs_channels: Minimum valid TXCS channels required
        timeout: Timeout per attempt

    Returns:
        dict with status and details
    """
    result = {
        'shot': shot,
        'verdict': 'UNKNOWN',
        'ts_times': [],
        'has_te': False,
        'has_ne': False,
        'has_ti': False,
        'has_ne_refl': False,
        'txcs_channels': 0,
        'txcs_valid_channels': 0,
        'valid_times': [],
        'h_mode_times': [],
        'sample_count': 0,
        'issues': [],
    }

    try:
        import signal as sig_module

        def handler(signum, frame):
            raise TimeoutError(f"Shot {shot} timed out")

        sig_module.signal(sig_module.SIGALRM, handler)
        sig_module.alarm(timeout)
    except (ImportError, AttributeError):
        pass  # Windows doesn't have SIGALRM

    try:
        # Step 1: Get TS time points (Te data availability)
        try:
            from MDSplus.connection import Connection
            conn = Connection('202.127.204.42')
            conn.openTree('TS_EAST', shot)
            ts_times = np.asarray(
                conn.get(r'dim_of(\Te_coreTS)').data(), dtype=np.float64
            ).flatten()
            conn.closeTree('TS_EAST', shot)
            result['ts_times'] = list(ts_times)
            result['sample_count'] = len(ts_times)
        except Exception as e:
            result['issues'].append(f'TS_EAST: {str(e)[:80]}')
            result['verdict'] = 'FAIL: no TS data'
            return result

        if len(ts_times) == 0:
            result['verdict'] = 'FAIL: TS has 0 time points'
            return result

        # Step 2: Try sample time points via readmds()
        if sample_times is None:
            # Sample up to 5 time points spread across the discharge
            n_samples = min(5, len(ts_times))
            indices = np.linspace(0, len(ts_times) - 1, n_samples, dtype=int)
            sample_times = [float(ts_times[i]) for i in indices]

        valid_count = 0
        best_txcs = 0
        h_mode_count = 0

        for t in sample_times:
            try:
                # Suppress verbose debug output from readmds()
                with open(os.devnull, 'w') as devnull:
                    old_stdout = sys.stdout
                    sys.stdout = devnull
                    try:
                        data, status, real_time = readmds(shot, t)
                    finally:
                        sys.stdout = old_stdout
            except Exception as e:
                result['issues'].append(f't={t:.2f}s: readmds() error: {str(e)[:80]}')
                continue

            if status.get('TS_status', 0) != 1:
                continue

            valid_count += 1

            # Check Te
            te_ok = ('TS' in data.get('Te', {}) and
                     len(data['Te']['TS'].get('data', [])) > 3)
            if te_ok:
                result['has_te'] = True

            # Check ne from TS
            ne_ts_ok = ('TS' in data.get('ne', {}) and
                        len(data['ne']['TS'].get('data', [])) > 3)
            if ne_ts_ok:
                result['has_ne'] = True

            # Check ne from Refl
            ne_refl_ok = ('Refl' in data.get('ne', {}) and
                          len(data['ne']['Refl'].get('data', [])) > 3)
            if ne_refl_ok:
                result['has_ne_refl'] = True

            # Check Ti from TXCS
            ti_ok = ('TXCS' in data.get('Ti', {}) and
                     len(data['Ti']['TXCS'].get('data', [])) >= min_txcs_channels)
            if ti_ok:
                result['has_ti'] = True
                n_ch = len(data['Ti']['TXCS']['data'])
                valid_ch = int(np.sum(np.isfinite(data['Ti']['TXCS']['data'])))
                result['txcs_channels'] = max(result['txcs_channels'], n_ch)
                result['txcs_valid_channels'] = max(result['txcs_valid_channels'], valid_ch)
                if valid_ch > best_txcs:
                    best_txcs = valid_ch

            # Check H98 for H-mode identification
            h98 = status.get('H98', None)
            if h98 is not None and h98 > 1.0:
                h_mode_count += 1

            if te_ok and (ne_ts_ok or ne_refl_ok) and ti_ok:
                result['valid_times'].append(real_time)
                if h98 is not None and h98 > 1.0:
                    result['h_mode_times'].append(real_time)

        result['sample_count'] = valid_count

        # Verdict
        if result['valid_times']:
            h_str = f' (H-mode={len(result["h_mode_times"])})' if result['h_mode_times'] else ''
            result['verdict'] = (f'PASS: {len(result["valid_times"])}/{valid_count}'
                                 f' valid time points{h_str}')
        elif result['has_te'] and result['has_ne'] and not result['has_ti']:
            result['verdict'] = 'PARTIAL: Te+ne OK, no Ti (TXCS missing)'
        elif result['has_te'] and not result['has_ne']:
            result['verdict'] = 'PARTIAL: Te OK, no ne'
        elif result['has_te'] and result['has_ne'] and result['has_ti']:
            result['verdict'] = 'PARTIAL: some data but no valid Ti profile'
        else:
            result['verdict'] = 'FAIL: insufficient diagnostics'

        # Try to release alarm
        try:
            sig_module.alarm(0)
        except Exception:
            pass

    except TimeoutError:
        result['verdict'] = 'FAIL: timeout'
        result['issues'].append('MDSplus connection timed out')
    except Exception as e:
        result['verdict'] = f'FAIL: {str(e)[:60]}'
        result['issues'].append(str(e)[:120])

    return result


def format_result(r, verbose=False):
    """Format a single shot result for display."""
    verdict = r['verdict']
    is_pass = verdict.startswith('PASS')
    is_partial = verdict.startswith('PARTIAL')
    icon = 'PASS' if is_pass else ('PARTIAL' if is_partial else 'FAIL')

    lines = [f"  Shot {r['shot']:>6}  [{icon}] {verdict}"]
    if r['valid_times']:
        lines.append(f"    Valid times: {r['valid_times'][:5]}"
                     f"{'...' if len(r['valid_times']) > 5 else ''}")
    if r['h_mode_times']:
        lines.append(f"    H-mode times (H98>1): {r['h_mode_times'][:3]}"
                     f"{'...' if len(r['h_mode_times']) > 3 else ''}")
    lines.append(f"    Te={r['has_te']}, ne_TS={r['has_ne']}, "
                 f"ne_Refl={r['has_ne_refl']}, Ti={r['has_ti']}")
    if r['txcs_channels'] > 0:
        lines.append(f"    TXCS: {r['txcs_channels']} total, "
                     f"{r['txcs_valid_channels']} valid channels")
    if r['issues'] and verbose:
        for issue in r['issues'][:3]:
            lines.append(f"    WARN: {issue}")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Find EAST shots with complete diagnostic data for ONETWO pipeline')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--start', type=int, help='Start shot number')
    group.add_argument('--shots', type=int, nargs='+', help='Specific shot list')
    parser.add_argument('--end', type=int, default=None, help='End shot (with --start)')
    parser.add_argument('--step', type=int, default=1, help='Step size (default 1)')
    parser.add_argument('--min-txcs', type=int, default=3,
                        help='Minimum TXCS valid channels (default 3)')
    parser.add_argument('--samples', type=int, default=5,
                        help='Time points to sample per shot (default 5)')
    parser.add_argument('--pass-only', action='store_true',
                        help='Only show PASS shots')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed issues')
    parser.add_argument('--output', type=str, default=None,
                        help='Save PASS shot list to file')
    parser.add_argument('--h-mode-only', action='store_true',
                        help='Only show shots with H-mode time points')
    args = parser.parse_args()

    if args.shots:
        shot_list = list(args.shots)
    else:
        end = args.end if args.end else args.start + 99
        shot_list = list(range(args.start, end + 1, args.step))

    print(f"{'='*70}")
    print(f"Range: {len(shot_list)} shots ({shot_list[0]}-{shot_list[-1]})")
    print(f"Min TXCS channels: {args.min_txcs}")
    print(f"Time samples per shot: {args.samples}")
    print(f"{'='*70}")

    passed = []
    partial = []
    failed = []

    for i, shot in enumerate(shot_list):
        pct = (i + 1) / len(shot_list) * 100
        sys.stdout.write(f"\r[{i+1}/{len(shot_list)} {pct:.0f}%] Shot {shot}...")
        sys.stdout.flush()

        result = check_shot(shot, min_txcs_channels=args.min_txcs)

        verdict = result['verdict']
        if verdict.startswith('PASS'):
            is_h_mode = bool(result.get('h_mode_times'))
            if args.h_mode_only and not is_h_mode:
                continue
            passed.append(result)
            if not args.pass_only:
                print()
                print(format_result(result, args.verbose))
            else:
                h_str = ' [H-mode]' if is_h_mode else ''
                print(f"\n  PASS Shot {shot} "
                      f"({len(result['valid_times'])} time pts, "
                      f"TXCS={result['txcs_valid_channels']}ch){h_str}")
        elif verdict.startswith('PARTIAL'):
            partial.append(result)
            if not args.pass_only:
                print()
                print(format_result(result, args.verbose))
        else:
            failed.append(result)
            if not args.pass_only and len(failed) <= 10:
                print()
                print(format_result(result, args.verbose))
            elif not args.pass_only and len(failed) == 11:
                print(f"\n  ... (remaining failures hidden, use --pass-only)")

    print(f"\n{'='*70}")
    print(f"Scan complete: {len(shot_list)} shots")
    print(f"  PASS    {len(passed):>4}  - complete diagnostic coverage")
    print(f"  PARTIAL {len(partial):>4}  - missing Ti or ne")
    print(f"  FAIL    {len(failed):>4}  - critical diagnostics missing")
    print(f"{'='*70}")

    if passed:
        print(f"\nPASS shots ({len(passed)}):")
        print(f"{'Shot':>8}  {'Times':>6}  {'H-mode':>7}  {'TXCS_ch':>7}  {'Range'}")
        print(f"{'-'*8}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*20}")
        for r in passed:
            n_times = len(r['valid_times'])
            n_h = len(r.get('h_mode_times', []))
            txcs = r['txcs_valid_channels']
            ts_t = r.get('ts_times', [])
            t_range = f'{ts_t[0]:.1f}-{ts_t[-1]:.1f}s' if ts_t else 'N/A'
            print(f"{r['shot']:>8}  {n_times:>6}  {n_h:>7}  {txcs:>7}  {t_range}")

    # Stats on missing diagnostics
    if partial or failed:
        te_ok = sum(1 for r in partial + failed if r['has_te'])
        ne_ok = sum(1 for r in partial + failed if r['has_ne'] or r['has_ne_refl'])
        ti_ok = sum(1 for r in partial + failed if r['has_ti'])
        total = len(partial) + len(failed)
        print(f"\nPartial/Fail breakdown ({total} shots):")
        print(f"  Has Te:  {te_ok}/{total} ({te_ok/total*100:.0f}%)")
        print(f"  Has ne:  {ne_ok}/{total} ({ne_ok/total*100:.0f}%)")
        print(f"  Has Ti:  {ti_ok}/{total} ({ti_ok/total*100:.0f}%)")

    if args.output and passed:
        with open(args.output, 'w') as f:
            for r in passed:
                f.write(f"{r['shot']}\n")
        print(f"\nPASS shots saved to: {args.output}")


if __name__ == '__main__':
    main()
