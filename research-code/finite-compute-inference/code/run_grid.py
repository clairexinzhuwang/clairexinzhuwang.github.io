"""Full simulation grid under the FROZEN Algorithm 1 (alg_paper.run_algorithm1).
Usage: python3 -u run_grid.py <kernel: auc|triplet|rank> [R] [cpus]
Writes ../grid_<kernel>.json with one record per (n, B, scheme)."""
import sys, os, json, time, numpy as np, ray
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alg_paper import run_algorithm1, default_t0, default_m
from kernels import AUCKernel, TripletKernel, RankKernel
import dgp as AU, dgp as M
from dgp import generate_data, get_svec_indices, svec
import paths as P

K_A = 16                       # conditional tuples per anchor stream (frozen)
# The ARCHIVED grid, in which the minibatch size grows with the sample size.
# It is kept exactly as it was run so that the tables it produced remain
# reproducible; the fixed-B study is a separate design and lives in
# run_fixed_B_grid.py, which never writes here.
GRIDS = {
    'auc':     [(1000,128),(1000,256),(10000,512),(10000,1024),(100000,2048),(100000,4096),(1000000,8192)],
    'triplet': [(1000,128),(1000,256),(10000,512),(10000,1024),(100000,2048),(100000,4096),(1000000,8192),(1000000,16384)],
    'rank':    [(100,32),(100,64),(100,100),(1000,128),(1000,256),(1000,512),(10000,256),(10000,512),(10000,1024),
                (100000,512),(100000,1024),(100000,2048),(1000000,2048),(1000000,4096),(1000000,8192)],
}
# Names the output file.  This grid is the one the archived tables report and
# writes grid_<kernel>.json; any other grid must set a tag and write its own
# file, so that two grids can never accumulate in one place.
GRID_TAG = ""

def build(kern_name, n, seed):
    rng = np.random.default_rng(seed)
    if kern_name == 'auc':
        p = 3; dgp = AU.make_dgp(p); m_ = int(round(0.4 * n)); Xp, Xn = AU.draw(m_, n - m_, *dgp, rng)
        k = AUCKernel(Xp, Xn); return k, np.zeros(p), [k.m, k.nn], rng, 'log'
    if kern_name == 'triplet':
        p, K = 4, 5; cm = M.aniso_class_means(p, K, np.random.default_rng(12345)); r, c, s = get_svec_indices(p)
        X, Y, ci = generate_data(n, p, K, cm, rng); k = TripletKernel(X, Y, ci); return k, svec(np.eye(p), r, c, s), n, rng, 'log'
    p = 2; X = rng.standard_normal((n, p)); y = X @ np.array([0.5, 1.0]) + rng.standard_normal(n)
    k = RankKernel(X, y); return k, k.theta_ols, n, rng, 'rank'

def one(kern_name, n, B, scheme, seed):
    t = time.time()
    kern, theta0, q, rng, rule = build(kern_name, n, seed)
    T = int((n / B) * np.log2(n)); t0 = default_t0(T, rule=rule, n=n); m0 = m = default_m(B, kern.n_ref, getattr(kern, 'h', 1.0))
    o = run_algorithm1(kern, theta0, B, T, t0, m0, m, q, scheme=scheme, rng=rng, k_per_stream=K_A)
    qs = q if isinstance(q, list) else [q]
    cost = B * T + m0 + m + 2 * K_A * sum(qs)
    return {'theta': o['theta'], 'v_full': np.diag(o['Sigma_full']), 'v_main': np.diag(o['Sigma_main']),
            'v_dat': np.diag(o['Sigma_dat']), 'v_alg_full': np.diag(o['alg_full']), 'v_alg_main': np.diag(o['alg_main']),
            'pilot_ok': o['pilot_ok'], 'pilot_iters': o['pilot_iters'], 'pilot_gn': o['pilot_gradnorm'],
            'A0_floor': o['A0_floor_active'], 'AT_floor': o['AT_floor_active'], 'A0_kappa': o['A0_kappa'], 'AT_kappa': o['AT_kappa'],
            'proj': float(np.mean([o.get(f'zeta_projected_k{i}', False) for i in range(len(kern.samples))])),
            'moved': float(np.mean([o.get(f'zeta_proj_moved_k{i}', 0.0) for i in range(len(kern.samples))])),
            't0': t0, 'T': T, 'N0': B * t0, 'm0': m0, 'm': m, 'q': qs, 'cost': cost, 'runtime': time.time() - t}

@ray.remote
def one_remote(kern_name, n, B, scheme, seed):
    return one(kern_name, n, B, scheme, seed)

def _safe_cpus(requested, per_worker_gb=0.8, reserve_gb=3.0):
    """Cap workers by free RAM, not just core count: the n=1e6 rank pilot is a
    10^7-tuple batch and 8 workers once drove the machine into swap."""
    try:
        import psutil
        avail = psutil.virtual_memory().available / 2**30
    except Exception:
        return requested
    fit = max(1, int((avail - reserve_gb) / per_worker_gb))
    if fit < requested:
        print(f"  [mem] {avail:.1f} GB free -> {fit} workers instead of {requested}", flush=True)
    return min(requested, fit)


def main():
    kern_name = sys.argv[1]; R = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    cpus = _safe_cpus(int(sys.argv[3]) if len(sys.argv) > 3 else 10)
    ts = {'auc': AU.compute_theta_star_auc(*AU.make_dgp(3), M=int(2e8))[0],
          'triplet': np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'theta_star_p4K5_3e8.npz'))['theta_star'],
          'rank': np.array([0.5, 1.0])}[kern_name]
    print(f"=== {kern_name}: R={R}, k_a={K_A}, theta*={np.round(ts,4)} ===", flush=True)
    ray.init(num_cpus=cpus, include_dashboard=False, ignore_reinit_error=True, log_to_driver=False)
    out_path = P.grid(kern_name, GRID_TAG)
    records = json.load(open(out_path)) if os.path.exists(out_path) else []
    done = {(r['n'], r['B'], r['scheme']) for r in records}
    hdr = f"{'n':>8}{'B':>7}{'T':>6}{'t0':>5}{'sch':>4} | {'|bias|':>9}{'Var_emp':>10}{'Var_dat':>10}{'Var_sgd':>10}{'cov':>7}{'VR':>6} | {'pilot':>6}{'fl':>4}{'proj':>5} | {'t/rep':>6}"
    print(hdr, flush=True)
    for n, B in GRIDS[kern_name]:
        # At B = 1 a size-one minibatch without replacement is the
        # with-replacement draw: draw_wor(1, rng) returns the same tuple from
        # the same rng state for every kernel (verified bit-level, values and
        # stream), so the two schemes are one computation and it is run once.
        for scheme in (('WR',) if B == 1 else ('WR', 'WOR')):
            if (n, B, scheme) in done: continue
            t = time.time()
            res = ray.get([one_remote.remote(kern_name, n, B, scheme, 1000 + s) for s in range(R)])
            th = np.array([r['theta'] for r in res]); ve = th.var(0, ddof=1)
            vf = np.array([r['v_full'] for r in res]); vm = np.array([r['v_main'] for r in res])
            vd = np.array([r['v_dat'] for r in res]); va = np.array([r['v_alg_full'] for r in res]); vam = np.array([r['v_alg_main'] for r in res])
            covf = np.abs(th - ts) < 1.96 * np.sqrt(np.maximum(vf, 0)); covm = np.abs(th - ts) < 1.96 * np.sqrt(np.maximum(vm, 0))
            # serial timing on an idle machine: a few extra replications after the parallel block
            tr = [one(kern_name, n, B, scheme, 90000 + s)['runtime'] for s in range(3 if n >= 100000 else 5)]
            rec = {'n': n, 'B': B, 'scheme': scheme, 'T': res[0]['T'], 't0': res[0]['t0'], 'N0': res[0]['N0'],
                   'm0': res[0]['m0'], 'm': res[0]['m'], 'q': res[0]['q'], 'k_a': K_A, 'R': R, 'cost_tuples': res[0]['cost'],
                   'bias_percoord': (th.mean(0) - ts).tolist(), 'bias': float(np.abs(th.mean(0) - ts).mean()),
                   'var_emp_percoord': ve.tolist(), 'var_emp': float(ve.mean()),
                   'var_dat_percoord': vd.mean(0).tolist(), 'var_dat': float(vd.mean()),
                   'var_sgd_percoord': va.mean(0).tolist(), 'var_sgd': float(va.mean()), 'var_sgd_mainfactor': float(vam.mean()),
                   'var_total': float(vf.mean()), 'var_total_mainfactor': float(vm.mean()),
                   'coverage': float(covf.mean()), 'coverage_percoord': covf.mean(0).tolist(),
                   'coverage_mainfactor': float(covm.mean()),
                   'VR': float(np.mean(ve / vf.mean(0))), 'VR_mainfactor': float(np.mean(ve / vm.mean(0))),
                   'pilot_ok_frac': float(np.mean([r['pilot_ok'] for r in res])), 'pilot_iters_mean': float(np.mean([r['pilot_iters'] for r in res])),
                   'pilot_gradnorm_max': float(max(r['pilot_gn'] for r in res)),
                   'A0_floor_frac': float(np.mean([r['A0_floor'] for r in res])), 'AT_floor_frac': float(np.mean([r['AT_floor'] for r in res])),
                   'A0_kappa_max': float(max(r['A0_kappa'] for r in res)), 'AT_kappa_max': float(max(r['AT_kappa'] for r in res)),
                   'proj_frac': float(np.mean([r['proj'] for r in res])), 'proj_moved_mean': float(np.mean([r['moved'] for r in res])),
                   'time_serial_med': float(np.median(tr)), 'time_parallel_mean': float(np.mean([r['runtime'] for r in res])),
                   'wall_s': time.time() - t}
            records.append(rec); json.dump(records, open(out_path, 'w'), indent=1)
            print(f"{n:>8}{B:>7}{rec['T']:>6}{rec['t0']:>5}{scheme:>4} | {rec['bias']:>9.2e}{rec['var_emp']:>10.3e}{rec['var_dat']:>10.3e}{rec['var_sgd']:>10.3e}{rec['coverage']:>7.3f}{rec['VR']:>6.2f} | {rec['pilot_ok_frac']:>6.2f}{rec['A0_floor_frac']+rec['AT_floor_frac']:>4.1f}{rec['proj_frac']:>5.2f} | {rec['time_serial_med']:>6.2f}   (wall {rec['wall_s']:.0f}s)", flush=True)
    ray.shutdown()
    print("DONE", kern_name, flush=True)

if __name__ == '__main__':
    main()
