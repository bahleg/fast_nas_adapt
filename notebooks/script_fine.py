import os

SEEDS = (0, 13, 42)

for seed in SEEDS:
    for parameter_opt_type in ['MI', 'CE']:
        for gamma_opt_type in ['MI', 'CE']:
            for lam in [0.0, 0.25, 0.5, 0.75, 1.0][::-1]:
                if gamma_opt_type == 'CE' and parameter_opt_type == 'CE' and lam != 0:
                    continue
                for file in ['run_dartslike_mi.py', 'run_dartslike_fine.py']:
                    cmd = f'python {file} --param_opt_type={parameter_opt_type} \
                        --gamma_opt_type={gamma_opt_type} --seed={seed} --lam={lam}'
                    if 'mi' in file:
                        cmd += ' --layer_wise=True'
                    print(cmd)
                    os.system(cmd)

