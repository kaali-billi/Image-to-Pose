import argparse
import time
import numpy as np
from scipy.stats import norm
from utils_gpu import *
import os
from tqdm import tqdm
import random


def main(pcd_file, fib, thresh, Hyp_last):
    # --------------------------------------------------------------------------
    # USER CONFIGURATION
    parser = argparse.ArgumentParser(description='PLuM Registration (GPU Accelerated)')
    parser.add_argument('--LOOKUP_TABLE_FILE', type=str, default='DC/Reward_8124_050mm_sig01.txt') # Reward File / Lookup File
    parser.add_argument('--LOOKUP_TO_MODEL', type=int, default=[4, 4.42, 1.78]) # Overlap Reward Model with Input (needs manual finetuning)
    parser.add_argument('--MAX_XYZ', type=int, default=[8, 12, 4]) # Max XYZ same as the lookup file
    parser.add_argument('--LOOKUP_STEP_SIZE', type=int, default=0.05) # Same as reward file
    parser.add_argument("--SPHERE_POINTS", type=int, default=fib) # No. of random rotations sampled
    parser.add_argument("--THRESHOLD", type=int, default=thresh) # AD-hoc threshold for accuracy
    parser.add_argument('--SEARCH_ROT_SIGMA', type=float, default=5) # Sigma same as Lookup
    parser.add_argument('--SEARCH_ITER', type=int, default=30)
    parser.add_argument('--SEARCH_RESAMPLE', type=int, default=2000)
    # sf = Scaling factor of the model used for lookup
    # Rescale input to correct size of the lookup file
    ptCloud, cen, rat = read_pc(pcd_file, n=2048, sf=5.720)
    npts = len(ptCloud)
    parser.add_argument('--NPTS', type=int, default=npts)
    params = parser.parse_args()

    # ------------------------------------------------------------------------
    # SETUP
    # ------------------------------------------------------------------------
    pointsPerMeter = 1.0 / params.LOOKUP_STEP_SIZE
    numXYZ = [round(params.MAX_XYZ[i] * pointsPerMeter + 1) for i in range(3)]

    # READ THE LOOKUP TABLE (keep on CPU initially, will be moved to GPU in evidence calc)
    lookupTable = readLookupTable(params.LOOKUP_TABLE_FILE)

    # ------------------------------------------------------------------------
    # START TIMING
    # ------------------------------------------------------------------------
    start = time.time()
    max_hyps = []

    for g in range(5):
        resample_count = 2000  # Local variable, starts fresh each g
        maxElement = 0
        hypotheses = generate_rotations_from_fibonacci(fib, g)
        hypotheses[fib - 1] = Hyp_last
        hypotheses_torch = torch.tensor(hypotheses, dtype=torch.float32, device=device)

        for iteration in range(1, params.SEARCH_ITER + 1):
            # ================================================================
            # GPU-ACCELERATED EVIDENCE CALCULATION
            # ================================================================
            evidence = calculate_evidence_gpu(
                lookupTable,
                hypotheses_torch,
                ptCloud,
                params.MAX_XYZ,
                numXYZ,
                pointsPerMeter,
                params.LOOKUP_TO_MODEL
            )
            evidence_torch = torch.tensor(evidence, dtype=torch.float32, device=device)
            last_evidence_torch = evidence_torch
            last_hypotheses_torch = hypotheses_torch

            # Hypothesis sampling
            try:
                # Pass resample_count in and get updated value back
                hypotheses_torch, maxElement, maxElementIndex, should_break, resample_count = \
                    parallel_hypothesis_sampling_gpu3(
                        hypotheses_torch,
                        evidence_torch,
                        iteration,
                        params,
                        thresh,
                        npts,
                        device,
                        resample_count  # ✅ Pass current count
                    )
            except ValueError as e:
                print(f"Error: {e}")
                exit(1)

            if should_break:
                break

            numberOfHypotheses = hypotheses_torch.shape[0]

        hypotheses_final = hypotheses_torch.cpu().numpy()

        if maxElement / npts < thresh:
            max_hyps.append((hypotheses_final[maxElementIndex], maxElement))
            #print(f"  → Hypothesis stored for g={g}")
        else:
            #print(f"  → Breaking at g={g} - threshold exceeded")
            break

    end = time.time()

    # ================================================================
    # FINAL RESULT SELECTION (Using stored evidence)
    # ================================================================

    if g == 5:
        best = max(max_hyps, key=lambda X: X[1])
        hyp = best[0]
        maxElement = best[1]
        #print(f"\nCompleted all 4 iterations. Best from stored hypotheses:")
    else:
        # Use stored evidence (no recomputation needed)
        evidence_list = last_evidence_torch.cpu().numpy().tolist()
        maxElement = max(evidence_list)
        maxElementIndex = evidence_list.index(maxElement)
        hypotheses_final = last_hypotheses_torch.cpu().numpy()
        hyp = hypotheses_final[maxElementIndex]
        #print(f"\nBroke at g={g}. Using best from final iteration:")

    TT = end - start

    ort = vis_PF_SA(
        TT, iteration, g, hyp, maxElement, cen,
        rescale_pc(ptCloud, cen),
        "DC/files/DC_2048_cen.pcd",
        ret=True
    )

    return hyp, (maxElement / npts), params, TT, ort, rat


if __name__ == "__main__":
    plum = []
    avg = []
    tt = []
    dir = f'../SIM_DC_TEST/FT_REC/' # Path to completed Point Clouds
    jp = os.listdir(dir)
    prev = [0, 0, 0]
    y = 0
    H = []
    pbar = tqdm(jp, desc=f"Evd")

    for f in pbar:
        file = os.path.join(dir, f)
        T, evd, P, TT, ort, terf = main(file, 10000, 160, prev)
        prev = T
        avg.append(evd)
        plum.append(T)
        tt.append(TT)
        H.append(terf)
        y += 1
        pbar.set_description(f"Evd: {evd:.4f}  Mean: {np.mean(np.array(avg)):.4f}")

    gv = np.asarray(avg)
    perf = np.asarray(H)

    np.savetxt(f'../SIM_DC_TEST/ROTATIONS_PRED_GPU_FT_OPT.npy', plum, delimiter=',') # saved file with Euler rotations : X,Y,Z

    print("Average Reward per point:", np.median(gv), np.average(gv),
          np.min(gv), np.argmin(gv), np.max(gv), np.argmax(gv))
    print("Average, min, max time taken:", np.average(np.array(tt)),
          np.min(np.array(tt)), np.max(np.array(tt)))
    print("ORTS FILES SAVED, STARTING METRICS GENERATION:")

    save_args_to_file(P, f'../SIM_DC_TEST/PRED_config_GPU_FT_OPT.txt')

    import matplotlib.pyplot as plt

    time_steps = range(len(tt))

    # Create the plot with dual y-axes
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plot reward per point on the first y-axis
    color1 = '#2E86AB'
    ax1.set_xlabel('Time Step', fontsize=13)
    ax1.set_ylabel('Reward per Point', fontsize=13, color=color1)
    ax1.plot(time_steps, tt, linewidth=1.5, color=color1, label='Reward per Point')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Create second y-axis for perf
    ax2 = ax1.twinx()
    color2 = '#A23B72'
    ax2.set_ylabel('Perf', fontsize=13, color=color2)
    ax2.plot(time_steps, perf, linewidth=1.5, color=color2, label='Perf', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Title
    plt.title('Reward per Point and Perf over Time Steps (GPU Accelerated)',
              fontsize=15, fontweight='bold')

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)

    plt.tight_layout()
    plt.show()
