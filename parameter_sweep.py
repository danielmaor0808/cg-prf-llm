# parameter_sweep.py
import numpy as np

def parameter_grid():
    for num_top_docs in np.arange(0, 0.5, 1):
        for num_llm_docs in np.arange(10, 10.5, 1):
            if num_top_docs + num_llm_docs != 10:
                continue
            for beta_iter in np.arange(0.2, 0.85, 0.1):
                for theta_iter in np.arange(5, 75, 5):
                    for b1_iter in np.arange(0.9, 0.95, 0.1):
                        for k_iter in np.arange(0.9, 2.35, 0.2):
                            yield {
                                "num_top_docs": num_top_docs,
                                "num_llm_docs": num_llm_docs,
                                "beta": beta_iter,
                                "theta": theta_iter,
                                "b1": b1_iter,
                                "k": k_iter
                            }
