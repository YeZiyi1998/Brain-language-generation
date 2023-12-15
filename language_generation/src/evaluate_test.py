from jupyter_utils import get_iterate_results, show_significance, scatter_plot_comparation, get_comparation_cases,plot_context_length2performance,language_evaluate_mask_with_sig,get_results
result_1 = get_iterate_results(base_path = '../results/', para_str = 'llama-7b_lr1e-5_tid1,2_b2', dataset_name='Pereira', print_log=True)
result_2 = get_iterate_results(base_path = '../results/', para_str = 'llama-7b_lr1e-5_tid1,2_b2_random', dataset_name='Pereira', print_log=True)
show_significance(result_1, result_2)
