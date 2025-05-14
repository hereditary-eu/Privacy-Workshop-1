# https://github.com/schneiderkamplab/syntheval/blob/main/src/syntheval/metrics/privacy

from syntheval import SynthEval

#import pdb

from copy import deepcopy

def calculate_metric(args, _real_data, _synthetic):
    # Deep copy to avoid modifying inputs
    real_data = deepcopy(_real_data)
    synthetic = deepcopy(_synthetic)

    # Ensure all column names are native Python strings
    real_data.columns = real_data.columns.map(str)
    synthetic.columns = synthetic.columns.map(str)

    # Initialize evaluator
    evaluator = SynthEval(real_data)

    # Evaluate with the specified resampling strategy
    evaluator.evaluate(synthetic, nnaa={"n_resample": 30})

    # Get the average NNAA (nearest neighbor adversarial accuracy) score
    result = evaluator._raw_results.get('nnaa', {}).get('avg', None)

    if result is None:
        raise ValueError("NNAA evaluation failed or did not return 'avg' score.")

    # Return 1 - result as a utility-oriented score (lower = worse privacy)
    return 1 - result