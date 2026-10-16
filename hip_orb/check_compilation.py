"""Untimed real-graph prediction and head-gradient checks for compiler candidates."""
import argparse
import copy
import json
from pathlib import Path

import torch

from hip_orb.benchmark_training import model_hash
from hip_orb.compilation import compiler_candidate
from hip_orb.data import OrbMolDataset, collate_samples, unpack_hessian
from hip_orb.graphs import canonical_message_graph, refresh_message_vectors
from hip_orb.model import HIPOrb
from hip_orb.preparation import prepare_batch
from hip_orb.profile_batch_steps import read_pool


def discrepancy(actual, expected):
    return {"max_abs": float((actual-expected).abs().max()),
            "relative_l2": float((actual-expected).norm()/expected.norm().clamp_min(1e-12))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--variants', nargs='+', default=['shape2', 'compile2', 'fusion2', 'compile_fusion2'])
    p.add_argument('--precision', choices=('float32-high', 'float32-highest'), default='float32-high')
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(7)
    torch.use_deterministic_algorithms(True)
    model = HIPOrb.pretrained('cuda', weights_path=a.checkpoint, precision=a.precision)
    initial = copy.deepcopy(model.head.state_dict())
    frozen = model_hash(model.base)
    dataset = OrbMolDataset(a.index, training_only=True)
    result = {"complete": False, "checks": [], "precision": torch.get_float32_matmul_precision(),
              "atol": 1e-4, "rtol": 1e-3, "deterministic_reductions": True,
              "head_only_candidates_share_identical_frozen_features": True}
    try:
        for requested in (2, 50, 350):
            samples, n = read_pool(dataset, requested, 2)
            batch = prepare_batch(collate_samples(samples), seed=7)
            graph = refresh_message_vectors(canonical_message_graph(
                model.adapter.from_ase_atoms_list(batch['molecules'], device='cuda')))
            sizes = torch.full((len(samples),), n, dtype=torch.long)
            target = unpack_hessian(batch['hessian_upper'].to('cuda'), n)
            rotation = batch['rotation'].to('cuda')
            target = torch.einsum('ab,kibjd,cd->kiajc', rotation,
                target.reshape(-1, n, 3, n, 3), rotation).reshape(-1, 3*n, 3*n)
            reference_features = model.features(graph.clone(), sizes=sizes)
            def evaluate(features=None):
                model.head.load_state_dict(initial)
                model.head.zero_grad(set_to_none=True)
                features = model.features(graph.clone(), sizes=sizes) if features is None else features
                predicted = torch.stack(model.head(features))
                error = predicted - target
                parameters = tuple(model.head.parameters())
                mse = torch.autograd.grad(error.square().mean(), parameters, retain_graph=True)
                mae = torch.autograd.grad(error.abs().mean(), parameters)
                return (features.nodes.detach(), features.edges.detach(), features.charges.detach(),
                        predicted.detach(), torch.cat([g.reshape(-1) for g in mse]).detach(),
                        torch.cat([g.reshape(-1) for g in mae]).detach())
            expected = evaluate(reference_features)
            for name in ['eager_repeat', *a.variants]:
                backbone = name in ('compile2', 'compile_fusion2', 'compile_triton2')
                inputs = name in ('fusion2', 'compile_fusion2')
                with compiler_candidate(model, backbone=backbone, pair_inputs=inputs,
                                        shaped_conditioning=name == 'shape2',
                                        triton_pairs=name in ('triton2', 'compile_triton2')):
                    actual = evaluate(reference_features if name in ('fusion2', 'triton2') else None)
                row = {'variant': name, 'natoms': n, 'batch_size': len(samples)}
                for key, candidate, reference in zip(('nodes', 'edges', 'charges', 'hessian',
                    'mse_gradients', 'mae_gradients'), actual, expected):
                    row[key] = discrepancy(candidate, reference)
                    try:
                        torch.testing.assert_close(candidate, reference, atol=1e-4, rtol=1e-3)
                    except AssertionError as error:
                        row.setdefault('failures', {})[key] = str(error)
                row['passed'] = not row.get('failures')
                result['checks'].append(row)
                a.report.write_text(json.dumps(result, indent=2)+'\n')
                print(json.dumps(row), flush=True)
        result['frozen_backbone_unchanged'] = model_hash(model.base) == frozen
        result['backbone_has_gradients'] = any(p.grad is not None for p in model.base.parameters())
        assert result['frozen_backbone_unchanged'] and not result['backbone_has_gradients']
        result['all_checks_passed'] = all(row['passed'] for row in result['checks'])
        result['backbone_comparison_controls'] = {
            str(row['natoms']): row['passed'] for row in result['checks'] if row['variant'] == 'eager_repeat'}
        result['complete'] = True
    except Exception as error:
        result['failure'] = str(error)
        raise
    finally:
        dataset.close()
        a.report.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
