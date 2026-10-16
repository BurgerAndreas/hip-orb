"""GPU correctness checks for the dedicated frozen-feature gather kernel."""
import json
import torch
from hip_orb.model import directed_pair_inputs
from hip_orb.fused_pair_inputs import fused_directed_pair_inputs


def main():
    torch.manual_seed(11)
    checks = []
    for d in (8, 256):
        for pairs in (0, 1, 7, 41):
            for nedges in (0, 5):
                nodes = torch.randn(13, d, device='cuda')
                edges = torch.randn(nedges, d, device='cuda')
                charges = torch.randn(13, device='cuda')
                i, j = torch.randint(13, (2, pairs), device='cuda')
                f, r = torch.randint(-1, nedges, (2, pairs), device='cuda')
                delta = torch.randn(pairs, 3, device='cuda')
                distance = delta.norm(dim=-1, keepdim=True).clamp_min(0.1)
                inputs = (nodes, edges, charges, i, j, f, r, delta, distance)
                expected = directed_pair_inputs(*inputs)
                actual = fused_directed_pair_inputs(*inputs)
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
                checks.append({'dimension': d, 'pairs': pairs, 'message_edges': nedges,
                    'max_abs': float((actual-expected).abs().max()) if pairs else 0.})
    print(json.dumps({'passed': True, 'checks': checks}), flush=True)


if __name__ == '__main__':
    main()
