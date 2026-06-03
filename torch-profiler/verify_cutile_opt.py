import torch
import torch.nn as nn
import cuda.tile as ct
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

class CuTileRMSNormOptimized(nn.Module):
    def __init__(self, original_norm):
        super().__init__()
        self.weight = original_norm.weight
        self.variance_epsilon = original_norm.variance_epsilon
        self.hidden_size = original_norm.weight.shape[0]
        self.block_rows = 64
        self.block_cols = 128
        self.kernel = self._compile_kernel(self.hidden_size, self.block_rows, self.block_cols)
        self.weight_2d = self.weight.unsqueeze(0)

    def _compile_kernel(self, cols, block_rows, block_cols):
        @ct.kernel
        def rmsnorm_kernel(inp: ct.Tile, weight: ct.Tile, out: ct.Tile, eps: float):
            pid = ct.bid(0)
            running_sum_sq = ct.zeros(shape=(block_rows, 1), dtype=ct.float32)
            num_col_tiles = ct.cdiv(cols, block_cols)
            for k in range(num_col_tiles):
                x = ct.load(inp, (pid, k), shape=(block_rows, block_cols))
                x_f32 = ct.astype(x, ct.float32)
                x_sq = ct.mul(x_f32, x_f32)
                tile_sum_sq = ct.sum(x_sq, axis=1, keepdims=True)
                running_sum_sq = ct.add(running_sum_sq, tile_sum_sq)
            mean_sq = ct.truediv(running_sum_sq, float(cols))
            inv_rms = ct.rsqrt(ct.add(mean_sq, eps))
            for k in range(num_col_tiles):
                x = ct.load(inp, (pid, k), shape=(block_rows, block_cols))
                x_f32 = ct.astype(x, ct.float32)
                w = ct.load(weight, (0, k), shape=(1, block_cols))
                w_f32 = ct.astype(w, ct.float32)
                x_norm = ct.mul(x_f32, inv_rms)
                res = ct.mul(x_norm, w_f32)
                res_f16 = ct.astype(res, ct.float16)
                ct.store(out, (pid, k), res_f16)
        return rmsnorm_kernel

    def forward(self, hidden_states):
        input_shape = hidden_states.shape
        x_2d = hidden_states.view(-1, self.hidden_size)
        original_rows = x_2d.shape[0]
        pad_rows = (self.block_rows - (original_rows % self.block_rows)) % self.block_rows
        if pad_rows > 0:
            x_2d = torch.nn.functional.pad(x_2d, (0, 0, 0, pad_rows))
        padded_rows = x_2d.shape[0]
        y_2d = torch.empty_like(x_2d)
        grid = (padded_rows // self.block_rows, 1, 1)
        stream = torch.cuda.current_stream()
        ct.launch(stream, grid, self.kernel, (x_2d, self.weight_2d, y_2d, self.variance_epsilon))
        if pad_rows > 0:
            y_2d = y_2d[:original_rows, :]
        return y_2d.view(input_shape)

def test_correctness():
    print("Testing Optimized cuTile RMSNorm correctness...")
    hidden_size = 896
    eps = 1e-6
    
    orig_norm = Qwen2RMSNorm(hidden_size, eps=eps).cuda().half()
    torch.nn.init.normal_(orig_norm.weight, mean=1.0, std=0.1)
    
    cutile_opt_norm = CuTileRMSNormOptimized(orig_norm)
    
    for b in [1, 2, 4]:
        for s in [64, 128, 256, 512, 1024]:
            x = torch.randn(b, s, hidden_size, device="cuda", dtype=torch.float16)
            
            with torch.no_grad():
                out_orig = orig_norm(x)
                out_cutile = cutile_opt_norm(x)
                
            max_diff = (out_orig - out_cutile).abs().max().item()
            is_close = torch.allclose(out_orig, out_cutile, atol=1e-3, rtol=1e-3)
            
            print(f"Batch={b}, Seq={s} | Max Diff: {max_diff:.6f} | Match: {'OK' if is_close else 'FAIL'}")
            assert is_close, f"Outputs do not match! Max diff: {max_diff}"
            
    print("All checks passed successfully!")

if __name__ == "__main__":
    test_correctness()
