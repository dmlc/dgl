#!/usr/bin/env python3
# coding: utf-8

"""PyPTO concat golden reference implementation.

- golden 必须是纯 PyTorch 实现，禁止引入 pypto。
- 导出函数 concat_golden() 供 test_concat.py 调用。
"""

import torch

# ─────────────────────────────────────────────
# Golden 参考实现（纯 torch）
# ─────────────────────────────────────────────

def concat_golden(tensors, dim=0):
    """PyTorch 参考实现。

    沿 dim 轴拼接一组张量，与 torch.cat 语义一致。

    Args:
        tensors: list[torch.Tensor] 待拼接张量列表。
        dim: 拼接轴，默认 0。

    Returns:
        torch.Tensor 拼接结果。
    """
    return torch.cat(tensors, dim=dim)


# ==========================================
# 验证
# ==========================================

def _validate():
    """自动生成的验证函数 - 运行时动态生成验证报告"""

    print("=" * 60)
    print("concat_golden 验证报告")
    print("=" * 60)

    # -- 1. 典型 case 验证（来自算子规格中的典型配置）--
    print("\n[典型 case 验证]")
    cases = [
        ("功能_P0", [torch.randn(256, 1024), torch.randn(256, 1024)], 0, (512, 1024)),
        ("性能_P0", [torch.randn(1024, 512), torch.randn(1024, 512)], -1, (1024, 1024)),
        ("多输入_P1", [torch.randn(128, 64) for _ in range(4)], 0, (512, 64)),
    ]
    for name, tensors, dim, expected_shape in cases:
        out = concat_golden(tensors, dim)
        assert out.shape == torch.Size(expected_shape), \
            f"{name}: shape {out.shape} != {expected_shape}"
        assert torch.equal(out, torch.cat(tensors, dim=dim)), \
            f"{name}: 结果与 torch.cat 不一致"
        print(f"  {name}: dim={dim} {[t.shape for t in tensors]} → {out.shape} ... ✓ PASS")

    # -- 2. 泛化 case 验证（来自算子规格中的动态轴范围）--
    print("\n[泛化 case 验证]")
    gen_cases = [
        ([1, 64, 32], [2, 64, 32], 0),   # dim=0 中间轴拼接
        ([64, 1, 32], [64, 2, 32], 1),   # dim=1 中间轴拼接
        ([64, 32, 1], [64, 32, 2], -1),  # dim=-1 尾轴拼接
        ([1, 1, 1], [1, 1, 1], 0),       # 最小值边界
    ]
    for shapes in gen_cases:
        tensors = [torch.randn(*s) for s in shapes[0:2]]
        dim = shapes[2]
        out = concat_golden(tensors, dim)
        assert torch.equal(out, torch.cat(tensors, dim=dim))
        print(f"  dim={dim} {shapes[0]} + {shapes[1]} → {tuple(out.shape)} ... ✓ PASS")

    # -- 3. 值域检查（从公式推导）--
    print("\n[值域检查]")
    t = [torch.randn(3, 5), torch.randn(4, 5)]
    out = concat_golden(t, 0)
    assert out.shape[0] == 3 + 4
    assert torch.equal(out[:3], t[0]) and torch.equal(out[3:], t[1])
    print("  out[:3]==t[0], out[3:]==t[1] ... ✓ PASS")

    # -- 4. 数值稳定性检查 --
    print("\n[数值稳定性检查]")
    big = [torch.full((2, 4), 1e30), torch.full((2, 4), -1e30)]
    out_big = concat_golden(big, 0)
    assert not torch.isnan(out_big).any() and not torch.isinf(out_big).any()
    zero = [torch.zeros(2, 3), torch.zeros(2, 3)]
    out_zero = concat_golden(zero, 0)
    assert (out_zero == 0).all()
    print("  大值/零值输入无 NaN/Inf，逐位保持 ... ✓ PASS")

    # -- 5. API 对比（如适用）--
    print("\n[API 对比]")
    a = torch.randn(8, 16)
    b = torch.randn(8, 16)
    assert torch.equal(concat_golden([a, b], 0), torch.cat([a, b], 0))
    assert torch.equal(concat_golden([a, b], 1), torch.cat([a, b], 1))
    print("  concat_golden 与 torch.cat 全 dim 一致 ... ✓ PASS")

    print("\n" + "=" * 60)
    print("验证完成")
    print("=" * 60)


if __name__ == "__main__":
    _validate()
