"""Test Ascend/NPU device support for DGL."""
import torch
import dgl


def _check_npu_available():
    """Check if NPU is available. Works with torch_npu >= 2.5.1 without explicit import."""
    return hasattr(torch, 'npu') and torch.npu.is_available()


def test_device_context_mapping():
    """Test that NPU device type is correctly mapped."""
    from dgl._ffi.runtime_ctypes import DGLContext
    
    # Check that npu is in the STR2MASK
    assert "npu" in DGLContext.STR2MASK
    assert DGLContext.STR2MASK["npu"] == 12
    assert "ascend" in DGLContext.STR2MASK
    assert DGLContext.STR2MASK["ascend"] == 12

    # Check that 12 is in MASK2STR
    assert 12 in DGLContext.MASK2STR
    assert DGLContext.MASK2STR[12] == "npu"
    
    print("Device context mapping test passed!")


def test_graph_to_npu():
    """Test graph.to() with NPU device (if available)."""
    # Create a simple graph
    src = torch.tensor([0, 1, 2])
    dst = torch.tensor([1, 2, 3])
    g = dgl.graph((src, dst))
    
    # Add some node features
    g.ndata['h'] = torch.randn(4, 5)
    
    print(f"Initial device: {g.device}")
    assert g.device == torch.device('cpu')
    
    # Check if NPU is available (torch_npu >= 2.5.1 auto-registers with torch)
    has_npu = _check_npu_available()
    
    if has_npu:
        # Move to NPU
        g_npu = g.to(torch.device('npu:0'))
        print(f"After to(npu:0): {g_npu.device}")
        assert g_npu.device == torch.device('npu', 0)
        
        # Check node features are also on NPU
        assert g_npu.ndata['h'].device == torch.device('npu', 0)
        
        # Move back to CPU
        g_cpu = g_npu.to(torch.device('cpu'))
        print(f"After to(cpu): {g_cpu.device}")
        assert g_cpu.device == torch.device('cpu')
        
        print("Graph NPU transfer test passed!")
    else:
        print("NPU not available, skipping transfer test")


def test_backend_functions():
    """Test backend device_type and device_id functions."""
    from dgl import backend as F
    
    # Test CPU
    cpu_device = torch.device('cpu')
    assert F.device_type(cpu_device) == 'cpu'
    assert F.device_id(cpu_device) == 0
    
    # Test CUDA (if available)
    if torch.cuda.is_available():
        cuda_device = torch.device('cuda:0')
        assert F.device_type(cuda_device) == 'cuda'
        assert F.device_id(cuda_device) == 0
    
    # Test NPU (if available)
    if _check_npu_available():
        npu_device = torch.device('npu:0')
        assert F.device_type(npu_device) == 'npu'
        assert F.device_id(npu_device) == 0
    
    print("Backend device functions test passed!")


def test_to_backend_ctx():
    """Test to_backend_ctx function with different device types."""
    from dgl import backend as F
    from dgl._ffi.runtime_ctypes import DGLContext
    
    # Test CPU context (device_type=1)
    cpu_ctx = DGLContext(1, 0)
    torch_cpu = F.to_backend_ctx(cpu_ctx)
    assert torch_cpu == torch.device('cpu')
    
    # Test CUDA context (device_type=2)
    if torch.cuda.is_available():
        cuda_ctx = DGLContext(2, 0)
        torch_cuda = F.to_backend_ctx(cuda_ctx)
        assert torch_cuda == torch.device('cuda', 0)
    
    # Test NPU context (device_type=12)
    if _check_npu_available():
        npu_ctx = DGLContext(12, 0)
        torch_npu_dev = F.to_backend_ctx(npu_ctx)
        assert torch_npu_dev == torch.device('npu', 0)
    
    print("to_backend_ctx test passed!")


if __name__ == "__main__":
    test_device_context_mapping()
    test_backend_functions()
    test_to_backend_ctx()
    test_graph_to_npu()
    print("\nAll tests passed!")
