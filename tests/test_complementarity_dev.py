import math
import torch
from benchmarks.complementarity_dev import complementary_fixture, run


def test_fixture():
    train, dev = complementary_fixture()
    assert len(train) == len(dev) == 1024
    for data in (train, dev[:512]):
        records = [data[i:i+32] for i in range(0,len(data),32)]
        assert {r[4:12] for r in records} == {b'........'}
        assert all(r[12] == r[0]+23 for r in records)
        assert {r[12] for r in records} == {88,89}
    assert not set(train[i:i+32] for i in range(0,len(train),32)) & set(dev[i:i+32] for i in range(0,512,32))
    assert dev[512:544]*16 == dev[512:]
    assert 128 not in train


def test_complementary_harness():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        r = run(42,b'abcd'*16,b'abce'*8,epochs=1)
    finally:
        torch.set_num_threads(old)
    assert len(r['backbones']) == 4
    for b in r['backbones'].values():
        assert b['optimizer_steps'] == 2
        assert set(b['arms']) == {'neural','fixed','global','confidence'}
        for name, arm in b['arms'].items():
            assert arm['bytes'] == 31 and math.isfinite(arm['bpb'])
            assert arm['hebbian_updates'] == 0
            assert arm['arbitration_updates'] == (31 if name in ('global','confidence') else 0)
