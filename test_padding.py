from batching import pad_2d


def test_pad_2d_shapes_and_values():
  x, lens = pad_2d([[1, 2], [3]], pad_id=0)
  assert x.shape == (2, 2)
  assert lens.tolist() == [2, 1]
  assert x.tolist() == [[1, 2], [3, 0]]
