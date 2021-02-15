import pytest

from core.surfaces import *


# def test_construct_ris():
#     ris = RIS(np.array([1, 2, 0]), )
#
#     assert id(ris) == ris.id
#
#
# def test_construct_ris_illegal_num_elements1():
#     with pytest.raises(ValueError):
#         ris = RIS(np.array([1, 2, 0]), )
#
# def test_construct_ris_illegal_num_elements2():
#     with pytest.raises(ValueError):
#         ris = RIS(np.array([1, 2, 0]), )
#
# def test_construct_ris_illegal_position1():
#     with pytest.raises(ValueError):
#         ris = RIS(np.array([1, 2, 0, 124]), )
#
# def test_construct_ris_illegal_position2():
#     with pytest.raises(ValueError):
#         ris = RIS(444444, )
#
# def test_ris_phase_attr():
#     phase = np.arange(0,10).reshape(2,5)
#     ris = RIS(np.array([1, 2, 0]), )
#     ris.phase = phase
#     assert np.array_equal(ris.phase, phase)
#
#
# def test_get_all_ris_positions():
#     pos = get_all_ris_positions(
#         [RIS(np.array([1, 2, 0]), ),
#          RIS(np.array([3, 4, 0]), ),
#          RIS(np.array([5, 6, 0]), ),
#          RIS(np.array([7, 8, 0]), )
#          ])
#
#     expected = np.array([
#         [1,2,0],
#         [3,4,0],
#         [5,6,0],
#         [7,8,0]
#     ])
#     assert np.array_equal(pos, expected)
#
#
# def test_get_all_ris_empty():
#     pos = get_all_ris_positions([])
#     assert np.array_equal(pos, np.array([]))
#





def test_RIS_constructor():
    r1 = RIS(np.array([0, 0, 0]),
             (2, 2),
             (1, 1),
             (1, 1),
             (1, 1),
             (2, 2),
             ('binary', {}),
             ('discrete', {'values':[1, np.pi]})
             )

    r1.set_random_state()
    print(r1.get_phase('2D'))

