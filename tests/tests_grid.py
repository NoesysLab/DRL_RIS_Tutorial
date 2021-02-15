from core.experiment_setups import *
from core.surfaces import RIS
from utils.plotting import plot_Grid



r1 = RIS(np.array([2, 1, 0]), )
r2 = RIS(np.array([4, -1, 0]), )
r3 = RIS(np.array([6, 1, 0]), )
r4 = RIS(np.array([8, -1, 0]), )

ris_list = [r1,r2,r3,r4]

tx = np.array([[0,0,0], [1,1,0]])
rx = np.array([[10,0,0], [10,1,0], [10,-1,0]])



def test_construct_grid():
    grid = PositionGrid(ris_list, tx, rx)


def test_plot_ascii():

    grid = PositionGrid(ris_list, tx, rx)
    plot = grid.plot_ascii(width_chars=20, height_chars=7, return_str=True)
    print(plot)
    assert plot=='''........*......*...o
....................
....................
x..................o
....................
....................
..x.*......*.......o
'''



def test_grid_from_ascii():
    ascii = '''........*......*...o
....................
....................
x..................o
....................
....................
..x.*......*.......o
'''
    grid = PositionGrid.from_ascii(ascii, 10, 0, 0, 0)
    res = grid.plot_ascii(width_chars=20, height_chars=7, return_str=True)
    assert ascii==res


def test_grid_from_ascii_different_values():
    ascii = '''........*......*...o
....................
....................
x..................o
....................
....................
..x.*......*.......o
'''
    grid = PositionGrid.from_ascii(ascii, [1, 2, 3, 4], [10, 20, 30, 40], [10, 20], [1, 2, 3], [1, 2, 3, 4])
    res = grid.plot_ascii(width_chars=20, height_chars=7, return_str=True)
    assert ascii == res


def test_grid_from_ascii_scaling():
    ascii = '''........*......*...o
....................
....................
x..................o
....................
....................
..x.*......*.......o
'''
    grid = PositionGrid.from_ascii(ascii, 10, 0, 0, 0, scale_x=3, scale_y=2)
    res = grid.plot_ascii(width_chars=20, height_chars=7, return_str=True)
    print(res)
    assert ascii == res


def test_plot():
    grid = PositionGrid(ris_list, tx, rx)
    plot_Grid(grid)