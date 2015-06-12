#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 2015-05-29, dyens
#
import sys
import math

__author__ = 'mkompan'

import matplotlib.pylab as pl
import numpy



def bb1(g1, g2):
    return 2.33333333333333 * g1 ** 2 + 1.16666666666667 * g1 * g2 - g1 + 0.25 * g2 ** 2


def bb2(g1, g2):
    return 2.0 * g1 * g2 + 0.583333333333333 * g2 ** 2 - g2


def checkArea(x, y):
#    return 2 > x > -1 and -2 < y < 2
#    return 20 > x > -20 and -20 < y < 20
    return 50 > x > -200 and -1000 < y < 1000


xReg, yReg = ((-0.2, 1.2), (-2., 1.))
Y, X = numpy.mgrid[yReg[0]:yReg[1]:1000j, xReg[0]:xReg[1]:1000j]

U = -bb1(X, Y)
V = -bb2(X, Y)
speed = numpy.sqrt(U * U + V * V)
XPlus = numpy.array([0, xReg[1]])
XMinus = numpy.array([xReg[0], 0])

pl.figure()
ax = pl.subplot(111)
print ax.axis()

grid = pl.grid()
poly = pl.Polygon([(0, 0), (xReg[1], 0), (xReg[1], -xReg[1] / 9 * 12.)], color='k', alpha=0.2)
ax.add_patch(poly)
poly = pl.Polygon([(0, 0), (xReg[0], -xReg[0] * 4), (xReg[0], yReg[0]), (xReg[1], -xReg[1] * 2.)], color='k', alpha=0.4)
ax.add_patch(poly)

pl.plot(XPlus, XPlus / 9 * (-12.), 'k--')
pl.plot(XPlus, XPlus * 0, 'k--')
pl.plot(XPlus, -XPlus * 2, 'k--')
pl.plot(XMinus, -XMinus * 4, 'k--')
pl.axis([xReg[0], xReg[1], yReg[0], yReg[1]])
#pl.streamplot(X, Y, U, V, color=speed)
pl.streamplot(X, Y, U, V, color='k')
#pl.colorbar()
pl.plot([0], [0], 'wo', ms=8)
pl.plot([3. / 7, 9. / 11], [0, -12. / 11], 'ko', ms=8)
pl.plot([12. / 17], [-12. / 17], 'kD', ms=8)
annotationSize = 15
ax.annotate('O', xy=(0, 0.), size=annotationSize, xycoords='data',
            xytext=( 0.03, 0.1), textcoords='data',
            bbox=dict(boxstyle="round", fc='1.'))
ax.annotate('A', xy=(3. / 7, 0.), size=annotationSize, xycoords='data',
            xytext=(3. / 7 + 0.03, 0.1), textcoords='data',
            bbox=dict(boxstyle="round", fc='1.'))
ax.annotate('B', xy=(12. / 17, -12. / 17), size=annotationSize, xycoords='data',
            xytext=(12. / 17 + 0.03, -12. / 17 + 0.1), textcoords='data',
            bbox=dict(boxstyle="round", fc='1.'))
ax.annotate('C', xy=(9. / 11, -12. / 11), size=annotationSize, xycoords='data',
            xytext=(9. / 11 + 0.03, -12. / 11 + 0.1), textcoords='data',
            bbox=dict(boxstyle="round", fc='1.'))

# ax.annotate('seddle points', xy=(3./7, 0.),  size=annotationSize, xycoords='data',
#             xytext=(-0.1, 0.5), textcoords='data',
#             bbox=dict(boxstyle="round", fc='1.'),
#             arrowprops=dict(arrowstyle="simple", shrinkA=10,
#                             shrinkB=10, color='k', alpha=0.7,
#                             connectionstyle="arc3,rad=.2"))
# ax.annotate('seddle points', xy=(9./11, -12./11.), size=annotationSize, xycoords='data',
#             xytext=(-0.1, 0.5), textcoords='data',
#             arrowprops=dict(arrowstyle="simple", shrinkA=10,
#                             shrinkB=10, color='k', alpha=0.7,
#                             connectionstyle="arc3,rad=.2"))
ax.annotate('Unphysical region', xy=(0, -1.5), size=annotationSize, xycoords='data',
            xytext=(-0.1, -1.5), textcoords='data',
            bbox=dict(boxstyle="round", fc="0.7"),
)
# ax.annotate('IR-attractive point', xy=(12./17, -12./17.),  size=annotationSize, xycoords='data',
#             xytext=(0.3, 0.7), textcoords='data',
#             bbox=dict(boxstyle="round", fc='1.'),
#             arrowprops=dict(arrowstyle="simple",
#                             shrinkB=10, color='k', alpha=0.7))
ax.annotate('Basin of attraction', xy=(0.7, -0.25), size=annotationSize, xycoords='data',
            xytext=(0.7, -0.25), textcoords='data',
            bbox=dict(boxstyle="round", fc="0.9"),
)

pl.xlabel('$g_1/\\varepsilon$', size=annotationSize)
pl.ylabel('$g_2/\\varepsilon$', size=annotationSize)

ax.set_xticks(numpy.array([0., 3./7, 0.5, 12. / 17, 9./11, 1.]))
ax.set_xticklabels(['0.', '${3}/{7}$', '0.5', '${12}/{17}$', '${9}/{11}$', '1.'])
ax.set_yticks(numpy.array([-2.0, -12./11, -1, -12. / 17, 0., 1.]))
ax.set_yticklabels(['-2.', '$-{12}/{11}$', '-1.', '$-{12}/{17}$', '0.', '1.'])

pl.show()
sys.exit()


