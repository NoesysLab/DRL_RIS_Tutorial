import numpy as np


import globals

def calc_pathloss(tx_location,
                  rx_location,
                  isLOS,):

    myDist = np.sqrt(np.sum(np.power(tx_location-rx_location, 2), axis=1)) # todo: check dimensions

    if isLOS:
        pathlossDB = globals.pathlossCoefficientLOS\
                     + 10*globals.pathlossExponentLOS\
                     *np.log10(myDist/globals.referenceDistance)
    else:
        pathlossDB = globals.pathlossCoefficientNLOS \
                     + 10 * globals.pathlossExponentNLOS \
                     * np.log10(myDist / globals.referenceDistance)

    myPthLoss = np.power(10, (pathlossDB/10))
    return myPthLoss