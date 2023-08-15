import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
from helper import get_times, get_oppositions, bestMarsOrbitParams


if __name__ == "__main__":

    # Import oppositions data from the CSV file provided
    data = np.genfromtxt(
        "../data/01_data_mars_opposition_updated.csv",
        delimiter=",",
        skip_header=True,
        dtype="int",
    )
    

    #Extract times from the data in terms of number of days.
    #"times" is a numpy array of length 12. The first time is the reference
    #time and is taken to be "zero". That is times[0] = 0.0
    times = get_times(data)
    assert len(times) == 12, "times array is not of length 12"

    # # Extract angles from the data in degrees. "oppositions" is
    # # a numpy array of length 12.
    oppositions = get_oppositions(data)
    assert len(oppositions) == 12, "oppositions array is not of length 12"

    # # Call the top level function for optimization
    # # The angles are all in degrees
    r, s, c, e1, e2, z, errors, maxError = bestMarsOrbitParams(
        times, oppositions
    )

    assert max(list(map(abs, errors))) == maxError, "maxError is not computed properly!"
    print(
        "Fit parameters: r = {:.4f}, s = {:.4f}, c = {:.4f}, e1 = {:.4f}, e2 = {:.4f}, z = {:.4f}".format(
            r, s, c, e1, e2, z
        )
    )
    print("The maximum angular error in degree = {:2.4f}".format(maxError))
