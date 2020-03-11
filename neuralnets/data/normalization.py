import numpy

def channel_minmax_scale(x, m=None, post_norm="zero"):

    if not post_norm in ["zero", "mean", "no"]:
        raise ValueError("%s not recognized." % post_norm)

    if m is None:
        flat_x = x.reshape(x.shape[0], -1)
        mi = flat_x.min(axis=1)
        ma = flat_x.max(axis=1)
    else:
        x_copy = x.copy()
        x_copy[~m] = numpy.nan
        flat_x = x_copy.reshape(x_copy.shape[0], -1)
        
        nan_check = numpy.isnan(flat_x).all(axis=1)
        if nan_check.any():
            flat_x[nan_check] = 0.0
            
        mi = numpy.nanmin(flat_x, axis=1)
        ma = numpy.nanmax(flat_x, axis=1)

    denom = ma - mi
    denom[denom <= 0.0] = 1.0

    normed = ((x.T-mi)/denom).T

    if numpy.isnan(normed).any():
        raise ValueError("normed contains nan")

    if m is not None:
        if post_norm == "mean":
            for i in range(normed.shape[0]):
                if m[i].any():
                    me = numpy.mean(normed[i][m[i]])
                    if numpy.isnan(me):
                        raise ValueError("here", me)
                    if (~m[i]).any():
                        normed[i][~m[i]] = me
        elif post_norm == "zero":
            normed[~m[i]] = 0.0

    return normed