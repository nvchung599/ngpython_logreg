from general import *
#https://docs.scipy.org/doc/numpy/reference/generated/numpy.meshgrid.html

toggle = 1

if toggle == 0:
    x = np.arange(1,10)
    y = np.arange(1,10)
    xx, yy = np.meshgrid(x, y)
    xy_pairs = np.dstack([xx, yy]).reshape(-1, 2)

    mask = xy_pairs[:,1]>5
    print(mask.shape)
    xy_pairs_masked = xy_pairs[mask]
    print(xy_pairs_masked.shape)


    #plt.scatter(xy_pairs[:,0], xy_pairs[:,1], marker='o')
    plt.scatter(xy_pairs_masked[:,0], xy_pairs_masked[:,1], marker='o')
    plt.show()

if toggle == 1:
    print(np.arange(1,5,1))
