from scipy import spatial as sp

def getAverageFromNeighbours(data,point,velocity, kNearest=3):
    """
    Data is a array_like
    x,y,z is the point to predict
    """
    x,y,z = point

    sp.KDTree(data)

    _, neighbours = sp.query([[x,y,z]], k=kNearest) # 3 Neighbours

    totalVel = 0
    for i in neighbours:
        totalVel += velocity[i]
    
    return (totalVel / kNearest)

    
    


    