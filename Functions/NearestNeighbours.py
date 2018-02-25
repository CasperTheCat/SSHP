from scipy import spatial as sp

def getAverageFromNeighbours(data,point,velocity, kNearest=3):
    """
    Data is a array_like
    x,y,z is the point to predict
    """
    x,y,z = point

    tree = sp.KDTree(data)

    dist, neighbours = tree.query([[x,y,z]], k=kNearest) # 3 Neighbours

    totalVel = 0
    usedVels = 0

    #for i in neighbours:
    for i in range(0, neighbours.shape[1]):
        if(dist[0][i] != 0):
            totalVel += velocity[neighbours[0][i]]
            usedVels += 1
    
    return (totalVel / usedVels)

    
def getAverageFromNeighboursFromTree(data,point,velocity, tree, kNearest=3):
    """
    Data is a array_like
    x,y,z is the point to predict
    """
    x,y,z = point

    dist, neighbours = tree.query([[x,y,z]], k=kNearest) # 3 Neighbours

    totalVel = 0
    usedVels = 0

    #for i in neighbours:
    for i in range(0, neighbours.shape[1]):
        if(dist[0][i] != 0):
            totalVel += velocity[neighbours[0][i]]
            usedVels += 1
    
    #print("Summed to " + str(totalVel))
    #print("Divisor of " + str(usedVels))
    #print("Returning " + str(totalVel / usedVels))
    return (totalVel / usedVels)
    


    