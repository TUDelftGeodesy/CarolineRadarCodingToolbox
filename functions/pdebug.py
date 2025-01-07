from matplotlib.patches import Polygon as mplPolygon

def plot_polygon(polygon,IGRS,ax,plt):
    # fig, ax = plt.subplots()
    
    # Extract coordinates from the polygon
    x, y = polygon.exterior.xy

    # Create a matplotlib Polygon patch
    mpl_polygon = mplPolygon(list(zip(x, y)), edgecolor='black', facecolor='none')
    
    # Plot the point
    ax.plot(IGRS.x,IGRS.y,'ro')
    
    # Add the patch to the axes
    ax.add_patch(mpl_polygon)
    
    
    # Set the x and y axis limits appropriately
    ax.set_xlim(min(x)-1, max(x)+1)
    ax.set_ylim(min(y)-1, max(y)+1)
    
    # plt.axis('equal')  # Equal aspect ratio for x and y axes
    # plt.show()
    
    return
    
import matplotlib.pyplot as plt

def plotorbit(orbit,):
    
    """ input:
            - orbit:     3xN array
    """
    
    
    x = orbit[0,:]
    y = orbit[1,:]
    z = orbit[2,:]
    
   
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z)
    ax.scatter(0,0,0,'k')
    
    return
    

    
    