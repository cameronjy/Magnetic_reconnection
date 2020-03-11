import yt
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as spc
import sys
import getopt
plt.rcParams.update({'font.size': 20})

#Derived fields that will get loaded along with the dataset.
#Number Density
def _nd(field, data):
    return(data["particle_weight"] * 1/(14.69692/6144 * yt.units.meter)**2 * 1/yt.units.meter)
#Kinetic Stress-Energy Tensor xx Component
def _pxx(field, data):
    gamma = (data["particle_momentum_x"])**2+(data["particle_momentum_y"])**2+(data["particle_momentum_z"])**2
    gamma = gamma/(me*c)**2
    gamma = np.sqrt(1+gamma)
    return(me * data["numberDensity"] * (data["particle_momentum_x"]/gamma/me)**2 * gamma)
#Total Magnetic Pressure
def _BPressure(field, data):
    return((data["Bx"]**2+data["By"]**2+data["Bz"]**2)/(2*mu))
#Magnetic Pressure in the X Direction
def _BxPressure(field, data):
    return(data["Bx"]**2/(2*mu))
#Total Electric Pressure
def _EPressure(field, data):
    return((data["Ex"]**2+data["Ey"]**2+data["Ez"]**2)*epsilon/2)
#Electric Pressure in the x Direction
def _ExPressure(field, data):
    return(data["Ex"]**2 *epsilon/2)
#Electromagnetic Stress-Energy Tensor xx Component
def _EMEnergyDensity(field, data):
    return(data["BPressure"]+data["EPressure"]-2*data["BxPressure"]-2*data["ExPressure"])

#Analytic Maxwell Juttner Distribution of u_x for a three dimensional distribution boosted in the x direction by beta.
def MJD(x,beta,theta):
    Gamma = 1/np.sqrt(1-beta**2)
    return( 1/(2 * me * c * Gamma**3 * spc.kn(2,1/theta)) * (Gamma * np.sqrt(1. + (x/(me * c))**2) + theta) * np.e**(-Gamma/theta * (np.sqrt(1. + (x/(me * c))**2) - beta * x/(me * c)))) * yt.units.kilogram*yt.units.meter/yt.units.second
#Fitting routine to find theta of a given distribution.
def thetaFit(x,f,beta,theta0):
    theta = theta0
    delta = theta/5
    R = MJD(x,beta,theta)
    R = np.abs((R - f))**2
    R = sum(R)
    oldR = R * 10
    while(R != oldR):
        oldR = R
        theta += delta
        R = MJD(x,beta,theta)
        R = np.abs((R - f))**2
        R = sum(R)
        if(R>oldR):
            delta = delta * -1/2
    return(theta)

#Relevant Physical Constants
#Electron Mass
me = 9.10938356*10**-31 * yt.units.kg
#Electron Charge
e = 1.60217662E-19 * yt.units.coulomb
#Speed of Light
c = 299792458 * yt.units.meter/yt.units.second
#Vacuum Permeability of Free Space
mu = 1.25663706212e-6 * yt.units.kg * yt.units.meter / (yt.units.Coulomb**2)
#Vacuum Permitivity of Free Space
epsilon = 8.88541878128e-12 * yt.units.coulomb**2/(yt.units.kg * yt.units.meter/yt.units.second**2 * yt.units.meter**2)

def main(argv):

    #Parse arguments for Simulation Size
    try:
        opts, args = getopt.getopt(argv,"h",["help","sigma=","lambdae=","beta=","xBins=","zBins=","plotfile="])
    except:
        print("Harris-Sheet-Diagnostics.py --sigma --lambdae --beta --xBins --zBins --plotfile")
        print("All Inputs are Required")
        print("Unrecognized Symbol or Missing Argument")
        sys.exit(1)

    #Magnetization Sigma
    sigma = None
    #Skin Depth
    lambdae = None
    #Current Sheet Drift Velocity
    beta = None
    #Desired number of bins in diagnostic domain
    xBins = None
    zBins = None
    #plotfile number
    plotfile = None
    
    for opt, arg in opts:
        if opt == "-h" or opt == "--help":
            print("Harris-Sheet-Diagnostics.py --sigma --lambdae --beta --xBins --zBins --plotfile")
            print("All Inputs are Required")
        elif opt == "--sigma":
            sigma = float(arg)
        elif opt == "--lambdae":
            lambdae = float(arg)
        elif opt == "--beta":
            beta = float(arg)
        elif opt == "--xBins":
            xBins = int(arg)
        elif opt == "--zBins":
            zBins = int(arg)
        elif opt == "--plotfile":
            plotfile = arg

    if (sigma == None or lambdae == None or beta == None or xBins == None or zBins == None or plotfile == None):
        print("Harris-Sheet-Diagnostics.py --sigma --lambdae --beta --xBins --zBins --plotfile")
        print("All Inputs are Required")
        sys.exit(1)

    #Load data and derived fields
    ds = yt.load("diags/plotfiles/plt"+arg)
    ds.add_field(("all","numberDensity"), function = _nd, units = "1/m**3", particle_type = True)
    ds.add_field(("all","pxx"), function = _pxx, units = "kg/(m*s**2)", particle_type = True)
    ds.add_field(("boxlib","BPressure"), function = _BPressure, units = "kg/(m*s**2)", particle_type = False)
    ds.add_field(("boxlib","EPressure"), function = _EPressure, units = "kg/(m*s**2)", particle_type = False)
    ds.add_field(("boxlib","BxPressure"), function = _BxPressure, units = "kg/(m*s**2)", particle_type = False)
    ds.add_field(("boxlib","ExPressure"), function = _ExPressure, units = "kg/(m*s**2)", particle_type = False)
    ds.add_field(("boxlib","EMEnergyDensity"), function = _EMEnergyDensity, units = "kg/(m*s**2)", particle_type = False )

    Lx = (float(ds.domain_right_edge[0]) - float(ds.domain_left_edge[0])) * yt.units.meter
    Lz = (float(ds.domain_right_edge[1]) - float(ds.domain_left_edge[1])) * yt.units.meter
    xCell = ds.domain_dimensions[0]
    zCell = ds.domain_dimensions[1]

    #Distance of current sheet from z axis.
    xcs = Lx/4
    #Domain Lower Left
    lowerLeft = (-xcs-Lx/xCell*xBins/2, 0,-0.5)
    #Domain Upper Right
    upperRight = (-xcs+Lx/xCell*xBins/2, Lx/xCell*zBins,0.5)
    #Center of Domain
    center = tuple((lowerLeft[x]+upperRight[x])/2 for x in range(3))
    #Position of the center of each bin
    xpos = np.arange(lowerLeft[0]+(upperRight[0]-lowerLeft[0])/xBins/2,upperRight[0], (upperRight[0]-lowerLeft[0])/xBins) * yt.units.meter
    #Location of the current sheet.

    lambdae *= yt.units.meter
    #Sheet Number Density
    nd = c**2 * me * epsilon / (lambdae**2 * e**2)
    #Background Number Density
    nb = nd/5
    #Peak Magnetic Field
    B0 = np.sqrt(mu * sigma * nb * me * c**2)
    #Current Sheet Half Thickness
    delta = sigma * c * me/(B0 * e)/3

    #Number Density Analytic Profile
    n = nb + (nd-nb)*(np.cosh((xpos-xcs)/delta)**-1+np.cosh((xpos+xcs)/delta)**-1)
    #Magnetic Field Analytic Profile
    B = B0*(-1 - np.tanh((xpos-xcs)/delta)+np.tanh((xpos+xcs)/delta))
    #Drift Beta Analytic Profile 
    beta = 0 + (0.3-0)*(np.cosh((xpos-xcs)/delta)**-1-np.cosh((xpos+xcs)/delta)**-1)
    #Theta Analytic Profile
    theta = 30/4 * (51/50 - (-1 - np.tanh((xpos-xcs)/delta)+np.tanh((xpos+xcs)/delta))**2)/((1+4*(np.cosh((xpos-xcs)/delta)**-1+np.cosh((xpos+xcs)/delta)**-1))*np.sqrt(1-beta**2))

    #Create relevant domains
    cg = ds.covering_grid(left_edge=lowerLeft, dims = (xBins,zBins,1), level=0)
    dd = ds.box(lowerLeft, upperRight, ds=ds, field_parameters=None, data_source=None)

    #Store diagnostic grids to bin particles
    totalParticles = int(dd[("all","particle_position_x")].size)
    thetaGrid = np.zeros(xBins)
    thetaGridCount = np.zeros(xBins)
    thetaGridCount = thetaGridCount.astype(int)
    particleCell = np.zeros(totalParticles)
    particleCell = particleCell.astype(int)
    pxxGrid = np.zeros(xBins) * yt.units.joule/yt.units.meter**3
    ndGrid = np.zeros(xBins) * yt.units.meter**-3

    #Load particle data for binning
    pos = dd[("all","particle_position_x")]
    particleMomentumX = dd[("all","particle_momentum_x")]
    pxx = dd[("all","pxx")]
    particleMomentumY = dd[("all","particle_momentum_y")]
    gamma = dd[("all","particle_momentum_x")]**2+dd[("all","particle_momentum_y")]**2+dd[("all","particle_momentum_z")]**2
    gamma = gamma/(me*c)**2
    gamma = np.sqrt(1+gamma)
    nd = dd[("all","numberDensity")]

    #Bin Particle Data
    for x in range(pos.size):
        i = int(np.ceil((pos[x]-center[0])/(upperRight[0]-lowerLeft[0])*xBins) + (xBins/2-1))
        particleCell[x] = i
        thetaGridCount[i] += 1
        pxxGrid[i] += pxx[x]/zBins
        ndGrid[i] += nd[x]

    thetaGridBins = np.empty(xBins,dtype=object)
    for i in range(xBins):
        thetaGridBins[i] = np.zeros(thetaGridCount[i])
    thetaGridCount = np.zeros(xBins)
    thetaGridCount = thetaGridCount.astype(int)
    for x in range(totalParticles):
        thetaGridBins[particleCell[x]][thetaGridCount[particleCell[x]]] = particleMomentumX[x]
        thetaGridCount[particleCell[x]] += 1

    #Fit binned momenta to distributions
    for i in range(xBins):
        y, x = np.histogram(thetaGridBins[i], bins=400, density = True)
        x = (x[:-1] + x[1:])/2 * yt.units.kilogram*yt.units.meter/yt.units.second
        thetaGrid[i] = (thetaFit(x,y,0,(1.52-0.15)/2))
        z = MJD(x,0.0,thetaGrid[i])

    plt.plot(xpos,pxxGrid)
    plt.plot(xpos,cg["Bz"][:,0,0]**2/(2*mu))
    plt.plot(xpos,pxxGrid+cg["Bz"][:,0,0]**2/(2*mu))
    plt.plot(xpos,me*c**2*2*n*theta*np.sqrt(1-beta**2)+B**2/(2*mu))
    plt.title('Value of Energy Density Calculated By Theta Over Current Sheet')
    plt.gca().set_ylabel('$J/m^3$')
    plt.gca().set_xlabel('$x (m)$')
    plt.gcf().set_facecolor('white')
    plt.legend(['Pxx Computed With Momentum Sums','Computed Exx','Computed Pxx+Exx','Analytic Expectation of Pxx+Exx'])
    plt.gcf().set_size_inches((24,13.5))
    plt.savefig("EnergyDensityTheta.png")
    plt.close()
    
    plt.plot(xpos,me*c**2 * thetaGrid * ndGrid*np.sqrt(1-beta**2))
    plt.plot(xpos,cg["Bz"][:,0,0]**2/(2*mu))
    plt.plot(xpos,me*c**2 * thetaGrid * ndGrid*np.sqrt(1-beta**2)+cg["Bz"][:,0,0]**2/(2*mu))
    plt.plot(xpos,me*c**2*2*n*theta*np.sqrt(1-beta**2)+B**2/(2*mu))
    plt.title('Value of Energy Density Calculated By Binning Over Current Sheet')
    plt.gca().set_ylabel('$J/m^3$')
    plt.gca().set_xlabel('$x (m)$')
    plt.gcf().set_facecolor('white')
    plt.legend(['Pxx Computed by Fitted $\\theta(x)$','Computed Exx','Computed Pxx+Exx','Analytic Expectation of Pxx+Exx'])
    plt.gcf().set_size_inches((24,13.5))
    plt.savefig("EnergyDensityBeta.png")
    plt.close()

    plt.plot(xpos,pxxGrid+cg["Bz"][:,0,0]**2/(2*mu))
    plt.plot(xpos,me*c**2 * thetaGrid * ndGrid*np.sqrt(1-beta**2)+cg["Bz"][:,0,0]**2/(2*mu))
    plt.plot(xpos,me*c**2*2*n*theta*np.sqrt(1-beta**2)+B**2/(2*mu))
    plt.title('Value of Energy Density Over Current Sheet')
    plt.gca().set_ylabel('$J/m^3$')
    plt.gca().set_xlabel('$x (m)$')
    plt.gcf().set_facecolor('white')
    plt.legend(['Computed by YT','Computed by Fitted $\\theta(x)$','Analytic Expectation'])
    plt.gcf().set_size_inches((24,13.5))
    plt.savefig("EnergyDensity.png")
    plt.close()

    curlB = (cg["Bz"][2:,:,:]-cg["Bz"][:-2,:,:])/(2*(upperRight[0]-lowerLeft[0])/xBins)
    plt.plot(xpos[1:-1],cg["Jy"][1:-1,0,0])
    plt.plot(xpos[1:-1],curlB[:,0,0]/mu)
    plt.title('Ampere\'s Law Check')
    plt.gca().set_xlabel('x')
    plt.gca().set_ylabel('Current Density $A/m^3$')
    plt.legend(['WarpX Gathered Jy','Grid Curl of B'])
    plt.gcf().set_size_inches((24,13.5))
    plt.savefig("AmperesLaw.png")
    plt.close()

if __name__ == "__main__":
    main(sys.argv[1:])
