# NNb.tools contains
# - purgeFolder()
# - clearDic()
# - arguments()
# - incrementFileName()
# - nameFile()
# - bSure()
# - getShortHalfAxis()
# - getEllipseOrdinate()
# - getBiggestArc()
# - checkFolder()
# - checkRunFolder()

#-------------------------
#
import numpy as np
import time as t
import os


#---------------------------------------------------------------------------------------------------
#
def purgeFolder(directory = 'data_folder', bVerbose = False, bPurgeList = False):
    '''
        Delete everithing there is in given folder and its children. The removal is permanent.
        
            
        Input :
            - directory : (str) folder that will be removed
            - bVerbose : (bool) activate messages
        
        Output :
            - indicator : (int)
            if folder doesn't exist will return -1, else returns 0.
        
        Example :
            - purgeFolder('data_folder/dummyFold')
            
        Requirements:
            - os.system()
            - os.path.exists()
            - time.strftime()
    '''
    
    if not os.path.exists(directory):
        if bVerbose:
            print("purgeFolder(): ./{} doesn't exist, nothing will be purged".format(directory))
        return -1
    else:
        purgeFile = "purgeList.txt"
        
        if bVerbose:
            print("purgeFolder(): ./{} is being purged.".format(directory), end='')
        
        if bPurgeList:
            if bVerbose:
                print(" Removed files will appear in {}.".format(purgeFile))
            os.system('echo "\n{}" >> {}'.format(t.strftime("%d-%m-%Y_%H:%M:%S"), purgeFile))
        else:
            if bVerbose:
                print('')
        
        # Folder removal
        if bPurgeList:
            os.system('rm -rv {} >> {}'.format(directory, purgeFile))
        else:
            os.system('rm -rv {}'.format(directory))
            
    if bVerbose:
        print("Purge finished")
        
    return 0


#---------------------------------------------------------------------------------------------------
# 
def clearDic(A, B):
    '''
        Return intersection and subtraction of two dictionnaries. A will impose its values.
        
        Input :
            - A : (dict) 
            - B : (dict)
            
        Output :
            - intersection : (dict) A ∩ B
            - difference : (dict) A - B (set theory)
            
        Example :
            -
        
        Requirements :
            - 
    '''
    
    keyIntersec = A.keys() & B.keys()
    intersection = {}
    for key in keyIntersec:
        intersection.update({key : A[key]})
    
    keyDiff = A.keys() ^ B.keys()
    difference = {}
    for key in keyDiff:
        if key in A.keys():
            difference.update({key : A[key]})
        else:
            difference.update({key : B[key]})
    
    return(intersection, difference)


#---------------------------------------------------------------------------------------------------
#
def arguments():
        """Returns tuple containing dictionary of calling function's
           named arguments and a list of calling function's unnamed
           positional arguments.
           
           src: http://kbyanc.blogspot.com/2007/07/python-aggregating-function-arguments.html
        """
        from inspect import getargvalues, stack
        posname, kwname, args = getargvalues(stack()[1][0])[-3:]
        posargs = args.pop(posname, [])
        args.update(args.pop(kwname, []))
        return args
        return args, posargs


#---------------------------------------------------------------------------------------------------
#
def incrementFileName(filename, directory = ''):
    '''
    Return a filename that is not present in ./directory/ by adding suffix '_k' with k an integer.
    
    Input :
        - filename : (str)
        - directory : (str)
        
    Output :
        - newFileName : (str)
        
    Example :
        - incrementFileName(filename = 'default.txt', directory = 'sub') returns 'default_2.txt'
        
    Requirements :
        - os.listdir()
        - os.path.join()
        - os.path.dirname()
        - os.path.basename()
    
    '''
    
    incorpDir = os.path.dirname(filename)
    directory = os.path.join(directory, incorpDir)
    
    if directory != '':
        listdir = os.listdir(directory)
    else :
        listdir = os.listdir()
    
    filename = os.path.basename(filename).split('.')
    base = ''
    for i in range(len(filename[:-1])):
        if i != 0:
            base += '.'
        base += filename[i]
        
    end = filename[-1]

    newFileName = '{}.{}'.format(base, end)

    cpt = 0
    while newFileName in listdir:
        newFileName = base + '_{}.{}'.format(cpt,end)
        cpt += 1
    
    if len(incorpDir) > 0:
        newFileName = os.path.join(incorpDir, newFileName)
    return(newFileName)


#---------------------------------------------------------------------------------------------------
#
def nameFile(elt):
    '''
        Transform any variable in a string friendly for file name
        
        Input :
            - elt : (*)
        
        Output :
            - name : (str)
        
        Example :
            - nameFile([3,2.5,16]) will return 'c3v-2d5v-16c'
        
        Deciphering :
            - numbers and letters are unaffected
            - ' ' is '-'
            if two spaces are consecutives only one is kept
            - '{' or '}' is 'a'
            - '(' or ')' is 'p'
            - '[' or ']' is 'c'
            - '.' is 'd'
            - ',' is 'v'
            - ':' is s
            - '-' is 't'
            - other characters are deleted
        
        Requirements :
            -
    '''
    strElt = str(elt)
    name = ""
    
    for i in range(len(strElt)):
        char = strElt[i]
        if char.isnumeric() or char.isalpha():
            name += char
        elif char in ['(',')']:
            name += 'p'
        elif char in ['[',']']:
            name += 'c'
        elif char in ['{','}']:
            name += 'a'
        elif char == ',':
            name += 'v'
        elif char == '.':
            name += 'd'
        elif char == ':':
            name += 's'
        elif char == '-':
            name += 't'
        elif char == ' ':
            if i > 0 and name[-1] != '-':
                name += '-'
        else:
            name += ''

    return name


#---------------------------------------------------------------------------------------------------
#
def bSure(eltU, size):
    '''
        Assure that the input is a list, and of length *size*, if not transform it.
        If len(eltU) < size, eltU will be extended with eltU[-1] while len(eltU) and *size* are different.
        Else, eltU will be shortened.
        
        
        Input :
            - eltU : (*) undetermined element
            - size : (uint) desired length of array
        
        Output :
            - eltU : (list)
            
        Example :
            - bSure(eltU = [3,2], size = 3)
            will return [3,2,2]
            - bSure(eltU = (3,2), size = 3)
            will return [(3,2), (3,2), (3,2)]
            
        Requirements :
            -
    '''
    
    if type(eltU) == type([]):
        l = len(eltU)
        L = []
        if size > l:
            for _ in range(l,size):
                L += [eltU[-1]]
            eltU += L
        else :
            eltU = eltU[:size]
    else:
        L = []
        for _ in range(size):
            L += [eltU]
        eltU = L
       
    return eltU


#---------------------------------------------------------------------------------------------------
#
def getShortHalfAxis(z_0, y_0, L = 200, bVerbose = True):
    '''
        Resolve ellipse equation : (z_0/a)^2 + (y_0/b)^2 = 1, with a^2 = (L/2)^2 + b^2
        
        Input :
            - z_0 : (float) abscissa
            - y_0 : (float) ordinate
            - L : (float) ellipse focal distance * 2
            - bVerbose : (str) activate display of warning
        
        Output :
            - b : (float) short half axis
            if an error is encountered during resolution, b set to 0.
            
        Example :
            - 
            
        Requirements :
            - numpy.sqrt()
            - numpy.float128
    '''
    
    # Resolution of (z/a)^2 + (y/b)^2 = 1, with a^2 = (L/2)^2 + b^2
#     B^2 - alpha1*B - alpha0 = 0 with B = b^2
    alpha1 = (z_0**2 + y_0**2 - (L/2)**2)
    alpha0 = (L / 2 * y_0)**2
    
    # Positive solution
    B1 = (alpha1 + np.sqrt(alpha1 ** 2 + 4 * alpha0, dtype = np.float128))/2
    
    # Protection of the result
    if B1 > 0:
        b = np.sqrt(B1)
    else :
        if bVerbose:
            print("getShortHalfAxis() : Warning B1 < 0\nb set to 0")
        b = 0
    
    return b


#---------------------------------------------------------------------------------------------------
#
def getEllipseOrdinate(z_0 = 20.0, b = 2, L = 200.0, bVerbose = True):
    '''
        Return the ordinate associated to abscissa z_0, relative to focus, for the ellipse characterized by b and L.

        Input :
            - z_0 : (float) absissa
            - b : (float) short half axis
            - L : (float) ellipse focal distance * 2
            - bVerbose : (str) activate display of warning
        
        Output :
            - y_0 : (float) ordinate
            if an error is encountered during resolution, y_0 set to 0
            
        Example :
            - getEllipseOrdinate(z_0 = 10.25, b = 2, L = 200, bVerbose = True)
        
        Requirements :
            - numpy.sqrt()
            - numpy.float128
    '''
    z_0a = L/2 - (z_0) # Coordinate of the maximum point of the ellipse portion
    
    # Positive solution
    Y_0 = b**2 * (1 - z_0a**2 / ((L / 2)**2 + b**2))
    
    # Protection of the result
    if Y_0 > 0:
        y_0 = np.sqrt(Y_0, dtype = np.float128)
    else :
        if bVerbose:
            print("getEllipseOrdinate() : Warning Y_0 <= 0\ny_0 set to 0")
        y_0 = 0
    
    return y_0


#---------------------------------------------------------------------------------------------------
#
def getBiggestArc(l = 10, z_start = 10, L = 200, tunnelHalfWidth = 2, bVerbose = True):
    '''
        Determine biggest ellipse arc, which end at z_start + l relatively to focus, fitting in 2 * tunnelHalfWidth.
        Consider that z_start+l is highest ordinate of arc.
        
        Input :
            - l : (float) length of elliptical arc 
            - z_start : (float) starting point of arc relative to focus
            - L : (float) ellipse focal distance * 2
            - tunnelHalfWidth : (float) ending point wanted for the arc
            - bVerbose : (str) activate display of warning
        
        Output :
            - b : (float) short half axis
            
        Example :
            -
        
        Requirements :
            - getShortHalfAxis()
    '''

    z_0 = L / 2 - (z_start + l) ; y_0 = tunnelHalfWidth # Coordinates of the maximum point of the ellipse portion
    
    b = getShortHalfAxis(z_0, y_0, L, bVerbose = bVerbose)
    
    return b


#---------------------------------------------------------------------------------------------------
# 
def tiltingAngle(dh, L):
    '''
                           dh
        Determine the    ______
        angle *t*, in    |    /
        radians, see     | t /
        figure :       L |--/
                         | /
                         |/
                         .
        Input :
            - dh : (float)
            - L : (float)
        
        Output :
            - theta : (float)(rad)
        
        Example:
            - tiltingAngle(.28, 200)
        
        Requirements :
            - numpy.arctan()
    '''
    theta = np.arctan(dh/L)
    return theta


#---------------------------------------------------------------------------------------------------
# 
def rotMatrix(t = np.pi/16, axis = 'x'):
    '''
        Generate a rotation matrix for a 3D cartesian coordinates system.
        It can be on axis rotation matrix or a composition of rotation matrix.
        
        Input :
            - t : (float)(array)(rad) angle of rotation.
            If the output is a composition of rotation matrix, dimension of *t* must be the same as *axis*. The angles must also appear in order of appearance in *axis*.
            
            - axis : (str) axis of rotation. 'X', 'x' and '0' are equivalent.
            In order to have multi rotation matrix, rotation axis must be concatenated like : 'xy' or 'yzx'. Rotation will be done in order of appearance. 
        
        Output :
            - R : (numpy.matrix) rotation matrix
            if syntax is invalid, will return identity matrix
        
        Example:
            - rotMatrix(np.pi/26, axis = '0')
            return Rx(np.pi/26)
            - rotMatrix([np.pi/4, np.pi/6], axis = 'xy')
            return Rx(np.pi/4)*Ry(np.pi/6)
        
        Requirements :
            - numpy.matrix()
            - numpy.cos()
            - numpy.sin()
            - numpy.float128
    '''
    R = np.matrix([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]], dtype = np.float128)
    
    if len(axis) > 1:
        for i in range(len(axis)):
            if axis[i] in ['x', 'X', '0']:
                tx = t[i]
            elif axis[i] in ['y', 'Y', '1']:
                ty = t[i]
            elif axis[i] in ['z', 'Z', '2']:
                tz = t[i]
    else:
        tx = t; ty = t; tz = t
    
    if 'x' in axis or 'X' in axis or '0' in axis:
        RX = np.matrix([[1,0,0],
                        [0, np.cos(tx, dtype = np.float128), -np.sin(tx, dtype = np.float128)],
                        [0, np.sin(tx, dtype = np.float128), np.cos(tx, dtype = np.float128)]], dtype = np.float128)
        R = RX*R
    
    if 'y' in axis or 'Y' in axis or '1' in axis:
        RY = np.matrix([[np.cos(ty, dtype = np.float128), 0, np.sin(ty, dtype = np.float128)],
                        [0,1,0],
                        [-np.sin(ty, dtype = np.float128), 0, np.cos(ty, dtype = np.float128)]], dtype = np.float128)
        R = RY*R
    
    if 'z' in axis or 'Z' in axis or '2' in axis:
        RZ = np.matrix([[np.cos(tz, dtype = np.float128), -np.sin(tz, dtype = np.float128), 0],
                        [np.sin(tz, dtype = np.float128), np.cos(tz, dtype = np.float128), 0],
                        [0,0,1]], dtype = np.float128)
        R = RZ*R

    return R


#---------------------------------------------------------------------------------------------------
#
def checkFolder(directory = 'data_folder', bVerbose = True, bVerboseE = False):
    '''
        Check existence of given folder.
        
        Input :
            - directory : (str) target folder
            - bVerbose : (bool) print creation indicator
            - bVerboseE : (bool) print existence indicator
            
        Output :
            - 
        
        Example :
            - checkFolder('data_folder')
        
        Requirements :
            - os.path.exists()
    '''
    if len(directory) > 0:
        if not os.path.exists(directory):
            if bVerbose:
                print('Creation of folder : {}'.format(directory))
            os.makedirs(directory)
        else:
            if bVerboseE:
                print('Folder *{}* exists'.format(directory))
    else:
        if bVerboseE:
            print('Directory name is empty')


#---------------------------------------------------------------------------------------------------
#
def checkRunFolder(directory = 'run_folder', bVerbose = True):
    '''
        Verify and create missing elements for folder *directory*, in order it to be operational for NNbar instrument simulation.
        Can generate folder, lbp_meters.off and LD2_ESS.comp.
        
        Input :
            - directory : (str) target folder
            - bVerbose : (bool) print information message
            
        Output :
            - 
        
        Example :
            - checkRunFolder('run_folderBis')
        
        Requirements :
            - checkFolder()
            - os.path.join()
            - os.path.isfile()
    '''
    
    # Check folder existence
    checkFolder(directory = directory)
    
    # Check mandatory files exitence
    ## LBP OFF file
    lbpBase = 'lbp_meters.off'
    lbp = os.path.join(directory, lbpBase)
    if not os.path.isfile(lbp):
        if bVerbose:
            print('Creation of file : {}'.format(lbpBase))
        f = open(lbp, 'w')
        f.write("OFF\n")
        f.write("32 16 0\n-0.540253 0.527000 3.710880\n-0.430901 -0.522500 2.670460\n-0.540253 -0.522500 3.710880\n-0.430901 0.527000 2.670460\n")
        f.write("-0.539635 0.528000 3.705000\n0.430901 0.528000 2.670460\n-0.430901 0.528000 2.670460\n0.539635 0.528000 3.705000\n")
        f.write("0.540253 0.527000 3.710880\n0.540253 -0.522500 3.710880\n0.430901 -0.522500 2.670460\n0.430901 0.527000 2.670460\n")
        f.write("-0.540253 -0.522500 3.710880\n-0.430901 -0.522500 2.670460\n0.430901 -0.522500 2.670460\n0.540253 -0.522500 3.710880\n")
        f.write("0.575062 -0.557500 3.707220\n0.731959 0.562000 5.200000\n0.731959 -0.557500 5.200000\n0.571330 0.562000 3.671720\n")
        f.write("-0.731959 0.562000 5.200000\n-0.575062 0.562000 3.707220\n-0.575062 -0.557500 3.707220\n-0.731959 -0.557500 5.200000\n")
        f.write("-0.731959 -0.557500 5.200000\n-0.575062 -0.557500 3.707220\n0.575062 -0.557500 3.707220\n0.731959 -0.557500 5.200000\n")
        f.write("-0.731959 0.563000 5.200000\n0.731959 0.563000 5.200000\n0.575062 0.563000 3.707220\n-0.575062 0.563000 3.707220\n")

        f.write("3 0 1 2\n3 0 3 1\n3 4 5 6\n3 4 7 5\n3 8 9 10\n3 11 8 10\n3 12 13 14\n3 15 12 14\n3 16 17 18\n")
        f.write("3 19 17 16\n3 20 21 22\n3 20 22 23\n3 24 25 26\n3 27 24 26\n3 28 29 30\n3 28 30 31")
        f.close()
    
    ## LD2_ESS component
    srcBase = 'LD2_ESS.comp'
    src = os.path.join(directory, srcBase)
    if not os.path.isfile(src):
        if bVerbose:
            print('Creation of file : {}'.format(srcBase))
        f = open(src, 'w')
        f.write("/*******************************************************************************\n*\n* McStas, neutron ray-tracing package\n")
        f.write("*         Copyright 1997-2002, All rights reserved\n*         Risoe National Laboratory, Roskilde, Denmark\n")
        f.write("*         Institut Laue Langevin, Grenoble, France\n*\n* Component: LD2_ESS\n*\n* %I\n* Written by: Matthew Frost\n")
        f.write("* Date: Sept 23, 2019\n* Origin:The University of Tennessee-Knoxville\n*\n")
        f.write("* A neutron source that provides events reflecting the phase space provided a by a \n")
        f.write("* source seen in Klinkby 2014 (https://arxiv.org/abs/1401.6003), then traced \n")
        f.write("* through the inner monolith shielding at ESS. All intensity originates at z=2.68 meters\n*\n")
        f.write("* There are no parameters, as the phase space cannot be altered and the intensity is fixed.\n*/ \n")
        f.write("DEFINE COMPONENT LD2_ESS\nDEFINITION PARAMETERS ()\nSETTING PARAMETERS ()\nOUTPUT PARAMETERS ()\n")
        f.write("/* Neutron parameters: (x,y,z,vx,vy,vz,t,sx,sy,sz,p) */\n")
        f.write("DECLARE\n%{\ndouble pmul, srcArea;\nint square;\ndouble rn,p_in,dv;\n\n")
        f.write("double v_dist[41][2] = { //Bin-normalized velocity distribution from original event output\n\t{   +0.00000000e+00,   +0.00000000e+00},\n")
        f.write("\t{   +1.00000000e+02,   +2.50614979e-02},\n\t{   +2.00000000e+02,   +1.48812038e-01},\n\t{   +3.00000000e+02,   +4.06283768e-01},\n")
        f.write("\t{   +4.00000000e+02,   +7.75966323e-01},\n\t{   +5.00000000e+02,   +1.27518496e+00},\n\t{   +6.00000000e+02,   +1.52600306e+00},\n")
        f.write("\t{   +7.00000000e+02,   +1.69156961e+00},\n\t{   +8.00000000e+02,   +1.90407912e+00},\n\t{   +9.00000000e+02,   +1.98775943e+00},\n")
        f.write("\t{   +1.00000000e+03,   +2.01200256e+00},\n\t{   +1.10000000e+03,   +1.90006687e+00},\n\t{   +1.20000000e+03,   +1.85266235e+00},\n")
        f.write("\t{   +1.30000000e+03,   +1.71530858e+00},\n\t{   +1.40000000e+03,   +1.68416473e+00},\n\t{   +1.50000000e+03,   +1.51967239e+00},\n")
        f.write("\t{   +1.60000000e+03,   +1.43081691e+00},\n\t{   +1.70000000e+03,   +1.33476925e+00},\n\t{   +1.80000000e+03,   +1.31846256e+00},\n")
        f.write("\t{   +1.90000000e+03,   +1.24733020e+00},\n\t{   +2.00000000e+03,   +1.22382339e+00},\n\t{   +2.10000000e+03,   +1.16483602e+00},\n")
        f.write("\t{   +2.20000000e+03,   +1.11414123e+00},\n\t{   +2.30000000e+03,   +1.08501019e+00},\n\t{   +2.40000000e+03,   +1.05723081e+00},\n")
        f.write("\t{   +2.50000000e+03,   +1.02192225e+00},\n\t{   +2.60000000e+03,   +9.43035359e-01},\n\t{   +2.70000000e+03,   +9.35017351e-01},\n")
        f.write("\t{   +2.80000000e+03,   +8.47482268e-01},\n\t{   +2.90000000e+03,   +7.78730450e-01},\n\t{   +3.00000000e+03,   +7.37994929e-01},\n")
        f.write("\t{   +3.10000000e+03,   +6.52325390e-01},\n\t{   +3.20000000e+03,   +6.05620013e-01},\n\t{   +3.30000000e+03,   +5.40267450e-01},\n")
        f.write("\t{   +3.40000000e+03,   +4.93186909e-01},\n\t{   +3.50000000e+03,   +4.45730563e-01},\n\t{   +3.60000000e+03,   +4.02213140e-01},\n")
        f.write("\t{   +3.70000000e+03,   +3.59298790e-01},\n\t{   +3.80000000e+03,   +3.08920642e-01},\n\t{   +3.90000000e+03,   +2.83930213e-01},\n")
        f.write("\t{   +4.00000000e+03,   +2.43306423e-01},\n};\n\n%}\n\n")
        
        f.write("INITIALIZE\n%{\n\np_in=8.25311333051e14/(mcget_ncount()); //Neutron Intensity (n/s) at 5MW operation\n\n")
        f.write("dv = v_dist[1][0]-v_dist[0][0];\n\n%}\n\nTRACE\n%{\n double v;\n int bin;\n\n double weighter,ysloper;\n\n")
        f.write(" t=0;\n z=2.68;\n\n x = 0.8 * (rand01() - 0.5);\n y = 0.6 * (rand01() - 0.5)-0.1;\n p = p_in;\n v = rand01();\n")
        f.write(" v*= v_dist[40][0]-v_dist[0][0];\n bin = (int) v/dv;\n\n ysloper = v_dist[bin][1];\n")
        f.write(" weighter = (v_dist[bin+1][1]-ysloper)/dv*(v-v_dist[bin][0])+ysloper;\n p*=weighter;\n\n rn = (rand01()-0.5)*0.08;\n")
        f.write(" vy= (rn+0.40407893*y+0.10528407)*v;\t//Vertical Phase Space Correlation\n rn = (rand01()-0.5)*0.09;\n")
        f.write(" vx = (rn+0.39032*x+0.000339573)*v;\t//Horizontal Phase Space Correlation\n\n vz=v*v-vx*vx-vy*vy;\n")
        f.write(" if (vz<0) vz=0;\t\t\t\n else vz = sqrt(vz);\t\t\t\n%}\n\nMCDISPLAY\n%{\n%}\n\nEND")
        f.close()
