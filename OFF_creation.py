# OFF contains
# - ypsilon() - should not be used
# - K_n() - should not be used
# - b_gen()
# - createShortHalfAxisArray()
# - ellipseResolution()
# - hyperbolaResolution()
# - createDoublePlanarNestedOFFwArray()
# - createSimpleDoublePlanarNestedOFFwArray()
# - createMonoPlanarNestedOFFwArray()
# - maskProcessing()
# - createToroidalNestedOFFwArray()
# - createNestedOFFwParam()

#-------------------------
# 
import numpy as np
import os
import copy as cp

#-------------------------
#
from tools import *


#---------------------------------------------------------------------------------------------------
# 
def ypsilon(z_pos, focal, b_halfaxis, bVerbose = True):
    '''
        Intermediary for b_gen()
        
        Requirements :
            - numpy.sqrt()
            - numpy.float128
    '''
    y = (1-z_pos**2/(focal**2+b_halfaxis**2))*b_halfaxis**2
    if (y > 0):
        y = np.sqrt(y, dtype = np.float128)
    else:
        if bVerbose:
            print('ypsilon() : warning y <= 0\ny set to 0')
        y = 0
    return y

def K_n(z_s, z_e, b_0, focal_l, n=1, bVerbose = True):
    '''
        Intermediary for b_gen()
    '''
    z_e_ellipse = z_e - focal_l 
    K_n = z_s**n/z_e**n*ypsilon(z_e_ellipse, focal_l, b_0, bVerbose = bVerbose)
    return K_n 

def b_gen(z_start, l, b_1, focal, n, bVerbose = True):
    '''
        Generation of *n*th correct short half axis. Based on O. Zimmer and R. Wagner algorithm.
    
        Requirements :
            - numpy.sqrt()
            - numpy.float128
    '''
    k_n = K_n(z_start, z_start+l, b_1, focal, n+1, bVerbose = bVerbose)
    C_1 =  focal**2 - (z_start- focal)**2 - k_n**2
    C_2 =  k_n**2 * focal**2 
    
    C_3 = C_1**2/4+C_2
    if C_3 > 0:
        B = np.sqrt(C_3, dtype = np.float128)-C_1/2
        if B >= 0:
            b = np.sqrt(B, dtype = np.float128)
        else:
            if bVerbose:
                print('b_gen() : warning B < 0\nb set to 0')
            b = 0
    else:
        if bVerbose:
            print('b_gen() : warning C_3 <= 0\nb set to 0') 
        b = 0
    
    return b


#---------------------------------------------------------------------------------------------------
# 
def createShortHalfAxisArray(z_start = 10, l = 10, L = 200, b_0 = (), b_1 = 1, nb_levels = 1, opticHalfWidth = 2,
                             bProtectSpacing = True, minSpacing = 0.01, bAdd = True,
                             bGetParamOnly = False, bVerbose = False, bWarnLimit = True,
                            ):
    
    '''
        Return an array of short half axis, starting with b_0, extended with short half axis generated starting from b_1 and going toward the inside.
        
        Input :
            - z_start : (float) starting point of arc relative to focus
            - l : (float) length of elliptical arc 
            - L : (float) ellipse focal distance * 2
            - b_0 : (tupple) tupple of short half axis
            - b_1 : (float) outer short half axis
            if b_1 = 'limit', b_1 will be set to biggest elliptical arc fitting into opticHalfWidth
            - nb_levels : (int) size of array output
            - opticHalfWidth : (float)
            if opticHalfWidth = 0, opticHalfWidth will adapat to fit biggest elliptic arc. In this case b_1 must not be set to 'limit'
            - bProtectSpacing : (bool) if True, consecutives levels can't be closer than *minSpacing* meter, such levels will be deleted.
            - minSpacing : (float) minimum space between too consecutives levels
            - bAdd : (bool) if bProtectSpacing, after deleting too packed levels, add levels spaced of *minSpacing* meter
            - bGetParamOnly : (bool) if True function only return *param* and do nothing except, if needed, adapting opticHalfWidth
            - bVerbose : (bool) activate display of messages
            - bWarnLimit : (bool) activate indicator (message) that maximum short axis is greater that strutural limits resulting in partial or non-existent levels
            *bVerbose* superseed this boolean
        
        Output :
            - b_array : (numpy.array) contains, as much as possible, concatenation of b_0, b_1 and generic generated b.
            - param : (dict) input values
            
        Example :
            - createShortHalfAxisArray(z_start = 10, l = 10, L = 200, b_0 = (3, 2.1), b_1 = 1.5, nb_levels = 8, opticHalfWidth = 2, bProtectSpacing = True, minSpacing = 1E-2)
            - createShortHalfAxisArray(z_start = 10, l = 10, L = 200, b_1 = 1, nb_levels = 10, opticHalfWidth = 2, bProtectSpacing = True, minSpacing = 0.01, bVerbose = True)
        
        Requirements :
            - getEllipseOrdinate()
            - getBiggestArc()
            - b_gen()
            - getShortHalfAxis()
            - numpy.zeros()
            - numpy.float128
            - numpy.append()
            - numpy.delete()
            - numpy.sqrt()
            - numpy.array()
    '''

    funcEllipseOrdi = getEllipseOrdinate
    funcArc = getBiggestArc
    funcBGen = b_gen
    funcHalfAxis = getShortHalfAxis
    
    param = {
        "z_start" : z_start, "l" : l, "L" : L, "b_0" : b_0, "b_1" : b_1, "nb_levels" : nb_levels,
        "opticHalfWidth" : opticHalfWidth,
        "bProtectSpacing" : bProtectSpacing, "minSpacing" : minSpacing, "bAdd" : bAdd,
        "bGetParamOnly" : bGetParamOnly, "bVerbose" : bVerbose, "bWarnLimit" : bWarnLimit,
    }
    
    
    if opticHalfWidth == 0: # Avoid display problems
        if b_1 != 'limit':
            if len(b_0) > 0 :
                b_max = max(max(b_0),b_1)
            else :
                b_max = b_1
        
            opticHalfWidth = funcEllipseOrdi(z_0 = z_start+l, b = b_max, L = L, bVerbose = bVerbose)
            param['opticHalfWidth'] = opticHalfWidth
    
    # To be used with clearDic() for intelligent calling of function
    if bGetParamOnly:
        return param
    
    # Calculate the biggest optics that can fit in the space defined by opticHalfWidth
    b_l = funcArc(l = l, z_start = z_start, L = L, tunnelHalfWidth = opticHalfWidth, bVerbose = bVerbose)
    
    # If one set b_1 at 'limit', automatically set b_1 to b_limit
    if (b_1 == 'limit'):
        b_1 = b_l
    
    # Find the maximum in all value given (b_0 is a tupple)
    if len(b_0) > 0 :
        b_max = max(max(b_0),b_1)
    else :
        b_max = b_1
    
    # Create the resulting array
    b_array = np.zeros(nb_levels, dtype = np.float128)
    
    # Indicate if one, or more, ellipse will be cut
    if bVerbose and bWarnLimit and (b_max > b_l):
        print("Warning : b_max({}) is higher than the limit b_limit({}).".format(b_max, b_l))
    
    
    # Begin the b_array with the values in b_0
    sizeB0 = len(b_0)
    
    if (sizeB0 >= nb_levels): # In case that the number of levels is lower than the dimension of b_0
        b_array[:nb_levels] = np.array(b_0[:nb_levels])
    else :
        b_array[:sizeB0] = np.array(b_0)
        b_array[sizeB0] = b_1
        
        # Create the remaining generic levels
        for n in range(nb_levels-sizeB0-1):
            b_array[n + sizeB0 + 1] = funcBGen(z_start, l, b_1, L/2, n, bVerbose)
    
    if bProtectSpacing:
        
        ## Removing levels
        cptDel = 0
        a_array = np.sqrt( (L / 2) ** 2 + b_array ** 2, dtype = np.float128)
        y_pos = np.array([])
        
        
        newNbLevels = param['nb_levels']
        i = 0
        while i != (len(b_array)): # delete the too close levels
            a = a_array[i]
            b = b_array[i]
            
            z_pos = z_start
            y_pos = np.append(y_pos,np.sqrt((1 - (z_pos - L / 2) **2 / (a ** 2))) * b)
            
            if (i > 0) and len(b_array) > 1 and abs(y_pos [i-1] - y_pos[i]) < minSpacing:
                b_array = np.delete(b_array,i)
                y_pos = np.delete(y_pos,i)
                newNbLevels -= 1
                i -= 1
                if cptDel == 0 and bVerbose: # Less messages
                    print("Warning : distance between consecutives levels was less than %0.1em. Faulty levels will be deleted."%minSpacing, end=' ')
                cptDel += 1
            i+=1
        iEnd = len(y_pos)-1
        
        if abs(y_pos[iEnd]) < minSpacing/2: # In case the one in the center is at a correct distance from neighbour but not of himself
            b_array = np.delete(b_array,iEnd)
            y_pos = np.delete(y_pos,iEnd)
            newNbLevels -= 1
            cptDel += 1
            
        if cptDel > 0 and bVerbose:
            print('{} levels have been deleted.'.format(cptDel))
        
        ## Adding levels
        if bAdd :
            diffLvl = nb_levels - newNbLevels
            if diffLvl > 0: # if we deleted levels, we replaced them by minSpacing-spaced levels if we can
                cptAdd = 0
                cptAddBis = 0
                for i in range(diffLvl):
                    y_minus = y_pos[-1] - minSpacing
                    if y_minus > 0:
                        b = funcHalfAxis(L/2-z_pos,y_minus)
                        b_array = np.append(b_array,b)
                        y_pos = np.append(y_pos, y_minus)
                        newNbLevels += 1
                        if cptAdd == 0 and bVerbose:
                            print('Constructing new levels with %0.1em spacing.'%minSpacing, end ='')
                            cptAddBis += 1
                        cptAdd += 1

                iEnd = len(y_pos)-1

                if abs(y_pos[iEnd]) < minSpacing/2:  # In case the one in the center is at a correct distance from neighbour but not of himself
                    b_array = np.delete(b_array,iEnd)
                    y_pos = np.delete(y_pos,iEnd)
                    newNbLevels -= 1
                    cptAdd -= 1

                if cptAdd > 0 and bVerbose:
                    print(' {} levels have been added.'.format(cptAdd))
                else :
                    if cptAddBis > 0 and bVerbose:
                        print(' {} levels have been added.'.format(cptAdd))
            
#             param['nb_levels'] = newNbLevels 
    return (np.array(b_array), param)


#---------------------------------------------------------------------------------------------------
#
def ellipseResolution(b_array, L, z_start, nb_segments, l, opticHalfWidth):
    '''
        Intermediary of create***NestedOFFwArray()
        
        Requirements :
            - numpy.flip()
            - numpy.sort()
            - numpy.sqrt()
            - numpy.vstack()
            - numpy.tile()
            - numpy.array()
            - numpy.arange()
            - numpy.ones()
            - numpy.float128
    
    '''
    b_array = np.flip(np.sort(b_array))
    
    # Management of ellipse properties
    b_array = b_array.reshape(-1, 1)
    b2_array = b_array**2 ; b2_array
    
    a2_array = (L/2)**2 + b_array**2 
    
    # Generation of evenly spaced circle in order to create the ellipse
    z_pos = z_start + np.arange(nb_segments+1, dtype = np.float128)* l/nb_segments
    
    zP = np.array([L/2 - z_pos for _ in range(len(b_array))], dtype = np.float128)
    zP2 = zP**2
    
    ## Resizing array for array composition
    a2_array = np.tile(a2_array, (1,len(z_pos)))
    
    # Finding the radius of every circle
    y2_array = (1-zP**2/a2_array) * b2_array
    y_array = np.sqrt(y2_array, dtype = np.float128)
    arrOW = opticHalfWidth * np.ones(len(y_array[0]), dtype = np.float128)
    y_array = np.vstack((arrOW, y_array))
    
    return z_pos, y_array    


def hyperbolaResolution(b_array, a_h, z_start, nb_segments, l, opticHalfWidth, bToro = False):
    '''
        Intermediary of create***NestedOFFwArray()
        
        Requirements :
            - numpy.flip()
            - numpy.sort()
            - numpy.sqrt()
            - numpy.vstack()
            - numpy.tile()
            - numpy.array()
            - numpy.arange()
            - numpy.ones()
            - numpy.float128
    
    '''
    b_array = np.flip(np.sort(b_array))

    c_array = np.sqrt(a_h**2 + b_array**2)
    #print(c_array)
    # Management of ellipse properties
    b_array = b_array.reshape(-1, 1)
    b2_array = b_array**2 ; b2_array
    
    a2_array = a_h**2 
    
    # Generation of evenly spaced circle in order to create the ellipse
    z_pos = z_start + np.arange(nb_segments+1, dtype = np.float128)* l/nb_segments
    #print(z_pos)

    zP = np.array([z_pos+c_array[idx] for idx in range(len(b_array))], dtype = np.float128)
    #zP = np.array([z_pos for idx in range(len(b_array))], dtype = np.float128)
    zP2 = zP**2
    #print(zP)
    ## Resizing array for array composition
    a2_array = np.tile(a2_array, (1,len(z_pos)))
    
    # Finding the radius of every circle
    #y2_array = (1-zP**2/a2_array) * b2_array

    y2_array = b2_array*((zP)**2/a2_array-1)
    #print(y2_array)
    
  
    #y2_array = b_halfaxis**2*((z_pos-z_center)**2/a2_array-1)

    y_array = np.sqrt(y2_array, dtype = np.float128)

    if bToro == False:
        arrOW = opticHalfWidth * np.ones(len(y_array[0]), dtype = np.float128)
        y_array = np.vstack((arrOW, y_array))
    
    return z_pos, y_array    


#---------------------------------------------------------------------------------------------------
#
def createDoublePlanarNestedOFFwArray(L = 200, b_array = np.array([1.]),
                                      z_start = 10,
                                      l = 10,
                                      nb_segments = 15,
                                      alphaRad = 0, axis = 'x', RC = np.matrix([0,-.28,0], dtype = np.float128),
                                      T = np.matrix([0,0,0], dtype = np.float128),
                                    
                                      filename='defaultFile.off', opticHalfWidth = 0,
                                      bBoundingBox = False,
                                      bWolter = False,
                                      bGetParamOnly = False, bVerbose = True,
                                     ):
    '''
        Write an OFF file of a double-planar nested optic based on a given array of short half axis.
        Optic can be rotated and translated.
        
        If opticHalfWidth != 0 and, resulting this, there are faces coinciding (located at opticHalfWidth -x or y), useless faces will be removed.
        So when rotating or translating the optic, algorithm may not suppress coincided faces.
        
        Input :
            - L : (float) ellipse focal distance * 2
            - b_array : (array) array of short half axis
            - z_start : (float) starting point of arc relative to focus
            - l : (float) length of elliptical arc 
            - nb_segments : (int)
            - alphaRad : (float)(array)(rad)
            see rotMatrix for more information
            - axis : (str)
            see rotMatrix for more information
            - RC : (numpy.matrix) rotation center. Can be row or column vector, will be correctly transposed to correct form
            - T : (numpy.matrix) translation vector. Can be row or column vector, will be correctly transposed to correct form
            - filename : (str) output for OFF file
            - opticHalfWidth : (float)
            if opticHalfWidth = 0, opticHalfWidth will adapat to fit biggest elliptic arc. In this case b_1 must not be set to 'limit'
            - bBoundingBox : (bool) generate a bounding box
            - bWolter : (bool) generate hyperbolic instead of elliptic optic (used for Wolter optics)
            - bGetParamOnly : (bool) if True function do nothing and only return *param*
            - bVerbose : (bool) display of messages
        
        Output :
            - param : (dict) input values
            
        Example :
            - createDoublePlanarNestedOFFwArray(L = 200, b_array = np.array([2.,1.,.5]), z_start = 10, l = 10, nb_segments = 15, filename='exampleFile.off', opticHalfWidth = 0, bBoundingBox = False, alphaRad = np.pi/26, axis = 'x', T = np.matrix([0,0,1]))
            - createDoublePlanarNestedOFFwArray(L = 200, b_array = np.array([2.,1.,.5]), z_start = 10, l = 10, nb_segments = 15, filename='exampleFile.off', opticHalfWidth = 0, bBoundingBox = False, alphaRad = [np.pi/26, np.pi/16], axis = 'xy', RC = '')
            - createDoublePlanarNestedOFFwArray(L = 200, b_array = np.array([2.,1.,.5]), z_start = 10, l = 10, nb_segments = 15, filename='exampleFile.off', opticHalfWidth = 0, bBoundingBox = False)
            - createDoublePlanarNestedOFFwArray(L = 200, b_array = createShortHalfAxisArray(z_start = 16.5, l = 10, L = 200, b_0 = (), b_1 = 'limit', nb_levels = 200, opticHalfWidth = 2, minSpacing = 1E-2, bProtectSpacing = True, bVerbose = True)[0], z_start = 16.5, l = 10, nb_segments = 15, filename='exampleFile.off', opticHalfWidth = 2, bBoundingBox = False, bVerbose = True)
        
        Requirements :
            - getEllipseOrdinate()
            - rotMatrix()
            - ellipseResolution()
            - hyperbolaResolution()
            - numpy.around()
            - numpy.logical_and()
            - numpy.shape()
            - numpy.zeros()
            - numpy.ones()
            - numpy.array()
            - numpy.tile()
            - numpy.arange()
            - numpy.min()
            - numpy.max()
            - numpy.vstack()
            - numpy.matrix()
            - numpy.float128
            - copy.deepcopy()    
    '''
    
    funcEllipseOrdi = getEllipseOrdinate
    
    if type(RC) is not str and np.shape(RC) != (3,1):
        RC = RC.T
        
    if np.shape(T) != (3,1):
        T = T.T
    
    param = {
        "L" : L, "l" : l, "z_start" : z_start, "opticHalfWidth" : opticHalfWidth,
        "b_array" : b_array, "nb_segments" : nb_segments,
        "alphaRad" : alphaRad, "axis" : axis, "RC" : RC,
        "T" : T,
        
        "bBoundingBox" : bBoundingBox, "filename" : filename,
        "bGetParamOnly" : bGetParamOnly, "bVerbose" : bVerbose,
    }
    
    bAdaptative = False
    if (opticHalfWidth == 0):
        bAdaptative = True
        opticHalfWidth = funcEllipseOrdi(z_0 = z_start+l, b = np.max(b_array), L = L, bVerbose = bVerbose)
        param['opticHalfWidth'] = opticHalfWidth
    # To be used with clearDic() for intelligent calling of function
    if bGetParamOnly:
        return param
    
    ##------------------------------------------------------------------------------------------------------------------------
    # Ellipse calculations
    #z_pos, y_array = ellipseResolution(b_array, L, z_start, nb_segments, l, opticHalfWidth)
    
    if bWolter == False:
        z_pos, y_array = ellipseResolution(b_array, L, z_start, nb_segments, l, opticHalfWidth) 
    else:
        #For Wolter Option L == a_h == short hyperbola half_axis
        z_pos, y_array = hyperbolaResolution(b_array, L, z_start, nb_segments, l, opticHalfWidth)
    ##------------------------------------------------------------------------------------------------------------------------
    # Points creation
    YY = np.vstack((y_array, np.flipud(-y_array)))
    
    X = np.array([])
    Y = np.array([])

    for i in range(len(YY[0])):
        YYY = np.tile(np.array([YY[:,i]]).transpose(), (1, len(YY)))
        XXX = YYY.transpose()
        if len(X) == 0:
            X = [XXX]
            Y = [YYY]
        else:
            X = np.vstack((X, [XXX]))
            Y = np.vstack((Y, [YYY]))
    
    Z = np.ones(np.shape(X))
    for i in range(len(X)):
        Z[i] *= z_pos[i]
        
    IND = np.arange(np.shape(Z)[0]*np.shape(Z)[1]*np.shape(Z)[2]).reshape(np.shape(Z)) # Corresponding indices
    
    
    ##------------------------------------------------------------------------------------------------------------------------
    # Rotation
    if type(alphaRad) is list:
        alphaRad = np.array(alphaRad) # for using abs()
        
    if np.max(abs(alphaRad)) != 0:
        # If wanted, setting rotation center to optic center
        if type(RC) is str:
            Va = []
            for a in range(len(X)):
                for l in range(len(X[a])):
                    for c in range(len(X[a][l])):
                        Va += [np.matrix([X[a,l,c], Y[a,l,c], Z[a,l,c]]).T] #Every coordinates of vertices in a matrix
            RC = np.sum(Va, axis = 0) / len(Va) # center of optic

        # Applying rotation
        R = rotMatrix(t = alphaRad, axis = axis)
        
        for a in range(len(X)):
            for l in range(len(X[a])):
                for c in range(len(X[a][l])):
                    V = np.matrix([X[a,l,c], Y[a,l,c], Z[a,l,c]]).T
                    
                    Vr = R*(V-RC)+RC # Move to rotation point, rotate, then cancel first translation
                    Vrt = Vr + T # Translation

                    X[a,l,c] = Vrt[0]
                    Y[a,l,c] = Vrt[1]
                    Z[a,l,c] = Vrt[2]
    
    elif (len(T) == 3) and not (np.min(T == np.matrix([0,0,0], dtype = np.float128).T)): # Only do translation
        for a in range(len(X)):
            for l in range(len(X[a])):
                for c in range(len(X[a][l])):
                    V = np.matrix([X[a,l,c], Y[a,l,c], Z[a,l,c]]).T
                    Vt = V + T # Translation

                    X[a,l,c] = Vt[0]
                    Y[a,l,c] = Vt[1]
                    Z[a,l,c] = Vt[2]
                    
                    
    ##------------------------------------------------------------------------------------------------------------------------
    # Width limit
    if bAdaptative:
        hoWXup = np.max(X)
        hoWXlow = np.min(X)
    
        hoWYup = np.max(Y)
        hoWYlow = np.min(Y)
    else:
        hoWXup = opticHalfWidth
        hoWXlow = -opticHalfWidth
    
        hoWYup = opticHalfWidth
        hoWYlow = -opticHalfWidth

        
    X[X > hoWXup] = hoWXup
    X[X < hoWXlow] = hoWXlow
    
    Y[Y > hoWYup] = hoWYup
    Y[Y < hoWYlow] = hoWYlow
    
#     Z[Z > z_start + l] = z_start + l
#     Z[Z < z_start] = z_start
    
    
    ##------------------------------------------------------------------------------------------------------------------------
    # Faces deletion
    MaskX = (abs(np.around(X, 8)) == round(opticHalfWidth, 8)) # avoid precision mistakes
    MaskY = (abs(np.around(Y, 8)) == round(opticHalfWidth, 8))
    
    MaskEtX = np.logical_and(MaskX[:-1], MaskX[1:])
    MaskEtY = np.logical_and(MaskY[:-1], MaskY[1:])
    
    if not bBoundingBox:
        MaskEtEtY = np.zeros(np.shape(MaskEtY), bool)
        for a in range(len(MaskEtY)):
            mid = int(len(MaskEtY[a])/2)
            MaskEtEtY[a,:mid] = np.logical_and(MaskEtY[a,1:mid+1], MaskEtY[a,:mid])
            MaskEtEtY[a,mid:] = np.logical_and(MaskEtY[a,mid:], MaskEtY[a,mid-1:-1])
            
            if np.min(MaskEtEtY[a]) == True: # If every faces are equal or greater than oW in the circle, enable to draw the "smallest" level
                MaskEtEtY[a,mid-1] = np.zeros(len(MaskEtEtY[a,mid-1]), bool)
                MaskEtEtY[a,mid] = np.zeros(len(MaskEtEtY[a,mid]), bool)
        
        MaskEtY = cp.deepcopy(MaskEtEtY)
                
        MaskEtEtX = np.zeros(np.shape(MaskEtX), bool)
        for a in range(len(MaskEtX)):
            mid = int(len(MaskEtX[a][0])/2)
            MaskEtEtX[a,:,:mid] = np.logical_and(MaskEtX[a,:,1:mid+1], MaskEtX[a,:,:mid]);
            MaskEtEtX[a,:,mid:] = np.logical_and(MaskEtX[a,:,mid:], MaskEtX[a,:,mid-1:-1])
            
            if np.min(MaskEtEtX[a,:]) == True:
                MaskEtEtX[a,:,mid-1] = np.zeros(len(MaskEtEtX[a,:,mid-1]), bool)
                MaskEtEtX[a,:,mid] = np.zeros(len(MaskEtEtX[a,:,mid]), bool)
        
        MaskEtX = cp.deepcopy(MaskEtEtX)
    
    ##------------------------------------------------------------------------------------------------------------------------
    # OFF Creation
    nbL = int(np.shape(YY)[0]/2)
    facesH = np.zeros((2*(nbL-1)*(2*nbL-1))*nb_segments, object)
    facesV = np.zeros((2*(nbL-1)*(2*nbL-1))*nb_segments, object)
    
    facesBB = np.zeros((4*(2*nbL-1))*nb_segments, object)
    
    
    cptFacesH = 0
    for a in range(len(X)-1):
        for l in range(1,len(X[a])-1):
            for c in range(len(X[a][l])-1):
                ## Horizontal
                if not(MaskEtX[a,l,c] and MaskEtX[a,l,c+1]) and not(MaskEtY[a,l,c] and MaskEtY[a,l,c+1]):
                    facesH[cptFacesH] = '4 {} {} {} {}\n'.format(IND[a,l,c], IND[a,l,c+1], IND[a+1,l,c+1], IND[a+1,l,c])
                    cptFacesH += 1
    
    cptFacesV = 0
    for a in range(len(X)-1):
        for l in range(len(X[a])-1):
            for c in range(1,len(X[a][l])-1):
                ## Vertical
                if not(MaskEtX[a,l,c] and MaskEtX[a,l+1,c]) and not(MaskEtY[a,l,c] and MaskEtY[a,l+1,c]):
                    facesV[cptFacesV] = '4 {} {} {} {}\n'.format(IND[a,l,c], IND[a,l+1,c], IND[a+1,l+1,c], IND[a+1,l,c])
                    cptFacesV += 1
    
    cptFacesBB = 0
    if bBoundingBox:
        for a in range(len(X)-1):
            for l in range(len(X[a])-1):
                for c in [0, len(X[a])-1]:
                    facesBB[cptFacesBB] = '4 {} {} {} {}\n'.format(IND[a,l,c], IND[a,l+1,c], IND[a+1,l+1,c], IND[a+1,l,c])
                    cptFacesBB += 1
        for a in range(len(X)-1):
            for l in [0, len(X[a])-1]:
                for c in range(len(X[a][l])-1):
                    facesBB[cptFacesBB] = '4 {} {} {} {}\n'.format(IND[a,l,c], IND[a,l,c+1], IND[a+1,l,c+1], IND[a+1,l,c])
                    cptFacesBB += 1

    f = open(filename,'w')
    # Write OFF-File Header
    f.write("OFF\n")
    f.write("# Double planar nested optic \n")
    
    f.write("# Number of vertex, number of faces, number of edges \n")

    nbFaces = cptFacesH+cptFacesV
    if bBoundingBox:
        nbFaces += cptFacesBB
    f.write("%d %d 0\n"%(np.shape(X)[0]*np.shape(X)[1]*np.shape(X)[2], nbFaces)) # Write the number of vertices (=corners) and faces ()


    for a in range(len(X)):
        for l in range(len(X[a])):
            for c in range(len(X[a][l])):
                f.write("%0.9f %0.9f %0.9f\n"%(X[a,l,c], Y[a,l,c], Z[a,l,c]-z_start))
        
    for i in range(cptFacesH):
        f.write(facesH[i])
    for i in range(cptFacesV):
        f.write(facesV[i])
        
    for i in range(cptFacesBB):
        f.write(facesBB[i])
        
    f.close()
    return(param)


#---------------------------------------------------------------------------------------------------
#
def createSimpleDoublePlanarNestedOFFwArray(L = 200,
                                            b_array = np.array([1.]),
                                            z_start = 10,
                                            l = 10,
                                            nb_segments = 15,
                                            opticHalfWidth = 0,
                                            filename='defaultFile.off',
                                            bBoundingBox = False,
                                            bWolter = False,
                                            fOffsetOrigin = 0,
                                            bGetParamOnly = False, bVerbose = True,
                                           ):
    '''
        Write an OFF file of a double-planar nested optic based on a given array of short half axis.
        This version doesn't define crosing points and doesn't support rotation.
        
        Input :
            - L : (float) ellipse focal distance * 2
            - b_array : (numpy.array) array of short half axis
            - z_start : (float) starting point of arc relative to focus
            - l : (float) length of elliptical arc 
            - nb_segments : (int)
            - filename : (str)
            - opticHalfWidth : (float)
            if opticHalfWidth = 0, opticHalfWidth will adapat to fit biggest elliptic arc. In this case b_1 must not be set to 'limit'
            - bBoundingBox : (bool)
            - bWolter : (bool) generate hyperbolic instead of elliptic optic (used for Wolter optics)
            - bGetParamOnly : (bool) if True function do nothing and only return *param*
            - bVerbose : (str) activate display of messages
        
        Output :
            - param : (dict) input values
            
        Example :
            - createSimpleDoublePlanarNestedOFFwArray(L = 200, b_array = np.array([2.,1.,.5]), z_start = 10, l = 10, nb_segments = 15, filename='exampleFile.off', opticHalfWidth = 0, bBoundingBox = False)
        
        Requirements :
            - getEllipseOrdinate()
            - numpy.ndenumerate()
            - numpy.array()
            - numpy.sqrt()
            - numpy.append()
    '''
    funcEllipseOrdi = getEllipseOrdinate
    
    param = {
        "L" : L, "l" : l, "z_start" : z_start, "opticHalfWidth" : opticHalfWidth,
        "b_array" : b_array, "nb_segments" : nb_segments,
        "bBoundingBox" : bBoundingBox, "filename" : filename,
        "bGetParamOnly" : bGetParamOnly, "bVerbose" : bVerbose,
    }
    
    if (opticHalfWidth == 0): 
        opticHalfWidth = funcEllipseOrdi(z_0 = z_start+l, b = np.max(b_array), L = L, bVerbose = bVerbose)
        param['opticHalfWidth'] = opticHalfWidth
    
    # To be used with clearDic() for intelligent calling of function
    if bGetParamOnly:
        return param
    
    # Init. routines
    # Calculation of long half axis from foci distance and short half axis
    a_array = np.sqrt( (L / 2) ** 2 + b_array ** 2)
    
    ## Numbers of vertices and faces for OFF File
    nb_mirrors = len(b_array)
    nb_vertex = 4 * nb_mirrors * (nb_segments + 1) * 2
    nb_faces = 2 * nb_mirrors * nb_segments * 2
    #Correction for bounding Box
    if bBoundingBox == True:
        nb_vertex += 8
        nb_faces += 2 + 2
    
    # Open OFF-File 
    f = open(filename,'w')
    # Write OFF-File Header
    f.write("OFF\n")
    f.write("%d %d 0\n"%(nb_vertex, nb_faces)) # Write the number of vertices (=corners) and faces ()
    
    #Check if bounding box should be written
    if bBoundingBox == True:
        z_end = z_start + l
        # Vertices of Bounding Box: X,Y,Z in meters
        f.write('%0.9f %0.9f %0.9f\n'%( opticHalfWidth, -opticHalfWidth, 0 + fOffsetOrigin)) # 0
        f.write('%0.9f %0.9f %0.9f\n'%( opticHalfWidth, -opticHalfWidth, z_end - z_start + fOffsetOrigin))   # 1
        f.write('%0.9f %0.9f %0.9f\n'%(-opticHalfWidth, -opticHalfWidth, z_end - z_start + fOffsetOrigin))   # 2
        f.write('%0.9f %0.9f %0.9f\n'%(-opticHalfWidth, -opticHalfWidth, 0 + fOffsetOrigin)) # 3
        f.write('%0.9f %0.9f %0.9f\n'%( opticHalfWidth,  opticHalfWidth, 0 + fOffsetOrigin)) # 4
        f.write('%0.9f %0.9f %0.9f\n'%( opticHalfWidth,  opticHalfWidth, z_end - z_start + fOffsetOrigin))   # 5
        f.write('%0.9f %0.9f %0.9f\n'%(-opticHalfWidth,  opticHalfWidth, z_end - z_start + fOffsetOrigin))   # 6
        f.write('%0.9f %0.9f %0.9f\n'%(-opticHalfWidth,  opticHalfWidth, 0 + fOffsetOrigin)) # 7
    
    
    # calculation of Vertices of the Mirrors
    #1st loop over nested mirrors layers
    aYPosV = []; aZPosV = []
    + fOffsetOrigin 
    cptYPos = 0
    for idx, b in np.ndenumerate(b_array): 
        a = a_array[idx]
        idx = idx[0]
        aiYPosV = np.array([]); aiZPosV = np.array([])
        # 2nd loop over mirror segments

        for i in range(nb_segments + 1):       
            #calculation of ellipse points at mirror position
            z_pos = z_start + i * (l / nb_segments)
            
            if bWolter == False:
                y_pos = (1 - (z_pos - L / 2) **2 / (a ** 2)) * (b ** 2)
            else:
                c = np.sqrt(L**2 + b**2)                
                y_pos = b**2*((z_pos+c)**2/L**2-1)    
                #print(c)
                #print(z_pos, y_pos)

    
        
        # Management of ellipse properties
        #b_array = b_array.reshape(-1, 1)
        #b2_array = b_array**2 ; b2_array
        
        #a2_array = a_h**2 
        
        # Generation of evenly spaced circle in order to create the ellipse
        #z_pos = z_start + np.arange(nb_segments+1, dtype = np.float128)* l/nb_segments
        #print(z_pos)

        #zP = np.array([z_pos+c_array[idx] for idx in range(len(b_array))], dtype = np.float128)
        #zP = np.array([z_pos for idx in range(len(b_array))], dtype = np.float128)
        #zP2 = zP**2
        #print(zP)
        ## Resizing array for array composition
        #a2_array = np.tile(a2_array, (1,len(z_pos)))
        
        # Finding the radius of every circle
        #y2_array = (1-zP**2/a2_array) * b2_array

        #y2_array = b2_array*((zP)**2/a2_array-1)
        #print(y2_array)
        
      
        #y2_array = b_halfaxis**2*((z_pos-z_center)**2/a2_array-1)

        #y_array = np.sqrt(y2_array, dtype = np.float128)

        #if bToro == False:
        #    arrOW = opticHalfWidth * np.ones(len(y_array[0]), dtype = np.float128)
        #    y_array = np.vstack((arrOW, y_array))
        
        #return z_pos, y_array  




            # make sure x is positive before taking the root
            if y_pos > 0:
                y_pos = np.sqrt(y_pos)
                if y_pos > opticHalfWidth:
                    y_pos = opticHalfWidth
            else:
                if cptYPos == 0 and bVerbose:
                    print('Warning ypos < 0: %0.9f\n'%(y_pos))
                cptYPos += 1

            aiYPosV = np.append(aiYPosV, y_pos); aiZPosV = np.append(aiZPosV, z_pos)
            #Write points
            #upper part of ellipse
            f.write('%0.9f %0.9f %0.9f\n'%( y_pos,  -opticHalfWidth, z_pos - z_start + fOffsetOrigin)) # 4*i +8
            f.write('%0.9f %0.9f %0.9f\n'%( y_pos,  opticHalfWidth, z_pos - z_start + fOffsetOrigin))   # 4*+1+8
            #lower part of ellipse
            f.write('%0.9f %0.9f %0.9f\n'%(-y_pos,  opticHalfWidth, z_pos - z_start + fOffsetOrigin))   # 4*+2+8
            f.write('%0.9f %0.9f %0.9f\n'%(-y_pos,  -opticHalfWidth, z_pos - z_start + fOffsetOrigin)) # 4*+3+8
        aYPosV += [aiYPosV]
        aZPosV += [aiZPosV]   
    
    aYPosV = np.array(aYPosV) ; aZPosV = np.array(aZPosV)
    
    ### 90° rotated optics
    aYPosH = []; aZPosH = []

    cptYpos = 0
    for idx, b in np.ndenumerate(b_array): 
        a = a_array[idx]
        idx = idx[0]
        aiYPosH = np.array([]); aiZPosH = np.array([])
        # 2nd loop over mirror segments 
        for i in range(nb_segments + 1):       
            #Calculate ellipse points at mirror position
            z_pos = z_start + i * (l / nb_segments)
            
            if bWolter == False:
                y_pos = (1 - (z_pos - L / 2) **2 / (a ** 2)) * (b ** 2)
            else:
                c = np.sqrt(L**2 + b**2)                
                y_pos = b**2*((z_pos+c)**2/L**2-1)  

            # make sure x is positive before taking the root

            if y_pos > 0:
                y_pos = np.sqrt(y_pos)
                if y_pos > opticHalfWidth:
                    y_pos = opticHalfWidth
            else:
                if cptYPos == 0 and bVerbose:
                    print('Warning ypos < 0: %0.9f\n'%(y_pos))
                cptYPos += 1
            
            aiYPosH = np.append(aiYPosH, y_pos); aiZPosH = np.append(aiZPosH, z_pos)
            
            #left part of ellipse
            f.write('%0.9f %0.9f %0.9f\n'%(-opticHalfWidth, -y_pos, z_pos - z_start + fOffsetOrigin)) # 4*+3+8
            f.write('%0.9f %0.9f %0.9f\n'%(opticHalfWidth, -y_pos, z_pos - z_start + fOffsetOrigin)) # 4*i +8
            #right part of ellipse
            f.write('%0.9f %0.9f %0.9f\n'%(opticHalfWidth, y_pos, z_pos - z_start + fOffsetOrigin))   # 4*+1+8
            f.write('%0.9f %0.9f %0.9f\n'%(-opticHalfWidth, y_pos, z_pos - z_start + fOffsetOrigin))   # 4*+2+8
        aYPosH += [aiYPosH]
        aZPosH += [aiZPosH]

    aYPosH = np.array(aYPosH) ; aZPosH = np.array(aZPosH)       
    
    
    # Write Faces (= indexes of vertices that constitute each face)
    index = 0  #index for bookkeeping
    if bBoundingBox == True:
        f.write('4 %d %d %d %d\n'%(0,1,2,3))  #/* bottom */
        f.write('4 %d %d %d %d\n'%(4,5,6,7))  #/* top */
        f.write('4 %d %d %d %d\n'%(0,1,5,4))  #/* left */
        f.write('4 %d %d %d %d\n'%(3,2,6,7))  #/* right */
        index = 8
        
    for i in range(nb_mirrors):    #loop over nested mirrors
        for j in range(nb_segments):  #loop over mirror segments#
            f.write('4 %d %d %d %d\n'%(index,   index+1, index+4+1, index+4))   # face of upper ellipse
            f.write('4 %d %d %d %d\n'%(index+2, index+3, index+4+3, index+4+2)) # face of lower ellipse
            index += 4
            if (j == nb_segments-1):
                index += 4
    
    ### 90° rotated optics
    for i in range(nb_mirrors):    #loop over nested mirrors
        for j in range(nb_segments):  #loop over mirror segments#
            f.write('4 %d %d %d %d\n'%(index,   index+1, index+4+1, index+4))   # face of left ellipse
            f.write('4 %d %d %d %d\n'%(index+2, index+3, index+4+3, index+4+2)) # face of right ellipse
            index += 4
            if (j == nb_segments-1):
                index += 4         
    #Close OFF-File 
    f.close()
    
#     print(aYPosH,aZPosH)
    return(param)


#---------------------------------------------------------------------------------------------------
#
def createMonoPlanarNestedOFFwArray(L = 200, b_array = np.array([1.]),
                                    z_start = 10,
                                    l = 10,
                                    nb_segments = 15,
                                    
                                    alphaRad = 0, axis = 'x', RC = np.matrix([0,-.28,0], dtype = np.float128),
                                    T = np.matrix([0,0,0], dtype = np.float128),
                                    
                                    bHorizontal = True,
                                      
                                    filename = 'defaultFile.off', opticHalfWidth = 0,
                                    bBoundingBox = False, bWolter = False,
                                    bGetParamOnly = False, bVerbose = True, 
                                   ):
    '''
        Write an OFF file of a mono-planar nested optic based on a given array of short half axis. Direction of planes is chosen by bHorizontal.
        Optic can be rotated and translated.
        
        If opticHalfWidth != 0 and, resulting this, there are faces coinciding (located at opticHalfWidth -x or y), useless faces will be removed.
        So when rotating or translating the optic, algorithm may not suppress coincided faces.

        
        Input :
            - L : (float) ellipse focal distance * 2
            - b_array : (array) array of short half axis
            - z_start : (float) starting point of arc relative to focus
            - l : (float) length of elliptical arc 
            - nb_segments : (int)
            - alphaRad : (float)(array)(rad)
            see rotMatrix for more information
            - axis : (str)
            see rotMatrix for more information
            - RC : (numpy.matrix) rotation center. Can be row or column vector, will be correctly transposed to correct form
            - T : (numpy.matrix) translation vector. Can be row or column vector, will be correctly transposed to correct form
            - bHorizontal : (bool) if True build horizontal nested levels, if False vertical ones
            - filename : (str) output for OFF file
            - opticHalfWidth : (float)
            if opticHalfWidth = 0, opticHalfWidth will adapat to fit biggest elliptic arc. In this case b_1 must not be set to 'limit'
            - bBoundingBox : (bool) generate a bounding box
            - bWolter : (bool) generate hyperbolic instead of elliptic optic (used for Wolter optics) 
            - bGetParamOnly : (bool) if True function do nothing and only return *param*
            - bVerbose : (bool) activate display of messages
        
        Output :
            - param : (dict) input values
            
        Example :
            - createMonoPlanarNestedOFFwArray(bHorizontal = True, L = 200, b_array = np.array([2.,1.,.5]), z_start = 10, l = 10, nb_segments = 15, filename='exampleFile.off', opticHalfWidth = 0, bBoundingBox = False, alphaRad = np.pi/26, axis = 'x', T = np.matrix([0,0,1]))
            - createMonoPlanarNestedOFFwArray(bHorizontal = False, L = 200, b_array = np.array([2.,1.,.5]), z_start = 10, l = 10, nb_segments = 15, filename='exampleFile.off', opticHalfWidth = 0, bBoundingBox = False, alphaRad = [np.pi/26, np.pi/16], axis = 'xy', RC = '')
            - createMonoPlanarNestedOFFwArray(bHorizontal = True, L = 200, b_array = createShortHalfAxisArray(z_start = 16.5, l = 10, L = 200, b_0 = (), b_1 = 'limit', nb_levels = 200, opticHalfWidth = 2, minSpacing = 1E-2, bProtectSpacing = True, bVerbose = True)[0], z_start = 16.5, l = 10, nb_segments = 15, filename='exampleFile.off', opticHalfWidth = 2, bBoundingBox = False, bVerbose = True)
        
        Requirements :
            - getEllipseOrdinate()
            - ellipseResolution()
            - hyperbolaResolution()
            - rotMatrix()
            - numpy.around()
            - numpy.logical_and()
            - numpy.shape()
            - numpy.zeros()
            - numpy.ones()
            - numpy.array()
            - numpy.tile()
            - numpy.arange()
            - numpy.min()
            - numpy.max()
            - numpy.flip()
            - numpy.sort()
            - numpy.vstack()
            - numpy.sqrt()
            - numpy.matrix()
            - numpy.float128
            - copy.deepcopy()
    '''
    
    funcEllipseOrdi = getEllipseOrdinate
    
    if type(RC) is not str and np.shape(RC) != (3,1):
        RC = RC.T
        
    if np.shape(T) != (3,1):
        T = T.T
    
    param = {
        "L" : L, "l" : l, "z_start" : z_start, "opticHalfWidth" : opticHalfWidth,
        "b_array" : b_array, "nb_segments" : nb_segments,
        "alphaRad" : alphaRad, "axis" : axis, "RC" : RC,
        "T" : T,
        "bHorizontal" : bHorizontal,
        "bBoundingBox" : bBoundingBox, "filename" : filename,
        "bGetParamOnly" : bGetParamOnly, "bVerbose" : bVerbose,
    }
    
    bAdaptative = False
    if (opticHalfWidth == 0):
        bAdaptative = True
        opticHalfWidth = funcEllipseOrdi(z_0 = z_start+l, b = np.max(b_array), L = L, bVerbose = bVerbose)
        param['opticHalfWidth'] = opticHalfWidth
    # To be used with clearDic() for intelligent calling of function
    if bGetParamOnly:
        return param
    
    ##------------------------------------------------------------------------------------------------------------------------
    # Ellipse calculations
    #z_pos, y_array = ellipseResolution(b_array, L, z_start, nb_segments, l, opticHalfWidth)
    
    if bWolter == False:
        z_pos, y_array = ellipseResolution(b_array, L, z_start, nb_segments, l, opticHalfWidth) 
    else:
        #For Wolter Option L == a_h == short hyperbola half_axis
        z_pos, y_array = hyperbolaResolution(b_array, L, z_start, nb_segments, l, opticHalfWidth)


    ##------------------------------------------------------------------------------------------------------------------------
    # Points creation
    YY = np.vstack((y_array, np.flipud(-y_array)))
    
    X = np.array([])
    Y = np.array([])

    for i in range(len(YY[0])):
        YYY = np.tile(np.array([YY[:,i]]).transpose(), (1, 2))
        XXX = np.ones(np.shape(YYY))
        XXX[:,0] *= opticHalfWidth
        XXX[:,1] *= -opticHalfWidth
        
        if len(X) == 0:
            X = [XXX]
            Y = [YYY]
        else:
            X = np.vstack((X, [XXX]))
            Y = np.vstack((Y, [YYY]))
    
    Z = np.ones(np.shape(X))
    for i in range(len(X)):
        Z[i] *= z_pos[i]
        
    IND = np.arange(np.shape(Z)[0]*np.shape(Z)[1]*np.shape(Z)[2]).reshape(np.shape(Z))
    
    
    ##------------------------------------------------------------------------------------------------------------------------
    # Rotation
    if type(alphaRad) is list:
        alphaRad = np.array(alphaRad) # in order to use abs()
        
    if np.max(abs(alphaRad)) != 0:
        if not bHorizontal:
            alphaRad = -alphaRad
            if axis in ['X','x','0',0]:
                axis = 'y'
            elif axis in ['Y','y','1',1]:
                axis = 'x'

        # If wanted, setting rotation center to optic center
        if type(RC) is str:
            Va = []
            for a in range(len(X)):
                for l in range(len(X[a])):
                    for c in range(len(X[a][l])):
                        Va += [np.matrix([X[a,l,c], Y[a,l,c], Z[a,l,c]]).T] #Every coordinates of vertices in a matrix
            RC = np.sum(Va, axis = 0) / len(Va) # center of optic

        # Applying rotation
        R = rotMatrix(t = alphaRad, axis = axis)

        for a in range(len(X)):
            for l in range(len(X[a])):
                for c in range(len(X[a][l])):
                    V = np.matrix([X[a,l,c], Y[a,l,c], Z[a,l,c]]).T
                    Vr = R*(V-RC)+RC # Move to rotation point, rotate, then cancel first translation
                    Vrt = Vr + T

                    X[a,l,c] = Vrt[0]
                    Y[a,l,c] = Vrt[1]
                    Z[a,l,c] = Vrt[2]
    ### Applying only a translation                
    elif (len(T) == 3) and not (np.min(T == np.matrix([0,0,0], dtype = np.float128).T)):
        for a in range(len(X)):
            for l in range(len(X[a])):
                for c in range(len(X[a][l])):
                    V = np.matrix([X[a,l,c], Y[a,l,c], Z[a,l,c]]).T
                    Vt = V + T 

                    X[a,l,c] = Vt[0]
                    Y[a,l,c] = Vt[1]
                    Z[a,l,c] = Vt[2]
    
    
    ##------------------------------------------------------------------------------------------------------------------------
    # Width limit
    if bAdaptative:
        hoWXup = np.max(X)
        hoWXlow = np.min(X)
    
        hoWYup = np.max(Y)
        hoWYlow = np.min(Y)
    else:
        hoWXup = opticHalfWidth
        hoWXlow = -opticHalfWidth
    
        hoWYup = opticHalfWidth
        hoWYlow = -opticHalfWidth

    
    X[X > hoWXup] = hoWXup
    X[X < hoWXlow] = hoWXlow
    
    Y[Y > hoWYup] = hoWYup
    Y[Y < hoWYlow] = hoWYlow
    
#     Z[Z > z_start + l] = z_start + l
#     Z[Z < z_start] = z_start
    
    
    ##------------------------------------------------------------------------------------------------------------------------
    # Faces deletion
    MaskY = np.round(abs(Y), 8) == np.round(2, 8) 
    MaskEtY = np.logical_and(MaskY[:-1], MaskY[1:])
    MaskEtY = np.logical_and(MaskEtY[:,:,0], MaskEtY[:,:,1])
    
    MaskEtEtY = np.zeros(np.shape(MaskEtY), bool)
    for a in range(len(MaskEtY)):
        mid = int(len(MaskEtY[a])/2)
        MaskEtEtY[a,:mid] = np.logical_and(MaskEtY[a,1:mid+1], MaskEtY[a,:mid])
        MaskEtEtY[a,mid:] = np.logical_and(MaskEtY[a,mid:], MaskEtY[a,mid-1:-1])
        if min(MaskEtEtY[a]) == True: # If every faces are equal or greater than oW in the circle, enable to draw the "smallest" level
            MaskEtEtY[a,mid-1] = False
            MaskEtEtY[a,mid] = False
    
    
    ##------------------------------------------------------------------------------------------------------------------------
    # OFF Creation
    faces = np.zeros(2*(len(y_array)-1)*nb_segments, object) # Container of faces indexes for OFF file    
    facesBB = np.zeros(nb_segments*(2 + 2*(2*len(y_array)-1)), object)
    
    # Faces
    cptFaces = 0
    for a in range(len(X)-1):
        for l in range(1,len(X[a])-1):
            if not(MaskEtEtY[a,l]):
                faces[cptFaces] = '4 {} {} {} {}\n'.format(IND[a,l,0], IND[a,l,1], IND[a+1,l,1], IND[a+1,l,0])
                cptFaces += 1
    
    # Faces of bounding box
    cptFacesBB = 0
    if bBoundingBox:
        for a in range(len(X)-1):
            for l in [0, len(X[a])-1]:
                if not(MaskEtEtY[a,l]):
                    facesBB[cptFacesBB] = '4 {} {} {} {}\n'.format(IND[a,l,0], IND[a,l,1], IND[a+1,l,1], IND[a+1,l,0])
                    cptFacesBB += 1
        for a in range(len(X)-1):
            for l in range(len(y_array)-1):
                if not(MaskEtEtY[a,l]):
                    facesBB[cptFacesBB] = '4 {} {} {} {}\n'.format(IND[a,l,0], IND[a,l+1,0], IND[a+1,l+1,0], IND[a+1,l,0])
                    cptFacesBB += 1
                    facesBB[cptFacesBB] = '4 {} {} {} {}\n'.format(IND[a,l,1], IND[a,l+1,1], IND[a+1,l+1,1], IND[a+1,l,1])
                    cptFacesBB += 1
                
                if not(MaskEtEtY[a,-1-l]):
                    ll = 2*len(y_array)-1-l
                    facesBB[cptFacesBB] = '4 {} {} {} {}\n'.format(IND[a,ll,0], IND[a,ll-1,0], IND[a+1,ll-1,0], IND[a+1,ll,0])
                    cptFacesBB += 1
                    facesBB[cptFacesBB] = '4 {} {} {} {}\n'.format(IND[a,ll,1], IND[a,ll-1,1], IND[a+1,ll-1,1], IND[a+1,ll,1])
                    cptFacesBB += 1
            l += 1
            if not(MaskEtEtY[a,l]):
                facesBB[cptFacesBB] = '4 {} {} {} {}\n'.format(IND[a,l,0], IND[a,l+1,0], IND[a+1,l+1,0], IND[a+1,l,0])
                cptFacesBB += 1
                facesBB[cptFacesBB] = '4 {} {} {} {}\n'.format(IND[a,l,1], IND[a,l+1,1], IND[a+1,l+1,1], IND[a+1,l,1])
                cptFacesBB += 1
    

    f = open(filename,'w')
    # Write OFF-File Header
    f.write("OFF\n")
    if bHorizontal:
        f.write("# Mono planar (H) nested optic \n")
    else:
        f.write("# Mono planar (V) nested optic \n")
    
    f.write("# Number of vertex, number of faces, number of edges \n")
    f.write("%d %d 0\n"%(4*len(y_array)*len(Y), cptFaces+cptFacesBB)) # Write the number of vertices (=corners) and faces ()
    
    if bHorizontal:
        for a in range(len(X)):
            for l in range(len(X[a])):
                for c in range(len(X[a][l])):
                    # Vertices
                    f.write('%0.9f %0.9f %0.9f\n'%(X[a,l,c], Y[a,l,c], Z[a,l,c]-z_start))
    else:
        for a in range(len(X)):
            for l in range(len(X[a])):
                for c in range(len(X[a][l])):
                    # Vertices
                    f.write('%0.9f %0.9f %0.9f\n'%(Y[a,l,c], X[a,l,c], Z[a,l,c]-z_start))
        
    for i in range(cptFaces):
        f.write(faces[i])
    
    for i in range(cptFacesBB):
        f.write(facesBB[i])
        
    f.close()
    
    return(param)


#---------------------------------------------------------------------------------------------------
#
def maskProcessing(mask):
    '''
        Intermediary of createToroidalNestedOFFwArray()
    '''
    
    for k in range(len(mask)):
        for i in range(len(mask[k])):
            for j in range(len(mask[k,i])):
                if mask[k,i,j]:
                    if j is not len(mask[k,i])-1:
                        if not mask[k,i,j+1]:
                            mask[k,i,j] = False
                    else:
                        if not mask[k,i,0]:
                            mask[k,i,j] = False
    
    return mask


#---------------------------------------------------------------------------------------------------
#
def createToroidalNestedOFFwArray(L = 200, b_array = np.array([1.]),
                                  z_start = 10,
                                  l = 10,
                                  nb_segments = 15, nb_segments_T = 50,
                                  filename='defaultFile.off', 
                                  alphaRad = 0, axis = 'x', RC = np.matrix([0,-.28,0], dtype = np.float128).T,
                                  T = np.matrix([0,0,0], dtype = np.float128),
                                  
                                  opticHalfWidth = 0,
                                  bBoundingBox = False,
                                  bWolter = False,
                                  bGetParamOnly = False, bVerbose = True,
                                 ):
    '''
            Write an OFF file of a elliptic nested optic based on a given array of short half axis.
        Optic can be rotated and translated.
        
        If opticHalfWidth != 0 and, resulting this, there are faces coinciding (located at opticHalfWidth -x or y), useless faces will be removed.
        So when rotating or translating the optic, algorithm may not suppress coincided faces.
        
        Input :
            - L : (float) ellipse focal distance * 2
            - b_array : (array) array of short half axis
            - z_start : (float) starting point of arc relative to focus
            - l : (float) length of elliptical arc 
            - nb_segments : (int) longitudinal discretisation of optic
            - nb_segments_T : (int) transversal discretisation of optic
            - alphaRad : (float)(array)(rad)
            see rotMatrix for more information
            - axis : (str)
            see rotMatrix for more information
            - RC : (numpy.matrix) rotation center. Can be row or column vector, will be correctly transposed to correct form
            - T : (numpy.matrix) translation vector. Can be row or column vector, will be correctly transposed to correct form
            - filename : (str) output for OFF file
            - opticHalfWidth : (float)
            if opticHalfWidth = 0, opticHalfWidth will adapat to fit biggest elliptic arc. In this case b_1 must not be set to 'limit'
            - bBoundingBox : (bool) generate a bounding box
            - bWolter : (bool) generate hyperbolic instead of elliptic optic (used for Wolter optics)
            - bGetParamOnly : (bool) if True function do nothing and only return *param*
            - bVerbose : (bool) display of messages
        
        Output :
            - param : (dict) input values
            
        Example :
            - createToroidalNestedOFFwArray(L = 200, b_array = np.array([2.,1.,.5]), z_start = 10, l = 10, nb_segments = 15, filename='exampleFile.off', opticHalfWidth = 0, bBoundingBox = False, alphaRad = np.pi/26, axis = 'x', T = np.matrix([0,0,1]))
            - createToroidalNestedOFFwArray(L = 200, b_array = np.array([2.,1.,.5]), z_start = 10, l = 10, nb_segments = 15, filename='exampleFile.off', opticHalfWidth = 0, bBoundingBox = False, alphaRad = [np.pi/26, np.pi/16], axis = 'xy', RC = '')
            - createToroidalNestedOFFwArray(L = 200, b_array = createShortHalfAxisArray(z_start = 16.5, l = 10, L = 200, b_0 = (), b_1 = 'limit', nb_levels = 200, opticHalfWidth = 2, minSpacing = 1E-2, bProtectSpacing = True, bVerbose = True)[0], z_start = 16.5, l = 10, nb_segments = 15, filename='exampleFile.off', opticHalfWidth = 2, bBoundingBox = True, bVerbose = True)
        
        Requirements :
            - getEllipseOrdinate()
            - rotMatrix()
            - ellipseResolution()
            - hyperbolaResolution()
            - numpy.around()
            - numpy.logical_and()
            - numpy.shape()
            - numpy.zeros()
            - numpy.ones()
            - numpy.array()
            - numpy.tile()
            - numpy.arange()
            - numpy.min()
            - numpy.max()
            - numpy.sum()
            - numpy.vstack()
            - numpy.matrix()
            - numpy.sin()
            - numpy.cos()
            - numpy.float128
            - copy.deepcopy()
    '''
    
    funcEllipseOrdi = getEllipseOrdinate
    
    if type(RC) is not str and np.shape(RC) != (3,1):
        RC = RC.T
        
    if np.shape(T) != (3,1):
        T = T.T
    
    
    param = {
        "L" : L, "l" : l, "z_start" : z_start, "opticHalfWidth" : opticHalfWidth,
        "b_array" : b_array, "nb_segments" : nb_segments, "nb_segments_T" : nb_segments_T,
        "bBoundingBox" : bBoundingBox, "filename" : filename,
        "RC" : RC, "alphaRad" : alphaRad, "axis" : axis,
        "T" : T,
        "bGetParamOnly" : bGetParamOnly, "bVerbose" : bVerbose,
    }
    
    if (opticHalfWidth == 0): 
        param['opticHalfWidth'] = funcEllipseOrdi(z_0 = z_start+l, b = np.max(b_array), L = L, bVerbose = bVerbose)
    
    # To be used with clearDic() for intelligent calling of function
    if bGetParamOnly:
        return param
    
    ##------------------------------------------------------------------------------------------------------------------------
    # Ellipse calculations  
    
    b_array = b_array.reshape(-1, 1)
    if bWolter == False:
        # Management of ellipse properties
        b_array = b_array.reshape(-1, 1)
        b2_array = b_array**2 ; b2_array
        
        a2_array = (L/2)**2 + b_array**2 ; a2_array
        
        # Generation of evenly spaced circle in order to create the ellipse
        z_pos = z_start + np.arange(nb_segments+1, dtype = np.float128)* l/nb_segments
        
        zP = np.array([L/2 - z_pos for _ in range(len(b_array))], dtype = np.float128)
        zP2 = zP**2
        
        ## Resizing array for array composition
        a2_array = np.tile(a2_array, (1,len(z_pos)))
        
        # Finding the radius of every circle
        y2_array = (1-zP**2/a2_array) * b2_array
        y_array = np.sqrt(y2_array, dtype = np.float128)
        

        # Ellipse calculations
        #z_pos, y_array = ellipseResolution(b_array, L, z_start, nb_segments, l, opticHalfWidth)
       
        #z_pos, y_array = ellipseResolution(b_array, L, z_start, nb_segments, l, opticHalfWidth) 
    else:
        #For Wolter Option L == a_h == short hyperbola half_axis
        z_pos, y_array = hyperbolaResolution(b_array, L, z_start, nb_segments, l, opticHalfWidth, bToro = True)
    



    theta = 2*np.pi/nb_segments_T * np.arange(nb_segments_T, dtype = np.float128)
    
    # Creating correct sized array for points
    X = np.arange(len(b_array)*(nb_segments + 1) * len(theta), dtype = np.float128).reshape((len(b_array), nb_segments + 1, len(theta)))
    Y = np.arange(len(b_array)*(nb_segments + 1) * len(theta), dtype = np.float128).reshape((len(b_array), nb_segments + 1, len(theta)))
    Z = np.arange(len(b_array)*(nb_segments + 1) * len(theta), dtype = np.float128).reshape((len(b_array), nb_segments + 1, len(theta)))
    
    # Finding every points
    for k in range(len(b_array)):
        for i in range(nb_segments+1):
            X[k,i] = y_array[k,i] * np.sin(theta, dtype = np.float128) 
            Y[k,i] = y_array[k,i] * np.cos(theta, dtype = np.float128)
    
    ##------------------------------------------------------------------------------------------------------------------------
    # Rotation
    if type(alphaRad) is list:
        alphaRad = np.array(alphaRad) # in order to use abs()
        
    if np.max(abs(alphaRad)) != 0:
        # If wanted, setting rotation center to optic center
        if type(RC) is str:
            Vt = []
            for k in range(len(b_array)):
                for i in range(nb_segments+1):
                    for j in range(nb_segments_T):
                        Vt += [np.matrix([X[k,i,j], Y[k,i,j], z_pos[i]]).T] #Every coordinates of vertices in a matrix
            RC = np.sum(Vt, axis = 0) / len(Vt) # center of optic

    # Applying rotation

        R = rotMatrix(t = alphaRad, axis = axis)

        for k in range(len(b_array)):
            for i in range(nb_segments+1):
                for j in range(nb_segments_T):
                    V = np.matrix([X[k,i,j], Y[k,i,j], z_pos[i]]).T
                    Vr = R*(V-RC)+RC # Move to rotation point, rotate, then cancel first translation

                    X[k,i,j] = Vr[0]
                    Y[k,i,j] = Vr[1]
                    Z[k,i,j] = Vr[2]
    elif (len(T) == 3) and not (np.min(T == np.matrix([0,0,0], dtype = np.float128).T)):
        for a in range(len(X)):
            for l in range(len(X[a])):
                for c in range(len(X[a][l])):
                    V = np.matrix([X[a,l,c], Y[a,l,c], z_pos[l]]).T
                    Vt = V + T 

                    X[a,l,c] = Vt[0]
                    Y[a,l,c] = Vt[1]
                    Z[a,l,c] = Vt[2]
                    
    else:
        for k in range(len(b_array)):
            for i in range(nb_segments+1):
                for j in range(nb_segments_T):
                    Z[k,i,j] = z_pos[i]
    
    
    ##------------------------------------------------------------------------------------------------------------------------
    # Width limit
    if (opticHalfWidth == 0):
        hoWXup = np.max(X)
        hoWXlow = np.min(X)
    
        hoWYup = np.max(Y)
        hoWYlow = np.min(Y)
    else:
        hoWXup = opticHalfWidth
        hoWXlow = -opticHalfWidth
    
        hoWYup = opticHalfWidth
        hoWYlow = -opticHalfWidth
    
    X[X > hoWXup] = hoWXup
    X[X < hoWXlow] = hoWXlow
    
    Y[Y > hoWYup] = hoWYup
    Y[Y < hoWYlow] = hoWYlow
    
#     Z[Z > z_start + l] = z_start + l
#     Z[Z < z_start] = z_start
    
    
    ##------------------------------------------------------------------------------------------------------------------------
    # Faces deletion
    MaskX = (abs(np.around(X, 8)) == round(opticHalfWidth, 8))
    MaskEtX = MaskX[:, :-1, :] & MaskX[:, 1:, :]
    MaskEtX = maskProcessing(MaskEtX)
    
    MaskY = (abs(np.around(Y, 8)) == round(opticHalfWidth, 8))
    MaskEtY = MaskY[:, :-1, :] & MaskY[:, 1:, :]
    MaskEtY = maskProcessing(MaskEtY)
    
    
    ##------------------------------------------------------------------------------------------------------------------------
    # OFF Creation
    faces = np.zeros(len(b_array) * (nb_segments+1) * nb_segments_T, object) # Container of faces indexes for OFF file
    
    index_offsetBB = 0 # Offset in case, one wants a bounding box
    if bBoundingBox:
        index_offsetBB = 8
        
    cptFaces = 0
    # Faces
    for k in range(len(b_array)):
        index_offset = (nb_segments_T*(nb_segments+1))*k + index_offsetBB
        for i in range(nb_segments):
            j = nb_segments_T-1
            if not bBoundingBox:
                if k > 0:
                    if not MaskEtX[k,i,j] and not MaskEtY[k,i,j]:
                        faces[cptFaces] = '4 {} {} {} {}\n'.format(index_offset+i*nb_segments_T+j, index_offset+(i+1)*nb_segments_T+j, index_offset+(i+1)*nb_segments_T, index_offset+i*nb_segments_T)
                        cptFaces += 1
                else:
                    faces[cptFaces] = '4 {} {} {} {}\n'.format(index_offset+i*nb_segments_T+j, index_offset+(i+1)*nb_segments_T+j, index_offset+(i+1)*nb_segments_T, index_offset+i*nb_segments_T)
                    cptFaces += 1
            else:
                if not MaskEtX[k,i,j] and not MaskEtY[k,i,j]:
                    faces[cptFaces] = '4 {} {} {} {}\n'.format(index_offset+i*nb_segments_T+j, index_offset+(i+1)*nb_segments_T+j, index_offset+(i+1)*nb_segments_T, index_offset+i*nb_segments_T)
                    cptFaces += 1

    for k in range(len(b_array)):
        index_offset = (nb_segments_T*(nb_segments+1))*k + index_offsetBB
        for i in range(nb_segments):
            for j in range(nb_segments_T-1):
                if not bBoundingBox:
                    if k > 0:
                        if not MaskEtX[k,i,j] and not MaskEtY[k,i,j]:
                            faces[cptFaces] = '4 {} {} {} {}\n'.format(index_offset+i*nb_segments_T+j, index_offset+(i+1)*nb_segments_T+j, index_offset+(i+1)*nb_segments_T+1+j, index_offset+i*nb_segments_T+1+j)
                            cptFaces += 1
                    else:
                        faces[cptFaces] = '4 {} {} {} {}\n'.format(index_offset+i*nb_segments_T+j, index_offset+(i+1)*nb_segments_T+j, index_offset+(i+1)*nb_segments_T+1+j, index_offset+i*nb_segments_T+1+j)
                        cptFaces += 1
                else:
                    if not MaskEtX[k,i,j] and not MaskEtY[k,i,j]:
                        faces[cptFaces] = '4 {} {} {} {}\n'.format(index_offset+i*nb_segments_T+j, index_offset+(i+1)*nb_segments_T+j, index_offset+(i+1)*nb_segments_T+1+j, index_offset+i*nb_segments_T+1+j)
                        cptFaces += 1


    f = open(filename,'w')
    # Write OFF-File Header
    f.write("OFF\n")
    f.write("# Elliptic nested optic \n")
    f.write("# Number of vertex, number of faces, number of edges \n")
    nbPoints = len(b_array) * (nb_segments+1) * nb_segments_T
    nbFaces = cptFaces
    if bBoundingBox:
        nbPoints += 8
        nbFaces += 4
        
    f.write("%d %d 0\n"%(nbPoints, nbFaces)) # Write the number of vertices (=corners) and faces ()
    
    if bBoundingBox:
        f.write('%0.9f %0.9f %0.9f\n'%(hoWXup, hoWYlow, 0))
        f.write('%0.9f %0.9f %0.9f\n'%(hoWXup, hoWYlow, l))
        f.write('%0.9f %0.9f %0.9f\n'%(hoWXlow, hoWYlow, l))
        f.write('%0.9f %0.9f %0.9f\n'%(hoWXlow, hoWYlow, 0))
        
        f.write('%0.9f %0.9f %0.9f\n'%(hoWXup, hoWYup, 0))
        f.write('%0.9f %0.9f %0.9f\n'%(hoWXup, hoWYup, l))
        f.write('%0.9f %0.9f %0.9f\n'%(hoWXlow, hoWYup, l))
        f.write('%0.9f %0.9f %0.9f\n'%(hoWXlow, hoWYup, 0))
    
    #Vertices
    for k in range(len(b_array)):
        for i in range(nb_segments+1):
            for j in range(nb_segments_T):
                f.write('%0.9f %0.9f %0.9f\n'%(X[k,i,j], Y[k,i,j], Z[k,i,j]-z_start))
        
    if bBoundingBox:
        f.write('4 %d %d %d %d\n'%(0,1,2,3))
        f.write('4 %d %d %d %d\n'%(4,5,6,7))
        f.write('4 %d %d %d %d\n'%(0,1,5,4))
        f.write('4 %d %d %d %d\n'%(3,2,6,7))
        
    for i in range(cptFaces):
        f.write(faces[i])
        
    f.close()

    return(param)


#---------------------------------------------------------------------------------------------------
#
def createNestedOFFwParam(bPrintB = False, bPrintOFF = False, printOFFSoft = 'meshlab', bTmpDir = True, tmpDir = os.path.join(os.path.expanduser("~"),'tmpOFF'),
                          bSubFold = True, subFold = 'OFF_File', bAutoName = True, OFFname = 'DefaultOFF.off', run_folder = 'run_folder',
                          bGetParamOnly = False, bVerbose = True, bMono = False,
                          funcGenD = createSimpleDoublePlanarNestedOFFwArray, funcGenM = createMonoPlanarNestedOFFwArray,
                          **kwargs
                         ):
    '''
        Write an OFF file of a nested optic, which can be mono or double planar. An array of short half axis will be generated thanks to given parameters.
        
        Requirements :
            - createShortHalfAxisArray()
            - createDoublePlanarNestedOFFwArray()
            - createMonoPlanarNestedOFFwArray()
            - createToroidalNestedOFFwArray()
        
        Input :
            - bPrintB : (bool) display the generated array of short half axis
            - bPrintOFF : (bool) display the OFF file with *printOFFSoft*
            - printOFFSoft : (str) name of rendering software (will be called using os.system)
            - bTmpDir : (bool) in case *printOFFSoft* doesn't have acces to current folder, will create a temporary OFF file in *tmpDir*
            - tmpDir : (str) path where to copy and display OFF file            
            - bSubFold : (bool) generate the OFF file in a sub folder
            - subFold : (str) name of desired sub folder
            - bAutoName : (bool) if True, generate an automatic name with all information needed to recreate file, else use *OFFname*
            - OFFname : (str) 
            - run_folder : (str) path where McStasScript runs
            - bMono : (bool) choose between mono or double planar optics            
            - bGetParamOnly : (bool) if True function do nothing and only return *param*
            - bVerbose : (str) activate display of messages
            - funcGenD : (func) generator of double planar OFF file
            - funcGenM : (func) generator of mono planar OFF file
            - funcHalfAxisArray : (func) generator of short half axis array
            - kwargs : (dict) contains arguments that must be applied to required function (for setting optics properties). For more information, see help of required functions.
        
        Output :
            - mcStasFileName : (str)('""') filename adapted to McStasScript functions
            - merge : (dictionnary) input values
            
        Example :
            - createNestedOFFwParam(bPrintB = False, bPrintOFF = True, printOFFSoft = 'meshlab', bTmpDir = True, bSubFold = True, subFold = 'exampleOFF', bAutoName = True, bVerbose = True, bMono = False, funcGenD = createDoublePlanarNestedOFFwArray, b_1 = 2.5, opticHalfWidth = 2, z_start = 16.5, l = 10, L = 200, nb_levels = 12, bProtectSpacing = True, minSpacing = 1E-2, nb_segments = 15, bBoundingBox = False)
            - createNestedOFFwParam(bPrintB = True, bSubFold = True, subFold = 'exampleOFF', bVerbose = False, bMono = True, bHorizontal = False, b_1 = 2.5, opticHalfWidth = 2, z_start = 16.5, l = 10, L = 200, nb_levels = 6, bProtectSpacing = True, minSpacing = 5E-3, nb_segments = 15, bBoundingBox = True)
        Requirements :
            - arguments()
            - checkFolder()
            - createSimpleDoublePlanarNestedOFFwArray()
            - createDoublePlanarNestedOFFwArray()
            - createToroidalNestedOFFwArray()
            - createMonoPlanarNestedOFFwArray()
            - os.path.join()
            - os.system()
        '''
    
    # Gathering all inputs if there are transcendental arguments
    argIn = cp.deepcopy(arguments())
    kwargs.update(argIn)
    
    ##------------------------------------------------------##
    ##                 Saving Parameters                     #
    ##------------------------------------------------------##
    funcHalfAxisArray = createShortHalfAxisArray
    
    param = {
        "bPrintB" : bPrintB, "bPrintOFF" : bPrintOFF, "printOFFSoft" : printOFFSoft,
        "bSubFold" : bSubFold, "subFold" : subFold, "bAutoName" : bAutoName, "OFFname" : OFFname,
        "bTmpDir" : bTmpDir, "tmpDir" : tmpDir, "run_folder" : run_folder,
        "bGetParamOnly" : bGetParamOnly, "bVerbose" : bVerbose,
        "bMono" : bMono, "funcGenD" : funcGenD, "funcGenM" : funcGenM,
    }
    
    ## Changing OFF file folder name for better organisation
    param['subFold'] += '_{}'.format(funcGenD.__name__)
    subFold = param['subFold']
    
    
    #------------------------------------------------------
    # Applying the arguments to the correct function
    
    pGen = funcHalfAxisArray(bGetParamOnly = True, bVerbose = bVerbose)
    if not bMono:
        pCreate = funcGenD(bGetParamOnly = True, bVerbose = bVerbose)
    else:
        pCreate = funcGenM(bGetParamOnly = True, bVerbose = bVerbose)
    
    argGen, arg = clearDic(kwargs, pGen)
    argGen = {**pGen, **argGen}
    argCreate, arg = clearDic(kwargs, pCreate)
    argCreate = {**pCreate, **argCreate}

    
    merge = {**argGen, **argCreate, **param}
    
    
    if (bGetParamOnly): # In order to only get the used parameters
        return(merge)
    
    ##------------------------------------------------------##
    ##                 Initialisation                        #
    ##------------------------------------------------------##
    
    if bAutoName: # naming file thanks to its properties
        b_ind = '{}m'.format(nameFile(merge['b_1']))
        if (len(merge['b_0']) > 0):
            b_ind += '_Wb0'
            for i in range(min(len(merge['b_0']),merge['nb_levels'])):
                b_ind += '-{}'.format(nameFile(merge['b_0'][i]))
        name = 'NO'
        
        if not bMono and merge['funcGenD'] == createSimpleDoublePlanarNestedOFFwArray:
            name += '_DPS'
        elif not bMono and merge['funcGenD'] == createDoublePlanarNestedOFFwArray:
            name += '_DP'
        elif not bMono and merge['funcGenD'] == createToroidalNestedOFFwArray:
            name += '_E'
        elif bMono and merge['funcGenM'] == createMonoPlanarNestedOFFwArray:
            name += '_MP'
#         elif bMono and merge['funcGenM'] == createSimpleMonoPlanarNestedOFFwArray:
#             name += '_MPS'
        
            
        
        name += '_l{}m_L{}m_zS{}m_b{}_ohW{}m_nbS{}_nbL{}_bB{}_bMs{}-{}'.format(nameFile(merge['l']),nameFile(merge['L']),nameFile(merge['z_start']),b_ind,
                                                                                                    nameFile(merge['opticHalfWidth']),
                                                                                                    nameFile(merge['nb_segments']),nameFile(merge['nb_levels']),
                                                                                                    nameFile(merge['bBoundingBox'])[0], nameFile(merge['bProtectSpacing'])[0], nameFile(merge['minSpacing']))
        if bMono:
            name += '_bH{}'.format(nameFile(nameFile(merge['bHorizontal'])))
        
        name += '.off'
    else:
        name = OFFname
        
    if (bSubFold): # In order to have better file management
        joinPath = os.path.join(run_folder,subFold)
        checkFolder(joinPath)# check if the folder doesn't exist
            
        abr_filename = os.path.join(subFold,name)
    else :
        abr_filename = name
        
    filename = os.path.join(run_folder, abr_filename)
    argCreate['filename'] = filename
    
    ##------------------------------------------------------##
    ##                    Generation                         #
    ##------------------------------------------------------##
    ## Array of short half axis 
    
    # Generation of an array of short half axis
    argCreate['b_array'], pGen = funcHalfAxisArray(**argGen)

    if (bPrintB):
        print(argCreate['b_array'])
    
    ## Generation of a double planar nested optic
    if not bMono:
        pCreate = funcGenD(**argCreate)
    else:
        pCreate = funcGenM(**argCreate)
        
    ## Display
    if (bPrintOFF):
        if bTmpDir :
            if bVerbose:
                print('Create temporary file and folder at {}'.format(tmpDir))
            checkFolder(tmpDir, bVerbose = False) # Check if folder exist
            
            os.system('cp {} {}'.format(filename,tmpDir))
            if bVerbose:
                print('Opening file. Temporary file and folder will be deleted after closing.')
            file = os.path.join(tmpDir,name)
            os.system("{} {} ; rm -v {} ; rmdir {}".format(printOFFSoft,file,file, tmpDir))
        else:
            if bVerbose:
                print('Displaying {}'.format(filename))
            os.system("{} {}".format(printOFFSoft, filename))
        
    #------------------------------------------------------
    # Merging the dictionaries of parameters of both called functions 
    merge = {**pGen, **pCreate, **param}
    
    mcStasFileName = '"'+abr_filename+'"'

    return(mcStasFileName, merge) # Returning the path that can be used by McStasScript


