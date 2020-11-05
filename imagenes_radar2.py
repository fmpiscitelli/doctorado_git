# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 10:50:24 2020

@author: Fran
"""


import matplotlib.colors as colors
from matplotlib import pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import pyart
import glob
import matplotlib.gridspec as gridspec

direc='C:/Users/Fran/Dropbox/Fran/doctorado/paper_tesis/datos_radar/nc_todos_vcrudo/'
salida= 'C:/Users/Fran/Dropbox/Fran/doctorado/paper_tesis/fig_casosbase_vmedio/'


archivos=['cfrad.20091117_174348.000_to_20091117_174737.000_INTA_Parana_SUR.nc', 
          'cfrad.20091117_182346.000_to_20091117_182734.000_INTA_Parana_SUR.nc',
          'cfrad.20091117_214352.000_to_20091117_214740.000_INTA_Parana_SUR.nc',
          'cfrad.20091129_175346.000_to_20091129_175730.999_INTA_Parana_SUR.nc',
          'cfrad.20120204_194431.000_to_20120204_194728.000_INTA_Parana_SUR.nc',
          'cfrad.20120204_181435.000_to_20120204_181732.000_INTA_Parana_SUR.nc',
          'cfrad.20121219_211433.000_to_20121219_211707.999_INTA_Parana_SUR.nc',
          'cfrad.20150809_020433.000_to_20150809_020729.445_INTA_Parana_SUR.nc',
          'cfrad.20151119_012430.000_to_20151119_012732.000_INTA_Parana_SUR.nc',
          'cfrad.20151119_011433.000_to_20151119_011731.000_INTA_Parana_SUR.nc',
          'cfrad.20160207_213433.000_to_20160207_213734.555_INTA_Parana_SUR.nc',
          'cfrad.20160207_224434.000_to_20160207_224734.001_INTA_Parana_SUR.nc',
          'cfrad.20160207_234457.000_to_20160207_234735.000_INTA_Parana_SUR.nc',
          'cfrad.20160207_235434.000_to_20160207_235733.995_INTA_Parana_SUR.nc',
          'cfrad.20161223_003434.000_to_20161223_003710.000_INTA_Parana_SUR.nc',
          'cfrad.20170103_230434.000_to_20170103_230731.389_INTA_Parana_SUR.nc',
          'cfrad.20170103_232434.000_to_20170103_232731.444_INTA_Parana_SUR.nc',
          'cfrad.20170103_232434.000_to_20170103_232731.444_INTA_Parana_SUR.nc']

deltax=[0.35,0.35,0.8,0.8,0.5,0.8,0.5,0.5,0.6,0.5,0.7,0.8,0.6,0.5,0.7,0.9,0.8,0.4]
deltay=[0.35,0.35,0.6,0.8,0.5,0.6,0.5,0.5,0.6,0.5,0.7,0.8,0.6,0.5,0.7,0.9,0.9,0.4]
xce=[-60.40,-60.45,-60.1,-60.25,-60.7,-60.0,-60.85,-61.35,-60.6,-61.15,-60.65,-60.65,-60.5,-60.785,-60.56,-60.20,-60.7,-60.05]
yce=[-31.75,-31.70,-31.4,-32.2,-31.3,-31.2,-31.4,-31.7,-31.2,-31.75,-32.15,-32.1,-32.1,-31.95,-32.2,-31.5,-31.42,-31.55]
v1=[-60.45,-60.46,-60.29,-60.4,-60.8,-60.1,-60.9,-61.43,-60.77,-61.28,-60.78,-60.67,-60.7,-60.95,-60.7,-60.285,-60.75,-60.15]
v2=[-60.45,-60.46,-60.29,-60.4,-60.8,-60.1,-60.9,-61.43,-60.77,-61.28,-60.78,-60.67,-60.7,-60.95,-60.7,-60.3,-60.75,-60.15]
v3=[-31.742,-31.625,-31.25,-32.3,-31.24,-31.17,-31.4,-31.59,-31.16,-31.56,-32.17,-32.1,-31.95,-31.96,-32.23,-31.2,-31.06,-31.55]
v4=[-31.78,-31.685,-31.365,-32.36,-31.28,-31.25,-31.46,-31.66,-31.23,-31.7,-32.3,-32.21,-32.1,-32.06,-32.31,-31.3,-31.2,-31.66]
v5=[-60.36,-60.385,-60.02,-60.4,-60.8,-60.0,-60.7,-61.31,-60.59,-61.1,-60.6,-60.55,-60.55,-60.85,-60.6,-60.17,-60.55,-60.05]
v6=[-60.45,-60.46,-60.29,-60.3,-60.68,-60.1,-60.9,-61.43,-60.77,-61.28,-60.78,-60.67,-60.7,-60.95,-60.7,-60.285,-60.75,-60.15]
v7=[-31.742,-31.625,-31.25,-32.3,-31.23,-31.2,-31.34,-31.51,-31.12,-31.56,-32.17,-32.1,-31.95,-31.96,-32.23,-31.25,-31.06,-31.55]
v8=[-31.742,-31.625,-31.25,-32.3,-31.22,-31.17,-31.4,-31.59,-31.16,-31.56,-32.17,-32.1,-31.95,-31.96,-32.23,-31.2,-31.06,-31.55]
v9=[-60.36,-60.385,-60.02,-60.3,-60.68,-60.0,-60.7,-61.31,-60.59,-61.1,-60.6,-60.55,-60.55,-60.85,-60.6,-60.2,-60.55,-60.05]
v10=[-60.36,-60.385,-60.02,-60.3,-60.68,-60.0,-60.7,-61.31,-60.59,-61.1,-60.6,-60.55,-60.55,-60.85,-60.6,-60.17,-60.55,-60.05]
v11=[-31.785,-31.685,-31.365,-32.36,-31.276,-31.265,-31.4,-31.57,-31.2,-31.7,-32.3,-32.21,-32.1,-32.06,-32.31,-31.35,-31.15,-31.66]
v12=[-31.742,-31.625,-31.25,-32.3,-31.23,-31.2,-31.34,-31.51,-31.12,-31.56,-32.17,-32.1,-31.95,-31.96,-32.23,-31.25,-31.06,-31.55]
v13=[-60.45,-60.46,-60.29,-60.4,-60.8,-60.1,-60.9,-61.43,-60.77,-61.28,-60.78,-60.67,-60.7,-60.95,-60.7,-60.3,-60.75,-60.15]
v14=[-60.36,-60.385,-60.02,-60.3,-60.68,-60.0,-60.7,-61.31,-60.59,-61.1,-60.6,-60.55,-60.55,-60.85,-60.6,-60.2,-60.55,-60.05]
v15=[-31.785,-31.685,-31.365,-32.36,-31.286,-31.25,-31.46,-31.66,-31.23,-31.7,-32.3,-32.21,-32.1,-32.06,-32.1,-31.3,-31.15,-31.66]
v16=[-31.785,-31.685,-31.365,-32.36,-31.276,-31.265,-31.4,-31.57,-31.2,-31.7,-32.3,-32.21,-32.1,-32.06,-32.1,-31.35,-31.15,-31.66]
horas=['1743', '1843', '2143', '1753', '1913', '1813', '2114', '0244', '0124', '0114', '2114', '2244', '2344', '2354', '0134', '2304', '2344', '2344']
elevaciones=[2.3,3.5,3.5,1.3,0.5,1.3,3.5,1.3,2.3,2.3,2.3,1.3,1.3,0.5,2.3,1.3,3.5,3.5]
filas_panel=[0,0,0,1,1,1,2,2,2,0,0,0,1,1,1,2,2,2]
cols_panel=[0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2]
orden=['a)','b)','c)','d)','e)','f)','g)','h)','i)','j)','k)','l)','m)','n)','o)','p)','q)','r)']
index=[2,3,3,1,1,0,2,1,2,2,2,1,1,0,2,1,3,1]

def xy_a_latlon(xc,yc):
    dx= xc - 0
    dy= yc - 0
    lat_radar=-31.858334
    lon_radar=-60.539722     
    dlatc=dy/111000      
    dlonc=dx/(111000*np.cos((np.pi/180)*lat_radar))

    latc=lat_radar  + dlatc
    lonc= lon_radar + dlonc

    return latc, lonc
#a esta funcion la voy a llamar al final del script


#===========================================================================================================
# OTRAS FUNCIONES CONTENIDAS EN ESTE MODULO
#===========================================================================================================   

#From Pyart order to ARE (Azmimuth , range , elevation )

def order_variable ( radar , var_name , undef )  :  

   import numpy as np
   import numpy.ma as ma
   #import warnings 
   #import matplotlib.pyplot as plt

   #From azimuth , range -> azimuth , range , elevation 

   if radar.ray_angle_res != None   :
      #print( radar.ray_angle_res , radar.ray_angle_res == None )
      ray_angle_res = np.unique( radar.ray_angle_res['data'] )
   else                             :
      print('Warning: ray_angle_res no esta definido, estimo la resolucion en radio como la diferencia entre los primeros angulos')
      ray_angle_res = np.min( np.abs( radar.azimuth['data'][1:] - radar.azimuth['data'][0:-1] ) )
      print('La resolucion en rango estimada es: ',ray_angle_res)


   if( np.size( ray_angle_res ) >= 2 )  :
      print('Warning: La resolucion en azimuth no es uniforme en los diferentes angulos de elevacion ')
      print('Warning: El codigo no esta preparado para considerar este caso y puede producir efectos indeseados ')
   ray_angle_res=np.nanmean( ray_angle_res )

   levels=np.sort( np.unique(radar.elevation['data']) )
   nb=radar.azimuth['data'].shape[0]

   order_azimuth=np.arange(0.0,360.0,ray_angle_res) #Asuming a regular azimuth grid

   na=np.size(order_azimuth)
   ne=np.size(levels)
   nr=np.size(radar.range['data'].data) 


   var = np.ones( (nb,nr) )

   if ( var_name == 'altitude' ) :
       var[:]=radar.gate_altitude['data']  
   elif( var_name == 'longitude' ) :
       var[:]=radar.gate_longitude['data']
   elif( var_name == 'latitude'  ) :
       var[:]=radar.gate_latitude['data']
   elif( var_name == 'x' )         :
       var[:]=radar.gate_x['data']
   elif( var_name == 'y' )         : 
       var[:]=radar.gate_y['data']
   else  :
       var[:]=radar.fields[var_name]['data'].data


   #Allocate arrays
   order_var    =np.zeros((na,nr,ne))
   order_time   =np.zeros((na,ne)) 
   azimuth_exact=np.zeros((na,ne))
   order_n      =np.zeros((na,nr,ne),dtype='int')
   
   current_lev = radar.elevation['data'][0]
   ilev = np.where( levels == current_lev )[0]

   for iray in range( 0 , nb )  :   #Loop over all the rays
 
     #Check if we are in the same elevation.
     if  radar.elevation['data'][iray] != current_lev  :
         current_lev = radar.elevation['data'][iray]
         ilev=np.where( levels == current_lev  )[0]

     #Compute the corresponding azimuth index.
     az_index = np.round( radar.azimuth['data'][iray] / ray_angle_res ).astype(int)
     #Consider the case when azimuth is larger than na*ray_angle_res-(ray_angle_res/2)
     if az_index >= na   :  
        az_index = 0

     tmp_var = var[iray,:]
     undef_mask = tmp_var == undef 
     tmp_var[ undef_mask ] = 0.0
    
     order_var [ az_index , : , ilev ] = order_var [ az_index , : , ilev ] + tmp_var
     order_n   [ az_index , : , ilev ] = order_n   [ az_index , : , ilev ] + np.logical_not(undef_mask).astype(int)

     order_time[ az_index , ilev ] = order_time[ az_index , ilev ] + radar.time['data'][iray]
     azimuth_exact[ az_index , ilev ] = azimuth_exact[ az_index , ilev ] + radar.azimuth['data'][ iray ]

   order_var[ order_n > 0 ] = order_var[ order_n > 0 ] / order_n[ order_n > 0 ]
   order_var[ order_n == 0] = undef

   return order_var , order_azimuth , levels , order_time , azimuth_exact

def order_variable_inv (  radar , var , undef )  :

   import numpy as np
   
   #From azimuth , range , elevation -> azimuth , range

   na=var.shape[0]
   nr=var.shape[1]
   ne=var.shape[2]

   nb=radar.azimuth['data'].shape[0]

   levels=np.sort( np.unique(radar.elevation['data']) )

   if radar.ray_angle_res != None   :
      #print( radar.ray_angle_res , radar.ray_angle_res == None )
      ray_angle_res = np.unique( radar.ray_angle_res['data'] )
   else                             :
      print('Warning: ray_angle_res no esta definido, estimo la resolucion en radio como la diferencia entre los primeros angulos')
      ray_angle_res = np.min( np.abs( radar.azimuth['data'][1:] - radar.azimuth['data'][0:-1] ) )
      print('La resolucion en rango estimada es: ',ray_angle_res)

   if( np.size( ray_angle_res ) >= 2 )  :
      print('Warning: La resolucion en azimuth no es uniforme en los diferentes angulos de elevacion ')
      print('Warning: El codigo no esta preparado para considerar este caso y puede producir efectos indesaedos ')
   ray_angle_res=np.nanmean( ray_angle_res )

   current_lev = radar.elevation['data'][0]
   ilev = np.where( levels == current_lev  )[0]

   output_var = np.zeros((nb,nr) )
   output_var[:] = undef

   for iray in range( 0 , nb )  :   #Loop over all the rays

      #Check if we are in the same elevation.
      if  radar.elevation['data'][iray] != current_lev  :
          current_lev = radar.elevation['data'][iray]
          ilev=np.where( levels == current_lev  )[0]

      #Compute the corresponding azimuth index.
      az_index = np.round( radar.azimuth['data'][iray] / ray_angle_res ).astype(int)
      #Consider the case when azimuth is larger than na*ray_angle_res-(ray_angle_res/2)
      if az_index >= na   :
         az_index = 0

      output_var[ iray , : ] = var[ az_index , : , ilev ]

   return output_var


def local_mean( array , kernel_x , kernel_y , undef ) :
    #Asumimos que hay condiciones ciclicas en el axis 0 pero no en el 1.
    #array es el array de datos de entrada
    #kernel_x es cual es el desplazamiento maximo (hacia cada lado) en la direccion de x
    #kernel_y es cual es el desplazamiento maximo (hacia cada lado) en la direccion de y
    #undef son los valores invalidos en el array de entrada.

    [nx,ny]=np.shape(array)
    arraym = np.zeros( np.shape(array) )
    countm = np.zeros( np.shape(array) )
    for ix in range(-kernel_x,kernel_x+1) :
        for iy in range(-kernel_y,kernel_y +1) :
          tmp_array = np.zeros( np.shape(array) )
          if iy > 0 :
             tmp_array[:,0+iy:] = array[:,0:-iy]
          if iy == 0 :
             tmp_array = np.copy(array)
          if iy < 0 :
             tmp_array[:,0:iy] = array[:,-iy:]
          tmp_array=np.roll( tmp_array , ix , axis=0 )
          mask = tmp_array != undef
          arraym[ mask ] = arraym[mask] + tmp_array[mask]
          countm[ mask ] = countm[mask] + 1
    mask = countm > 0
    arraym[mask] = arraym[mask] / countm[mask]
    arraym[~mask] = undef 

    return arraym


fig=plt.figure(figsize=(40,40))
anchos=[1,1,1] #lastres col tienen el mismo ancho
altos=[1,1,1,0.25] #las primeras 3 filas tienen el mismo ancho y la ultima 1cuarto (es la barra)
spec=fig.add_gridspec(ncols=3,nrows=4, width_ratios=anchos, height_ratios=altos)
for k in range(9):    

    file= direc +  archivos[k]
    anio = file[83:87] 
    mes = file[87:89] 
    dia = file[89:91] 
    hora_inicio = file[92:96]
    hora_fin = file[115:119]

    fecha=anio+'.'+mes+'.' + dia
    hora=hora_inicio+'_to_'+hora_fin
    
    radar = pyart.io.read(file)
    display = pyart.graph.RadarMapDisplay(radar)
    sweeps=radar.sweep_number
    undef = radar.fields['dBZ']['data'].fill_value
    
    print('Dealiasing')
    for i in range(0,len(sweeps)):
        levels=np.unique(radar.elevation['data'])#toma los elementos no repetidos de un array y los ordena.
        ele=str(levels[i]) #vector que corresponde al vector levels en cada elevacion i
        datos=radar.extract_sweeps([i])
        V=datos.fields['V']['data'] #extrae, si es posible, la variable viento radial en cada elevacion a un tiempo dado
        # Definimos el tamaño del kernel para la convolusion
        #Aplicamos un filtro para prevenir el aliasing
        nyq = radar.instrument_parameters['nyquist_velocity']['data'][0] #extraemos el valor de velocidad nyquist
        #vel_texture = pyart.retrieve.calculate_velocity_texture(radar, vel_field='V', wind_size=2, nyq=39.9)
        #radar.add_field('velocity_texture', vel_texture, replace_existing=True)
        #gatefilter = pyart.filters.GateFilter(radar)
        #gatefilter.exclude_above('velocity_texture', 2)
        Vc = pyart.correct.dealias_region_based(radar, vel_field='V', nyquist_vel=nyq,centered=True ) #, gatefilter=gatefilter)
        radar.add_field('Vc', Vc, replace_existing=True)
    
    #PASO LAS VARIABLES DEL FORMATO CFRADIAL A UN ARRAY AZIMUTH,RANGO,ELEVACION
    print('Ordenando variables')
    [dbz3d , azimuth , levels , time , azimuthe ] = order_variable ( radar , 'dBZ' , undef )  
    [v3d   , azimuth , levels , time , azimuthe ] = order_variable ( radar , 'Vc' , undef )  
    [x3d , azimuth , levels , time , azimuthe ] = order_variable ( radar , 'x' , undef )  
    [y3d , azimuth , levels , time , azimuthe ] = order_variable ( radar , 'y' , undef ) 
    [z3d , azimuth , levels , time , azimuthe ] = order_variable ( radar , 'altitude' , undef )
    
    rango=radar.range['data']
    
    #ELIMINAMOS OUTLIERS LOCALES. LOS PIXELES CUYO VALOR ESTA MUY LEJOS DE UNA MEDIA LOCAL
    #CALCULADA SOBRE UN DOMINIO DE 5 x 10 (radiales y azimuths) SON ELIMINADOS.
    #EN OCASIONES PUEDE QUE SEA NECESARIO HACER MAS DE UNA PASADA DE ESTE TIPO DE FILTROS PARA 
    #ELIMINAR OUTLIERS SUCESIVOS (grupos de 2/3 pixeles con valores anomalos)
    print('Sacando outliers')
    vm3d = np.zeros( np.shape(v3d) )
    vd3d = np.zeros( np.shape(v3d) )
        
    for ilev , my_lev in enumerate(levels) :
        # Definimos el kernel en size (esto es para calcular la media movil)
        kernel_x = 2
        kernel_y = 2 
        v2d = np.copy(v3d[:,:,ilev])
        # Esta funcion calcula la media local (una convolucion). La razon por la que no usamos
        #las funciones de convolucion es porque este campo tiene undef.
        vmean = local_mean( v2d , kernel_x , kernel_y , undef )
    
        
        mask = np.abs(v2d-vmean) > 10.0   #Si un pixel supera en 10 m/s a la media local lo elimino.
        v2d[mask] = undef
        v3d[:,:,ilev] = np.copy(v2d)
        vm3d[:,:,ilev] = np.copy(vmean)
        vd3d[:,:,ilev] = np.copy(v2d-vmean)
    
    xc=xce[k]
    yc=yce[k]
    dx=deltax[k]
    dy=deltay[k]    
    
    for ilev , my_lev in enumerate(levels):
    #for ilev in range(6,7) :
        ele=str(levels[ilev])
        x2d = x3d[:,:,ilev]
        y2d = y3d[:,:,ilev]
        z2d = z3d[:,:,ilev]
    
        v2d = np.copy(v3d[:,:,ilev])
        vm2d = np.copy(vm3d[:,:,ilev])
        dbz2d = np.copy(dbz3d[:,:,ilev])
    
        dbz2d[dbz2d==undef]=np.nan
        v2d[v2d==undef]=np.nan
        vm2d[vm2d==undef]=np.nan
        #Defino una mascara que me permite calcular las propiedades del campo de viento sobre la region de interes.
        mask = np.logical_and(x2d > xc-dx/2 , x2d < xc+dx/2 ) 
        mask = np.logical_and( mask , y2d > yc-dy/2 )
        mask = np.logical_and( mask , y2d < yc+dy/2)
        mean_vr = np.nanmean( v2d[mask] )
    
        
    x2d = x3d[:,:,0]
    y2d = y3d[:,:,0]
    lats=np.zeros((360,480))
    lons=np.zeros((360,480))
    for i in range(360):
        for j in range(480):
            lats[i,j]=xy_a_latlon(x2d[i,j],y2d[i,j])[0]
            lons[i,j]=xy_a_latlon(x2d[i,j],y2d[i,j])[1]

    dbz2d = np.copy(dbz3d[:,:,0])
    nyq = radar.instrument_parameters['nyquist_velocity']['data'][0]
    mascara_v = np.logical_or( vm3d > nyq , vm3d < -nyq )
    vm3d[mascara_v] = np.nan #NOS ASEGURAMOS DE NO CALCULAR CORTANTES EN LUGARES DONDE EL VIENTO NO ES VALIDO.
    
    #Creo la imagen PPI
    ax=fig.add_subplot(spec[filas_panel[k],cols_panel[k]])

    plt.pcolor(lons,lats,vm3d[:,:,index[k]],cmap='pyart_balance',vmin=-25,vmax=25)#;plt.colorbar()
    plt.contour(lons,lats,dbz2d,levels=[30,40,50],colors='k')
    plt.axis([xc-dx/2 , xc+dx/2 , yc-dy/2 , yc+dy/2])
    plt.tick_params(axis='both', labelsize=30)
    plt.grid()
    plt.plot(-60.539722, -31.858334, marker='o', color='r', markersize=20)
        
    plt.plot([v1[k],v2[k]],[v3[k],v4[k]],linewidth=9, color='r', linestyle='-.')
    plt.plot([v5[k],v6[k]],[v7[k],v8[k]],linewidth=9, color='r', linestyle='-.')
    plt.plot([v9[k],v10[k]],[v11[k],v12[k]],linewidth=9, color='r', linestyle='-.')
    plt.plot([v13[k],v14[k]],[v15[k],v16[k]],linewidth=9, color='r', linestyle='-.')
    plt.title(orden[k] +' ' + fecha +' '+ horas[k] + ' UTC' +' '+ str(elevaciones[k]) +' Deg', fontsize=37)
    if k==0 or k==3 or k==6:
        plt.ylabel('Latitude [°S]', fontsize=35)
    if k==6 or k==7 or k==8:
        plt.xlabel('Longitude [°W]', fontsize=35)
    plt.pcolor(lons,lats,vm3d[:,:,index[k]],cmap='pyart_balance',vmin=-25,vmax=25)#;plt.colorbar()
    plt.grid(True)

axes=plt.subplot(spec[3,:])
cbar=plt.colorbar(cax=axes, orientation='horizontal', shrink=0.6, ticks=np.arange(-25,26,5))
cbar.ax.tick_params(labelsize=22)
cbar.set_label(label='Corrected Radial Velocity [m/s]', fontsize=50)
plt.savefig(salida + "panel1_cb.png" , dpi=300)

fig=plt.figure(figsize=(40,40))
anchos=[1,1,1] #lastres col tienen el mismo ancho
altos=[1,1,1,0.25] #las primeras 3 filas tienen el mismo ancho y la ultima 1cuarto (es la barra)
spec=fig.add_gridspec(ncols=3,nrows=4, width_ratios=anchos, height_ratios=altos)
for l in range(9):    

    file= direc +  archivos[9+l]
    anio = file[83:87] 
    mes = file[87:89] 
    dia = file[89:91] 
    hora_inicio = file[92:96]
    hora_fin = file[115:119]

    fecha=anio+'.'+mes+'.' + dia
    hora=hora_inicio+'_to_'+hora_fin
    
    radar = pyart.io.read(file)
    sweeps=radar.sweep_number
    undef = radar.fields['dBZ']['data'].fill_value
    
    print('Dealiasing')
    for i in range(0,len(sweeps)):
        levels=np.unique(radar.elevation['data'])#toma los elementos no repetidos de un array y los ordena.
        ele=str(levels[i]) #vector que corresponde al vector levels en cada elevacion i
        datos=radar.extract_sweeps([i])
        V=datos.fields['V']['data'] #extrae, si es posible, la variable viento radial en cada elevacion a un tiempo dado
        # Definimos el tamaño del kernel para la convolusion
        #Aplicamos un filtro para prevenir el aliasing
        nyq = radar.instrument_parameters['nyquist_velocity']['data'][0] #extraemos el valor de velocidad nyquist
        #vel_texture = pyart.retrieve.calculate_velocity_texture(radar, vel_field='V', wind_size=2, nyq=39.9)
        #radar.add_field('velocity_texture', vel_texture, replace_existing=True)
        #gatefilter = pyart.filters.GateFilter(radar)
        #gatefilter.exclude_above('velocity_texture', 2)
        Vc = pyart.correct.dealias_region_based(radar, vel_field='V', nyquist_vel=nyq,centered=True ) #, gatefilter=gatefilter)
        radar.add_field('Vc', Vc, replace_existing=True)
    
    #PASO LAS VARIABLES DEL FORMATO CFRADIAL A UN ARRAY AZIMUTH,RANGO,ELEVACION
    print('Ordenando variables')
    [dbz3d , azimuth , levels , time , azimuthe ] = order_variable ( radar , 'dBZ' , undef )  
    [v3d   , azimuth , levels , time , azimuthe ] = order_variable ( radar , 'Vc' , undef )  
    [x3d , azimuth , levels , time , azimuthe ] = order_variable ( radar , 'x' , undef )  
    [y3d , azimuth , levels , time , azimuthe ] = order_variable ( radar , 'y' , undef ) 
    [z3d , azimuth , levels , time , azimuthe ] = order_variable ( radar , 'altitude' , undef )
    
    rango=radar.range['data']
    
    #ELIMINAMOS OUTLIERS LOCALES. LOS PIXELES CUYO VALOR ESTA MUY LEJOS DE UNA MEDIA LOCAL
    #CALCULADA SOBRE UN DOMINIO DE 5 x 10 (radiales y azimuths) SON ELIMINADOS.
    #EN OCASIONES PUEDE QUE SEA NECESARIO HACER MAS DE UNA PASADA DE ESTE TIPO DE FILTROS PARA 
    #ELIMINAR OUTLIERS SUCESIVOS (grupos de 2/3 pixeles con valores anomalos)
    print('Sacando outliers')
    vm3d = np.zeros( np.shape(v3d) )
    vd3d = np.zeros( np.shape(v3d) )
        
    for ilev , my_lev in enumerate(levels) :
        # Definimos el kernel en size (esto es para calcular la media movil)
        kernel_x = 2
        kernel_y = 2 
        v2d = np.copy(v3d[:,:,ilev])
        # Esta funcion calcula la media local (una convolucion). La razon por la que no usamos
        #las funciones de convolucion es porque este campo tiene undef.
        vmean = local_mean( v2d , kernel_x , kernel_y , undef )
    
        
        mask = np.abs(v2d-vmean) > 10.0   #Si un pixel supera en 10 m/s a la media local lo elimino.
        v2d[mask] = undef
        v3d[:,:,ilev] = np.copy(v2d)
        vm3d[:,:,ilev] = np.copy(vmean)
        vd3d[:,:,ilev] = np.copy(v2d-vmean)
    
    xc=xce[9+l]
    yc=yce[9+l]
    dx=deltax[9+l]
    dy=deltay[9+l]    
    
    for ilev , my_lev in enumerate(levels):
    #for ilev in range(6,7) :
        ele=str(levels[ilev])
        x2d = x3d[:,:,ilev]
        y2d = y3d[:,:,ilev]
        z2d = z3d[:,:,ilev]
    
        v2d = np.copy(v3d[:,:,ilev])
        vm2d = np.copy(vm3d[:,:,ilev])
        dbz2d = np.copy(dbz3d[:,:,ilev])
    
        dbz2d[dbz2d==undef]=np.nan
        v2d[v2d==undef]=np.nan
        vm2d[vm2d==undef]=np.nan
        #Defino una mascara que me permite calcular las propiedades del campo de viento sobre la region de interes.
        mask = np.logical_and(x2d > xc-dx/2 , x2d < xc+dx/2 ) 
        mask = np.logical_and( mask , y2d > yc-dy/2 )
        mask = np.logical_and( mask , y2d < yc+dy/2)
        mean_vr = np.nanmean( v2d[mask] )
    
        
    x2d = x3d[:,:,0]
    y2d = y3d[:,:,0]
    lats=np.zeros((360,480))
    lons=np.zeros((360,480))
    for i in range(360):
        for j in range(480):
            lats[i,j]=xy_a_latlon(x2d[i,j],y2d[i,j])[0]
            lons[i,j]=xy_a_latlon(x2d[i,j],y2d[i,j])[1]

    dbz2d = np.copy(dbz3d[:,:,0])
    nyq = radar.instrument_parameters['nyquist_velocity']['data'][0]
    mascara_v = np.logical_or( vm3d > nyq , vm3d < -nyq )
    vm3d[mascara_v] = np.nan #NOS ASEGURAMOS DE NO CALCULAR CORTANTES EN LUGARES DONDE EL VIENTO NO ES VALIDO.
    
    #Creo la imagen PPI
    ax=fig.add_subplot(spec[filas_panel[l],cols_panel[l]])

    plt.pcolor(lons,lats,vm3d[:,:,index[9+l]],cmap='pyart_balance',vmin=-25,vmax=25)#;plt.colorbar()
    plt.contour(lons,lats,dbz2d,levels=[30,40,50],colors='k')
    plt.axis([xc-dx/2 , xc+dx/2 , yc-dy/2 , yc+dy/2])
    plt.tick_params(axis='both', labelsize=30)
    plt.grid()
    plt.plot(-60.539722, -31.858334, marker='o', color='r', markersize=20)

    plt.plot([v1[9+l],v2[9+l]],[v3[9+l],v4[9+l]],linewidth=9, color='r', linestyle='-.')
    plt.plot([v5[9+l],v6[9+l]],[v7[9+l],v8[9+l]],linewidth=9, color='r', linestyle='-.')
    plt.plot([v9[9+l],v10[9+l]],[v11[9+l],v12[9+l]],linewidth=9, color='r', linestyle='-.')
    plt.plot([v13[9+l],v14[9+l]],[v15[9+l],v16[9+l]],linewidth=9, color='r', linestyle='-.')
    plt.title(orden[9+l] +' ' + fecha +' '+ horas[9+l] + ' UTC' +' '+ str(elevaciones[9+l]) +' Deg', fontsize=37)
    if l==0 or l==3 or l==6:
        plt.ylabel('Latitude [°S]', fontsize=35)
    if l==6 or l==7 or l==8:
        plt.xlabel('Longitude [°W]', fontsize=35)
    plt.pcolor(lons,lats,vm3d[:,:,index[9+l]],cmap='pyart_balance',vmin=-25,vmax=25)#;plt.colorbar()
    plt.grid(True)

axes=plt.subplot(spec[3,:])
cbar=plt.colorbar(cax=axes, orientation='horizontal', shrink=0.6, ticks=np.arange(-25,26,5))
cbar.ax.tick_params(labelsize=22)
cbar.set_label(label='Corrected Radial Velocity [m/s]', fontsize=50)
plt.savefig(salida + "panel2_cb.png" , dpi=300)