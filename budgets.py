#!usr/bin/env python3

import numpy as np
import xarray as xr
import pandas as pd
from calc import g, R_e, R_d, absolute_vorticity 

#----------------------------
# budget term calculations
#----------------------------

# note these are budget terms and require transport of a quntity phi

def flux_divergence(u, v, phi):

    lats_rad = np.deg2rad(u.latitude) # in radians
    
    adv_x, adv_y = u * phi, v * phi
    flux_div = 1/(R_e * np.cos(lats_rad)) * (adv_x.differentiate('longitude') * 180/np.pi + (adv_y * np.cos(lats_rad)).differentiate('latitude') * 180/np.pi)

    return flux_div 


def divergence(u, v, phi):

    lats_rad = np.deg2rad(u.latitude) # in radians

    div = phi * (1/(R_e * np.cos(lats_rad)) * (u.differentiate('longitude') * 180/np.pi + (v * np.cos(lats_rad)).differentiate('latitude') * 180/np.pi))
    
    return div


def advection(u, v, phi):

    lats_rad = np.deg2rad(u.latitude) # in radians
    coslat = np.cos(lats_rad)
    dy = np.deg2rad(R_e)
    dx = np.deg2rad(R_e * coslat)

    adv = u * phi.differentiate('longitude')/dx + v * phi.differentiate('latitude')/dy
    
    return adv


def vertical_advection(w, phi):

    vert_flux_div = (w * phi).differentiate('level') / 100
    vert_div = phi * w.differentiate('level') / 100
    vert_adv = w * phi.differentiate('level') / 100

    return vert_flux_div, vert_div, vert_adv


def advection_lineint(u, v, phi, column_integrate=False):
    """
    Calculates advection through boundaries of a box, splitting into each edge
    """

    adv_x = u * phi
    adv_y = v * phi

    adv, adv_south, adv_east, adv_north, adv_west = boundary_flux(adv_x, adv_y, column_integrate=column_integrate)
    
    return adv, adv_south, adv_east, adv_north, adv_west


def boundary_flux(phi_x, phi_y, column_integrate=False):
    """
    Calculates flux through a rectangular boundary
    phi_x and phi_y are the vector components of the flux, where the flux can be simple advection (e.g. u*phi)
    or something more calculated like "Z" in the circulation budget which also has tilting/friction.
    """
    lats = np.deg2rad(phi_x.latitude) # in radians
    coslat = np.cos(lats)

    min_lon, max_lon = phi_x['longitude'].min(), phi_x['longitude'].max()
    min_lat, max_lat = phi_x['latitude'].min(), phi_x['latitude'].max()
    
    # convert lon/lat to x/y
    dy = np.deg2rad(R_e)
    dx = np.deg2rad(R_e * coslat)
    dx_north = dx.sel(latitude=max_lat)
    dx_south = dx.sel(latitude=min_lat)

    box_area = R_e**2 * np.deg2rad(max_lon - min_lon) * (np.sin(np.deg2rad(max_lat)) - np.sin(np.deg2rad(min_lat)))#get_box_area(phi)

    # note the east/west fluxes have an extra -ve sign since ERA5 data has latitude ordered from 90 to -90 (reverse)
    flux_south = ((phi_y.sel(latitude=min_lat).integrate('longitude') * dx_south)) / box_area # n = (0, 1)
    flux_east  = ((phi_x.sel(longitude=max_lon).integrate('latitude') * dy)) / box_area # n = (-1, 0)
    flux_north = -((phi_y.sel(latitude=max_lat).integrate('longitude') * dx_north)) / box_area # n = (0, -1)
    flux_west  = -((phi_x.sel(longitude=min_lon).integrate('latitude') * dy)) / box_area # n = (1, 0)
    
    total_flux = flux_south + flux_east + flux_north + flux_west

    if column_integrate:
        total_flux = total_flux.integrate('level') / g * 100
        flux_south = flux_south.integrate('level') / g * 100
        flux_east = flux_east.integrate('level') / g * 100
        flux_north = flux_north.integrate('level') / g * 100
        flux_west = flux_west.integrate('level') / g * 100
    
    return total_flux, flux_south, flux_east, flux_north, flux_west
    

def tilting(u, v, w, rv):
    """
    For relative vorticity/circulation budget only
    """
    # THIS NEEDS TO BE DONE IN SPHERICAL COORDS (not sure how?)

    lats_rad = np.deg2rad(u.latitude) # in radians
    coslat = np.cos(lats_rad)

    # add coordinates to the data in radians 
    u = add_rad_coords(u)
    v = add_rad_coords(v)
    w = add_rad_coords(w)
    rv = add_rad_coords(rv)
    
    dw_dy = w.differentiate('latrad') / R_e
    dv_dp = v.differentiate('level') / 100 # data in hPa, convert to Pa
    rv_x = dw_dy - dv_dp # x component of relative vorticity
    
    dw_dx = w.differentiate('lonrad') / (R_e * coslat)
    du_dp = u.differentiate('level') / 100
    rv_y = du_dp - dw_dx # y component of relative vorticity
    
    tilt_x = -w * rv_x
    tilt_y = -w * rv_y
    
    # flux divergence of the above tilting vector
    return 1/(R_e * coslat) * ((tilt_y * coslat).differentiate('latrad') + tilt_x.differentiate('lonrad'))

def tilting_comps(u, v, w, rv):
    """
    For relative vorticity/circulation budget only
    COMPONENTS, not the budget term
    """

    lats_rad = np.deg2rad(u.latitude) # in radians
    coslat = np.cos(lats_rad)

    u = add_rad_coords(u)
    v = add_rad_coords(v)
    w = add_rad_coords(w)
    rv = add_rad_coords(rv)
    
    dw_dy = w.differentiate('latrad') / R_e
    dv_dp = v.differentiate('level') / 100 # data in hPa, convert to Pa
    rv_x = dw_dy - dv_dp
    
    dw_dx = w.differentiate('lonrad') / (R_e * coslat)
    du_dp = u.differentiate('level') / 100
    rv_y = du_dp - dw_dx
    
    tilt_x = -w * rv_x
    tilt_y = -w * rv_y

    return tilt_x, tilt_y

def friction(u_param, v_param):

    lats_rad = np.deg2rad(u_param.latitude) # in radians
    coslat = np.cos(lats_rad)

    u_param = add_rad_coords(u_param)
    v_param = add_rad_coords(v_param)

    # take the curl of the friction for the budget 
    F_fri_x = -v_param
    F_fri_y = u_param

    F_fri = 1/(R_e * coslat) * ((F_fri_y * coslat).differentiate('latrad') + F_fri_x.differentiate('lonrad'))
    
    return F_fri

#----------------------
# circulation budget
#----------------------


def circulation_budget(u, v, w, rv, calc_fric=False, u_param=None, v_param=None, column_integrate=False, area_mean = True):
    """
    Main circulation budget function
    """

    av = absolute_vorticity(rv)
    
    dav_dt = av.differentiate('time', datetime_unit = 's')
    flux_div = flux_divergence(u, v, av)
    div = divergence(u, v, av)
    adv = advection(u, v, av)
    tilt = tilting(u, v, w, rv)
    
    # residual
    res = dav_dt + flux_div + tilt
    
    # calculate friction if necessary
    if calc_fric:
        if u_param is None or v_param is None:
            raise Exception('if friction is specified, need to add u_param and v_param as arguments')
        fric = friction(u_param, v_param)
        res_fric = res + fric
    else:
        fric = xr.zeros_like(dav_dt)
        res_fric = xr.zeros_like(dav_dt)

    if column_integrate:
        dav_dt = dav_dt.integrate('level') / g * 100 # could use tcwv I guess
        flux_div = flux_div.integrate('level') / g * 100
        div = div.integrate('level') / g * 100
        adv = adv.integrate('level') / g * 100
        tilt = tilt.integrate('level') / g * 100
        res = res.integrate('level') / g * 100
        fric = fric.integrate('level') / g * 100
        res_fric = res_fric.integrate('level') / g * 100

    if area_mean:
        dav_dt = spherical_area_mean(dav_dt)
        flux_div = spherical_area_mean(flux_div)
        div = spherical_area_mean(div)
        adv = spherical_area_mean(adv)
        tilt = spherical_area_mean(tilt)
        res = spherical_area_mean(res)
        fric = spherical_area_mean(fric)
        res_fric = spherical_area_mean(res_fric)

    return dav_dt, flux_div, div, adv, tilt, res, fric, res_fric


def circulation_budget_layer(u, v, w, rv, min_level, max_level, area_mean=True):
    """
    calculate a mass-weightyed circulation budget
    basically just an auxilliary function which cuts the relevant levels and passes to main function
    """

    u = u.sel(level=slice(min_level, max_level))
    v = v.sel(level=slice(min_level, max_level))
    w = w.sel(level=slice(min_level, max_level))
    rv = rv.sel(level=slice(min_level, max_level))

    return circulation_budget(u, v, w, rv, column_integrate=True, area_mean=area_mean)


def absolute_vorticity_flux(u, v, w, rv, fric=False, u_tend=None, v_tend=None):
    """
    The calculates "Z" in the circulation budget.
    Note it's not just a horizontal flux
    """

    av = absolute_vorticity(rv)
    
    adv_x, adv_y = u * av, v * av

    tilt_x, tilt_y = tilting_comps(u, v, w, rv)

    # parameterised friction
    if fric:
        # friction part of Z is the cross product of the param terms
        fric_x = -v_tend
        fric_y = u_tend
    else:
        fric_x = 0
        fric_y = 0

    flux_x = adv_x + tilt_x + fric_x
    flux_y = adv_y + tilt_y + fric_y

    return flux_x, flux_y, adv_x, adv_y, tilt_x, tilt_y, fric_x, fric_y

#---------------------
# Moisture budget
#---------------------

def moisture_budget(u, v, w, q, E, P, column_integrate=False, area_mean=True):
    """
    Budget for a single level. Note convergence & vertical advection cancel out.
    """

    dq_dt = q.differentiate('time', datetime_unit = 's')
    flux_div = flux_divergence(u, v, q)
    div = divergence(u, v, q)
    adv = advection(u, v, q)
    vert_flux_div, vert_div, vert_adv = vertical_advection(w, q)
    E_mean = E
    P_mean = P

    if column_integrate:
        dq_dt = dq_dt.integrate('level') / g * 100 # could use tcwv I guess
        flux_div = flux_div.integrate('level') / g * 100
        div = div.integrate('level') / g * 100
        adv = adv.integrate('level') / g * 100
        vert_flux_div = vert_flux_div.integrate('level') / g * 100
        vert_div = vert_div.integrate('level') / g * 100
        vert_adv = vert_adv.integrate('level') / g * 100

    if area_mean:
        dq_dt = spherical_area_mean(dq_dt)
        flux_div = spherical_area_mean(flux_div)
        div = spherical_area_mean(div)
        adv = spherical_area_mean(adv)
        vert_flux_div = spherical_area_mean(vert_flux_div)
        vert_div = spherical_area_mean(vert_div)
        vert_adv = spherical_area_mean(vert_adv)
        E_mean = spherical_area_mean(E)
        P_mean = spherical_area_mean(P)


    return dq_dt, flux_div, div, adv, vert_flux_div, vert_div, vert_adv, E_mean, P_mean


def moisture_budget_layer(u, v, w, q, E, P, min_level, max_level, area_mean = True):

    """
    calculates budget in a layer
    Same as regular budget, but need to do vertical fluxes into/out of layer too.
    """

    u = u.sel(level=slice(min_level, max_level))
    v = v.sel(level=slice(min_level, max_level))
    w = w.sel(level=slice(min_level, max_level))
    q = q.sel(level=slice(min_level, max_level))

    dq_dt, flux_div, div, adv, _, _, _, E_mean, P_mean = moisture_budget(u, v, w, q, E, P, column_integrate=True)

    v_flux = w * q
    net_vert_adv = (v_flux.sel(level=max_level) - v_flux.sel(level=min_level)) / g

    if area_mean:
        net_vert_adv = spherical_area_mean(net_vert_adv)
        
    return dq_dt, flux_div, div, adv, net_vert_adv, E_mean, P_mean


#------------------------
# Supporting functions
#------------------------

def add_rad_coords(data):
    """
    Adds new coords for latitude/longitude in radiations for differentiation/integration
    """
    data = data.assign_coords(lonrad = ('longitude', np.deg2rad(data.longitude.values)))
    data = data.assign_coords(latrad = ('latitude', np.deg2rad(data.latitude.values)))
    return data


# def spherical_winds(u, v):
#     """
#     Converts u=dx/dt, v=dy/dt to spherical counterparts u_lon = dlambda/dt, u_lat = dphi/dt
#     """
#     coslat = np.cos(np.deg2rad(u.latitude))
#     u_lon = 1/(R_e*coslat) * u
#     u_lat = 1/(R_e) * v
#     return u_lon, u_lat

    
def moving_average(data, n=24):
    data = xr.DataArray(data, dims = data.dims)#['time'])
    data = data.fillna(0)
    data = data.rolling(time=n, center=True).mean().dropna('time')
    return data


def spherical_area_mean(data):
    """
    takes the mean over a box while accounting for spherical geometry (scale by cos latitude)
    """
    coslat = np.cos(np.deg2rad(data.latitude))
    data_mean = data * coslat
    data_mean = -data_mean.integrate(['latitude', 'longitude']) * (np.pi/180)**2 # negative sign is because lats are ordered reversely in ERA5
    box_area = get_box_area(data)
    data_mean /= box_area # normalise by box area
    
    return data_mean


def get_box_area(data):
    """
    get the size of the box in spherical coords
    """
    min_lon, max_lon = data['longitude'].min(), data['longitude'].max()
    min_lat, max_lat = data['latitude'].min(), data['latitude'].max()

    area = np.deg2rad(max_lon - min_lon) * (np.sin(np.deg2rad(max_lat)) - np.sin(np.deg2rad(min_lat)))
    return area

