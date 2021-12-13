#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameters for the Building Automation System (BAS) case. There parameters
were obtained from the following reference:
    
    N. Cauchi and A. Abate.  Benchmarks for cyber-physical systems: A modular 
    model libraryfor building automation systems. In ADHS, volume 51 of 
    IFAC-PapersOnLine, pages 49–54. Elsevier, 2018.
    
The GitLab repository of these benchmarks can be accessed via the URL below:
    https://github.com/natchi92/BASBenchmarks
"""

class parameters(object):
    
    def __init__(self):
        
        """
        Adopted from the BAC Benchmarks by
            N. Cauchi and A. Abate.  Benchmarks for cyber-physical systems: A 
            modular model libraryfor building automation systems. In ADHS, 
            volume 51 of IFAC-PapersOnLine, pages 49–54. Elsevier, 2018.    
        
        For the original GitHub repository, see:
            https://github.com/natchi92/BASBenchmarks
        """
        
        #--------- Material properties   -------
        
        self.Materials = {'air': {}, 'water': {}, 'concrete': {}}
        
        self.Materials['air']['Cpa']   = 1204/1e3       # thermal capacity of air [J/kgK]
        self.Materials['air']['rhoa']  = 1.2            # density of air [kg/m3]
        self.Materials['water']['Cpw'] = 4180/1e3       # thermal capacity of water [J/kgK]
        self.Materials['water']['rhow']= 1000/1e3       # density of water [kg/m3]
        self.Materials['concrete']['Cn'] = 880/1e3      # thermal capacity of concrete [J/kgK]
        self.Materials['concrete']['rhon'] = 2.4e3/1e3  # density of concrete [kg/m3]
        
        
        #--------- Boiler Parameters  -------
        
        self.Boiler = {}
        
        # Gas Boiler, AMBI simulator
        self.Boiler['taub']     = 60*5                  # Time constant of boiler [s]
        self.Boiler['kb']       = 348.15 - 273.15       # Steady-state temperature of the boiler (75degC) [K]
        self.Boiler['Tswbss']   = 75
        self.Boiler['sigma']    = 0.5
        
        self.Splitter = {'uv': 0}                       # Water flow Splitter [-]
        
        #--------- Fan Coil Unit (FCU) Parameters -------
        
        self.FCU = {}
        
        self.FCU['Vfcu']   = 0.06                       # Volume of fcu [m3]
        self.FCU['Afcu']   = 0.26                       # Contact area for conduction in fcu [m2]
        self.FCU['mfcu_l'] = 0.16                       # FCU mass air flow when fan is in low mode [m3/s]
        self.FCU['mfcu_m'] = 0.175                      # FCU mass air flow when fan is in med mode [m3/s]
        self.FCU['mfcu_h'] = 0.19                       # FCU mass air flow when fan is in high mode [m3/s]
        
        #--------- Air Handling Unit (AHU) Parameters     -------#
        
        self.AHU = {'sa': {}, 'rw': {}}
        
        self.AHU['sa']['k0']        = 2.0167e-05        # Constant k0 = Cpa/(Cpa*rho_a*Vahu)
        self.AHU['sa']['k1']        = 0.0183            # Constant k1 = (UA)a/(Cpa*rho_a*Vahu)
        self.AHU['rw']['k0']        = 0.0109            # Constant k0 = Cpw/(Cpw*rho_h*Vahu)
        self.AHU['rw']['k1']        = 0.0011            # Constant k1 = (UA)a/(Cpw*rho_h*Vahu)
        self.AHU['rw']['alpha3']    = 0.0011            # Constant alpha_3
        self.AHU['rw']['Trwss']     = 35                # AHU return water steady state temperature [deg C]
        self.AHU['w_a']             = 1/60              # Nominal rate of water flow [m3/min]
        self.AHU['w_max']           = 10/60             # Max rate of water flow [m3/min]
        self.AHU['sa']['sigma']     =0.1
        self.AHU['rw']['sigma']     =0.1
        
        self.Mixer = {'um': 0.5}

        #--------- Radiator Parameters-------#
        
        self.Radiator = {'Zone1': {}, 'Zone2': {}, 'rw': {}}
        
        self.Radiator['k0']             = 0.0193        # Constant k0 = Cpw/(Cpw*rho_h*Vr)
        self.Radiator['k1']             = 8.900e-04     # Constant k1 = (UA)r/(Cpw*rho_h*Vr)
        self.Radiator['w_r']            = 5/60          # Nominal rate of water flow [m3/min]
        self.Radiator['w_max']          = 10/60         # Max rate of water flow [m3/min]
        self.Radiator['Zone1']['Prad']  = 800/1000
        self.Radiator['Zone2']['Prad']  = 600/1000      # Rated output power of radiator [kW]
        self.Radiator['alpha2']         = 0.0250        # Coefficients of Qrad [-]
        self.Radiator['alpha1']         = -0.02399
        self.Radiator['Trwrss']         = 35            # Radiator steady state temperature [deg C]
        self.Radiator['rw']['sigma']    = 0.1
        
        #--------- Zone Parameters    -------#
        #--------- Zone 1             -------#
        
        self.Zone1 = {'Tz': {}, 'Te': {}}
        
        self.Zone1['Cz']    =  51.5203                  # Thermal capacitance of zone [J/kgK]
        self.Zone1['Cn']    =  52.3759                  # Thermal capacitance of neighbouring wall [J/kgK]
        self.Zone1['Rout']  =  7.20546                  # Resistance of walls connected to outside [K/W]
        self.Zone1['Rn']    =  5.4176                   # Resistance of walls connected to neighbouring wall [K/W]
        self.Zone1['mu']    =  0.000199702104146        # Q_occ coefficient in \muCO_2 + \zeta [-]
        self.Zone1['zeta']  =  0.188624079752965
        self.Zone1['alpha'] =  0.044                    # Absorptivity coefficient of wall [W/m2K]
        self.Zone1['A_w']   =  1.352                    # Area of window [m2]
        self.Zone1['iota']  =  0.359444194048431        # Q_solar coefficient in \alpha A_w(\iota T_out + \gamma) [-]
        self.Zone1['gamma'] =  1.622446039686184
        self.Zone1['m']     = 10/60                     # Fixed mass supply air [m3/min]
        self.Zone1['Twss']  = 18                        # Zone 1 walls steady-state temperature [deg C]
        self.Zone1['Tsp']   = 20                        # Zone 1 desired temperature set-point  [deg C]
        self.Zone1['Tz']['sigma'] = 0.02
        self.Zone1['Te']['sigma'] = 0.01
        
        #--------- Zone 2             -------#
        
        self.Zone2 = {'Tz': {}, 'Te': {}}
        
        self.Zone2['Cz']    =  50.2437                 # Thermal capacitance of zone [J/kgK]
        self.Zone2['Cn']    =  53.3759                 # Thermal capacitance of neighbouring wall [J/kgK]
        self.Zone2['Rout']  =  7.4513                  # Resistance of walls connected to outside [K/W]
        self.Zone2['Rn']    =  5.7952                  # Resistance of walls connected to neighbouring wall [K/W]
        self.Zone2['mu']    =  5.805064e-6             # Q_occ coefficient in \muCO_2 + \zeta [-]
        self.Zone2['zeta']  =  -0.003990
        self.Zone2['alpha'] =  0.044                   # Absorptivity coefficient of wall [W/m2K]
        self.Zone2['A_w']   =  10.797                  # Area of window [m2]
        self.Zone2['iota']  =  0.03572                 # Q_solar coefficient in \alpha A_w(\iota T_out + \gamma) [-]
        self.Zone2['gamma'] =  0.06048
        self.Zone2['m']     =  10/60                   # Fixed mass supply air [m3/min]
        self.Zone2['Twss']  =  18                      # Zone 2 walls steady-state temperature [deg C]
        self.Zone2['Tsp']   =  20                      # Zone 2 desired temperature set-point  [deg C]
        self.Zone2['Tz']['sigma'] = .02
        self.Zone2['Te']['sigma'] = .01